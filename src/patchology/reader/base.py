"""Backend-agnostic slide reader that follows the OpenSlide object conventions.

Every backend exposes the same surface as :class:`openslide.OpenSlide`:

* ``dimensions``, ``level_count``, ``level_dimensions``, ``level_downsamples``
* ``properties`` keyed with the ``openslide.*`` property names
* ``associated_images`` as a mapping of name to RGBA :class:`PIL.Image.Image`
* ``read_region(location, level, size)`` where ``location`` is in level-0
  pixels and the result is an RGBA image, transparent outside the slide
* ``get_thumbnail(size)`` and ``get_best_level_for_downsample(downsample)``

On top of that, a few numpy helpers used by the tiling pipeline are provided.
"""

from __future__ import annotations

import abc
import functools
import logging
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from typing import Self

import cv2
import numpy as np
from PIL import Image, ImageCms

from patchology.color import apply_transform, icc_to_srgb

logger = logging.getLogger(__name__)

# Property names, identical to the constants exported by openslide-python.
PROPERTY_NAME_COMMENT = "openslide.comment"
PROPERTY_NAME_VENDOR = "openslide.vendor"
PROPERTY_NAME_QUICKHASH1 = "openslide.quickhash-1"
PROPERTY_NAME_BACKGROUND_COLOR = "openslide.background-color"
PROPERTY_NAME_OBJECTIVE_POWER = "openslide.objective-power"
PROPERTY_NAME_MPP_X = "openslide.mpp-x"
PROPERTY_NAME_MPP_Y = "openslide.mpp-y"
PROPERTY_NAME_BOUNDS_X = "openslide.bounds-x"
PROPERTY_NAME_BOUNDS_Y = "openslide.bounds-y"
PROPERTY_NAME_BOUNDS_WIDTH = "openslide.bounds-width"
PROPERTY_NAME_BOUNDS_HEIGHT = "openslide.bounds-height"

# Relative slack when matching a requested downsample to a pyramid level, so
# that e.g. a level with downsample 4.0002 is used for a request of 4.0.
LEVEL_DOWNSAMPLE_TOLERANCE = 0.01


class SlideOpenError(RuntimeError):
    """Raised when no backend could open a slide."""


class SlideReader(abc.ABC):
    """Abstract whole slide image reader with an OpenSlide-compatible API."""

    #: Short backend identifier, e.g. ``"openslide"`` or ``"fastslide"``.
    backend: str

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    # --- OpenSlide API -------------------------------------------------------

    @property
    @abc.abstractmethod
    def level_count(self) -> int:
        """Number of levels in the slide pyramid."""

    @property
    @abc.abstractmethod
    def level_dimensions(self) -> tuple[tuple[int, int], ...]:
        """``(width, height)`` of each level, level 0 first."""

    @property
    @abc.abstractmethod
    def level_downsamples(self) -> tuple[float, ...]:
        """Downsample factor of each level relative to level 0."""

    @property
    @abc.abstractmethod
    def properties(self) -> Mapping[str, str]:
        """Slide metadata, using ``openslide.*`` keys where applicable."""

    @property
    @abc.abstractmethod
    def associated_images(self) -> Mapping[str, Image.Image]:
        """Label, macro, thumbnail etc. as RGBA images."""

    @abc.abstractmethod
    def read_region(
        self, location: tuple[int, int], level: int, size: tuple[int, int]
    ) -> Image.Image:
        """Read an RGBA region.

        ``location`` is the top-left corner in level-0 pixels, ``size`` is in
        pixels of ``level``. Pixels outside the slide are fully transparent.
        """

    @abc.abstractmethod
    def close(self) -> None:
        """Release the underlying slide handle."""

    @property
    def dimensions(self) -> tuple[int, int]:
        """``(width, height)`` of level 0."""
        return self.level_dimensions[0]

    def get_best_level_for_downsample(self, downsample: float) -> int:
        """Return the highest-resolution level whose downsample is <= ``downsample``.

        Mirrors ``openslide_get_best_level_for_downsample`` exactly.
        """
        downsamples = self.level_downsamples
        if downsample < downsamples[0]:
            return 0
        for i in range(1, self.level_count):
            if downsample < downsamples[i]:
                return i - 1
        return self.level_count - 1

    def get_thumbnail(self, size: tuple[int, int]) -> Image.Image:
        """Return an RGB thumbnail that fits within ``size``, keeping aspect ratio."""
        downsample = max(dim / thumb for dim, thumb in zip(self.dimensions, size))
        level = self.get_best_level_for_downsample(downsample)
        tile = self.read_region((0, 0), level, self.level_dimensions[level])
        thumb = Image.new("RGB", tile.size, self._background_rgb())
        thumb.paste(tile, None, tile)
        thumb.thumbnail(size, Image.Resampling.LANCZOS)
        return thumb

    # --- Extras --------------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    @property
    def mpp(self) -> tuple[float, float] | None:
        """``(mpp_x, mpp_y)`` in microns per level-0 pixel, if known."""
        try:
            mpp_x = float(self.properties[PROPERTY_NAME_MPP_X])
            mpp_y = float(self.properties[PROPERTY_NAME_MPP_Y])
        except (KeyError, ValueError):
            return None
        if mpp_x <= 0 or mpp_y <= 0:
            return None
        return mpp_x, mpp_y

    @property
    def icc_profile(self) -> bytes | None:
        """Embedded ICC colour profile of the slide, or None if it has none."""
        return None

    def apply_icc(self, rgb: np.ndarray) -> np.ndarray:
        """Convert ``(H, W, 3)`` uint8 pixels from the slide's ICC profile to sRGB.

        Returns ``rgb`` unchanged if the slide has no usable profile.
        """
        transform = self._icc_transform
        return rgb if transform is None else apply_transform(rgb, transform)

    @functools.cached_property
    def _icc_transform(self) -> ImageCms.ImageCmsTransform | None:
        profile = self.icc_profile
        if profile is None:
            logger.info("%s has no ICC profile; colours are left as stored", self._path.name)
            return None
        try:
            transform = icc_to_srgb(profile)
        except (ImageCms.PyCMSError, OSError) as e:
            logger.warning(
                "%s has an unusable ICC profile, skipping calibration: %s", self._path, e
            )
            return None
        logger.info(
            "%s: applying embedded ICC profile (%d bytes) -> sRGB", self._path.name, len(profile)
        )
        return transform

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """``(x, y, width, height)`` of the non-empty area in level-0 pixels.

        Falls back to the full slide when the format does not report bounds.
        """
        props = self.properties
        try:
            return (
                int(props[PROPERTY_NAME_BOUNDS_X]),
                int(props[PROPERTY_NAME_BOUNDS_Y]),
                int(props[PROPERTY_NAME_BOUNDS_WIDTH]),
                int(props[PROPERTY_NAME_BOUNDS_HEIGHT]),
            )
        except (KeyError, ValueError):
            return (0, 0, *self.dimensions)

    def read_region_rgb(
        self, location: tuple[int, int], level: int, size: tuple[int, int]
    ) -> np.ndarray:
        """Like :meth:`read_region` but returns an ``(H, W, 3)`` uint8 array.

        Transparent pixels are composited onto the slide background colour.
        """
        rgba = np.asarray(self.read_region(location, level, size))
        return _composite(rgba, self._background_rgb())

    def read_region_scaled(
        self, location: tuple[int, int], downsample: float, size: tuple[int, int]
    ) -> np.ndarray:
        """Read ``size`` output pixels at an arbitrary ``downsample`` from level 0.

        The closest pyramid level at or above the requested resolution is read
        and resized, so ``downsample`` need not match a level exactly.
        """
        level = self._level_for_downsample(downsample)
        level_downsample = self.level_downsamples[level]
        factor = downsample / level_downsample
        read_size = (max(1, round(size[0] * factor)), max(1, round(size[1] * factor)))
        rgb = self.read_region_rgb(location, level, read_size)
        if read_size != tuple(size):
            interpolation = cv2.INTER_AREA if factor > 1 else cv2.INTER_LINEAR
            rgb = cv2.resize(rgb, tuple(size), interpolation=interpolation)
        return rgb

    def read_at_downsample(self, downsample: float, chunk_size: int = 2048) -> np.ndarray:
        """Read the whole slide at ``downsample`` as an ``(H, W, 3)`` uint8 array.

        The image is assembled from chunks so memory stays bounded even when
        the closest pyramid level is much larger than the output.
        """
        width, height = self.dimensions
        out_w = max(1, round(width / downsample))
        out_h = max(1, round(height / downsample))
        out = np.empty((out_h, out_w, 3), dtype=np.uint8)
        for oy in range(0, out_h, chunk_size):
            for ox in range(0, out_w, chunk_size):
                cw = min(chunk_size, out_w - ox)
                ch = min(chunk_size, out_h - oy)
                location = (round(ox * downsample), round(oy * downsample))
                out[oy : oy + ch, ox : ox + cw] = self.read_region_scaled(
                    location, downsample, (cw, ch)
                )
        return out

    def _level_for_downsample(self, downsample: float) -> int:
        return self.get_best_level_for_downsample(downsample * (1 + LEVEL_DOWNSAMPLE_TOLERANCE))

    def _background_rgb(self) -> tuple[int, int, int]:
        value = self.properties.get(PROPERTY_NAME_BACKGROUND_COLOR, "ffffff")
        try:
            return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))
        except (ValueError, IndexError):
            return (255, 255, 255)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({str(self._path)!r})"


def level_properties(
    level_dimensions: tuple[tuple[int, int], ...], level_downsamples: tuple[float, ...]
) -> dict[str, str]:
    """Build the ``openslide.level-count`` / ``openslide.level[i].*`` properties."""
    props = {"openslide.level-count": str(len(level_dimensions))}
    for i, ((w, h), ds) in enumerate(zip(level_dimensions, level_downsamples)):
        props[f"openslide.level[{i}].width"] = str(w)
        props[f"openslide.level[{i}].height"] = str(h)
        props[f"openslide.level[{i}].downsample"] = repr(float(ds))
    return props


def _composite(rgba: np.ndarray, background: tuple[int, int, int]) -> np.ndarray:
    rgb = rgba[..., :3]
    alpha = rgba[..., 3]
    if alpha.min() == 255:
        return np.ascontiguousarray(rgb)
    a = alpha[..., None].astype(np.float32) / 255.0
    bg = np.asarray(background, dtype=np.float32)
    return (rgb * a + bg * (1.0 - a)).round().astype(np.uint8)
