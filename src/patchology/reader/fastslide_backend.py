"""FastSlide backend, adapted to the OpenSlide conventions.

Differences from FastSlide's native API that this adapter hides:

* ``read_region`` takes level-0 coordinates (FastSlide uses level-native ones).
  A level-0 location that falls between pixels of a downsampled level is
  snapped down to the pixel grid; OpenSlide interpolates instead, so the two
  backends can differ by up to one level pixel at levels other than 0.
* Regions extending past the slide edge are padded with transparent pixels
  instead of being clipped, and negative coordinates are allowed.
* Results are RGBA :class:`PIL.Image.Image` objects.
* ``properties`` exposes the ``openslide.*`` keys; FastSlide's own metadata is
  kept under ``fastslide.*``.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from pathlib import Path

import fastslide
import numpy as np
from PIL import Image

from patchology.reader.base import (
    PROPERTY_NAME_BOUNDS_HEIGHT,
    PROPERTY_NAME_BOUNDS_WIDTH,
    PROPERTY_NAME_BOUNDS_X,
    PROPERTY_NAME_BOUNDS_Y,
    PROPERTY_NAME_MPP_X,
    PROPERTY_NAME_MPP_Y,
    PROPERTY_NAME_OBJECTIVE_POWER,
    PROPERTY_NAME_QUICKHASH1,
    PROPERTY_NAME_VENDOR,
    SlideOpenError,
    SlideReader,
    level_properties,
)

# Per-slide LRU cache of decoded tiles, as OpenSlide has by default: neighbouring
# tiles share source tiles (~35% hits without overlap, ~75% at 50%). 64 MB is as
# fast as larger caches; FastSlide's global cache would hold up to 1 GB.
TILE_CACHE_BYTES = 64 << 20


class FastSlideReader(SlideReader):
    backend = "fastslide"

    def __init__(self, path: str | Path) -> None:
        super().__init__(path)
        try:
            slide = fastslide.FastSlide.from_file_path(str(self._path), cache=TILE_CACHE_BYTES)
        except RuntimeError as e:
            raise SlideOpenError(f"FastSlide cannot open {self._path}: {e}") from e
        # FastSlide may hand back an unusable object for files it fails to
        # parse instead of raising, so probe it before accepting it.
        try:
            level_count = slide.level_count
            _ = slide.dimensions
            dtype = slide.dtype
            channels = slide.num_channels
        except RuntimeError as e:
            slide.close()
            raise SlideOpenError(f"FastSlide cannot open {self._path}: {e}") from e
        if level_count < 1 or dtype != "uint8" or channels not in (1, 3, 4):
            slide.close()
            raise SlideOpenError(
                f"FastSlide opened {self._path} but it is not an 8-bit "
                f"brightfield image (dtype={dtype}, channels={channels})"
            )
        self._slide = slide
        self._level_dimensions = tuple(tuple(d) for d in slide.level_dimensions)
        self._level_downsamples = tuple(float(d) for d in slide.level_downsamples)
        self._properties = self._build_properties()
        self._associated = _AssociatedImages(slide.associated_images)

    @property
    def level_count(self) -> int:
        return len(self._level_dimensions)

    @property
    def level_dimensions(self) -> tuple[tuple[int, int], ...]:
        return self._level_dimensions

    @property
    def level_downsamples(self) -> tuple[float, ...]:
        return self._level_downsamples

    @property
    def properties(self) -> Mapping[str, str]:
        return self._properties

    @property
    def associated_images(self) -> Mapping[str, Image.Image]:
        return self._associated

    @property
    def icc_profile(self) -> bytes | None:
        return self._slide.icc_profile or None

    def read_region(
        self, location: tuple[int, int], level: int, size: tuple[int, int]
    ) -> Image.Image:
        return Image.fromarray(self._read_rgba(location, level, size), "RGBA")

    def read_region_rgb(
        self, location: tuple[int, int], level: int, size: tuple[int, int]
    ) -> np.ndarray:
        # Skip the PIL round trip; only out-of-slide pixels need compositing.
        rgba = self._read_rgba(location, level, size)
        rgb = rgba[..., :3]
        outside = rgba[..., 3] == 0
        if outside.any():
            rgb = rgb.copy()
            rgb[outside] = self._background_rgb()
        return np.ascontiguousarray(rgb)

    def close(self) -> None:
        self._slide.close()

    def _read_rgba(
        self, location: tuple[int, int], level: int, size: tuple[int, int]
    ) -> np.ndarray:
        if not 0 <= level < self.level_count:
            raise ValueError(f"level {level} out of range [0, {self.level_count})")
        width, height = size
        downsample = self._level_downsamples[level]
        level_w, level_h = self._level_dimensions[level]
        lx = math.floor(location[0] / downsample)
        ly = math.floor(location[1] / downsample)

        out = np.zeros((height, width, 4), dtype=np.uint8)
        x0, y0 = max(lx, 0), max(ly, 0)
        x1, y1 = min(lx + width, level_w), min(ly + height, level_h)
        if x1 <= x0 or y1 <= y0:
            return out
        image = self._slide.read_region((x0, y0), level, (x1 - x0, y1 - y0))
        pixels = _to_rgba(image.to_interleaved().numpy())
        ph, pw = pixels.shape[:2]
        out[y0 - ly : y0 - ly + ph, x0 - lx : x0 - lx + pw] = pixels
        return out

    def _build_properties(self) -> dict[str, str]:
        slide = self._slide
        native = dict(slide.properties)
        props = {f"fastslide.{k}": str(v) for k, v in native.items()}
        props[PROPERTY_NAME_VENDOR] = str(slide.format).lower()
        props.update(level_properties(self._level_dimensions, self._level_downsamples))

        mpp_x, mpp_y = slide.mpp
        if mpp_x > 0 and mpp_y > 0:
            props[PROPERTY_NAME_MPP_X] = repr(float(mpp_x))
            props[PROPERTY_NAME_MPP_Y] = repr(float(mpp_y))
        objective = native.get("objective_magnification")
        if objective:
            props[PROPERTY_NAME_OBJECTIVE_POWER] = f"{float(objective):g}"
        quickhash = slide.quickhash
        if quickhash:
            props[PROPERTY_NAME_QUICKHASH1] = quickhash

        (bx, by), (bw, bh) = slide.bounds
        if (bx, by, bw, bh) != (0, 0, *self._level_dimensions[0]):
            props[PROPERTY_NAME_BOUNDS_X] = str(bx)
            props[PROPERTY_NAME_BOUNDS_Y] = str(by)
            props[PROPERTY_NAME_BOUNDS_WIDTH] = str(bw)
            props[PROPERTY_NAME_BOUNDS_HEIGHT] = str(bh)
        return props


class _AssociatedImages(Mapping[str, Image.Image]):
    """Lazy mapping of FastSlide associated images to RGBA PIL images."""

    def __init__(self, native: fastslide.AssociatedImages) -> None:
        self._native = native
        self._keys = tuple(native.keys())

    def __getitem__(self, key: str) -> Image.Image:
        if key not in self._keys:
            raise KeyError(key)
        pixels = self._native[key].to_interleaved().numpy()
        return Image.fromarray(_to_rgba(pixels), "RGBA")

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)


def _to_rgba(pixels: np.ndarray) -> np.ndarray:
    if pixels.ndim == 2:
        pixels = pixels[..., None]
    channels = pixels.shape[2]
    if channels == 4:
        return pixels
    if channels == 1:
        pixels = np.repeat(pixels, 3, axis=2)
    elif channels != 3:
        raise ValueError(f"unsupported channel count {channels}")
    alpha = np.full(pixels.shape[:2] + (1,), 255, dtype=np.uint8)
    return np.concatenate([pixels, alpha], axis=2)
