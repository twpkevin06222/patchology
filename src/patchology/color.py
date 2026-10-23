"""Colour calibration (embedded ICC profile to sRGB) and Macenko stain normalisation."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageCms

# Reference H&E optical density vectors (columns: hematoxylin, eosin) and 99th
# percentile stain concentrations, from the reference Macenko implementation.
REFERENCE_STAIN_MATRIX = np.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]])
REFERENCE_MAX_CONCENTRATIONS = np.array([1.9705, 1.0308])

# cmsFLAGS_NOCACHE (ImageCms.Flags.NOCACHE in newer Pillow).
_LCMS_NOCACHE = 0x0040

# Fewer stained pixels than this cannot give a stable stain estimate.
MIN_STAINED_PIXELS = 100

StainMethod = Literal["macenko"]


def icc_to_srgb(profile: bytes) -> ImageCms.ImageCmsTransform:
    """Build a perceptual transform from an embedded ICC ``profile`` to sRGB.

    Raises:
        ImageCms.PyCMSError: ``profile`` is not a valid RGB ICC profile.
    """
    source = ImageCms.ImageCmsProfile(BytesIO(profile))
    # No 1-pixel cache, so threads reading tiles can share one transform safely.
    return ImageCms.buildTransform(
        source, ImageCms.createProfile("sRGB"), "RGB", "RGB", flags=_LCMS_NOCACHE
    )


def apply_transform(rgb: np.ndarray, transform: ImageCms.ImageCmsTransform) -> np.ndarray:
    """Apply an ICC ``transform`` to an ``(H, W, 3)`` uint8 RGB array."""
    return np.asarray(ImageCms.applyTransform(Image.fromarray(rgb), transform))


@dataclass(frozen=True)
class Stains:
    """H&E stain vectors and their 99th percentile concentrations.

    Attributes:
        matrix: ``(3, 2)`` unit optical density vectors, hematoxylin then eosin.
        max_concentrations: ``(2,)`` 99th percentile concentration of each stain.
    """

    matrix: np.ndarray
    max_concentrations: np.ndarray

    @classmethod
    def estimate(
        cls,
        rgb: np.ndarray,
        alpha: float = 1.0,
        beta: float = 0.15,
        background: float = 240.0,
    ) -> Stains:
        """Estimate stains from RGB pixels with the Macenko method.

        Args:
            rgb: ``(..., 3)`` uint8 pixels, e.g. one image or stacked tiles.
            alpha: Percentile of the angle distribution taken as each stain's
                extreme, which makes the estimate robust to outliers.
            beta: Pixels with optical density below this in any channel are
                treated as background and ignored.
            background: Transmitted light intensity (Io).

        Raises:
            ValueError: Too few stained pixels to estimate the stains.
        """
        pixels = np.asarray(rgb).reshape(-1, 3)
        # Clipped (0) channels have unknown OD; on dark nuclei they inflate the
        # 99th percentile concentration and wash out normalised tiles.
        od = _optical_density(pixels[(pixels > 0).all(axis=1)], background)
        stained = od[(od > beta).all(axis=1)]
        if len(stained) < MIN_STAINED_PIXELS:
            raise ValueError(
                f"only {len(stained)} stained pixels (need {MIN_STAINED_PIXELS}) to estimate stains"
            )
        # The two largest principal directions span the H&E plane.
        _, eigvecs = np.linalg.eigh(np.cov(stained.T))
        plane = eigvecs[:, 1:3]
        # Point the main direction into positive OD so angles do not wrap at ±pi.
        if plane[:, 1].sum() < 0:
            plane[:, 1] *= -1
        projected = stained @ plane
        angles = np.arctan2(projected[:, 1], projected[:, 0])
        lo, hi = np.percentile(angles, [alpha, 100 - alpha])
        v_lo = plane @ np.array([np.cos(lo), np.sin(lo)])
        v_hi = plane @ np.array([np.cos(hi), np.sin(hi)])
        # Hematoxylin absorbs more red light than eosin.
        h, e = (v_lo, v_hi) if v_lo[0] > v_hi[0] else (v_hi, v_lo)
        matrix = np.stack([h, e], axis=1)
        matrix /= np.linalg.norm(matrix, axis=0)

        concentrations = np.linalg.lstsq(matrix, od.T, rcond=None)[0]
        return cls(matrix, np.percentile(concentrations, 99, axis=1))

    def to_dict(self) -> dict:
        return {
            "matrix": self.matrix.round(6).tolist(),
            "max_concentrations": self.max_concentrations.round(6).tolist(),
        }


REFERENCE_STAINS = Stains(REFERENCE_STAIN_MATRIX, REFERENCE_MAX_CONCENTRATIONS)

type StainSource = Stains | np.ndarray | Image.Image | str | Path


def load_stains(source: StainSource) -> Stains:
    """Return ``source`` as :class:`Stains`, estimating them from an RGB image or path.

    Raises:
        FileNotFoundError: ``source`` is a path that does not exist.
        ValueError: The image cannot be read or has too few stained pixels.
    """
    if isinstance(source, Stains):
        return source
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"stain reference not found: {path}")
        try:
            with Image.open(path) as image:
                source = image.convert("RGB")
        except OSError as e:
            raise ValueError(f"cannot read stain reference {path}: {e}") from e
    if isinstance(source, Image.Image):
        source = source.convert("RGB")
    rgb = np.asarray(source)
    if rgb.ndim < 2 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"stain reference must be uint8 RGB, got {rgb.dtype} {rgb.shape}")
    try:
        return Stains.estimate(rgb)
    except ValueError as e:
        raise ValueError(f"cannot estimate stains from reference: {e}") from e


def macenko_normalize(
    rgb: np.ndarray,
    source: Stains,
    target: Stains = REFERENCE_STAINS,
    background: float = 240.0,
) -> np.ndarray:
    """Re-render ``rgb`` from ``source`` stains with ``target`` stains (Macenko).

    Returns an array of the same shape as ``rgb``, uint8.
    """
    # uint8 channels have 256 possible ODs: look them up instead of a log per pixel.
    od = -np.log((np.arange(256) + 1) / background)[np.asarray(rgb).reshape(-1, 3)]
    # The pseudo-inverse of the (3, 2) stain matrix gives the same least-squares
    # concentrations as solving per tile, folded with the rescaling into one matrix.
    scale = target.max_concentrations / source.max_concentrations
    concentrations = od @ (np.linalg.pinv(source.matrix).T * scale)
    out = background * np.exp(-(concentrations @ target.matrix.T))
    return np.clip(out, 0, 255).round().astype(np.uint8).reshape(rgb.shape)


@dataclass(frozen=True)
class ColorConfig:
    """Colour processing applied to tiles, in this order.

    Attributes:
        icc: Convert pixels from the slide's embedded ICC profile to sRGB.
            Slides without a profile are left unchanged. Also applied to the
            detection thumbnail, so masks and stain estimates see the same
            colours as the tiles.
        stain_normalization: ``"macenko"`` to normalise H&E stains, or None.
        stain_target: What to normalise to: :class:`Stains`, or a reference
            RGB image (``(..., 3)`` uint8 array, e.g. a single pixel or a
            tile; PIL image; or image path) whose stains are estimated.
            Defaults to :data:`REFERENCE_STAINS`; pass your own reference
            image to match your data.
        stain_sample_tiles: Number of selected tiles, spread evenly over the
            slide, from which the slide's own stains are estimated.
    """

    icc: bool = True
    stain_normalization: StainMethod | None = None
    stain_target: StainSource = REFERENCE_STAINS
    stain_sample_tiles: int = 64

    def __post_init__(self) -> None:
        if self.stain_normalization not in (None, "macenko"):
            raise ValueError(
                f"stain_normalization must be 'macenko' or None, got {self.stain_normalization!r}"
            )
        if self.stain_sample_tiles <= 0:
            raise ValueError(f"stain_sample_tiles must be positive, got {self.stain_sample_tiles}")

    @functools.cached_property
    def target_stains(self) -> Stains:
        """``stain_target`` as :class:`Stains`, estimated on first use."""
        return load_stains(self.stain_target)


def _optical_density(rgb: np.ndarray, background: float) -> np.ndarray:
    pixels = np.asarray(rgb, dtype=np.float64).reshape(-1, 3)
    return -np.log((pixels + 1) / background)
