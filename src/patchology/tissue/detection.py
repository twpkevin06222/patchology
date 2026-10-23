"""Threshold-based tissue detection on a low-resolution slide image."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

ThresholdMethod = Literal["triangle", "otsu", "otsu_gray", "fixed"]


@dataclass(frozen=True)
class TissueConfig:
    """Settings for :func:`detect_tissue`.

    Sizes are physical (microns) so results do not depend on ``thumbnail_mpp``.

    Attributes:
        method: How to threshold HSV saturation, where stained tissue is
            saturated and glass is not. ``"triangle"`` suits slides that are
            mostly background, which is typical, and keeps faint tissue such
            as fat. ``"otsu"`` splits saturation into two classes; on slides
            with dense tissue it can split dense from faint tissue instead of
            tissue from glass. ``"otsu_gray"`` applies Otsu to inverted
            grayscale, for weakly stained slides. ``"fixed"`` thresholds
            saturation at ``fixed_threshold``.
        thumbnail_mpp: Resolution (microns per pixel) at which the slide is
            read for detection. The resulting mask is also used to filter
            tiles, so it bounds how precisely tissue fraction is measured.
        fixed_threshold: Saturation cut-off in [0, 255] for ``"fixed"``.
        min_saturation: Lower bound on the threshold. Saturation of
            near-white glass is noise (a one-level change in one RGB channel
            moves it by ~1) and differs between decoders, so automatic
            thresholds below ~6 track that noise rather than tissue.
        smoothing_um: Median filter size applied before thresholding.
        closing_um: Morphological closing that joins nearby tissue fragments.
        min_tissue_area_um2: Connected regions smaller than this are dropped.
        max_hole_area_um2: Holes inside tissue smaller than this are filled.
    """

    method: ThresholdMethod = "triangle"
    thumbnail_mpp: float = 8.0
    fixed_threshold: int = 20
    min_saturation: int = 7
    smoothing_um: float = 56.0
    closing_um: float = 128.0
    min_tissue_area_um2: float = 50_000.0
    max_hole_area_um2: float = 50_000.0


def detect_tissue(rgb: np.ndarray, mpp: float, config: TissueConfig = TissueConfig()) -> np.ndarray:
    """Return a boolean tissue mask for an ``(H, W, 3)`` uint8 RGB image.

    Args:
        rgb: Low-resolution image of the slide, background white.
        mpp: Microns per pixel of ``rgb``.
        config: Detection settings.
    """
    if config.method == "otsu_gray":
        signal = 255 - cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    elif config.method in ("triangle", "otsu", "fixed"):
        signal = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1]
    else:
        raise ValueError(f"unknown threshold method {config.method!r}")
    smooth_px = _odd(config.smoothing_um / mpp)
    if smooth_px > 1:
        signal = cv2.medianBlur(signal, smooth_px)

    if config.method == "fixed":
        threshold = config.fixed_threshold
    else:
        flag = cv2.THRESH_TRIANGLE if config.method == "triangle" else cv2.THRESH_OTSU
        threshold, _ = cv2.threshold(signal, 0, 255, cv2.THRESH_BINARY + flag)
    if config.method != "otsu_gray":
        threshold = max(threshold, config.min_saturation)
    mask = signal > threshold

    close_px = _odd(config.closing_um / mpp)
    if close_px > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)

    pixel_area = mpp * mpp
    mask = remove_small_objects(mask, config.min_tissue_area_um2 / pixel_area)
    mask = fill_small_holes(mask, config.max_hole_area_um2 / pixel_area)
    return mask


def remove_small_objects(mask: np.ndarray, min_area_px: float) -> np.ndarray:
    """Drop 8-connected foreground components smaller than ``min_area_px``."""
    if min_area_px <= 1 or not mask.any():
        return mask
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area_px
    keep[0] = False
    return keep[labels]


def fill_small_holes(mask: np.ndarray, max_area_px: float) -> np.ndarray:
    """Fill background components enclosed by tissue that are smaller than ``max_area_px``."""
    if max_area_px <= 0:
        return mask
    holes = ~mask
    _, labels, stats, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), connectivity=4)
    fill = stats[:, cv2.CC_STAT_AREA] < max_area_px
    # Background connected to the image border is not a hole.
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    fill[border] = False
    fill[0] = False
    return mask | fill[labels]


def _odd(size: float) -> int:
    """Round a kernel size to the nearest odd integer >= 1."""
    n = max(1, round(size))
    return n if n % 2 else n + 1
