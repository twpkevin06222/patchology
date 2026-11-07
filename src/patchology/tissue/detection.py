"""Threshold-based tissue detection on a low-resolution slide image."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

ThresholdMethod = Literal["triangle", "otsu", "otsu_gray", "fixed"]

# Black fiducials and dark slide edges are grey, but HSV saturation, (max - min) / max,
# blows up on dark pixels and passes them as tissue. Absolute chroma (max - min) stays
# low on them (<= ~30) and high on dark H&E tissue (>= ~45 on the TIGER val set). Strong
# IHC stain can be dark grey too, so grey regions also need to be flat (3x3 px intensity
# SD: <= ~5.5 on artifacts, >= ~9 on IHC tissue in ACROBAT val) or touch the border.
DARK_MAX_VALUE = 160
DARK_MAX_CHROMA = 25
DARK_MAX_TEXTURE = 8.0
DARK_OPENING_UM = 40.0


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
        remove_dark_objects: Treat dark grey regions of at least
            ``min_tissue_area_um2`` that are flat or touch the image border,
            such as fiducial markers, labels and slide edges, and their colour
            fringe as background.
    """

    method: ThresholdMethod = "triangle"
    thumbnail_mpp: float = 8.0
    fixed_threshold: int = 20
    min_saturation: int = 7
    smoothing_um: float = 56.0
    closing_um: float = 128.0
    min_tissue_area_um2: float = 50_000.0
    max_hole_area_um2: float = 50_000.0
    remove_dark_objects: bool = True


def detect_tissue(rgb: np.ndarray, mpp: float, config: TissueConfig = TissueConfig()) -> np.ndarray:
    """Return a boolean tissue mask for an ``(H, W, 3)`` uint8 RGB image.

    Args:
        rgb: Low-resolution image of the slide, background white.
        mpp: Microns per pixel of ``rgb``.
        config: Detection settings.
    """
    if config.remove_dark_objects:
        rgb = np.where(dark_objects(rgb, mpp, config)[..., None], np.uint8(255), rgb)
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


def dark_objects(rgb: np.ndarray, mpp: float, config: TissueConfig = TissueConfig()) -> np.ndarray:
    """Return a mask of solid dark grey regions in ``rgb``, grown to cover their colour fringe."""
    pixels = rgb.astype(np.int16)
    value = pixels.max(axis=2)
    dark = ((value < DARK_MAX_VALUE) & (value - pixels.min(axis=2) < DARK_MAX_CHROMA)).astype(
        np.uint8
    )
    # Opening drops scattered grey pixels inside tissue, such as pigment and nuclei.
    open_px = _odd(DARK_OPENING_UM / mpp)
    if open_px > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px))
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    big = stats[:, cv2.CC_STAT_AREA] >= config.min_tissue_area_um2 / (mpp * mpp)
    big[0] = False

    # Markers, clips and labels are flat; strongly stained IHC tissue is grainy.
    v = value.astype(np.float32)
    mean = cv2.blur(v, (3, 3))
    texture = np.sqrt(np.maximum(cv2.blur(v * v, (3, 3)) - mean * mean, 0))
    flat = np.zeros(count, bool)
    for i in np.flatnonzero(big):
        x, y, w, h = stats[i, :4]
        region = labels[y : y + h, x : x + w] == i
        flat[i] = np.median(texture[y : y + h, x : x + w][region]) < DARK_MAX_TEXTURE

    # Grainy slide edges lie along the image border or beside a flat artifact.
    fringe_px = 2 * _odd(config.smoothing_um / mpp) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fringe_px, fringe_px))
    seeds = cv2.dilate(flat[labels].astype(np.uint8), kernel)
    seeds[[0, -1], :] = seeds[:, [0, -1]] = 1
    touching = np.zeros(count, bool)
    touching[labels[seeds > 0]] = True
    artifacts = (big & (flat | touching))[labels]
    return cv2.dilate(artifacts.astype(np.uint8), kernel).astype(bool)


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
