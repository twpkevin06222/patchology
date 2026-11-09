"""Threshold-based tissue detection on a low-resolution slide image."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import cv2
import numpy as np

ThresholdMethod = Literal["triangle", "otsu", "otsu_gray", "fixed"]


@dataclass(frozen=True)
class GreyObjectConfig:
    """Thresholds of the grey-object rules of :func:`detect_tissue`.

    Saturation, (max - min) / max, blows up on dark pixels, so black fiducials,
    labels and slide edges pass as tissue; glass and coverslip edges leave light
    grey bands that pass too. Both are grey, so both rules test absolute chroma
    (max - min of RGB). Defaults sit inside gaps measured on the TIGER and ACROBAT
    validation sets (noted per attribute); recalibrate with
    ``patchology.utilities.inspect_tissue``.

    Attributes:
        max_value: Dark rule: pixels darker than this (max of RGB) can be dark
            objects. Also the light-band rule's lower bound on a region's median
            value, so dark grey regions are left to the dark rule.
        max_chroma: Dark rule: pixels with chroma below this can be dark objects
            (artifacts <= ~30, dark H&E tissue >= ~45).
        max_texture: Dark rule: a dark region is flat, so an artifact, when its
            median 3x3 px intensity SD is below this (artifacts <= ~5.5, strongly
            stained IHC tissue >= ~9). Grainy dark regions are still removed when
            they touch the image border or a flat artifact.
        opening_um: Dark rule: opening that drops scattered grey pixels inside
            tissue (pigment, nuclei) before regions are formed.
        band_max_chroma: Light-band rule: a tissue region is unstained grey when
            its 99th-percentile chroma is below this (glass bands <= 16,
            straight tissue strips >= 26).
        straight_min_fraction: Light-band rule: an unstained region is dropped
            when at least this fraction of its outline lies on straight runs
            (glass bands >= 0.53); the image border counts as straight.
        straight_min_segment_um: Length from which an outline run counts as straight.
        straight_tolerance_um: Deviation allowed along a straight run.
    """

    max_value: int = 160
    max_chroma: int = 25
    max_texture: float = 8.0
    opening_um: float = 40.0
    band_max_chroma: int = 20
    straight_min_fraction: float = 0.5
    straight_min_segment_um: float = 300.0
    straight_tolerance_um: float = 16.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not value > 0:
                raise ValueError(f"GreyObjectConfig.{name} must be positive, got {value}")
        if self.straight_min_fraction > 1:
            raise ValueError(
                f"straight_min_fraction must be in (0, 1], got {self.straight_min_fraction}"
            )


@dataclass(frozen=True)
class GreyRegion:
    """A region a grey-object rule measured, as reported by :func:`tissue_regions`.

    Attributes:
        rule: ``"dark"`` (dark grey candidates) or ``"band"`` (every tissue
            region, tested as a light grey straight band).
        bbox: ``(x, y, w, h)`` in thumbnail pixels.
        area_um2: Region area.
        value: Median of max(R, G, B).
        chroma: Median (dark) or 99th percentile (band) of max - min of RGB.
        texture: Dark only: median 3x3 px intensity SD; NaN for bands.
        straight_fraction: Band only: outline fraction on straight runs; NaN
            for dark regions.
        touching: Dark only: touches the image border or a flat artifact.
        removed: The rule made the region background.
    """

    rule: Literal["dark", "band"]
    bbox: tuple[int, int, int, int]
    area_um2: float
    value: float
    chroma: float
    texture: float
    straight_fraction: float
    touching: bool
    removed: bool


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
            fringe as background; also drop unstained grey tissue regions
            outlined mostly by straight lines, such as glass edge bands
            (:func:`straight_grey_objects`).
        grey_objects: Thresholds of those two rules.
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
    grey_objects: GreyObjectConfig = GreyObjectConfig()


def detect_tissue(rgb: np.ndarray, mpp: float, config: TissueConfig = TissueConfig()) -> np.ndarray:
    """Return a boolean tissue mask for an ``(H, W, 3)`` uint8 RGB image.

    Args:
        rgb: Low-resolution image of the slide, background white.
        mpp: Microns per pixel of ``rgb``.
        config: Detection settings.
    """
    return tissue_regions(rgb, mpp, config)[0]


def tissue_regions(
    rgb: np.ndarray, mpp: float, config: TissueConfig = TissueConfig()
) -> tuple[np.ndarray, list[GreyRegion]]:
    """:func:`detect_tissue`, plus every region the grey-object rules measured.

    For calibrating :class:`GreyObjectConfig`; the regions are empty when
    ``remove_dark_objects`` is off.
    """
    regions: list[GreyRegion] = []
    if config.remove_dark_objects:
        dark, regions = _dark_objects(rgb, mpp, config)
        rgb = np.where(dark[..., None], np.uint8(255), rgb)
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
    if config.remove_dark_objects:
        bands, band_regions = _straight_grey_objects(mask, rgb, mpp, config.grey_objects)
        mask &= ~bands
        regions += band_regions
    mask = fill_small_holes(mask, config.max_hole_area_um2 / pixel_area)
    return mask, regions


def dark_objects(rgb: np.ndarray, mpp: float, config: TissueConfig = TissueConfig()) -> np.ndarray:
    """Return a mask of solid dark grey regions in ``rgb``, grown to cover their colour fringe."""
    return _dark_objects(rgb, mpp, config)[0]


def _dark_objects(
    rgb: np.ndarray, mpp: float, config: TissueConfig
) -> tuple[np.ndarray, list[GreyRegion]]:
    grey = config.grey_objects
    pixels = rgb.astype(np.int16)
    value = pixels.max(axis=2)
    chroma = value - pixels.min(axis=2)
    dark = ((value < grey.max_value) & (chroma < grey.max_chroma)).astype(np.uint8)
    # Opening drops scattered grey pixels inside tissue, such as pigment and nuclei.
    open_px = _odd(grey.opening_um / mpp)
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
    medians = {}
    for i in np.flatnonzero(big):
        x, y, w, h = stats[i, :4]
        region = labels[y : y + h, x : x + w] == i
        box = (slice(y, y + h), slice(x, x + w))
        medians[i] = [np.median(a[box][region]) for a in (value, chroma, texture)]
    flat = np.zeros(count, bool)
    for i, (_, _, region_texture) in medians.items():
        flat[i] = region_texture < grey.max_texture

    # Grainy slide edges lie along the image border or beside a flat artifact.
    fringe_px = 2 * _odd(config.smoothing_um / mpp) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fringe_px, fringe_px))
    seeds = cv2.dilate(flat[labels].astype(np.uint8), kernel)
    seeds[[0, -1], :] = seeds[:, [0, -1]] = 1
    touching = np.zeros(count, bool)
    touching[labels[seeds > 0]] = True
    removed = big & (flat | touching)
    regions = [
        GreyRegion(
            rule="dark",
            bbox=tuple(int(n) for n in stats[i, :4]),
            area_um2=float(stats[i, cv2.CC_STAT_AREA] * mpp * mpp),
            value=float(region_value),
            chroma=float(region_chroma),
            texture=float(region_texture),
            straight_fraction=float("nan"),
            touching=bool(touching[i]),
            removed=bool(removed[i]),
        )
        for i, (region_value, region_chroma, region_texture) in medians.items()
    ]
    return cv2.dilate(removed[labels].astype(np.uint8), kernel).astype(bool), regions


def straight_grey_objects(
    mask: np.ndarray, rgb: np.ndarray, mpp: float, config: GreyObjectConfig = GreyObjectConfig()
) -> np.ndarray:
    """Return the components of ``mask`` that are light grey (no stain) and whose outline
    runs mostly along long straight lines, such as bands along glass or coverslip edges."""
    return _straight_grey_objects(mask, rgb, mpp, config)[0]


def _straight_grey_objects(
    mask: np.ndarray, rgb: np.ndarray, mpp: float, config: GreyObjectConfig
) -> tuple[np.ndarray, list[GreyRegion]]:
    mask = mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    pixels = rgb.astype(np.int16)
    value = pixels.max(axis=2)
    chroma = value - pixels.min(axis=2)
    drop = np.zeros(count, bool)
    regions = []
    for i in range(1, count):
        x, y, w, h = stats[i, :4]
        region = labels[y : y + h, x : x + w] == i
        box = (slice(y, y + h), slice(x, x + w))
        region_chroma = np.percentile(chroma[box][region], 99)
        region_value = np.median(value[box][region])
        straight = _straight_fraction(region, mpp, config)
        # Dark grey is left to the dark rule, whose texture test keeps strong IHC stain.
        grey = region_chroma < config.band_max_chroma and region_value >= config.max_value
        drop[i] = grey and straight >= config.straight_min_fraction
        regions.append(
            GreyRegion(
                rule="band",
                bbox=(int(x), int(y), int(w), int(h)),
                area_um2=float(stats[i, cv2.CC_STAT_AREA] * mpp * mpp),
                value=float(region_value),
                chroma=float(region_chroma),
                texture=float("nan"),
                straight_fraction=float(straight),
                touching=False,
                removed=bool(drop[i]),
            )
        )
    return drop[labels], regions


def _straight_fraction(region: np.ndarray, mpp: float, config: GreyObjectConfig) -> float:
    """Fraction of the outline of ``region`` on straight runs of ``straight_min_segment_um``."""
    # Padding closes the outline along the image border, which counts as straight.
    padded = np.pad(region, 1).astype(np.uint8)
    contours, _ = cv2.findContours(padded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=len)
    corners = cv2.approxPolyDP(contour, config.straight_tolerance_um / mpp, True)
    corners = corners[:, 0].astype(float)
    sides = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
    return sides[sides >= config.straight_min_segment_um / mpp].sum() / cv2.arcLength(contour, True)


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
