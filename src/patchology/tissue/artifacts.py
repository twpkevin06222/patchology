"""Artifact detection: pen markings and out-of-focus (blurred) regions."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from patchology.reader.base import SlideReader
from patchology.tissue.detection import remove_small_objects
from patchology.tissue.mask import SlideMask

# Saturation above which a pixel at blur resolution counts as stained.
STAINED_MIN_SATURATION = 7

# HSV ranges in OpenCV units (H in [0, 180), S and V in [0, 255]) as
# (h_min, h_max, s_min, s_max, v_min, v_max). Hue ranges where h_min > h_max
# wrap around 180. Chosen to stay clear of hematoxylin (purple, H ~125-165)
# and eosin (pink, H ~150-180 with low saturation). Black ink is near-neutral,
# which separates it from tissue folds: those are as dark but strongly
# saturated purple (S ~170).
PEN_HSV_RANGES: dict[str, tuple[int, int, int, int, int, int]] = {
    "blue": (95, 125, 70, 255, 30, 235),
    "green": (35, 90, 40, 255, 30, 235),
    "black": (0, 180, 0, 110, 0, 100),
    "red": (172, 8, 150, 255, 50, 235),
}


@dataclass(frozen=True)
class ArtifactConfig:
    """Settings for :func:`detect_pen_marks` and :func:`detect_blur`.

    Attributes:
        pen: Remove pen markings from the tissue mask.
        pen_colors: Ink colours to detect, keys of :data:`PEN_HSV_RANGES`.
            ``"red"`` is off by default because strongly eosinophilic tissue
            can fall in its range.
        pen_min_area_um2: Ink regions smaller than this are ignored.
        pen_dilation_um: Grow detected ink by this margin, to catch faint
            stroke edges.
        blur: Remove out-of-focus regions. This reads the slide at
            ``blur_mpp``, so it is much slower than tissue detection.
        blur_mpp: Resolution at which focus is measured. Must be fine
            enough that in-focus tissue shows cellular detail.
        blur_window_um: Side length of the square cells focus is scored on.
        blur_threshold: Cells whose focus score is below this are blurred.
            The score (see :func:`blur_scores`) compares fine detail to
            overall contrast, so it does not depend on tissue density. On a
            20x Aperio slide at the default ``blur_mpp``, in-focus cells
            scored >= 11 (1st percentile); with simulated defocus of 1.5 um
            (Gaussian sigma) 95% of cells scored below 4, and at 2 um 99%
            scored below 2. Inspect :func:`blur_scores` to tune it on your
            data.
    """

    pen: bool = True
    pen_colors: tuple[str, ...] = ("blue", "green", "black")
    pen_min_area_um2: float = 10_000.0
    pen_dilation_um: float = 64.0
    blur: bool = False
    blur_mpp: float = 1.0
    blur_window_um: float = 128.0
    blur_threshold: float = 4.0

    def __post_init__(self) -> None:
        unknown = set(self.pen_colors) - PEN_HSV_RANGES.keys()
        if unknown:
            raise ValueError(
                f"unknown pen colours {sorted(unknown)}, known: {sorted(PEN_HSV_RANGES)}"
            )


def detect_pen_marks(
    rgb: np.ndarray, mpp: float, config: ArtifactConfig = ArtifactConfig()
) -> np.ndarray:
    """Return a boolean mask of pen ink in an ``(H, W, 3)`` uint8 RGB image."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    ink = np.zeros(rgb.shape[:2], dtype=bool)
    for colour in config.pen_colors:
        h_min, h_max, s_min, s_max, v_min, v_max = PEN_HSV_RANGES[colour]
        hue = (h >= h_min) & (h <= h_max) if h_min <= h_max else (h >= h_min) | (h <= h_max)
        ink |= hue & (s >= s_min) & (s <= s_max) & (v >= v_min) & (v <= v_max)

    # Opening removes isolated pixels (e.g. single dark nuclei clusters).
    ink = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    ink = remove_small_objects(ink.astype(bool), config.pen_min_area_um2 / (mpp * mpp))
    dilate_px = round(config.pen_dilation_um / mpp)
    if dilate_px > 0 and ink.any():
        size = 2 * dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        ink = cv2.dilate(ink.astype(np.uint8), kernel).astype(bool)
    return ink


def blur_scores(
    slide: SlideReader,
    tissue: SlideMask,
    slide_mpp: float,
    config: ArtifactConfig = ArtifactConfig(),
    chunk_cells: int = 16,
) -> tuple[np.ndarray, SlideMask]:
    """Score focus on a grid of square cells covering the slide.

    The slide is read at ``config.blur_mpp`` in chunks. Each cell is scored
    as ``var(Laplacian) / (var(intensity) + 1)`` over its stained tissue
    pixels: the ratio of fine detail to overall contrast, which drops when
    the image is out of focus regardless of how dense the tissue is. Glass,
    and stained pixels next to glass, are excluded, so the tissue border
    and empty space such as fat cell interiors do not affect the score.

    Returns:
        ``(scores, valid)``: per-cell focus score as a float array (NaN where
        a cell has too little tissue), and a mask of the scored cells.
        Both use the same grid, with ``valid.downsample`` giving its scale.
    """
    downsample = config.blur_mpp / slide_mpp
    cell = max(4, round(config.blur_window_um / config.blur_mpp))
    width, height = slide.dimensions
    out_w = max(1, round(width / downsample))
    out_h = max(1, round(height / downsample))
    grid_w = -(-out_w // cell)
    grid_h = -(-out_h // cell)
    cell_downsample = downsample * cell

    # Coarse tissue presence per cell, used to skip chunks without tissue.
    tissue_cells = tissue.sample((cell_downsample, cell_downsample), (grid_h, grid_w))
    scores = np.full((grid_h, grid_w), np.nan, dtype=np.float64)
    chunk = cell * chunk_cells

    for gy in range(0, grid_h, chunk_cells):
        for gx in range(0, grid_w, chunk_cells):
            cells_here = tissue_cells[gy : gy + chunk_cells, gx : gx + chunk_cells]
            if not cells_here.any():
                continue
            ox, oy = gx * cell, gy * cell
            cw, ch = min(chunk, out_w - ox), min(chunk, out_h - oy)
            location = (round(ox * downsample), round(oy * downsample))
            rgb = slide.read_region_scaled(location, downsample, (cw, ch))
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
            lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)

            # Score stained pixels only, and drop those within 2 px of glass:
            # the Laplacian there responds to the tissue edge, not focus.
            region = tissue.sample((downsample, downsample), (ch, cw), location)
            saturation = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1]
            stained = (region & (saturation > STAINED_MIN_SATURATION)).astype(np.uint8)
            region = cv2.erode(stained, np.ones((5, 5), np.uint8)).astype(bool)
            nh, nw = cells_here.shape
            weight = _cells(region.astype(np.float32), nh, nw, cell)
            n = weight.sum(axis=(1, 3))
            # Require a quarter of the cell to be tissue for a stable estimate.
            ok = n >= 0.25 * cell * cell
            with np.errstate(invalid="ignore", divide="ignore"):
                lap_var = _weighted_var(_cells(lap, nh, nw, cell), weight, n)
                gray_var = _weighted_var(_cells(gray, nh, nw, cell), weight, n)
                score = lap_var / (gray_var + 1.0)
            scores[gy : gy + nh, gx : gx + nw] = np.where(ok, score, np.nan)

    valid = SlideMask(~np.isnan(scores), (cell_downsample, cell_downsample))
    return scores, valid


def detect_blur(
    slide: SlideReader,
    tissue: SlideMask,
    slide_mpp: float,
    config: ArtifactConfig = ArtifactConfig(),
) -> SlideMask:
    """Return a mask of tissue cells whose focus score is below ``config.blur_threshold``."""
    scores, valid = blur_scores(slide, tissue, slide_mpp, config)
    blurred = valid.array & (np.nan_to_num(scores, nan=np.inf) < config.blur_threshold)
    return SlideMask(blurred, valid.downsample)


def _cells(image: np.ndarray, nh: int, nw: int, cell: int) -> np.ndarray:
    """Zero-pad ``image`` to ``nh x nw`` cells and view it as ``(nh, cell, nw, cell)``."""
    pad_h, pad_w = nh * cell - image.shape[0], nw * cell - image.shape[1]
    return np.pad(image, ((0, pad_h), (0, pad_w))).reshape(nh, cell, nw, cell)


def _weighted_var(values: np.ndarray, weight: np.ndarray, n: np.ndarray) -> np.ndarray:
    mean = (values * weight).sum(axis=(1, 3)) / n
    return (values * values * weight).sum(axis=(1, 3)) / n - mean * mean
