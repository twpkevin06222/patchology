"""Regions of interest from GeoJSON annotations (e.g. QuPath exports)."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Collection, Sequence
from pathlib import Path

import cv2
import numpy as np

from patchology.tissue.mask import SlideMask

logger = logging.getLogger(__name__)

# Level-0 pixels per ROI mask pixel; 16 is ~4 um at 0.25 mpp.
ROI_DOWNSAMPLE = 16.0

# Characters that may not touch the slide name in an annotation file name,
# so "sample" matches "sample.svs_roi.geojson" but not "sample-2_roi.geojson".
_NAME_CHARS = "A-Za-z0-9-"


def roi_matches_slide(roi_path: str | Path, slide_path: str | Path) -> bool:
    """True if ``roi_path``'s file name contains ``slide_path``'s name as a whole token."""
    stem = re.escape(Path(slide_path).stem)
    pattern = rf"(?<![{_NAME_CHARS}]){stem}(?![{_NAME_CHARS}])"
    return re.search(pattern, Path(roi_path).name) is not None


def find_roi(directory: str | Path, slide_path: str | Path) -> Path:
    """Return the single ``*.geojson`` in ``directory`` whose name contains the slide name.

    Raises:
        FileNotFoundError: No matching annotation.
        ValueError: More than one matching annotation.
    """
    directory = Path(directory)
    matches = sorted(p for p in directory.glob("*.geojson") if roi_matches_slide(p, slide_path))
    if not matches:
        raise FileNotFoundError(f"no ROI annotation for {Path(slide_path).name} in {directory}")
    if len(matches) > 1:
        names = ", ".join(p.name for p in matches)
        raise ValueError(f"several ROI annotations for {Path(slide_path).name}: {names}")
    return matches[0]


type Polygon = list[np.ndarray]
"""Exterior ring followed by any hole rings, each ``(N, 2)`` level-0 ``(x, y)``."""


def load_polygons(path: str | Path) -> list[Polygon]:
    """Read the polygons of a GeoJSON file (see :func:`load_classified_polygons`)."""
    return load_classified_polygons(path)[0]


def load_classified_polygons(path: str | Path) -> tuple[list[Polygon], list[str | None]]:
    """Read the polygons of a GeoJSON file and each one's class name.

    Accepts a Feature, FeatureCollection, list of features or bare geometry
    with level-0 pixel coordinates. The class is the feature's QuPath
    ``properties.classification.name`` (or a plain string
    ``classification``), None when absent. Non-polygon geometries (points,
    lines) are skipped with a warning.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ValueError: The file is not valid GeoJSON or contains no polygons.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"annotation not found: {path}")
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"cannot read annotation {path}: {e}") from e

    polygons, classes = [], []
    for geometry, name in _geometries(data):
        kind = geometry.get("type")
        if kind == "Polygon":
            parts = [geometry["coordinates"]]
        elif kind == "MultiPolygon":
            parts = geometry["coordinates"]
        else:
            logger.warning("%s: skipping %s geometry, only polygons are used", path.name, kind)
            continue
        polygons += [[np.asarray(r, float)[:, :2] for r in rings] for rings in parts]
        classes += [name] * len(parts)
    if not polygons:
        raise ValueError(f"{path} contains no Polygon or MultiPolygon geometry")
    return polygons, classes


def rasterize_polygons(
    polygons: list[Polygon],
    shape: tuple[int, int],
    scale: tuple[float, float],
    origin: tuple[float, float] = (0.0, 0.0),
    labels: Sequence[int] | None = None,
) -> np.ndarray:
    """Draw ``polygons`` into an ``(H, W)`` uint8 mask, 0 outside.

    Each polygon is drawn with its value from ``labels`` (default 1); where
    polygons overlap the later one wins. Mask pixel ``(i, j)`` covers
    level-0 ``origin + (j, i) / scale``; holes stay empty.
    """
    mask = np.zeros(shape, np.uint8)
    region = np.empty_like(mask)
    offset, factor = np.asarray(origin), np.asarray(scale)
    # One fillPoly per polygon fills its exterior and leaves its holes empty.
    # shortcut: fillPoly includes boundary pixels, so shapes grow by up to one
    # mask pixel; draw at a finer scale if that matters.
    for rings, label in zip(polygons, labels or [1] * len(polygons), strict=True):
        region[:] = 0
        cv2.fillPoly(region, [np.round((r - offset) * factor).astype(np.int32) for r in rings], 1)
        mask[region > 0] = label
    return mask


def load_roi(
    path: str | Path,
    slide_dimensions: tuple[int, int],
    downsample: float = ROI_DOWNSAMPLE,
    exclude_classes: Collection[str] = (),
) -> SlideMask:
    """Rasterise the polygons of a GeoJSON file into a slide mask.

    Args:
        exclude_classes: Leave out objects of these annotation class names
            (see :func:`load_classified_polygons`); unclassified objects are kept.

    Raises:
        ValueError: See :func:`load_classified_polygons`, or every object is excluded.
    """
    polygons, classes = load_classified_polygons(path)
    if exclude_classes:
        polygons = [p for p, c in zip(polygons, classes) if c not in exclude_classes]
        if not polygons:
            raise ValueError(f"{Path(path).name}: every object is in {sorted(exclude_classes)}")
    width, height = slide_dimensions
    shape = (max(1, round(height / downsample)), max(1, round(width / downsample)))
    mask = rasterize_polygons(polygons, shape, (shape[1] / width, shape[0] / height))
    roi = SlideMask.from_array(mask, slide_dimensions)
    logger.info(
        "loaded ROI %s: %d polygon(s), %.1f%% of slide",
        Path(path).name,
        len(polygons),
        100 * roi.area_fraction(),
    )
    return roi


def _geometries(data, name: str | None = None) -> list[tuple[dict, str | None]]:
    """``(geometry, class name)`` pairs; a Feature's class applies to all its geometries."""
    if isinstance(data, list):
        return [g for item in data for g in _geometries(item, name)]
    kind = data.get("type")
    if kind == "FeatureCollection":
        return _geometries(data.get("features", []), name)
    if kind == "Feature":
        cls = (data.get("properties") or {}).get("classification")
        name = cls.get("name") if isinstance(cls, dict) else cls
        return _geometries(data["geometry"], name) if data.get("geometry") else []
    if kind == "GeometryCollection":
        return _geometries(data.get("geometries", []), name)
    return [(data, name)]
