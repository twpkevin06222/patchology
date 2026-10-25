"""Convert ASAP XML annotations to GeoJSON.

Each ``<Annotation>`` becomes a Feature whose QuPath-style
``properties.classification.name`` is its ``PartOfGroup``, so the output works
with :class:`patchology.SegmentationTileGenerator` (``class_map``) and as a ROI
(``roi_exclude_classes``). Coordinates stay level-0 pixels. Output files are
named ``<xml stem>.geojson`` so they match their slide by name::

    uv run python -m patchology.utilities.xml_to_geojson annotations.xml [out_dir]
    uv run python -m patchology.utilities.xml_to_geojson xml_dir/ [out_dir]
"""

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from loguru import logger

# shortcut: a Spline is kept as the polygon through its control points, not the
# smooth curve ASAP draws; upgrade if spline annotations need exact boundaries.
POLYGON_TYPES = {"Polygon", "Rectangle", "Spline"}
POINT_TYPES = {"Dot", "PointSet"}


def asap_to_geojson(xml_path: str | Path) -> dict:
    """Read an ASAP XML file into a GeoJSON FeatureCollection.

    Points (``Dot``, ``PointSet``) become MultiPoint features, which tiling
    skips. Polygons with fewer than 3 vertices are dropped with a warning.

    Raises:
        ValueError: The file is not ASAP XML.
    """
    xml_path = Path(xml_path)
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as e:
        raise ValueError(f"cannot read {xml_path}: {e}") from e
    if root.tag != "ASAP_Annotations":
        raise ValueError(f"{xml_path} is not ASAP XML (root <{root.tag}>)")

    features = []
    for ann in root.iter("Annotation"):
        kind, name = ann.get("Type"), ann.get("Name")
        coords = sorted(ann.iter("Coordinate"), key=lambda c: int(c.get("Order", 0)))
        points = [[float(c.get("X")), float(c.get("Y"))] for c in coords]
        if kind in POLYGON_TYPES:
            if len(points) < 3:
                logger.warning(
                    "{}: skipping {} with {} vertices", xml_path.name, name, len(points)
                )
                continue
            geometry = {
                "type": "Polygon",
                "coordinates": [points + points[:1]],
            }  # closed ring
        elif kind in POINT_TYPES:
            geometry = {"type": "MultiPoint", "coordinates": points}
        else:
            logger.warning(
                "{}: skipping {} of unknown type {}", xml_path.name, name, kind
            )
            continue
        group = ann.get("PartOfGroup")
        properties = {"name": name}
        if group and group != "None":
            properties["classification"] = {"name": group}
        features.append(
            {"type": "Feature", "geometry": geometry, "properties": properties}
        )
    return {"type": "FeatureCollection", "features": features}


def convert(xml_path: str | Path, out_dir: str | Path | None = None) -> Path:
    """Write ``<out_dir>/<xml stem>.geojson`` (default: next to the XML) and return its path."""
    xml_path = Path(xml_path)
    out_dir = Path(out_dir) if out_dir is not None else xml_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{xml_path.stem}.geojson"
    collection = asap_to_geojson(xml_path)
    out_path.write_text(json.dumps(collection))
    logger.info(
        "{}: wrote {} features to {}",
        xml_path.name,
        len(collection["features"]),
        out_path,
    )
    return out_path


def main(source: Path, out_dir: Path | None = None) -> None:
    for xml_path in sorted(source.glob("*.xml")) if source.is_dir() else [source]:
        convert(xml_path, out_dir)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else None)
