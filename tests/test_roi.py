import json

import pytest

from patchology import TileGenerator, TilingConfig
from patchology.tissue import find_roi, load_roi, roi_matches_slide

from conftest import ArrayReader

CONFIG = TilingConfig(tile_size=128, tile_mpp=1.0, tissue_overlap=0.9)


@pytest.mark.parametrize(
    "roi_name, matches",
    [
        ("sample.svs_roi_annotations.geojson", True),
        ("sample_roi.geojson", True),
        ("roi sample.geojson", True),
        ("sampleX_roi.geojson", False),
        ("sample-2_roi.geojson", False),
        ("1sample_roi.geojson", False),
        ("other.geojson", False),
    ],
)
def test_roi_name_matching(roi_name, matches):
    assert roi_matches_slide(roi_name, "/data/sample.svs") is matches


def _square(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]


def _write(path, data):
    path.write_text(json.dumps(data))
    return path


def test_load_roi_polygon_with_hole(tmp_path):
    feature = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [_square(0, 0, 800, 800), _square(200, 200, 600, 600)],
        },
    }
    roi = load_roi(_write(tmp_path / "a.geojson", feature), (1600, 1600), downsample=16)
    assert roi.array[2, 2] and not roi.array[25, 25] and not roi.array[75, 75]
    # fillPoly includes boundary pixels: allow a one-pixel ring.
    assert roi.area_fraction() == pytest.approx((800**2 - 400**2) / 1600**2, rel=0.12)


def test_load_roi_feature_collection_and_multipolygon(tmp_path):
    data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": [[_square(0, 0, 400, 400)], [_square(800, 800, 1200, 1200)]],
                },
            },
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [5, 5]}},
        ],
    }
    roi = load_roi(_write(tmp_path / "b.geojson", data), (1600, 1600), downsample=16)
    assert roi.area_fraction() == pytest.approx(2 * 400**2 / 1600**2, rel=0.12)


def test_load_roi_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_roi(tmp_path / "missing.geojson", (10, 10))
    bad = tmp_path / "bad.geojson"
    bad.write_text("{")
    with pytest.raises(ValueError, match="cannot read"):
        load_roi(bad, (10, 10))
    with pytest.raises(ValueError, match="no Polygon"):
        load_roi(
            _write(tmp_path / "pt.geojson", {"type": "Point", "coordinates": [1, 1]}), (10, 10)
        )


def test_find_roi(tmp_path):
    _write(tmp_path / "s1.svs_roi.geojson", {})
    _write(tmp_path / "s1-2_roi.geojson", {})
    assert find_roi(tmp_path, "s1.svs").name == "s1.svs_roi.geojson"
    with pytest.raises(FileNotFoundError):
        find_roi(tmp_path, "s3.svs")
    _write(tmp_path / "s1_other.geojson", {})
    with pytest.raises(ValueError, match="several"):
        find_roi(tmp_path, "s1.svs")


def test_generator_tiles_only_inside_roi(slide: ArrayReader, tmp_path):
    # Tissue spans x 1024-3072; the ROI keeps its left half.
    feature = {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [_square(0, 0, 2048, 3072)]},
    }
    roi_path = _write(tmp_path / "synthetic_roi.geojson", feature)
    full = TileGenerator(slide, tiling=CONFIG)
    gen = TileGenerator(slide, tiling=CONFIG, roi=roi_path)
    assert 0 < len(gen) < len(full)
    assert (gen.coords[:, 0] + 256 <= 2048 + 128).all()
    assert gen.roi_path == roi_path
    (_, tile), *_ = gen
    assert tile.shape == (128, 128, 3)

    # The ROI directory form finds the same file.
    assert TileGenerator(slide, tiling=CONFIG, roi=tmp_path).roi_path == roi_path


def test_generator_rejects_roi_for_other_slide(slide: ArrayReader, tmp_path):
    feature = {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [_square(0, 0, 10, 10)]},
    }
    roi_path = _write(tmp_path / "another-slide_roi.geojson", feature)
    with pytest.raises(ValueError, match="does not name slide"):
        TileGenerator(slide, roi=roi_path)
    assert TileGenerator(slide, roi=roi_path, check_roi_name=False).roi is not None


def test_generator_roi_excludes_classes(slide: ArrayReader, tmp_path):
    # Tissue spans x 1024-3072: class01 covers the left half, class02 the right.
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [_square(x0, 0, x1, 3072)]},
            "properties": {"classification": {"name": name}},
        }
        for name, x0, x1 in [("class01", 0, 2048), ("class02", 2048, 4096)]
    ]
    roi_path = _write(tmp_path / "synthetic_roi.geojson", features)
    both = TileGenerator(slide, tiling=CONFIG, roi=roi_path)
    right = TileGenerator(slide, tiling=CONFIG, roi=roi_path, roi_exclude_classes={"class01"})
    assert 0 < len(right) < len(both)
    assert (right.coords[:, 0] >= 2048 - 128).all()
    with pytest.raises(ValueError, match="every object"):
        TileGenerator(slide, roi=roi_path, roi_exclude_classes={"class01", "class02"})
