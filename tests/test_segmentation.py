import json

import numpy as np
import pytest

from patchology import SegmentationTileGenerator, TileGenerator, TilingConfig

from conftest import ArrayReader

# 128 px tiles at 1 mpp on a 0.5 mpp slide: 256 level-0 px per tile.
CONFIG = TilingConfig(tile_size=128, tile_mpp=1.0, tissue_overlap=0.9)


def _write_gt(tmp_path, polygons, name="synthetic_annotations.geojson", classes=None):
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": rings},
            "properties": {"classification": {"name": c}} if c else {},
        }
        for rings, c in zip(polygons, classes or [None] * len(polygons))
    ]
    path = tmp_path / name
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return path


def _square(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]


@pytest.fixture
def gt(tmp_path):
    # Inside the tissue (1024-3072 x 512-2048), aligned with the 256 px tile grid
    # of the tissue-detected candidates only loosely, so overlaps vary.
    return _write_gt(tmp_path, [[_square(1536, 1024, 2048, 1536)]])


def test_yields_tile_and_aligned_mask(slide: ArrayReader, gt):
    gen = SegmentationTileGenerator(slide, gt, object_mask_overlap=0.2, tiling=CONFIG)
    items = list(gen)
    assert len(items) == len(gen) > 0
    for (x, y), tile, mask in items:
        assert tile.shape == (128, 128, 3) and mask.shape == (128, 128) and mask.dtype == np.uint8
        assert mask.mean() >= 0.2
        # Mask pixel (i, j) is level-0 (x + 2j, y + 2i): check it against the square.
        ys, xs = np.nonzero(mask)
        assert (x + 2 * xs >= 1536 - 2).all() and (x + 2 * xs <= 2048 + 2).all()
        assert (y + 2 * ys >= 1024 - 2).all() and (y + 2 * ys <= 1536 + 2).all()
    np.testing.assert_allclose(gen.object_fraction, [m.mean() for _, _, m in items])


def test_overlap_threshold_and_zero_keeps_all(slide: ArrayReader, gt):
    every = SegmentationTileGenerator(slide, gt, object_mask_overlap=0.0, tiling=CONFIG)
    some = SegmentationTileGenerator(slide, gt, object_mask_overlap=0.2, tiling=CONFIG)
    full = SegmentationTileGenerator(slide, gt, object_mask_overlap=1.0, tiling=CONFIG)
    assert len(every) == len(TileGenerator(slide, tiling=CONFIG))
    assert len(every) > len(some) >= len(full)
    assert (full.object_fraction == 1.0).all()


def test_hole_is_empty_in_mask(slide: ArrayReader, tmp_path):
    gt = _write_gt(tmp_path, [[_square(1024, 512, 3072, 2048), _square(1536, 1024, 2560, 1536)]])
    gen = SegmentationTileGenerator(slide, gt, object_mask_overlap=0.0, tiling=CONFIG)
    inside_hole = [
        i
        for i, (x, y) in enumerate(gen.coords)
        if 1536 <= x and x + 256 <= 2560 and 1024 <= y and y + 256 <= 1536
    ]
    # The hole edge may leave a one-pixel line on tiles that share it (fillPoly).
    assert inside_hole and all(gen.tile_mask(i)[1:-1, 1:-1].sum() == 0 for i in inside_hole)


def test_validation(slide: ArrayReader, gt, tmp_path):
    with pytest.raises(ValueError, match="object_mask_overlap"):
        SegmentationTileGenerator(slide, gt, object_mask_overlap=1.5)
    other = _write_gt(tmp_path, [[_square(0, 0, 10, 10)]], name="other-slide_annotations.geojson")
    with pytest.raises(ValueError, match="does not name slide"):
        SegmentationTileGenerator(slide, other)
    assert SegmentationTileGenerator(slide, other, check_ground_truth_name=False).polygons


def test_metadata_records_ground_truth(slide: ArrayReader, gt):
    gen = SegmentationTileGenerator(slide, gt, object_mask_overlap=0.2, tiling=CONFIG)
    meta = gen.metadata()
    assert meta["ground_truth"]["object_mask_overlap"] == 0.2
    assert meta["num_tiles"] == len(gen)


def test_class_map_labels_mask(slide: ArrayReader, tmp_path):
    # Left half class01, right half class02, a corner of class03 that is not mapped.
    gt = _write_gt(
        tmp_path,
        [
            [_square(1024, 512, 2048, 2048)],
            [_square(2048, 512, 3072, 2048)],
            [_square(0, 0, 64, 64)],
        ],
        classes=["class01", "class02", "class03"],
    )
    gen = SegmentationTileGenerator(
        slide, gt, object_mask_overlap=0.0, class_map={"class01": 1, "class02": 7}, tiling=CONFIG
    )
    assert gen.classes == ["class01", "class02"]
    values = {int(v) for i in range(len(gen)) for v in np.unique(gen.tile_mask(i))}
    assert values <= {0, 1, 7} and {1, 7} <= values
    for i, (x, y) in enumerate(gen.coords):
        inside = 1024 <= x and x + 256 <= 3072 and 512 <= y and y + 256 <= 2048
        if inside and (x + 256 <= 2048 or x >= 2048):
            assert (gen.tile_mask(i)[1:-1, 1:-1] == (1 if x < 2048 else 7)).all()
    assert gen.metadata()["ground_truth"]["class_map"] == {"class01": 1, "class02": 7}


def test_class_map_validation(slide: ArrayReader, gt):
    with pytest.raises(ValueError, match="labels must be integers"):
        SegmentationTileGenerator(slide, gt, class_map={"class01": 0})
    with pytest.raises(ValueError, match="no object of classes"):
        SegmentationTileGenerator(slide, gt, class_map={"class01": 1})


def test_roi_class_restricts_tiles_and_ignores_outside(slide: ArrayReader, tmp_path):
    # ROI box half a tile off the 256 px grid, with a tumour spilling past its right edge.
    path = _write_gt(
        tmp_path,
        [[_square(1152, 640, 2176, 1664)], [_square(1664, 1152, 2688, 1664)]],
        classes=["roi", "tumor"],
    )
    edge = TilingConfig(tile_size=128, tile_mpp=1.0, tissue_overlap=0.1)  # keep edge tiles
    gen = SegmentationTileGenerator(
        slide, path, object_mask_overlap=0.0, class_map={"tumor": 1}, roi_class="roi", tiling=edge
    )
    items = list(gen)
    assert items and gen.roi_polygons is not None and gen.classes == ["tumor"]
    masks = [m for _, _, m in items]
    assert set(np.unique(masks)) <= {0, 1, 255} and any((m == 255).any() for m in masks)
    for (x, y), _, mask in items:
        # Mask pixel (i, j) is level-0 (x + 2j, y + 2i); 2 px slack for fillPoly's boundary.
        xs = np.nonzero(mask == 1)[1]
        assert (x + 2 * xs <= 2176 + 2).all()  # tumour outside the ROI is never labelled
        assert x < 2176 and y < 1664 and x + 256 > 1152 and y + 256 > 640  # tiles touch the ROI
    # object_fraction counts labelled pixels only, not ignored ones.
    np.testing.assert_allclose(gen.object_fraction, [(m == 1).mean() for m in masks])
    assert gen.metadata()["ground_truth"]["ignore_index"] == 255


def test_roi_class_validation(slide: ArrayReader, gt, tmp_path):
    with pytest.raises(ValueError, match="no object of roi_class"):
        SegmentationTileGenerator(slide, gt, roi_class="roi", tiling=CONFIG)
    only_roi = _write_gt(tmp_path, [[_square(1024, 512, 2048, 1536)]], classes=["roi"])
    with pytest.raises(ValueError, match="no ground truth"):
        SegmentationTileGenerator(slide, only_roi, roi_class="roi", tiling=CONFIG)
    with pytest.raises(ValueError, match="also a class_map label"):
        SegmentationTileGenerator(slide, gt, roi_class="roi", class_map={"a": 255}, tiling=CONFIG)
    with pytest.raises(ValueError, match="ignore_index must be"):
        SegmentationTileGenerator(slide, gt, roi_class="roi", ignore_index=0, tiling=CONFIG)


def test_overlay_draws_ground_truth_per_label(slide: ArrayReader, tmp_path):
    gt = _write_gt(
        tmp_path,
        [[_square(1536, 1024, 2048, 1536)], [_square(2304, 1024, 2816, 1536)]],
        classes=["tumour", "stroma"],
    )
    gen = SegmentationTileGenerator(
        slide, gt, class_map={"tumour": 1, "stroma": 2}, object_mask_overlap=0.0, tiling=CONFIG
    )
    plain = np.asarray(gen.overlay(ground_truth=False, tiles=False, tissue=False))
    auto = np.asarray(gen.overlay(tiles=False, tissue=False))
    custom = np.asarray(
        gen.overlay(tiles=False, tissue=False, label_colors={1: (255, 0, 0), 2: (0, 0, 255)})
    )
    # Thumbnail pixel at the centre of each square, level-0 (1792, 1280) and (2560, 1280).
    dx, dy = 4096 / plain.shape[1], 3072 / plain.shape[0]
    tumour, stroma = (round(1280 / dy), round(1792 / dx)), (round(1280 / dy), round(2560 / dx))
    assert (auto[tumour] != plain[tumour]).any() and (auto[tumour] != auto[stroma]).any()
    assert custom[tumour][0] > custom[tumour][2] and custom[stroma][2] > custom[stroma][0]


def test_make_palette_distinct_colours():
    from patchology.visualize import make_palette

    for n in (3, 10, 40):
        palette = make_palette(range(1, n + 1))
        assert len(palette) == n and len(set(palette.values())) == n
