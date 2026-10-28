import numpy as np
import pytest
from test_segmentation import _square, _write_gt

from patchology import DetectionTileGenerator, TileGenerator, TilingConfig
from patchology.tiling.generator import _BoxGrid

from conftest import ArrayReader

# 128 px tiles at 1 mpp on a 0.5 mpp slide: 256 level-0 px per tile, so a tile
# pixel is 2 level-0 px. Tissue spans 1024-3072 x 512-2048.
CONFIG = TilingConfig(tile_size=128, tile_mpp=1.0, tissue_overlap=0.9)


def _cells(tmp_path, boxes, classes=None):
    return _write_gt(tmp_path, [[_square(*b)] for b in boxes], classes=classes)


def test_boxes_are_coco_tile_pixels_and_cut_objects_need_overlap(slide: ArrayReader, tmp_path):
    # 40 x 40 cell across the x = 1536 tile edge: 90% in the left tile, 10% in the right.
    gt = _cells(tmp_path, [(1500, 1100, 1540, 1140)])
    gen = DetectionTileGenerator(slide, gt, min_box_visibility=0.6, tiling=CONFIG)
    assert gen.coords.tolist() == [[1280, 1024]]
    _, tile, target = gen[0]
    assert tile.shape == (128, 128, 3)
    # [x, y, w, h] in tile pixels, clipped at the tile edge.
    np.testing.assert_allclose(target["boxes"], [[110, 38, 18, 20]])
    assert target["boxes"].dtype == np.float32 and target["labels"].tolist() == [1]

    both = DetectionTileGenerator(slide, gt, min_box_visibility=0.05, tiling=CONFIG)
    assert both.coords.tolist() == [[1280, 1024], [1536, 1024]]
    np.testing.assert_allclose(both.tile_target(1)["boxes"], [[0, 38, 2, 20]])
    np.testing.assert_array_equal(both.object_count, [1, 1])


def test_n_objects_and_zero_keeps_empty_tiles(slide: ArrayReader, tmp_path):
    gt = _cells(
        tmp_path,
        [(1100, 1100, 1120, 1120), (1200, 1200, 1220, 1220), (2100, 1100, 2120, 1120)],
        classes=["class01", "class01", "class02"],
    )
    labels = {"class01": 0, "class02": 3}
    two = DetectionTileGenerator(slide, gt, min_objects=2, class_map=labels, tiling=CONFIG)
    assert two.coords.tolist() == [[1024, 1024]]
    assert two.tile_target(0)["labels"].tolist() == [0, 0]
    one = DetectionTileGenerator(slide, gt, min_objects=1, class_map=labels, tiling=CONFIG)
    assert [one.tile_target(i)["labels"].tolist() for i in range(len(one))] == [[0, 0], [3]]

    every = DetectionTileGenerator(slide, gt, min_objects=0, tiling=CONFIG)
    assert len(every) == len(TileGenerator(slide, tiling=CONFIG))
    empty = every.tile_target(every.index_of(2560, 1536))
    assert empty["boxes"].shape == (0, 4) and empty["labels"].shape == (0,)
    meta = every.metadata()["ground_truth"]
    assert meta["min_objects"] == 0 and meta["min_box_visibility"] == 0.6


def test_roi_class_keeps_tiles_wholly_inside(slide: ArrayReader, tmp_path):
    # ROI half a tile off the grid; cells on every tile of it.
    cells = [
        (cx, cy, cx + 20, cy + 20) for cx in range(1100, 2200, 128) for cy in range(700, 1700, 128)
    ]
    gt = _cells(
        tmp_path, [(1152, 640, 2176, 1664), *cells], classes=["roi"] + ["cell"] * len(cells)
    )
    edge = TilingConfig(tile_size=128, tile_mpp=1.0, tissue_overlap=0.1)
    gen = DetectionTileGenerator(slide, gt, roi_class="roi", tiling=edge)
    assert len(gen) and gen.classes == ["cell"] * len(cells)
    for x, y in gen.coords:
        assert 1152 <= x and x + 256 <= 2176 and 640 <= y and y + 256 <= 1664


def test_validation(slide: ArrayReader, tmp_path):
    gt = _cells(tmp_path, [(1500, 1100, 1540, 1140)], classes=["cell"])
    for kwargs, match in [
        ({"min_box_visibility": 1.5}, "min_box_visibility"),
        ({"min_objects": -1}, "min_objects"),
        ({"min_objects": 1.5}, "min_objects"),
        ({"class_map": {"cell": -1}}, "labels must be integers"),
        ({"class_map": {"other": 1}}, "no object of classes"),
    ]:
        with pytest.raises(ValueError, match=match):
            DetectionTileGenerator(slide, gt, tiling=CONFIG, **kwargs)


def test_box_grid_finds_every_box_a_tile_touches():
    rng = np.random.default_rng(0)
    xy = rng.uniform(-100, 3000, (500, 2))
    boxes = np.hstack([xy, xy + rng.uniform(1, 60, (500, 2))])
    boxes[0] = [10, 10, 2900, 2900]  # one box spanning many cells
    grid = _BoxGrid(boxes, 256.0)
    for x, y in rng.uniform(-200, 3000, (200, 2)):
        touching = np.flatnonzero(
            (boxes[:, 0] < x + 256)
            & (boxes[:, 2] > x)
            & (boxes[:, 1] < y + 256)
            & (boxes[:, 3] > y)
        )
        found = grid.query(x, y, 256.0)
        assert set(touching) <= set(found) and (np.diff(found) > 0).all()
