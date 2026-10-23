import numpy as np
import pytest

from patchology import SlideMask, TilingConfig, TilingPipeline
from patchology.visualize import (
    HEX_TILE_COLOR,
    TILE_COLOR,
    TISSUE_COLOR,
    draw_mask,
    draw_tiles,
    visualize_result,
    visualize_slide,
)

from conftest import ArrayReader

GRAY = np.full((100, 200, 3), 128, dtype=np.uint8)


def _mask(slide_dims=(2000, 1000)):
    array = np.zeros((10, 20), bool)
    array[2:8, 5:15] = True
    return SlideMask.from_array(array, slide_dims)


def test_draw_mask_tints_inside_only():
    out = draw_mask(GRAY, (10.0, 10.0), _mask(), alpha=0.5, outline=0)
    assert (out[50, 100] == [64, 164, 64]).all()
    assert (out[5, 5] == 128).all()
    assert (GRAY == 128).all()  # input untouched


def test_draw_mask_outline_only():
    out = draw_mask(GRAY, (10.0, 10.0), _mask(), alpha=0, outline=1)
    assert (out[50, 100] == 128).all()
    assert (out[20, 100] == TISSUE_COLOR).all()  # top edge of the mask


def test_draw_tiles_boxes_at_scale():
    coords = np.array([[100, 200], [1000, 500]])
    out = draw_tiles(GRAY, (10.0, 10.0), coords, tile_size=200)
    # First box spans (10, 20)-(30, 40) in image pixels.
    assert (out[20, 10:31] == TILE_COLOR).all()
    assert (out[20:41, 30] == TILE_COLOR).all()
    assert (out[30, 20] == 128).all()  # interior untouched without fill
    filled = draw_tiles(GRAY, (10.0, 10.0), coords, tile_size=200, fill_alpha=1.0)
    assert (filled[30, 20] == TILE_COLOR).all()


def test_draw_tiles_hexagon():
    # Tile (0, 20)-(60, 80): centre (30, 50), circumradius 60 / sqrt(3) ~ 34.6.
    out = draw_tiles(GRAY, (1.0, 1.0), np.array([[0, 20]]), 60, HEX_TILE_COLOR, shape="hex")
    outline = (out == HEX_TILE_COLOR).all(axis=2)
    assert outline[50, 0] and outline[50, 60]  # flat sides on the tile's left and right edges
    assert outline[15, 30] and outline[85, 30]  # vertices beyond the tile's top and bottom
    assert not outline[22, 2]  # tile corner lies outside the hexagon
    assert not outline[50, 30]


def test_draw_tiles_bad_shape():
    with pytest.raises(ValueError, match="shape"):
        draw_tiles(GRAY, (1.0, 1.0), np.array([[0, 0]]), 10, shape="circle")


def test_draw_tiles_empty():
    assert (draw_tiles(GRAY, (1.0, 1.0), np.empty((0, 2)), 10) == GRAY).all()


def test_visualize_slide_mask_and_grid(slide: ArrayReader):
    mask = np.zeros((96, 128), np.uint8)
    mask[16:64, 32:96] = 1
    image = visualize_slide(
        slide, tissue_mask=mask, coords=np.array([[1024, 512]]), tile_size=512, max_size=512
    )
    assert image.size == (512, 384)
    pixels = np.asarray(image)
    assert (pixels[64, 128:192] == TILE_COLOR).all()  # tile at 1024/8, 512/8


def test_visualize_slide_requires_tile_size(slide: ArrayReader):
    with pytest.raises(ValueError, match="tile_size"):
        visualize_slide(slide, coords=np.array([[0, 0]]))


def test_visualize_result_layers(slide: ArrayReader):
    result = TilingPipeline(tiling=TilingConfig(tile_size=128, tile_mpp=1.0)).run(slide)
    full = np.asarray(visualize_result(result))
    mask_only = np.asarray(visualize_result(result, tiles=False))
    grid_only = np.asarray(visualize_result(result, tissue=False, artifacts=False))
    assert full.shape == result.thumbnail.shape
    assert (mask_only == TILE_COLOR).all(axis=2).sum() == 0
    assert (grid_only == TILE_COLOR).all(axis=2).sum() > 0
    assert (grid_only == TISSUE_COLOR).all(axis=2).sum() == 0

    hex_result = TilingPipeline(tiling=TilingConfig(tile_size=128, tile_mpp=1.0, grid="hex")).run(
        slide
    )
    hex_grid = np.asarray(visualize_result(hex_result, tissue=False, artifacts=False))
    assert (hex_grid == HEX_TILE_COLOR).all(axis=2).sum() > 0
    assert (hex_grid == TILE_COLOR).all(axis=2).sum() == 0
    square_on_hex = np.asarray(
        visualize_result(hex_result, tissue=False, artifacts=False, tile_shape="square")
    )
    assert (square_on_hex == TILE_COLOR).all(axis=2).sum() > 0

    sharp = visualize_result(result, slide=slide, max_size=1024)
    assert max(sharp.size) == 1024
    assert result.overlay(tiles=False).size == full.shape[1::-1]
