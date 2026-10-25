import json

import cv2
import numpy as np
import pytest

from patchology import ArtifactConfig, SlideMask, TilingConfig, TilingPipeline
from patchology.tiling import tile_fractions

from conftest import ArrayReader, make_slide_image

# Tissue occupies (1024, 512)-(3072, 2048) on the default synthetic slide.
TISSUE = (1024, 512, 3072, 2048)


def _inside_tissue(coords, size, margin=128):
    x0, y0, x1, y1 = TISSUE
    return (
        (coords[:, 0] >= x0 - margin).all()
        and (coords[:, 1] >= y0 - margin).all()
        and (coords[:, 0] + size <= x1 + margin).all()
        and (coords[:, 1] + size <= y1 + margin).all()
    )


@pytest.mark.parametrize("grid", ["square", "hex"])
def test_tiles_cover_tissue_only(slide: ArrayReader, grid):
    config = TilingConfig(tile_size=128, tile_mpp=1.0, grid=grid, tissue_overlap=0.9)
    result = TilingPipeline(tiling=config).run(slide)
    assert result.level0_tile_size == 256
    assert len(result) > 0
    assert _inside_tissue(result.coords, 256)
    assert (result.tissue_fraction >= 0.9).all()
    # The tissue is 2048 x 1536 level-0 px: 8 x 6 non-overlapping 256 px tiles.
    if grid == "square":
        assert len(result) == pytest.approx(48, abs=6)


def test_tile_overlap_controls_density(slide: ArrayReader):
    base = TilingConfig(tile_size=128, tile_mpp=1.0, tissue_overlap=0.9)
    dense = TilingConfig(tile_size=128, tile_mpp=1.0, tile_overlap=0.5, tissue_overlap=0.9)
    n_base = len(TilingPipeline(tiling=base).run(slide))
    n_dense = len(TilingPipeline(tiling=dense).run(slide))
    assert n_dense > 3 * n_base


def test_tile_mpp_sets_footprint(slide: ArrayReader):
    result = TilingPipeline(tiling=TilingConfig(tile_size=256, tile_mpp=2.0)).run(slide)
    assert result.level0_tile_size == 1024
    tile = result.read_tile(slide, 0)
    assert tile.shape == (256, 256, 3)


def test_custom_mask_replaces_detection(slide: ArrayReader):
    # Only the left quarter of the tissue is "annotated".
    custom = np.zeros((3072 // 32, 4096 // 32), np.uint8)
    custom[512 // 32 : 2048 // 32, 1024 // 32 : 1536 // 32] = 1
    config = TilingConfig(tile_size=128, tile_mpp=1.0, tissue_overlap=0.9)
    result = TilingPipeline(tiling=config).run(slide, tissue_mask=custom)
    assert len(result) > 0
    assert (result.coords[:, 0] >= 1024).all()
    assert (result.coords[:, 0] + 256 <= 1536).all()


def test_pen_marks_removed_from_tiles():
    image = make_slide_image()
    cv2.rectangle(image, (1024, 1000), (3072, 1200), (40, 140, 70), -1)  # green ink band
    slide = ArrayReader(image)
    config = TilingConfig(tile_size=128, tile_mpp=1.0, tissue_overlap=0.9)
    with_pen = TilingPipeline(tiling=config).run(slide)
    without = TilingPipeline(tiling=config, artifacts=ArtifactConfig(pen=False)).run(slide)
    assert len(with_pen) < len(without)
    tops, bottoms = with_pen.coords[:, 1], with_pen.coords[:, 1] + 256
    assert not ((bottoms > 1000) & (tops < 1200)).any()


def test_blur_removal_runs_in_pipeline():
    image = make_slide_image()
    image[512:2048, 2048:3072] = cv2.GaussianBlur(image[512:2048, 2048:3072], (0, 0), 4)
    slide = ArrayReader(image)
    config = TilingConfig(tile_size=128, tile_mpp=1.0, tissue_overlap=0.9)
    result = TilingPipeline(tiling=config, artifacts=ArtifactConfig(blur=True)).run(slide)
    assert len(result) > 0
    assert (result.coords[:, 0] + 256 <= 2048 + 256).all()


@pytest.mark.parametrize("grid", ["square", "hex"])
def test_tissue_overlap_threshold(slide: ArrayReader, grid):
    # Tissue covers x < 288: the 256 px tile column starting at x=256 overlaps
    # it by 32 / 256 = 12.5%.
    mask = np.zeros((3072 // 16, 4096 // 16), bool)
    mask[:, : 288 // 16] = True
    no_pen = ArtifactConfig(pen=False)

    def run(overlap):
        config = TilingConfig(tile_size=128, tile_mpp=1.0, grid=grid, tissue_overlap=overlap)
        return TilingPipeline(tiling=config, artifacts=no_pen).run(slide, tissue_mask=mask)

    assert TilingConfig().tissue_overlap == 0.1
    lenient, strict = run(0.1), run(0.2)
    assert (lenient.tissue_fraction >= 0.1).all()
    assert (strict.tissue_fraction >= 0.2).all()
    border = lambda r: r.tissue_fraction < 0.2
    assert border(lenient).any()
    assert not border(strict).any()
    assert len(lenient) > len(strict)


def test_missing_mpp_requires_override(slide_image):
    slide = ArrayReader(slide_image, mpp=None)
    with pytest.raises(ValueError, match="MPP value was not provided"):
        TilingPipeline().run(slide)
    result = TilingPipeline(tiling=TilingConfig(slide_mpp=0.5)).run(slide)
    assert result.slide_mpp == 0.5


def test_implausible_mpp_rejected(slide_image):
    slide = ArrayReader(slide_image, mpp=1000)  # 10 px/cm TIFF resolution tag
    with pytest.raises(ValueError, match="MPP value was not provided"):
        TilingPipeline().run(slide)


def test_too_few_pyramid_levels_rejected(slide_image):
    slide = ArrayReader(slide_image, downsamples=(1, 4))
    with pytest.raises(ValueError, match="faulty magnification"):
        TilingPipeline().run(slide)


def test_tile_fractions_exact():
    array = np.zeros((10, 10), bool)
    array[:, :5] = True
    mask = SlideMask(array, (10.0, 10.0))
    coords = np.array([[0, 0], [25, 0], [50, 0], [0, 50]])
    np.testing.assert_allclose(tile_fractions(mask, coords, 50), [1.0, 0.5, 0.0, 1.0])


def test_tile_fractions_tiny_tiles_use_one_mask_pixel():
    mask = SlideMask(np.array([[True, False]]), (100.0, 100.0))
    np.testing.assert_allclose(tile_fractions(mask, np.array([[10, 10], [150, 10]]), 5), [1, 0])


def test_save_outputs(slide: ArrayReader, tmp_path):
    result = TilingPipeline(tiling=TilingConfig(tile_size=128, tile_mpp=2.0)).run(slide)
    result.save(tmp_path)
    for name in ("tiles.csv", "metadata.json", "tissue_mask.png", "overlay.png"):
        assert (tmp_path / name).exists()
    rows = (tmp_path / "tiles.csv").read_text().splitlines()
    assert rows[0] == "x,y,tissue_fraction"
    assert len(rows) == len(result) + 1
    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert meta["num_tiles"] == len(result)
    assert meta["tiling"]["tile_mpp"] == 2.0

    paths = result.save_tiles(slide, tmp_path / "tiles")
    assert len(paths) == len(result)
    assert paths[0].name == f"{result.coords[0][0]}_{result.coords[0][1]}.png"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tile_size": 0},
        {"tile_mpp": -1},
        {"tile_overlap": -0.1},
        {"tile_overlap": 1.0},
        {"tissue_overlap": 1.5},
        {"grid": "tri"},
    ],
)
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        TilingConfig(**kwargs)


def test_run_logs_tiling_parameters(slide: ArrayReader, caplog):
    caplog.set_level("INFO", logger="patchology")
    config = TilingConfig(tile_size=128, tile_mpp=1.0, tissue_overlap=0.9)
    result = TilingPipeline(tiling=config).run(slide)
    w, h = slide.dimensions
    assert f"slide {w} x {h} px at" in caplog.text
    assert "tile 128 px at 1 mpp = 256.0 level-0 px" in caplog.text
    assert f"generated {len(result)} tiles" in caplog.text


def test_bad_mask_path_errors(slide: ArrayReader, tmp_path):
    with pytest.raises(FileNotFoundError, match="tissue mask"):
        TilingPipeline().run(slide, tissue_mask=tmp_path / "missing.png")
    bad = tmp_path / "bad.png"
    bad.write_text("not an image")
    with pytest.raises(ValueError, match="cannot read tissue mask"):
        TilingPipeline().run(slide, tissue_mask=bad)
