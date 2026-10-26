import numpy as np
import pytest

from patchology import ColorConfig, TileGenerator, TilingConfig, TilingPipeline

from conftest import ArrayReader

CONFIG = TilingConfig(tile_size=128, tile_mpp=2.0)


def test_yields_same_tiles_as_pipeline(slide: ArrayReader):
    gen = TileGenerator(slide, tiling=CONFIG)
    expected = TilingPipeline(tiling=CONFIG).run(slide)
    items = list(gen)
    assert len(items) == len(gen) == len(expected) > 0
    (x, y), tile = items[0]
    assert (x, y) == tuple(expected.coords[0])
    np.testing.assert_array_equal(tile, expected.read_tile(slide, 0))
    assert gen[0][0] == (x, y)


def test_init_runs_pipeline_and_logs_summary(slide: ArrayReader, caplog):
    with caplog.at_level("INFO", logger="patchology.tiling.generator"):
        gen = TileGenerator(slide, tiling=CONFIG)
    assert "result" in gen.__dict__
    assert gen.dimensions == slide.dimensions
    assert gen.mpp == 0.5 and gen.mpp_xy == (0.5, 0.5)
    assert gen.properties["openslide.vendor"] == "synthetic"
    assert gen.level_count == 3 and gen.icc_profile is None
    assert gen.tissue_mask.area_fraction() > 0
    summary = caplog.records[-1].getMessage()
    assert "scanner: synthetic" in summary and f"generated tiles: {len(gen)}" in summary
    assert "tiling time: " in summary
    assert "slide MPP: 0.5000" in summary and "mm^2" in summary


def test_stain_normalization_and_override_mpp(slide: ArrayReader):
    color = ColorConfig(stain_normalization="macenko", stain_sample_tiles=4)
    gen = TileGenerator(
        slide, tiling=TilingConfig(tile_size=128, tile_mpp=2.0, slide_mpp=0.25), color=color
    )
    assert gen.mpp == 0.25
    (_, tile), *_ = gen
    assert tile.shape == (128, 128, 3)
    assert gen.source_stains is not None


def test_borrowed_reader_is_not_closed(slide: ArrayReader, monkeypatch):
    closed = []
    monkeypatch.setattr(slide, "close", lambda: closed.append(True))
    with TileGenerator(slide):
        pass
    assert not closed


def test_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        TileGenerator(tmp_path / "missing.svs")
