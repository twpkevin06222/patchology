from pathlib import Path

import numpy as np
import pytest

import patchology.reader as reader_module
from patchology import TileGenerator
from patchology.reader import SlideOpenError, open_slide

from conftest import ArrayReader


def test_read_region_pads_outside_slide_with_transparency(slide: ArrayReader):
    width, height = slide.dimensions
    region = np.asarray(slide.read_region((width - 10, height - 20), 0, (64, 64)))
    assert region.shape == (64, 64, 4)
    assert (region[:20, :10, 3] == 255).all()
    assert (region[20:, :, 3] == 0).all()
    assert (region[:, 10:, 3] == 0).all()

    region = np.asarray(slide.read_region((-16, -16), 0, (32, 32)))
    assert (region[:16, :, 3] == 0).all()
    assert (region[16:, 16:, 3] == 255).all()


def test_read_region_rgb_composites_background(slide: ArrayReader):
    rgb = slide.read_region_rgb((-8, -8), 0, (8, 8))
    assert rgb.shape == (8, 8, 3)
    assert (rgb == 255).all()


def test_best_level_matches_openslide_semantics(slide: ArrayReader):
    assert slide.level_downsamples == (1.0, 4.0, 16.0)
    assert slide.get_best_level_for_downsample(0.5) == 0
    assert slide.get_best_level_for_downsample(3.99) == 0
    assert slide.get_best_level_for_downsample(4.0) == 1
    assert slide.get_best_level_for_downsample(100) == 2


def test_read_region_scaled_any_downsample(slide: ArrayReader):
    for downsample in (1.0, 2.0, 4.0, 6.5, 16.0, 40.0):
        tile = slide.read_region_scaled((1024, 512), downsample, (100, 80))
        assert tile.shape == (80, 100, 3)
        assert tile.dtype == np.uint8


def test_read_at_downsample_assembles_chunks(slide: ArrayReader):
    whole = slide.read_at_downsample(8.0, chunk_size=128)
    single = slide.read_at_downsample(8.0, chunk_size=10_000)
    assert whole.shape == (384, 512, 3)
    assert np.abs(whole.astype(int) - single).mean() < 1.0


def test_read_at_downsample_bounds_level_reads(slide_image, monkeypatch):
    flat = ArrayReader(slide_image, downsamples=(1,))  # no pyramid: every read is level 0
    sizes = []
    read = flat.read_region_rgb
    monkeypatch.setattr(flat, "read_region_rgb", lambda *a: sizes.append(a[2]) or read(*a))
    thumb = flat.read_at_downsample(8.0, chunk_size=128)
    assert thumb.shape == (384, 512, 3)
    assert max(max(s) for s in sizes) <= 128


def test_thumbnail_keeps_aspect_ratio(slide: ArrayReader):
    assert slide.get_thumbnail((256, 256)).size == (256, 192)


def test_mpp_and_bounds(slide: ArrayReader, slide_image):
    assert slide.mpp == (0.5, 0.5)
    assert slide.bounds == (0, 0, 4096, 3072)
    assert ArrayReader(slide_image, mpp=None).mpp is None


def _failing(path):
    raise SlideOpenError("unsupported format")


def _patch_backends(monkeypatch, classes):
    monkeypatch.setattr(reader_module, "_reader_class", lambda name: classes[name])


def test_open_slide_falls_back(monkeypatch, tmp_path: Path, slide_image):
    path = tmp_path / "slide.svs"
    path.touch()
    working = lambda p: ArrayReader(slide_image, path=p)
    _patch_backends(monkeypatch, {"fastslide": _failing, "openslide": working})
    assert open_slide(path, "fastslide").backend == "array"
    with pytest.raises(SlideOpenError, match="fastslide: unsupported format"):
        open_slide(path, "fastslide", fallback=False)

    _patch_backends(monkeypatch, {"openslide": _failing, "fastslide": working})
    assert open_slide(path, "openslide").backend == "array"


def test_tile_generator_fallback_flag(monkeypatch, tmp_path: Path, slide_image):
    path = tmp_path / "slide.svs"
    path.touch()
    working = lambda p: ArrayReader(slide_image, path=p)
    _patch_backends(monkeypatch, {"openslide": _failing, "fastslide": working})
    assert TileGenerator(path).backend == "array"
    with pytest.raises(SlideOpenError, match="openslide: unsupported format"):
        TileGenerator(path, fallback=False)


def test_open_slide_reports_every_backend_error(monkeypatch, tmp_path: Path):
    path = tmp_path / "slide.svs"
    path.touch()
    _patch_backends(monkeypatch, {"fastslide": _failing, "openslide": _failing})
    with pytest.raises(SlideOpenError) as info:
        open_slide(path)
    assert "openslide:" in str(info.value) and "fastslide:" in str(info.value)


def test_open_slide_treats_missing_backend_as_unavailable(monkeypatch, tmp_path: Path, slide_image):
    path = tmp_path / "slide.svs"
    path.touch()

    def not_installed(p):
        raise ImportError("No module named 'fastslide'")

    working = lambda p: ArrayReader(slide_image, path=p)
    _patch_backends(monkeypatch, {"fastslide": not_installed, "openslide": working})
    assert open_slide(path, "fastslide").backend == "array"


def test_open_slide_argument_errors(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        open_slide(tmp_path / "missing.svs")
    with pytest.raises(ValueError, match="unknown backend"):
        open_slide(tmp_path, "bioformats")


def test_corrupt_file_fails_on_both_real_backends(tmp_path: Path):
    path = tmp_path / "corrupt.svs"
    path.write_bytes(b"not a slide")
    with pytest.raises(SlideOpenError):
        open_slide(path, "fastslide")
