"""Conformance of the real backends; needs ``$PATCHOLOGY_TEST_SLIDE``."""

import os

import numpy as np
import pytest

from patchology.reader import open_slide

pytest.importorskip("fastslide")


@pytest.fixture(scope="module")
def readers(real_slide_path):
    openslide_reader = open_slide(real_slide_path, "openslide", fallback=False)
    fastslide_reader = open_slide(real_slide_path, "fastslide", fallback=False)
    yield openslide_reader, fastslide_reader
    openslide_reader.close()
    fastslide_reader.close()


def test_geometry_matches(readers):
    os_reader, fs_reader = readers
    assert os_reader.dimensions == fs_reader.dimensions
    assert os_reader.level_count == fs_reader.level_count
    assert os_reader.level_dimensions == fs_reader.level_dimensions
    np.testing.assert_allclose(os_reader.level_downsamples, fs_reader.level_downsamples)


@pytest.mark.parametrize(
    "key",
    [
        "openslide.mpp-x",
        "openslide.mpp-y",
        "openslide.objective-power",
        "openslide.level-count",
        "openslide.level[0].width",
        "openslide.level[0].height",
    ],
)
def test_openslide_properties_match(readers, key):
    os_reader, fs_reader = readers
    if key not in os_reader.properties:
        pytest.skip(f"{key} not reported by OpenSlide for this slide")
    assert float(fs_reader.properties[key]) == pytest.approx(float(os_reader.properties[key]))


def test_quickhash_matches(readers):
    os_reader, fs_reader = readers
    key = "openslide.quickhash-1"
    assert fs_reader.properties[key] == os_reader.properties[key]


def test_associated_images_match(readers):
    os_reader, fs_reader = readers
    assert set(fs_reader.associated_images) <= set(os_reader.associated_images)
    for name, image in fs_reader.associated_images.items():
        assert image.mode == "RGBA"
        assert image.size == os_reader.associated_images[name].size


@pytest.mark.parametrize("level", [0, -1])
def test_read_region_pixels_match_at_level_origin(readers, level):
    os_reader, fs_reader = readers
    level = level % os_reader.level_count
    size = (512, 512)
    a = np.asarray(os_reader.read_region((0, 0), level, size)).astype(int)
    b = np.asarray(fs_reader.read_region((0, 0), level, size)).astype(int)
    assert a.shape == b.shape == (512, 512, 4)
    assert (a[..., 3] == b[..., 3]).all()
    assert np.abs(a[..., :3] - b[..., :3]).mean() < 2.0


def test_read_region_past_edge_is_padded_like_openslide(readers):
    os_reader, fs_reader = readers
    width, height = os_reader.dimensions
    for location in [(width - 100, height - 50), (-64, -32)]:
        a = np.asarray(os_reader.read_region(location, 0, (256, 256)))
        b = np.asarray(fs_reader.read_region(location, 0, (256, 256)))
        assert b.shape == (256, 256, 4)
        assert (a[..., 3] == b[..., 3]).all()


def test_fastslide_read_in_forked_child_raises_instead_of_hanging(readers, real_slide_path):
    readers[1].read_region((0, 0), 0, (64, 64))  # starts FastSlide's thread pool here
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            open_slide(real_slide_path, "fastslide", fallback=False).read_region(
                (0, 0), 0, (64, 64)
            )
            os.write(w, b"read")
        except RuntimeError as e:
            os.write(w, b"raised" if "forkserver" in str(e) else b"other")
        os._exit(0)
    os.waitpid(pid, 0)
    assert os.read(r, 16) == b"raised"
