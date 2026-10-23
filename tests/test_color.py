import json

import numpy as np
import pytest
from PIL import Image, ImageCms

from patchology import (
    REFERENCE_STAINS,
    ColorConfig,
    Stains,
    TilingConfig,
    TilingPipeline,
    macenko_normalize,
)

from conftest import ArrayReader

# Stains that differ from the reference but, like real H&E, absorb in every channel.
_OTHER = np.array([[0.45, 0.30], [0.80, 0.85], [0.40, 0.43]])
OTHER_STAINS = Stains(_OTHER / np.linalg.norm(_OTHER, axis=0), np.array([1.5, 1.2]))


def _render(stains: Stains, seed: int = 0) -> np.ndarray:
    """H&E image with random stain concentrations, scaled to ``stains``' maxima."""
    rng = np.random.default_rng(seed)
    conc = rng.random((2, 128 * 128)) * stains.max_concentrations[:, None]
    od = stains.matrix @ conc
    return np.clip(240 * np.exp(-od.T), 0, 255).round().astype(np.uint8).reshape(128, 128, 3)


def test_estimate_recovers_stain_vectors():
    stains = Stains.estimate(_render(OTHER_STAINS))
    np.testing.assert_allclose(stains.matrix, OTHER_STAINS.matrix, atol=0.05)


def test_normalize_maps_source_to_target_rendering():
    source = _render(OTHER_STAINS)
    normalized = macenko_normalize(source, Stains.estimate(source))
    target = Stains.estimate(_render(REFERENCE_STAINS))
    np.testing.assert_allclose(Stains.estimate(normalized).matrix, target.matrix, atol=0.05)
    assert normalized.shape == source.shape and normalized.dtype == np.uint8


def test_estimate_rejects_blank_image():
    with pytest.raises(ValueError, match="stained pixels"):
        Stains.estimate(np.full((32, 32, 3), 240, np.uint8))


class ProfileReader(ArrayReader):
    def __init__(self, image, profile):
        super().__init__(image)
        self._profile = profile

    @property
    def icc_profile(self):
        return self._profile


def test_icc_srgb_profile_is_identity(slide_image):
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    reader = ProfileReader(slide_image, srgb)
    patch = slide_image[500:600, 1000:1100]
    np.testing.assert_allclose(reader.apply_icc(patch), patch, atol=2)


def test_bad_icc_profile_warns_and_skips(slide_image, caplog):
    reader = ProfileReader(slide_image, b"not a profile")
    patch = slide_image[:10, :10]
    assert reader.apply_icc(patch) is patch
    assert "unusable ICC profile" in caplog.text


def test_icc_runs_before_stain_normalization(slide_image, monkeypatch):
    calls = []
    reader = ArrayReader(slide_image)
    monkeypatch.setattr(reader, "apply_icc", lambda rgb: calls.append("icc") or rgb)
    from patchology.tiling import tiler

    real = tiler.macenko_normalize
    monkeypatch.setattr(tiler, "macenko_normalize", lambda *a: calls.append("stain") or real(*a))
    color = ColorConfig(stain_normalization="macenko", stain_sample_tiles=4)
    result = TilingPipeline(tiling=TilingConfig(tile_size=128, tile_mpp=2.0), color=color).run(
        reader
    )
    calls.clear()
    result.read_tile(reader, 0)
    assert calls == ["icc", "stain"]


def test_pipeline_stain_normalization(slide: ArrayReader, tmp_path):
    color = ColorConfig(stain_normalization="macenko", stain_sample_tiles=8)
    result = TilingPipeline(tiling=TilingConfig(tile_size=128, tile_mpp=2.0), color=color).run(
        slide
    )
    assert result.source_stains is not None
    tile = result.read_tile(slide, 0)
    assert tile.shape == (128, 128, 3)
    result.save(tmp_path)
    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert meta["color"]["stain_normalization"] == "macenko"
    assert len(meta["color"]["source_stains"]["matrix"]) == 3


def test_color_config_validation():
    with pytest.raises(ValueError):
        ColorConfig(stain_normalization="reinhard")
    with pytest.raises(ValueError):
        ColorConfig(stain_sample_tiles=0)


def test_stain_target_sources(tmp_path):
    image = _render(OTHER_STAINS)
    path = tmp_path / "ref.png"
    Image.fromarray(image).save(path)
    expected = Stains.estimate(image).matrix
    for source in (image, Image.fromarray(image), path, str(path)):
        np.testing.assert_allclose(ColorConfig(stain_target=source).target_stains.matrix, expected)
    assert ColorConfig(stain_target=REFERENCE_STAINS).target_stains is REFERENCE_STAINS


def test_default_target_is_reference_stains():
    assert ColorConfig().target_stains is REFERENCE_STAINS


@pytest.mark.parametrize(
    "source, error",
    [
        ("missing.png", FileNotFoundError),
        (np.zeros((8, 8), np.uint8), ValueError),
        (np.full((32, 32, 3), 240, np.uint8), ValueError),
    ],
)
def test_bad_stain_target_fails_early(slide: ArrayReader, source, error):
    color = ColorConfig(stain_normalization="macenko", stain_target=source)
    with pytest.raises(error):
        TilingPipeline(color=color).run(slide)


def test_clipped_pixels_do_not_inflate_concentrations():
    image = _render(OTHER_STAINS)
    clean = Stains.estimate(image)
    clipped = image.copy()
    clipped[:8] = [30, 0, 60]  # 6% saturated, near-black nuclei
    stains = Stains.estimate(clipped)
    np.testing.assert_allclose(stains.max_concentrations, clean.max_concentrations, rtol=0.1)
