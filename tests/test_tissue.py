import cv2
import numpy as np
import pytest

from patchology.tissue import (
    ArtifactConfig,
    SlideMask,
    TissueConfig,
    blur_scores,
    detect_blur,
    detect_pen_marks,
    detect_tissue,
    load_mask,
)
from patchology.tissue.detection import fill_small_holes, remove_small_objects

from conftest import BACKGROUND, ArrayReader, make_slide_image, tissue_texture

THUMB_MPP = 8.0


def _thumbnail(height=400, width=500, seed=0):
    rng = np.random.default_rng(seed)
    image = np.full((height, width, 3), BACKGROUND, dtype=np.float32)
    image += rng.normal(0, 1.5, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("method", ["triangle", "otsu", "fixed"])
def test_detects_tissue_and_ignores_glass(method):
    thumb = _thumbnail()
    thumb[100:300, 150:350] = tissue_texture(200, 200)
    mask = detect_tissue(thumb, THUMB_MPP, TissueConfig(method=method))
    assert mask[110:290, 160:340].mean() > 0.99
    assert mask[:80].mean() == 0
    assert mask[:, :120].mean() == 0


def test_blank_slide_has_no_tissue():
    mask = detect_tissue(_thumbnail(), THUMB_MPP)
    assert not mask.any()


def test_small_debris_removed_and_small_holes_filled():
    thumb = _thumbnail()
    thumb[100:300, 150:350] = tissue_texture(200, 200)
    thumb[190:200, 240:250] = BACKGROUND  # 10x10 px hole = 6400 um^2
    thumb[20:24, 20:24] = tissue_texture(4, 4)  # 4x4 px speck = 1024 um^2
    mask = detect_tissue(thumb, THUMB_MPP)
    assert mask[190:200, 240:250].all()
    assert not mask[20:24, 20:24].any()


def test_remove_small_objects_and_fill_small_holes():
    mask = np.zeros((50, 50), bool)
    mask[5:25, 5:25] = True
    mask[40:42, 40:42] = True
    mask[12:14, 12:14] = False
    cleaned = remove_small_objects(mask, 10)
    assert not cleaned[40:42, 40:42].any()
    assert cleaned[5:25, 5:25].sum() == mask[5:25, 5:25].sum()
    filled = fill_small_holes(cleaned, 10)
    assert filled[5:25, 5:25].all()
    # Background touching the border is never a "hole".
    assert not fill_small_holes(cleaned, 10_000)[0, 0]


@pytest.mark.parametrize(
    ("colour", "ink"),
    [("blue", (40, 70, 170)), ("green", (40, 140, 70)), ("black", (45, 45, 55))],
)
def test_pen_marks_detected_on_glass_and_tissue(colour, ink):
    thumb = _thumbnail()
    thumb[100:300, 150:350] = tissue_texture(200, 200)
    stroke = np.zeros(thumb.shape[:2], np.uint8)
    cv2.line(stroke, (20, 20), (480, 380), 1, 10)
    stroke = stroke.astype(bool)
    thumb[stroke] = (0.85 * np.array(ink) + 0.15 * thumb[stroke]).astype(np.uint8)
    pen = detect_pen_marks(thumb, THUMB_MPP, ArtifactConfig(pen_dilation_um=0))
    assert pen[stroke].mean() > 0.95
    assert pen[~stroke].mean() < 0.01


def test_tissue_folds_are_not_black_ink():
    thumb = _thumbnail()
    thumb[100:300, 150:350] = tissue_texture(200, 200)
    # A fold: very dark but saturated purple.
    cv2.line(thumb, (150, 200), (350, 210), (24, 9, 26), 6)
    assert not detect_pen_marks(thumb, THUMB_MPP).any()


def test_unknown_pen_colour_rejected():
    with pytest.raises(ValueError, match="unknown pen colours"):
        ArtifactConfig(pen_colors=("purple",))


@pytest.fixture
def half_blurred_slide():
    image = make_slide_image((4096, 3072), tissue_box=(512, 512, 3072, 2048))
    right = image[512:2560, 2048:3584]
    image[512:2560, 2048:3584] = cv2.GaussianBlur(right, (0, 0), 4)
    tissue = np.zeros((3072 // 16, 4096 // 16), bool)
    tissue[32:160, 32:224] = True
    return ArrayReader(image), SlideMask.from_array(tissue, (4096, 3072))


def test_blur_scores_separate_sharp_from_blurred(half_blurred_slide):
    slide, tissue = half_blurred_slide
    scores, valid = blur_scores(slide, tissue, 0.5, ArtifactConfig(blur_mpp=1.0))
    cell = valid.downsample[0]
    columns = (np.arange(scores.shape[1]) + 0.5) * cell
    sharp = scores[:, (columns > 600) & (columns < 1900)]
    blurred = scores[:, (columns > 2200) & (columns < 3500)]
    assert np.nanmin(sharp) > 2 * np.nanmax(blurred)


def test_detect_blur_flags_only_blurred_half(half_blurred_slide):
    slide, tissue = half_blurred_slide
    blurred = detect_blur(slide, tissue, 0.5, ArtifactConfig(blur_mpp=1.0))
    on_tissue = blurred.like(tissue) & tissue.array
    xs = np.nonzero(on_tissue)[1] * tissue.downsample[0]
    assert len(xs) > 0
    assert xs.min() >= 2048 - 256  # one blur cell of slack at the boundary


def test_slide_mask_from_array_and_sampling():
    array = np.zeros((30, 40), bool)
    array[10:20, 10:30] = True
    mask = SlideMask.from_array(array, (4000, 3000))
    assert mask.downsample == (100.0, 100.0)
    fine = mask.sample((50.0, 50.0), (60, 80))
    assert fine.sum() == array.sum() * 4
    # A grid extending past the slide is False outside it.
    assert not mask.sample((200.0, 200.0), (20, 25))[:, 20:].any()


def test_slide_mask_set_operations():
    a = SlideMask.from_array(np.eye(10, dtype=bool) | np.eye(10, k=1, dtype=bool), (100, 100))
    b = SlideMask.from_array(np.eye(20, dtype=bool), (100, 100))
    assert (a - b).array.sum() == 9
    assert (a & b).array.sum() == 10
    assert (a | b).shape == (10, 10)


def test_load_mask_sources(tmp_path):
    array = np.zeros((30, 40), np.uint8)
    array[5:10, 5:10] = 255
    np.save(tmp_path / "mask.npy", array)
    cv2.imwrite(str(tmp_path / "mask.png"), array)
    for source in (array, tmp_path / "mask.npy", str(tmp_path / "mask.png")):
        mask = load_mask(source, (400, 300))
        assert mask.shape == (30, 40)
        assert mask.array.sum() == 25


def test_mask_with_wrong_aspect_ratio_warns(caplog):
    SlideMask.from_array(np.ones((10, 10)), (4000, 1000))
    assert "aspect ratio" in caplog.text
