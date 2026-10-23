from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from patchology.reader.base import SlideReader, level_properties

BACKGROUND = (242, 242, 240)
EOSIN = np.array([225, 140, 190], dtype=np.float32)
HEMATOXYLIN = np.array([90, 50, 140], dtype=np.float32)


class ArrayReader(SlideReader):
    """In-memory slide with a 1x/4x/16x pyramid, for tests."""

    backend = "array"

    def __init__(
        self,
        image: np.ndarray,
        mpp: float | None = 0.5,
        path: str | Path = "synthetic.tiff",
        downsamples: tuple[int, ...] = (1, 4, 16),
    ) -> None:
        super().__init__(path)
        self._levels = [image]
        for ds in downsamples[1:]:
            h, w = image.shape[0] // ds, image.shape[1] // ds
            self._levels.append(cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA))
        self._downsamples = tuple(image.shape[1] / level.shape[1] for level in self._levels)
        self._props = level_properties(self.level_dimensions, self._downsamples)
        self._props["openslide.vendor"] = "synthetic"
        if mpp is not None:
            self._props["openslide.mpp-x"] = str(mpp)
            self._props["openslide.mpp-y"] = str(mpp)

    @property
    def level_count(self) -> int:
        return len(self._levels)

    @property
    def level_dimensions(self) -> tuple[tuple[int, int], ...]:
        return tuple((lv.shape[1], lv.shape[0]) for lv in self._levels)

    @property
    def level_downsamples(self) -> tuple[float, ...]:
        return self._downsamples

    @property
    def properties(self) -> Mapping[str, str]:
        return self._props

    @property
    def associated_images(self) -> Mapping[str, Image.Image]:
        return {}

    def read_region(self, location, level, size) -> Image.Image:
        ds = self._downsamples[level]
        data = self._levels[level]
        lx, ly = int(np.floor(location[0] / ds)), int(np.floor(location[1] / ds))
        w, h = size
        out = np.zeros((h, w, 4), dtype=np.uint8)
        x0, y0 = max(lx, 0), max(ly, 0)
        x1, y1 = min(lx + w, data.shape[1]), min(ly + h, data.shape[0])
        if x1 > x0 and y1 > y0:
            out[y0 - ly : y1 - ly, x0 - lx : x1 - lx, :3] = data[y0:y1, x0:x1]
            out[y0 - ly : y1 - ly, x0 - lx : x1 - lx, 3] = 255
        return Image.fromarray(out, "RGBA")

    def close(self) -> None:
        pass


def tissue_texture(height: int, width: int, seed: int = 0) -> np.ndarray:
    """Pink stroma with dark purple nuclei, sharp at the pixel level."""
    rng = np.random.default_rng(seed)
    nuclei = (rng.random((height, width)) < 0.02).astype(np.float32)
    nuclei = cv2.GaussianBlur(nuclei, (0, 0), 1.5)
    nuclei = np.clip(nuclei / nuclei.max() * 3, 0, 1)[..., None]
    rgb = EOSIN * (1 - nuclei) + HEMATOXYLIN * nuclei
    rgb += rng.normal(0, 6, rgb.shape)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def make_slide_image(
    size: tuple[int, int] = (4096, 3072),
    tissue_box: tuple[int, int, int, int] = (1024, 512, 2048, 1536),
    seed: int = 0,
) -> np.ndarray:
    """White slide with one rectangular piece of tissue at ``(x, y, w, h)``."""
    width, height = size
    rng = np.random.default_rng(seed + 1)
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = BACKGROUND
    image = np.clip(image + rng.normal(0, 1.5, image.shape), 0, 255).astype(np.uint8)
    x, y, w, h = tissue_box
    image[y : y + h, x : x + w] = tissue_texture(h, w, seed)
    return image


@pytest.fixture
def slide_image() -> np.ndarray:
    return make_slide_image()


@pytest.fixture
def slide(slide_image: np.ndarray) -> ArrayReader:
    return ArrayReader(slide_image)


@pytest.fixture(scope="session")
def real_slide_path() -> Path:
    """A real WSI for backend conformance tests, from ``$PATCHOLOGY_TEST_SLIDE``."""
    value = os.environ.get("PATCHOLOGY_TEST_SLIDE")
    if not value:
        pytest.skip("set PATCHOLOGY_TEST_SLIDE to a whole slide image to run")
    path = Path(value)
    if not path.exists():
        pytest.skip(f"PATCHOLOGY_TEST_SLIDE={value} does not exist")
    return path
