"""Low-resolution masks that cover the full level-0 extent of a slide."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image

# Relative difference in aspect ratio between a custom mask and the slide
# above which we warn that the mask probably does not match the slide.
ASPECT_RATIO_TOLERANCE = 0.02


@dataclass(frozen=True)
class SlideMask:
    """A boolean mask spanning the whole slide at reduced resolution.

    Attributes:
        array: ``(H, W)`` bool array; True marks the region of interest.
        downsample: ``(x, y)`` level-0 pixels per mask pixel.
    """

    array: np.ndarray
    downsample: tuple[float, float]

    @classmethod
    def from_array(cls, array: np.ndarray, slide_dimensions: tuple[int, int]) -> SlideMask:
        """Wrap ``array`` as a mask covering a slide of ``slide_dimensions``.

        Non-boolean input is treated as True where it is non-zero. Colour
        images count a pixel as foreground if any channel is non-zero.
        """
        array = np.asarray(array)
        if array.ndim == 3:
            array = array.any(axis=2)
        if array.ndim != 2:
            raise ValueError(f"mask must be 2D, got shape {array.shape}")
        width, height = slide_dimensions
        mask_h, mask_w = array.shape
        slide_ratio = width / height
        mask_ratio = mask_w / mask_h
        if abs(mask_ratio - slide_ratio) / slide_ratio > ASPECT_RATIO_TOLERANCE:
            logger.warning(
                "mask aspect ratio {:.3f} differs from slide aspect ratio {:.3f}; "
                "the mask is stretched to cover the whole slide",
                mask_ratio,
                slide_ratio,
            )
        return cls(array.astype(bool), (width / mask_w, height / mask_h))

    @property
    def shape(self) -> tuple[int, int]:
        return self.array.shape

    def sample(
        self,
        downsample: tuple[float, float],
        shape: tuple[int, int],
        origin: tuple[float, float] = (0.0, 0.0),
    ) -> np.ndarray:
        """Sample this mask on another pixel grid with nearest-neighbour.

        The target grid has ``shape`` ``(H, W)`` pixels of ``downsample``
        level-0 pixels each, with its top-left corner at level-0 ``origin``.
        Pixels outside this mask are False.
        """
        height, width = shape
        xs = (origin[0] + (np.arange(width) + 0.5) * downsample[0]) / self.downsample[0]
        ys = (origin[1] + (np.arange(height) + 0.5) * downsample[1]) / self.downsample[1]
        xs, ys = np.floor(xs).astype(np.int64), np.floor(ys).astype(np.int64)
        inside_x = (xs >= 0) & (xs < self.shape[1])
        inside_y = (ys >= 0) & (ys < self.shape[0])
        out = self.array[
            np.ix_(np.clip(ys, 0, self.shape[0] - 1), np.clip(xs, 0, self.shape[1] - 1))
        ]
        return out & inside_y[:, None] & inside_x[None, :]

    def like(self, other: SlideMask) -> np.ndarray:
        """Sample this mask on ``other``'s pixel grid."""
        return self.sample(other.downsample, other.shape)

    def __and__(self, other: SlideMask) -> SlideMask:
        return SlideMask(self.array & other.like(self), self.downsample)

    def __or__(self, other: SlideMask) -> SlideMask:
        return SlideMask(self.array | other.like(self), self.downsample)

    def __sub__(self, other: SlideMask) -> SlideMask:
        return SlideMask(self.array & ~other.like(self), self.downsample)

    def area_fraction(self) -> float:
        return float(self.array.mean())

    def save(self, path: str | Path) -> None:
        """Save as an 8-bit PNG (0 / 255)."""
        Image.fromarray(self.array.astype(np.uint8) * 255).save(path)


type MaskSource = SlideMask | np.ndarray | Image.Image | str | Path


def load_mask(source: MaskSource, slide_dimensions: tuple[int, int]) -> SlideMask:
    """Load a custom tissue mask for a slide.

    ``source`` can be a :class:`SlideMask`, an array, a PIL image, or a path to
    a ``.npy`` file or any image PIL can read. Arrays and images are assumed to
    cover the full level-0 extent of the slide, at any resolution.
    """
    if isinstance(source, SlideMask):
        return source
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"tissue mask not found: {path}")
        try:
            if path.suffix == ".npy":
                array = np.load(path)
            else:
                with Image.open(path) as image:
                    array = np.asarray(image.convert("L"))
        except (OSError, ValueError) as e:
            raise ValueError(f"cannot read tissue mask {path}: {e}") from e
        logger.debug("loaded tissue mask {} ({} x {} px)", path, array.shape[1], array.shape[0])
        return SlideMask.from_array(array, slide_dimensions)
    if isinstance(source, Image.Image):
        return SlideMask.from_array(np.asarray(source.convert("L")), slide_dimensions)
    return SlideMask.from_array(source, slide_dimensions)
