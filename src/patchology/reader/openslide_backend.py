"""OpenSlide backend: a thin wrapper around :class:`openslide.OpenSlide`."""

from __future__ import annotations

import functools
import os
from collections.abc import Mapping
from pathlib import Path

import openslide
from PIL import Image

from patchology.reader import base
from patchology.reader.base import SlideOpenError, SlideReader


@functools.cache
def _shared_cache(pid: int) -> openslide.OpenSlideCache:
    # Keyed by pid: a forked worker must not use the parent's cache (or its lock).
    return openslide.OpenSlideCache(base.TILE_CACHE_BYTES)


class OpenSlideReader(SlideReader):
    backend = "openslide"

    def __init__(self, path: str | Path) -> None:
        super().__init__(path)
        try:
            self._slide = openslide.OpenSlide(self._path)
        except openslide.OpenSlideError as e:
            raise SlideOpenError(f"OpenSlide cannot open {self._path}: {e}") from e
        self._slide.set_cache(_shared_cache(os.getpid()))

    @property
    def level_count(self) -> int:
        return self._slide.level_count

    @property
    def level_dimensions(self) -> tuple[tuple[int, int], ...]:
        return self._slide.level_dimensions

    @property
    def level_downsamples(self) -> tuple[float, ...]:
        return self._slide.level_downsamples

    @property
    def properties(self) -> Mapping[str, str]:
        return self._slide.properties

    @property
    def associated_images(self) -> Mapping[str, Image.Image]:
        return self._slide.associated_images

    @property
    def icc_profile(self) -> bytes | None:
        profile = self._slide.color_profile
        return None if profile is None else profile.tobytes()

    def read_region(
        self, location: tuple[int, int], level: int, size: tuple[int, int]
    ) -> Image.Image:
        return self._slide.read_region(location, level, size)

    def get_best_level_for_downsample(self, downsample: float) -> int:
        return self._slide.get_best_level_for_downsample(downsample)

    def get_thumbnail(self, size: tuple[int, int]) -> Image.Image:
        return self._slide.get_thumbnail(size)

    def close(self) -> None:
        self._slide.close()
