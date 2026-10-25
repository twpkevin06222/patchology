"""One object per slide: its properties, masks and a generator of colour-processed tiles,
optionally paired with ground-truth segmentation masks from GeoJSON annotations."""

from __future__ import annotations

import dataclasses
import functools
import os
from collections.abc import Collection, Iterator, Mapping
from pathlib import Path
from types import TracebackType
from typing import Self

import numpy as np
from loguru import logger
from PIL import Image

from patchology.color import ColorConfig, Stains
from patchology.reader import Backend, SlideReader, open_slide
from patchology.reader.base import PROPERTY_NAME_VENDOR
from patchology.tiling.tiler import (
    TilingConfig,
    TilingPipeline,
    TilingResult,
    default_num_threads,
    ordered_map,
)
from patchology.tissue.artifacts import ArtifactConfig
from patchology.tissue.detection import TissueConfig
from patchology.tissue.mask import MaskSource, SlideMask
from patchology.tissue.roi import (
    find_roi,
    load_classified_polygons,
    load_roi,
    polygons_to_mask,
    rasterize_polygons,
    roi_matches_slide,
)


class TileGenerator:
    """Tile a slide: ICC calibration -> tissue -> artifacts -> tiles -> stain normalisation.

    Iterating yields ``((x, y), tile)`` with level-0 top-left coordinates and
    ``(tile_size, tile_size, 3)`` uint8 RGB tiles. The pipeline (tissue,
    artifacts, tile selection) runs in ``__init__``, which logs a slide summary,
    so iterating twice reads the tiles twice but detects tissue once.

    It is a map-style dataset (``len`` and integer indexing), so it can be
    wrapped in a ``torch.utils.data.Dataset`` and used with a ``DataLoader``
    with any ``num_workers``: each worker process reopens the slide instead
    of sharing the parent's file handle, and the tiling result is computed
    once in the parent (``len(gen)``, which the sampler calls) and copied to
    the workers.

    Example::

        with TileGenerator("slide.svs", color=ColorConfig(stain_normalization="macenko")) as gen:
            print(gen.dimensions, gen.mpp, len(gen))
            for (x, y), tile in gen:
                ...
    """

    def __init__(
        self,
        slide: str | Path | SlideReader,
        backend: Backend = "openslide",
        fallback: bool = True,
        tissue_mask: MaskSource | None = None,
        tiling: TilingConfig = TilingConfig(),
        tissue: TissueConfig = TissueConfig(),
        artifacts: ArtifactConfig = ArtifactConfig(),
        color: ColorConfig = ColorConfig(),
        roi: str | Path | SlideMask | None = None,
        check_roi_name: bool = True,
        roi_exclude_classes: Collection[str] = (),
        num_threads: int | None = None,
    ) -> None:
        """
        Args:
            slide: Slide path, or an open :class:`SlideReader` that the caller
                keeps ownership of (it is not closed by :meth:`close`).
            backend: Preferred backend when ``slide`` is a path.
            fallback: Try the other backend if ``backend`` cannot open the
                slide. Set False to pin the decoder, e.g. for reproducible tiles.
            tissue_mask: Optional custom tissue mask, see :meth:`TilingPipeline.run`.
            roi: Only tile tissue inside this region: a GeoJSON file, a
                directory holding one GeoJSON whose name contains the slide
                name, or a :class:`SlideMask`.
            check_roi_name: Refuse a GeoJSON file whose name does not contain
                the slide name, so a ROI is never applied to the wrong slide.
            roi_exclude_classes: Annotation class names whose objects are not
                part of the ROI, e.g. ``{"class01"}`` to tile only the others.
            num_threads: Threads reading tiles while iterating and in
                :meth:`save_tiles` (1 reads one at a time). None uses
                :func:`default_num_threads` (``$PATCHOLOGY_NUM_THREADS``, else
                ``min(8, CPUs - 1)``). Tiles still come
                out in order and at most ``2 * num_threads`` are held at once.
                Indexing (``gen[i]``, as a DataLoader does) is unaffected.

        Raises:
            ValueError: ``roi`` names another slide (see ``check_roi_name``),
                or ``roi_exclude_classes`` excludes every object.
        """
        num_threads = default_num_threads() if num_threads is None else num_threads
        if num_threads < 1:
            raise ValueError(f"num_threads must be at least 1, got {num_threads}")
        self._owns_slide = not isinstance(slide, SlideReader)
        self._slide = open_slide(slide, backend, fallback=fallback) if self._owns_slide else slide
        self._pid: int | None = os.getpid()
        # How a worker process reopens the slide.
        self._slide_path, self._backend = self._slide.path, self._slide.backend
        self.pipeline = TilingPipeline(
            tiling=tiling, tissue=tissue, artifacts=artifacts, color=color
        )
        self._tissue_mask = tissue_mask
        self.num_threads = num_threads
        self.roi_path: Path | None = None
        self.roi: SlideMask | None = None
        try:
            if isinstance(roi, SlideMask):
                self.roi = roi
            elif roi is not None:
                self.roi_path = self._resolve_roi(Path(roi), check_roi_name)
                self.roi = load_roi(
                    self.roi_path, self.slide.dimensions, exclude_classes=roi_exclude_classes
                )
            self._run_pipeline()
        except Exception:
            self.close()
            raise

    def _run_pipeline(self) -> None:
        """Compute :attr:`result` and log a summary of the slide and its tiles."""
        width, height = self.dimensions
        tissue_mm2 = self.tissue_mask.area_fraction() * width * height * self.mpp**2 / 1e6
        logger.info(
            "slide name: {} | slide dimensions: ({}, {}) | slide MPP: {:.4f} | "
            "scanner: {} | level count: {}\n"
            "generated tiles: {} | tissue area: {:.2f} mm^2",
            self.path.name,
            width,
            height,
            self.mpp,
            self.properties.get(PROPERTY_NAME_VENDOR, "unknown"),
            self.level_count,
            len(self),
            tissue_mm2,
        )

    def _resolve_roi(self, roi: Path, check_name: bool) -> Path:
        if roi.is_dir():
            return find_roi(roi, self.path)
        if check_name and not roi_matches_slide(roi, self.path):
            raise ValueError(
                f"ROI {roi.name} does not name slide {self.path.name}; "
                "pass check_roi_name=False to use it anyway"
            )
        return roi

    @property
    def slide(self) -> SlideReader:
        """The open slide, reopened once in each new process (DataLoader workers),
        since slide file handles must not be shared between processes."""
        if self._pid != os.getpid():
            self._slide = open_slide(self._slide_path, self._backend)
            self._pid, self._owns_slide = os.getpid(), True
        return self._slide

    def __getstate__(self) -> dict:
        # Pickled for spawned workers: send the tiling result, not the file handle.
        return {**self.__dict__, "result": self.result, "_slide": None, "_pid": None}

    @functools.cached_property
    def result(self) -> TilingResult:
        """The tiling result; computed in ``__init__``."""
        return self.pipeline.run(self.slide, self._tissue_mask, self.roi)

    def __iter__(self) -> Iterator[tuple[tuple[int, int], np.ndarray]]:
        return ordered_map(self.get_tile, range(len(self)), self.num_threads)

    def __len__(self) -> int:
        return len(self.result)

    def __getitem__(self, index: int) -> tuple[tuple[int, int], np.ndarray]:
        return self.get_tile(index)

    def get_tile(self, index: int) -> tuple[tuple[int, int], np.ndarray]:
        """``((x, y), tile)`` of selected tile ``index`` (negative counts from the end).

        Raises:
            IndexError: ``index`` is out of range.
        """
        x, y = self.result.coords[index]
        return (int(x), int(y)), self.result.read_tile(self.slide, index)

    def get_tile_at(self, x: int, y: int) -> np.ndarray:
        """The processed tile whose level-0 top-left corner is ``(x, y)``.

        Any coordinate can be read, not only selected tiles; see :meth:`index_of`.
        """
        return self.result.read_tile_at(self.slide, int(x), int(y))

    def index_of(self, x: int, y: int) -> int:
        """Index of the selected tile at level-0 ``(x, y)``.

        Raises:
            KeyError: No selected tile starts at ``(x, y)``.
        """
        hits = np.flatnonzero((self.result.coords == (x, y)).all(axis=1))
        if not len(hits):
            raise KeyError(f"no selected tile at ({x}, {y})")
        return int(hits[0])

    # --- Slide properties (no pipeline run needed) ---------------------------

    @property
    def path(self) -> Path:
        return self.slide.path

    @property
    def backend(self) -> str:
        return self.slide.backend

    @property
    def properties(self) -> Mapping[str, str]:
        """OpenSlide-style metadata (``openslide.*`` keys; FastSlide's under ``fastslide.*``)."""
        return self.slide.properties

    @property
    def dimensions(self) -> tuple[int, int]:
        return self.slide.dimensions

    @property
    def level_count(self) -> int:
        return self.slide.level_count

    @property
    def level_dimensions(self) -> tuple[tuple[int, int], ...]:
        return self.slide.level_dimensions

    @property
    def level_downsamples(self) -> tuple[float, ...]:
        return self.slide.level_downsamples

    @property
    def mpp(self) -> float:
        """Level-0 microns per pixel used for tiling (``TilingConfig.slide_mpp`` wins)."""
        return self.pipeline.slide_mpp(self.slide)

    @property
    def mpp_xy(self) -> tuple[float, float] | None:
        """``(mpp_x, mpp_y)`` as reported by the slide, or None."""
        return self.slide.mpp

    @property
    def objective_power(self) -> float | None:
        value = self.slide.properties.get("openslide.objective-power")
        return float(value) if value else None

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        return self.slide.bounds

    @property
    def icc_profile(self) -> bytes | None:
        return self.slide.icc_profile

    @property
    def associated_images(self) -> Mapping[str, Image.Image]:
        return self.slide.associated_images

    # --- Pipeline outputs ----------------------------------------------------

    @property
    def coords(self) -> np.ndarray:
        """``(N, 2)`` level-0 ``(x, y)`` of each tile's top-left corner."""
        return self.result.coords

    @property
    def tissue_fraction(self) -> np.ndarray:
        return self.result.tissue_fraction

    @property
    def tissue_mask(self) -> SlideMask:
        """Tissue after artifact removal."""
        return self.result.tissue_mask

    @property
    def artifact_mask(self) -> SlideMask | None:
        return self.result.artifact_mask

    @property
    def thumbnail(self) -> np.ndarray:
        """ICC-calibrated (if enabled) RGB thumbnail on the tissue mask's grid."""
        return self.result.thumbnail

    @property
    def source_stains(self) -> Stains | None:
        """The slide's estimated stains, when stain normalisation is enabled."""
        return self.result.source_stains

    def metadata(self) -> dict:
        return self.result.metadata()

    def overlay(self, **kwargs) -> Image.Image:
        return self.result.overlay(**kwargs)

    def save(self, out_dir: str | Path) -> None:
        """Write ``tiles.csv``, ``metadata.json``, ``tissue_mask.png`` and ``overlay.png``."""
        self.result.save(out_dir)

    def save_tiles(self, out_dir: str | Path, fmt: str = "png") -> list[Path]:
        return self.result.save_tiles(self.slide, out_dir, fmt, self.num_threads)

    # --- Lifecycle -----------------------------------------------------------

    def close(self) -> None:
        # A copy in another process never opened its own slide, so has nothing to close.
        if self._owns_slide and self._pid == os.getpid():
            self._slide.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"TileGenerator({str(self.path)!r}, backend={self.backend!r})"


class SegmentationTileGenerator(TileGenerator):
    """A :class:`TileGenerator` that also yields each tile's ground-truth mask.

    Tiles are selected as by :class:`TileGenerator` (tissue, artifacts, ROI),
    then kept only if at least ``object_mask_overlap`` of their area is
    covered by the ground-truth polygons. Iterating yields
    ``((x, y), tile, mask)``, where ``mask`` is a ``(tile_size, tile_size)``
    uint8 array aligned pixel for pixel with ``tile``: 0 for background and
    each object's label from ``class_map`` (1 for every object without one).

    A file that also outlines where it was annotated (e.g. TIGER ``roi``
    boxes) is handled with ``roi_class``: those objects become the ROI, and
    mask pixels outside them are ``ignore_index`` rather than background, so
    a loss can skip them (``CrossEntropyLoss(ignore_index=255)``).

    Example::

        with SegmentationTileGenerator(
            "sample.svs",
            "sample.svs_annotations.geojson",
            object_mask_overlap=0.2,
            class_map={"class01": 1, "class02": 2, "class03": 3},
        ) as gen:
            for (x, y), tile, mask in gen:
                ...
    """

    def __init__(
        self,
        slide: str | Path | SlideReader,
        ground_truth: str | Path,
        object_mask_overlap: float = 0.2,
        class_map: Mapping[str, int] | None = None,
        check_ground_truth_name: bool = True,
        roi_class: str | None = None,
        ignore_index: int = 255,
        **kwargs,
    ) -> None:
        """
        Args:
            slide: Slide path or open reader, see :class:`TileGenerator`.
            ground_truth: GeoJSON file of annotated objects, in level-0 pixels.
            object_mask_overlap: Keep a tile if at least this fraction of it
                is ground truth, in ``[0, 1]``; 0.2 means 20%. With 0 every
                selected tissue tile is kept, including those with an empty mask.
                Counts every labelled pixel, whatever its class.
            class_map: Annotation class name to mask label (1-255), e.g.
                ``{"class01": 1, "class02": 2}``. Objects of other classes, or
                unclassified, are left out with a warning. None gives a 0/1
                mask of all objects.
            check_ground_truth_name: Refuse a file whose name does not contain
                the slide name, so masks are never taken from another slide.
            roi_class: Class name of the objects in ``ground_truth`` that mark
                the annotated region rather than a label, e.g. ``"roi"`` for
                TIGER. Tiles come only from inside them (and inside ``roi``
                if also given), and mask pixels outside them are
                ``ignore_index``. None treats every object as ground truth.
            ignore_index: Mask value for pixels outside the ``roi_class``
                region, in ``[1, 255]`` and not used by ``class_map``. Unused
                without ``roi_class``.
            **kwargs: Passed to :class:`TileGenerator` (tiling, color, roi, ...).

        Raises:
            ValueError: Bad ``object_mask_overlap``, ``class_map`` or
                ``ignore_index``, unreadable file, no object of a mapped class
                or of ``roi_class``, or a file named for another slide.
        """
        if not 0.0 <= object_mask_overlap <= 1.0:
            raise ValueError(f"object_mask_overlap must be in [0, 1], got {object_mask_overlap}")
        if class_map is not None and not all(
            isinstance(v, int) and 1 <= v <= 255 for v in class_map.values()
        ):
            raise ValueError(f"class_map labels must be integers in [1, 255], got {class_map}")
        if roi_class is not None:
            if not 1 <= ignore_index <= 255:
                raise ValueError(f"ignore_index must be in [1, 255], got {ignore_index}")
            if class_map is not None and ignore_index in class_map.values():
                raise ValueError(f"ignore_index {ignore_index} is also a class_map label")
        super().__init__(slide, **kwargs)
        try:
            path = Path(ground_truth)
            if check_ground_truth_name and not roi_matches_slide(path, self.path):
                raise ValueError(
                    f"ground truth {path.name} does not name slide {self.path.name}; "
                    "pass check_ground_truth_name=False to use it anyway"
                )
            self.polygons, self.classes = load_classified_polygons(path)
            self.roi_polygons = None
            if roi_class is not None:
                self._split_roi(path, roi_class)
            if class_map is not None:
                self._select_classes(path, class_map)
        except Exception:
            self.close()
            raise
        self.ground_truth_path = path
        self.object_mask_overlap = object_mask_overlap
        self.class_map = None if class_map is None else dict(class_map)
        self.roi_class, self.ignore_index = roi_class, ignore_index
        self.labels = (
            [1] * len(self.polygons) if class_map is None else [class_map[c] for c in self.classes]
        )
        # (x0, y0, x1, y1) of each exterior ring, to skip polygons off a tile.
        self._bboxes = np.array([[*p[0].min(axis=0), *p[0].max(axis=0)] for p in self.polygons])
        logger.debug("loaded ground truth {}: {} polygon(s)", path.name, len(self.polygons))
        try:
            super()._run_pipeline()
        except Exception:
            self.close()
            raise

    def _run_pipeline(self) -> None:
        # Skipped in TileGenerator.__init__: selection needs the ground truth loaded above.
        pass

    def _split_roi(self, path: Path, roi_class: str) -> None:
        is_roi = [c == roi_class for c in self.classes]
        if not any(is_roi):
            raise ValueError(f"{path.name} has no object of roi_class {roi_class!r}")
        if all(is_roi):
            raise ValueError(f"{path.name} has only {roi_class!r} objects, no ground truth")
        self.roi_polygons = [p for p, r in zip(self.polygons, is_roi) if r]
        self.polygons = [p for p, r in zip(self.polygons, is_roi) if not r]
        self.classes = [c for c, r in zip(self.classes, is_roi) if not r]
        region = polygons_to_mask(self.roi_polygons, self.slide.dimensions)
        self.roi = region if self.roi is None else self.roi & region

    def _select_classes(self, path: Path, class_map: Mapping[str, int]) -> None:
        unmapped = sorted({str(c) for c in self.classes if c not in class_map})
        if unmapped:
            logger.warning("{}: ignoring objects of unmapped classes {}", path.name, unmapped)
        keep = [c in class_map for c in self.classes]
        if not any(keep):
            raise ValueError(f"{path.name} has no object of classes {sorted(class_map)}")
        self.polygons = [p for p, k in zip(self.polygons, keep) if k]
        self.classes = [c for c, k in zip(self.classes, keep) if k]

    @functools.cached_property
    def result(self) -> TilingResult:
        """Tissue tiles that overlap the ground truth enough; computed in ``__init__``."""
        result = TileGenerator.result.func(self)
        fractions = np.array(
            [self._labelled(self._mask(x, y, result)).mean() for x, y in result.coords],
            dtype=float,
        )
        keep = fractions >= self.object_mask_overlap
        self.object_fraction = fractions[keep]
        logger.info(
            "{}: kept {} of {} tissue tiles with >= {:.0f}% ground truth",
            self.path.name,
            int(keep.sum()),
            len(result),
            100 * self.object_mask_overlap,
        )
        if not keep.any():
            logger.warning(
                "{}: no tile overlaps the ground truth by {:.0f}%; lower object_mask_overlap",
                self.path.name,
                100 * self.object_mask_overlap,
            )
        return dataclasses.replace(
            result, coords=result.coords[keep], tissue_fraction=result.tissue_fraction[keep]
        )

    def tile_mask(self, index: int) -> np.ndarray:
        """Ground-truth mask of tile ``index``, ``(tile_size, tile_size)`` uint8 labels."""
        x, y = self.result.coords[index]
        return self._mask(x, y, self.result)

    def get_tile(self, index: int) -> tuple[tuple[int, int], np.ndarray, np.ndarray]:
        """``((x, y), tile, mask)`` of selected tile ``index``."""
        (x, y), tile = super().get_tile(index)
        return (x, y), tile, self.tile_mask(index)

    def get_tile_at(self, x: int, y: int) -> tuple[np.ndarray, np.ndarray]:
        """``(tile, mask)`` whose level-0 top-left corner is ``(x, y)``."""
        return super().get_tile_at(x, y), self._mask(int(x), int(y), self.result)

    def metadata(self) -> dict:
        meta = super().metadata()
        meta["ground_truth"] = {
            "path": self.ground_truth_path.name,
            "polygons": len(self.polygons),
            "object_mask_overlap": self.object_mask_overlap,
            "class_map": self.class_map,
            "roi_class": self.roi_class,
            "ignore_index": self.ignore_index if self.roi_class is not None else None,
        }
        return meta

    def _mask(self, x: float, y: float, result: TilingResult) -> np.ndarray:
        size = result.config.tile_size
        extent = result.level0_tile_size
        b = self._bboxes
        hit = (b[:, 0] < x + extent) & (b[:, 2] > x) & (b[:, 1] < y + extent) & (b[:, 3] > y)
        scale, origin = (size / extent,) * 2, (float(x), float(y))
        mask = np.zeros((size, size), np.uint8)
        if hit.any():
            polygons = [p for p, h in zip(self.polygons, hit) if h]
            labels = [v for v, h in zip(self.labels, hit) if h]
            mask = rasterize_polygons(polygons, (size, size), scale, origin, labels)
        if self.roi_polygons is not None:
            inside = rasterize_polygons(self.roi_polygons, (size, size), scale, origin)
            mask[inside == 0] = self.ignore_index
        return mask

    def _labelled(self, mask: np.ndarray) -> np.ndarray:
        """Pixels with a ground-truth label, i.e. neither background nor ignored."""
        if self.roi_polygons is None:
            return mask > 0
        return (mask > 0) & (mask != self.ignore_index)
