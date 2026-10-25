"""Tiling pipeline: tissue mask -> artifact removal -> grid -> filtered tiles."""

from __future__ import annotations

import csv
import json
import logging
import os
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from patchology.color import ColorConfig, Stains, macenko_normalize
from patchology.reader import Backend, SlideReader, open_slide
from patchology.tiling.grid import GridType, make_grid
from patchology.tissue.artifacts import ArtifactConfig, detect_blur, detect_pen_marks
from patchology.tissue.detection import TissueConfig, detect_tissue
from patchology.tissue.mask import MaskSource, SlideMask, load_mask
from patchology.visualize import visualize_result

logger = logging.getLogger(__name__)

NUM_THREADS_ENV = "PATCHOLOGY_NUM_THREADS"


def default_num_threads() -> int:
    """Threads for reading tiles: ``$PATCHOLOGY_NUM_THREADS`` if set, else
    ``min(8, usable CPUs - 1)`` and at least 1.

    Tile reads release the GIL in both backends and scale almost linearly; past
    ~8 the decoder and disk stop keeping up, and one CPU is left for the rest
    of the program.

    Raises:
        ValueError: ``$PATCHOLOGY_NUM_THREADS`` is not a positive integer.
    """
    value = os.environ.get(NUM_THREADS_ENV)
    if value:
        try:
            threads = int(value)
        except ValueError:
            threads = 0
        if threads < 1:
            raise ValueError(f"{NUM_THREADS_ENV} must be a positive integer, got {value!r}")
        return threads
    # CPUs this process may run on (container / taskset limits), not all CPUs.
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    return max(1, min(8, (cpus or 1) - 1))


def ordered_map[T, R](
    fn: Callable[[T], R], items: Iterable[T], num_threads: int = 1
) -> Iterator[R]:
    """Yield ``fn(item)`` for each item, in order, using ``num_threads`` threads.

    At most ``2 * num_threads`` results are in flight, so memory stays bounded
    however slowly the caller consumes them. Leaving the loop early cancels the
    reads not yet started; an exception in ``fn`` is raised at its position.
    """
    if num_threads <= 1:
        yield from map(fn, items)
        return
    pending: deque = deque()
    with ThreadPoolExecutor(num_threads, thread_name_prefix="patchology") as pool:
        try:
            for item in items:
                pending.append(pool.submit(fn, item))
                if len(pending) >= 2 * num_threads:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()
        finally:
            for future in pending:
                future.cancel()


# Relative amount by which tile_mpp may be finer than the slide's resolution
# before warning about upsampling, so 0.5 mpp on a 0.5026 mpp scan is quiet.
UPSAMPLE_TOLERANCE = 0.05

# Fewer levels than this means the thumbnail is read from (near) full resolution,
# which can take tens of GB of RAM on a whole slide.
MIN_PYRAMID_LEVELS = 3


@dataclass(frozen=True)
class TilingConfig:
    """Tile geometry and selection.

    Attributes:
        tile_size: Output tile side length in pixels at ``tile_mpp``.
        tile_mpp: Output resolution in microns per pixel. Tiles are read from
            the closest pyramid level at or above this resolution and
            resized, so any value works regardless of the slide's levels.
        tile_overlap: Fraction by which neighbouring tiles overlap, in
            ``[0, 1)``; 0.2 means 20%. The distance between tiles is
            ``tile_size * (1 - tile_overlap)``. On a hex grid this is the
            overlap between horizontal neighbours; rows overlap a little more
            (about 13% of the tile height at 0).
        grid: ``"square"`` or ``"hex"`` lattice of tile positions, see
            :mod:`patchology.tiling.grid`.
        tissue_overlap: Keep a tile if at least this fraction of its square
            footprint overlaps tissue (after artifact removal), on both square
            and hex grids. The default 0.1 keeps tiles on the tissue border
            that are mostly glass; raise it for cleaner tiles. With 0, any
            overlap at all is enough.
        slide_mpp: Level-0 resolution to assume instead of the value in the
            slide metadata; required if the slide does not report it.
    """

    tile_size: int = 256
    tile_mpp: float = 0.5
    tile_overlap: float = 0.0
    grid: GridType = "square"
    tissue_overlap: float = 0.1
    slide_mpp: float | None = None

    def __post_init__(self) -> None:
        if self.tile_size <= 0:
            raise ValueError(f"tile_size must be positive, got {self.tile_size}")
        if self.tile_mpp <= 0:
            raise ValueError(f"tile_mpp must be positive, got {self.tile_mpp}")
        if not 0.0 <= self.tile_overlap < 1.0:
            raise ValueError(f"tile_overlap must be in [0, 1), got {self.tile_overlap}")
        if not 0.0 <= self.tissue_overlap <= 1.0:
            raise ValueError(f"tissue_overlap must be in [0, 1], got {self.tissue_overlap}")
        if self.grid not in ("square", "hex"):
            raise ValueError(f"grid must be 'square' or 'hex', got {self.grid!r}")


@dataclass
class TilingResult:
    """Selected tiles of one slide, plus the masks used to select them.

    Attributes:
        coords: ``(N, 2)`` int64 level-0 ``(x, y)`` of each tile's top-left.
        tissue_fraction: ``(N,)`` tissue fraction of each tile.
        level0_tile_size: Tile footprint on level 0, in level-0 pixels.
        tissue_mask: Tissue after artifact removal; what tiles were scored on.
        artifact_mask: Removed artifacts (pen, blur), or None if disabled.
        thumbnail: RGB slide image on the same grid as ``tissue_mask``.
        color: Colour processing applied by :meth:`read_tile`.
        source_stains: The slide's estimated stains when stain normalisation
            is enabled, else None.
    """

    slide_path: Path
    backend: str
    slide_dimensions: tuple[int, int]
    slide_mpp: float
    config: TilingConfig
    coords: np.ndarray
    tissue_fraction: np.ndarray
    level0_tile_size: float
    tissue_mask: SlideMask
    artifact_mask: SlideMask | None
    thumbnail: np.ndarray = field(repr=False)
    color: ColorConfig = ColorConfig()
    source_stains: Stains | None = None

    def __len__(self) -> int:
        return len(self.coords)

    @property
    def downsample(self) -> float:
        """Level-0 pixels per output tile pixel."""
        return self.config.tile_mpp / self.slide_mpp

    def read_tile(self, slide: SlideReader, index: int) -> np.ndarray:
        """Read tile ``index`` as an ``(tile_size, tile_size, 3)`` uint8 array.

        The tile is ICC calibrated, then stain normalised, as set in ``color``.
        """
        x, y = self.coords[index]
        return self.read_tile_at(slide, int(x), int(y))

    def read_tile_at(self, slide: SlideReader, x: int, y: int) -> np.ndarray:
        """Read the tile whose level-0 top-left corner is ``(x, y)``, processed as
        :meth:`read_tile`; ``(x, y)`` need not be one of ``coords``."""
        size = self.config.tile_size
        try:
            tile = slide.read_region_scaled((x, y), self.downsample, (size, size))
        except Exception as e:  # backends raise their own error types
            raise RuntimeError(f"cannot read tile at ({x}, {y}) from {slide.path}") from e
        if self.color.icc:
            tile = slide.apply_icc(tile)
        if self.source_stains is not None:
            tile = macenko_normalize(tile, self.source_stains, self.color.target_stains)
        return tile

    def iter_tiles(
        self, slide: SlideReader, num_threads: int = 1
    ) -> Iterator[tuple[tuple[int, int], np.ndarray]]:
        """Yield ``((x, y), tile)`` for every selected tile, in order.

        ``num_threads`` > 1 reads tiles concurrently (see :func:`ordered_map`).
        """

        def read(i: int) -> tuple[tuple[int, int], np.ndarray]:
            x, y = self.coords[i]
            return (int(x), int(y)), self.read_tile(slide, i)

        return ordered_map(read, range(len(self)), num_threads)

    def save_tiles(
        self, slide: SlideReader, out_dir: str | Path, fmt: str = "png", num_threads: int = 1
    ) -> list[Path]:
        """Write each tile to ``out_dir/{x}_{y}.{fmt}`` and return the paths.

        With ``num_threads`` > 1, tiles are read and encoded concurrently.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        def save(i: int) -> Path:
            x, y = self.coords[i]
            path = out_dir / f"{int(x)}_{int(y)}.{fmt}"
            Image.fromarray(self.read_tile(slide, i)).save(path)
            return path

        paths = list(ordered_map(save, range(len(self)), num_threads))
        logger.info("%s: saved %d tiles to %s", self.slide_path.name, len(paths), out_dir)
        return paths

    def metadata(self) -> dict:
        stains = {"source_stains": None, "target_stains": None}
        if self.source_stains is not None:
            stains = {
                "source_stains": self.source_stains.to_dict(),
                "target_stains": self.color.target_stains.to_dict(),
            }
        return {
            "slide_path": str(self.slide_path),
            "backend": self.backend,
            "slide_dimensions": list(self.slide_dimensions),
            "slide_mpp": self.slide_mpp,
            "level0_tile_size": self.level0_tile_size,
            "num_tiles": len(self),
            "tiling": asdict(self.config),
            "color": {
                "icc": self.color.icc,
                "stain_normalization": self.color.stain_normalization,
                **stains,
            },
        }

    def to_csv(self, path: str | Path) -> None:
        """Write ``x, y, tissue_fraction`` rows in level-0 coordinates."""
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y", "tissue_fraction"])
            for (x, y), frac in zip(self.coords, self.tissue_fraction):
                writer.writerow([int(x), int(y), f"{frac:.4f}"])

    def save(self, out_dir: str | Path) -> None:
        """Write ``tiles.csv``, ``metadata.json``, ``tissue_mask.png`` and ``overlay.png``."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.to_csv(out_dir / "tiles.csv")
        (out_dir / "metadata.json").write_text(json.dumps(self.metadata(), indent=2))
        self.tissue_mask.save(out_dir / "tissue_mask.png")
        self.overlay().save(out_dir / "overlay.png")
        logger.info("%s: saved tiling result to %s", self.slide_path.name, out_dir)

    def overlay(self, **kwargs) -> Image.Image:
        """Thumbnail with tissue in green, artifacts in red and tiles in blue.

        Keyword arguments are passed to :func:`patchology.visualize.visualize_result`,
        e.g. ``overlay(tiles=False)`` for the masks only or
        ``overlay(slide=slide, max_size=4096)`` for a sharper image.
        """
        return visualize_result(self, **kwargs)


@dataclass(frozen=True)
class TilingPipeline:
    """Detect tissue, remove artifacts and select tiles from a slide.

    Example::

        pipeline = TilingPipeline(tiling=TilingConfig(tile_size=224, tile_mpp=0.5, grid="hex"))
        with open_slide("slide.svs", backend="fastslide") as slide:
            result = pipeline.run(slide)
            result.save("out/slide")
            for (x, y), tile in result.iter_tiles(slide):
                ...
    """

    tiling: TilingConfig = TilingConfig()
    tissue: TissueConfig = TissueConfig()
    artifacts: ArtifactConfig = ArtifactConfig()
    color: ColorConfig = ColorConfig()

    def run(
        self,
        slide: SlideReader,
        tissue_mask: MaskSource | None = None,
        roi: SlideMask | None = None,
    ) -> TilingResult:
        """Tile ``slide``.

        Args:
            slide: An open slide.
            tissue_mask: Optional custom tissue mask covering the whole slide
                (array, PIL image, ``.npy``/image path or :class:`SlideMask`).
                It replaces threshold-based detection. Artifact removal is
                still applied to it as configured in ``self.artifacts``.
            roi: Optional region of interest, e.g. from
                :func:`patchology.tissue.roi.load_roi`. Only tissue inside it
                is tiled.
        """
        if self.color.stain_normalization is not None:
            target = self.color.target_stains  # fail on a bad reference before any reads
            logger.info(
                "%s: stain target H %s, E %s, max concentrations %s",
                slide.path.name,
                np.round(target.matrix[:, 0], 3).tolist(),
                np.round(target.matrix[:, 1], 3).tolist(),
                np.round(target.max_concentrations, 3).tolist(),
            )
        slide_mpp = self.slide_mpp(slide)
        thumbnail, tissue, artifacts = self.detect(slide, tissue_mask, slide_mpp)
        clean = tissue - artifacts if artifacts is not None else tissue
        if roi is not None:
            clean = clean & roi

        cfg = self.tiling
        scale = cfg.tile_mpp / slide_mpp
        if scale < 1 - UPSAMPLE_TOLERANCE:
            logger.warning(
                "tile_mpp %.4g is finer than the slide's %.4g mpp; tiles will be upsampled",
                cfg.tile_mpp,
                slide_mpp,
            )
        level0_tile = cfg.tile_size * scale
        logger.info(
            "%s: slide %d x %d px at %.4g mpp (%s, %d levels); "
            "tile %d px at %.4g mpp = %.1f level-0 px (downsample %.3g), "
            "%s grid, tile_overlap %.2f, tissue_overlap %.2f",
            slide.path.name,
            *slide.dimensions,
            slide_mpp,
            slide.backend,
            slide.level_count,
            cfg.tile_size,
            cfg.tile_mpp,
            level0_tile,
            scale,
            cfg.grid,
            cfg.tile_overlap,
            cfg.tissue_overlap,
        )
        candidates = make_grid(cfg.grid, slide.bounds, level0_tile, cfg.tile_overlap)
        coords = np.round(candidates).astype(np.int64).reshape(-1, 2)
        fractions = tile_fractions(clean, coords, level0_tile)
        keep = fractions >= cfg.tissue_overlap
        if cfg.tissue_overlap == 0:
            keep &= fractions > 0
        num_kept = int(keep.sum())
        logger.info(
            "%s: generated %d tiles (%d grid candidates, %.1f%% tissue after artifact removal)",
            slide.path.name,
            num_kept,
            len(coords),
            100 * clean.area_fraction(),
        )
        if num_kept == 0:
            logger.warning(
                "%s: no tiles kept; check the tissue mask or lower tissue_overlap (%.2f)",
                slide.path.name,
                cfg.tissue_overlap,
            )

        result = TilingResult(
            slide_path=slide.path,
            backend=slide.backend,
            slide_dimensions=slide.dimensions,
            slide_mpp=slide_mpp,
            config=cfg,
            coords=coords[keep],
            tissue_fraction=fractions[keep],
            level0_tile_size=level0_tile,
            tissue_mask=clean,
            artifact_mask=artifacts,
            thumbnail=thumbnail,
            color=self.color,
        )
        if self.color.stain_normalization == "macenko":
            result.source_stains = self._estimate_stains(slide, result)
        return result

    def detect(
        self,
        slide: SlideReader,
        tissue_mask: MaskSource | None = None,
        slide_mpp: float | None = None,
    ) -> tuple[np.ndarray, SlideMask, SlideMask | None]:
        """Return ``(thumbnail, tissue, artifacts)`` without tiling.

        ``tissue`` is the detected (or custom) tissue before artifact removal;
        ``artifacts`` is None when pen and blur removal are both disabled.

        Raises:
            ValueError: The slide has fewer than ``MIN_PYRAMID_LEVELS`` levels.
        """
        if slide.level_count < MIN_PYRAMID_LEVELS:
            raise ValueError(
                f"{slide.path} has a faulty magnification pyramid: {slide.level_count} "
                f"level(s), at least {MIN_PYRAMID_LEVELS} are needed to tile without "
                "running out of memory; re-save it as a pyramidal slide"
            )
        slide_mpp = slide_mpp or self.slide_mpp(slide)
        thumbnail = slide.read_at_downsample(self.tissue.thumbnail_mpp / slide_mpp)
        if self.color.icc:
            thumbnail = slide.apply_icc(thumbnail)
        thumb_mpp = slide_mpp * slide.dimensions[0] / thumbnail.shape[1]

        if tissue_mask is not None:
            tissue = load_mask(tissue_mask, slide.dimensions)
        else:
            detected = detect_tissue(thumbnail, thumb_mpp, self.tissue)
            tissue = SlideMask.from_array(detected, slide.dimensions)

        logger.debug(
            "%s: thumbnail %d x %d px at %.4g mpp, %s tissue %.1f%%",
            slide.path.name,
            thumbnail.shape[1],
            thumbnail.shape[0],
            thumb_mpp,
            "custom" if tissue_mask is not None else "detected",
            100 * tissue.area_fraction(),
        )

        artifacts = None
        if self.artifacts.pen:
            pen = detect_pen_marks(thumbnail, thumb_mpp, self.artifacts)
            artifacts = SlideMask.from_array(pen, slide.dimensions)
            logger.debug("%s: pen marks %.1f%%", slide.path.name, 100 * artifacts.area_fraction())
        if self.artifacts.blur:
            blur = detect_blur(slide, tissue, slide_mpp, self.artifacts)
            logger.debug("%s: blurred %.1f%%", slide.path.name, 100 * blur.area_fraction())
            if artifacts is None:
                artifacts = SlideMask(blur.like(tissue), tissue.downsample)
            else:
                artifacts = artifacts | blur
        return thumbnail, tissue, artifacts

    def _estimate_stains(self, slide: SlideReader, result: TilingResult) -> Stains:
        if len(result) == 0:
            raise ValueError(f"{slide.path}: no tiles to estimate stains from")
        count = min(len(result), self.color.stain_sample_tiles)
        # Tiles are ICC calibrated here since source_stains is not set yet.
        indices = np.linspace(0, len(result) - 1, count).round().astype(int)
        tiles = np.stack(
            list(
                ordered_map(
                    lambda i: result.read_tile(slide, int(i)), indices, default_num_threads()
                )
            )
        )
        try:
            stains = Stains.estimate(tiles)
        except ValueError as e:
            raise ValueError(f"{slide.path}: cannot estimate stains: {e}") from e
        logger.info(
            "%s: macenko stains from %d tiles: H %s, E %s, max concentrations %s",
            slide.path.name,
            count,
            np.round(stains.matrix[:, 0], 3).tolist(),
            np.round(stains.matrix[:, 1], 3).tolist(),
            np.round(stains.max_concentrations, 3).tolist(),
        )
        return stains

    def slide_mpp(self, slide: SlideReader) -> float:
        """Level-0 microns per pixel: the configured override, else the slide metadata."""
        if self.tiling.slide_mpp is not None:
            return self.tiling.slide_mpp
        mpp = slide.mpp
        if mpp is None:
            raise ValueError(
                f"MPP value was not provided: {slide.path} does not report its resolution; "
                "set TilingConfig.slide_mpp"
            )
        mpp_x, mpp_y = mpp
        if abs(mpp_x - mpp_y) / mpp_x > 0.01:
            logger.warning(
                "%s has anisotropic pixels (%g x %g mpp); using x", slide.path, mpp_x, mpp_y
            )
        return mpp_x


def tile_fractions(mask: SlideMask, coords: np.ndarray, level0_tile_size: float) -> np.ndarray:
    """Fraction of each square tile footprint covered by ``mask``.

    Uses a summed-area table, so the cost is independent of tile size.
    Footprints smaller than one mask pixel are measured on that one pixel.
    """
    if len(coords) == 0:
        return np.empty(0)
    height, width = mask.shape
    integral = cv2.integral(mask.array.astype(np.uint8)).astype(np.int64)
    dsx, dsy = mask.downsample
    x0 = np.clip(np.round(coords[:, 0] / dsx).astype(np.int64), 0, width - 1)
    y0 = np.clip(np.round(coords[:, 1] / dsy).astype(np.int64), 0, height - 1)
    x1 = np.clip(np.round((coords[:, 0] + level0_tile_size) / dsx).astype(np.int64), x0 + 1, width)
    y1 = np.clip(np.round((coords[:, 1] + level0_tile_size) / dsy).astype(np.int64), y0 + 1, height)
    covered = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    return covered / ((x1 - x0) * (y1 - y0))


def tile_slide(
    path: str | Path,
    backend: Backend = "openslide",
    tissue_mask: MaskSource | None = None,
    tiling: TilingConfig = TilingConfig(),
    tissue: TissueConfig = TissueConfig(),
    artifacts: ArtifactConfig = ArtifactConfig(),
    color: ColorConfig = ColorConfig(),
) -> TilingResult:
    """Open ``path`` and run :class:`TilingPipeline` on it."""
    pipeline = TilingPipeline(tiling=tiling, tissue=tissue, artifacts=artifacts, color=color)
    with open_slide(path, backend) as slide:
        return pipeline.run(slide, tissue_mask)
