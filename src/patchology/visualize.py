"""Overlays of tissue masks and tile grids on slide thumbnails.

The building blocks :func:`draw_mask` and :func:`draw_tiles` draw onto an RGB
array whose scale is given as ``downsample`` (level-0 pixels per image pixel).
:func:`visualize_slide` and :func:`visualize_result` combine them and return a
:class:`PIL.Image.Image`, which displays inline in Jupyter or can be saved.

Tiles saved earlier can be shown without rerunning the pipeline::

    coords = np.loadtxt("out/tiles.csv", delimiter=",", skiprows=1, usecols=(0, 1))
    tile_size = json.load(open("out/metadata.json"))["level0_tile_size"]
    with open_slide("slide.svs") as slide:
        visualize_slide(slide, tissue_mask="out/tissue_mask.png", coords=coords,
                        tile_size=tile_size).show()
"""

from __future__ import annotations

import colorsys
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np
from PIL import Image

from patchology.reader import SlideReader
from patchology.tissue.mask import MaskSource, SlideMask, load_mask
from patchology.tissue.roi import Polygon, rasterize_polygons

if TYPE_CHECKING:
    from patchology.tiling.generator import TileGenerator
    from patchology.tiling.tiler import TilingResult

type Color = tuple[int, int, int]
type TileShape = Literal["square", "hex"]

TISSUE_COLOR: Color = (0, 200, 0)
ARTIFACT_COLOR: Color = (230, 0, 0)
TILE_COLOR: Color = (0, 0, 255)
HEX_TILE_COLOR: Color = (0, 0, 0)
GEOJSON_TILE_COLOR: Color = (0, 0, 0)

# matplotlib's tab10, used for up to 10 ground-truth labels.
TAB10: list[Color] = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
    (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207),
]  # fmt: skip


def make_palette(labels: Sequence[int]) -> dict[int, Color]:
    """A distinct colour per label: tab10 for up to 10 labels, else golden-ratio spaced hues."""
    labels = sorted(set(labels))
    if len(labels) <= len(TAB10):
        return dict(zip(labels, TAB10))
    # Golden-ratio steps put consecutive labels far apart on the colour wheel.
    hues = [colorsys.hsv_to_rgb(i * 0.618034 % 1, 0.9, 0.9) for i in range(len(labels))]
    return {label: tuple(round(c * 255) for c in rgb) for label, rgb in zip(labels, hues)}


def draw_labels(
    image: np.ndarray,
    downsample: tuple[float, float],
    polygons: list[Polygon],
    labels: Sequence[int],
    colors: Mapping[int, Color] | None = None,
    alpha: float = 0.5,
    outline: int = 2,
) -> np.ndarray:
    """Tint each level-0 polygon in its label's colour and outline it.

    Args:
        image: ``(H, W, 3)`` uint8 RGB thumbnail; not modified.
        downsample: ``(x, y)`` level-0 pixels per ``image`` pixel.
        polygons: Level-0 polygons, as loaded from GeoJSON.
        labels: Label of each polygon.
        colors: RGB colour per label; :func:`make_palette` by default.
        alpha: Tint opacity in [0, 1].
        outline: Contour thickness in pixels; 0 draws only the tint.
    """
    colors = colors or make_palette(labels)
    scale = (1 / downsample[0], 1 / downsample[1])
    out = image.copy()
    for label in sorted(set(labels)):
        chosen = [p for p, lab in zip(polygons, labels, strict=True) if lab == label]
        region = rasterize_polygons(chosen, out.shape[:2], scale)
        if alpha > 0:
            tinted = (
                out[region > 0].astype(np.float32) * (1 - alpha) + np.array(colors[label]) * alpha
            )
            out[region > 0] = tinted.round().astype(np.uint8)
        if outline > 0:
            contours, _ = cv2.findContours(region, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, colors[label], outline)
    return out


def draw_mask(
    image: np.ndarray,
    downsample: tuple[float, float],
    mask: SlideMask,
    color: Color = TISSUE_COLOR,
    alpha: float = 0.35,
    outline: int = 2,
) -> np.ndarray:
    """Tint ``mask`` onto ``image`` and outline its contours.

    Args:
        image: ``(H, W, 3)`` uint8 RGB thumbnail; not modified.
        downsample: ``(x, y)`` level-0 pixels per ``image`` pixel.
        mask: Mask to draw, at any resolution.
        color: RGB colour of the tint and outline.
        alpha: Tint opacity in [0, 1]; 0 draws only the outline.
        outline: Contour thickness in pixels; 0 draws only the tint.
    """
    region = mask.sample(downsample, image.shape[:2])
    out = image.copy()
    if alpha > 0:
        tinted = out[region].astype(np.float32) * (1 - alpha) + np.array(color) * alpha
        out[region] = tinted.round().astype(np.uint8)
    if outline > 0:
        contours, _ = cv2.findContours(
            region.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, color, outline)
    return out


def draw_tiles(
    image: np.ndarray,
    downsample: tuple[float, float],
    coords: np.ndarray,
    tile_size: float,
    color: Color = TILE_COLOR,
    thickness: int = 1,
    fill_alpha: float = 0.0,
    shape: TileShape = "square",
) -> np.ndarray:
    """Draw tile outlines onto ``image``.

    Args:
        image: ``(H, W, 3)`` uint8 RGB thumbnail; not modified.
        downsample: ``(x, y)`` level-0 pixels per ``image`` pixel.
        coords: ``(N, 2)`` level-0 ``(x, y)`` top-left corners.
        tile_size: Tile side length in level-0 pixels.
        color: RGB outline colour.
        thickness: Outline thickness in pixels.
        fill_alpha: Opacity of a fill inside each tile, 0 for outlines only.
        shape: ``"square"`` draws the tile itself. ``"hex"`` draws a
            pointy-top hexagon centred on the tile, ``tile_size`` wide; on a
            hex grid with ``tile_overlap=0`` these tessellate exactly.
    """
    out = image.copy()
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 2)
    if len(coords) == 0:
        return out
    polygons = _tile_polygons(coords, tile_size, shape) / np.asarray(downsample)
    polygons = np.round(polygons).astype(np.int32)
    if fill_alpha > 0:
        filled = np.zeros(out.shape[:2], dtype=np.uint8)
        cv2.fillPoly(filled, list(polygons), 1)
        region = filled.astype(bool)
        tinted = out[region].astype(np.float32) * (1 - fill_alpha) + np.array(color) * fill_alpha
        out[region] = tinted.round().astype(np.uint8)
    cv2.polylines(out, list(polygons), True, color, thickness)
    return out


def _tile_polygons(coords: np.ndarray, tile_size: float, shape: TileShape) -> np.ndarray:
    """``(N, K, 2)`` level-0 outline vertices of each tile."""
    half = tile_size / 2
    if shape == "square":
        offsets = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * half
    elif shape == "hex":
        radius = tile_size / math.sqrt(3)  # flat sides at the tile's left and right edges
        angles = np.deg2rad(np.arange(-90, 270, 60))
        offsets = np.stack([np.cos(angles), np.sin(angles)], axis=1) * radius
    else:
        raise ValueError(f"shape must be 'square' or 'hex', got {shape!r}")
    centres = coords + half
    return centres[:, None, :] + offsets[None, :, :]


def render_overlay(
    thumbnail: np.ndarray,
    slide_dimensions: tuple[int, int],
    *,
    tissue_mask: SlideMask | None = None,
    artifact_mask: SlideMask | None = None,
    coords: np.ndarray | None = None,
    tile_size: float | None = None,
    max_size: int | None = None,
    mask_alpha: float = 0.35,
    tile_thickness: int = 1,
    tile_shape: TileShape = "square",
    tile_color: Color | None = None,
    ground_truth: tuple[list[Polygon], Sequence[int]] | None = None,
    label_colors: Mapping[int, Color] | None = None,
) -> Image.Image:
    """Draw any combination of masks, ground truth and tiles over ``thumbnail``.

    Tiles are drawn in ``tile_color``, by default blue for squares and black
    for hexagons. ``ground_truth`` is ``(polygons, labels)`` drawn with
    :func:`draw_labels` in ``label_colors``.

    ``thumbnail`` must cover the whole slide. If ``max_size`` is given it is
    first resized so its longer side is ``max_size`` pixels, which makes small
    tiles legible on a low-resolution thumbnail.
    """
    image = thumbnail
    if max_size is not None:
        image = _fit(image, max_size)
    width, height = slide_dimensions
    downsample = (width / image.shape[1], height / image.shape[0])

    if tissue_mask is not None:
        image = draw_mask(image, downsample, tissue_mask, TISSUE_COLOR, mask_alpha)
    if artifact_mask is not None:
        image = draw_mask(image, downsample, artifact_mask, ARTIFACT_COLOR, max(mask_alpha, 0.5))
    if ground_truth is not None:
        image = draw_labels(image, downsample, *ground_truth, label_colors)
    if coords is not None:
        if tile_size is None:
            raise ValueError("tile_size (in level-0 pixels) is required to draw coords")
        if tile_color is None:
            tile_color = HEX_TILE_COLOR if tile_shape == "hex" else TILE_COLOR
        image = draw_tiles(
            image, downsample, coords, tile_size, tile_color, tile_thickness, shape=tile_shape
        )
    return Image.fromarray(image)


def visualize_slide(
    slide: SlideReader,
    *,
    tissue_mask: MaskSource | None = None,
    artifact_mask: MaskSource | None = None,
    coords: np.ndarray | None = None,
    tile_size: float | None = None,
    max_size: int = 2048,
    mask_alpha: float = 0.35,
    tile_thickness: int = 1,
    tile_shape: TileShape = "square",
    tile_color: Color | None = None,
) -> Image.Image:
    """Read a thumbnail of ``slide`` and overlay masks and/or tiles on it.

    Args:
        slide: Open slide.
        tissue_mask: Mask drawn in green; anything :func:`load_mask` accepts.
        artifact_mask: Mask drawn in red.
        coords: ``(N, 2)`` level-0 tile corners.
        tile_size: Tile side length in level-0 pixels; required with ``coords``.
        max_size: Longer side of the thumbnail in pixels.
        mask_alpha: Opacity of the tissue tint.
        tile_thickness: Box outline thickness in pixels.
        tile_shape: ``"square"`` or ``"hex"`` outlines; see :func:`draw_tiles`.
        tile_color: Outline colour; blue for squares, black for hexagons.
    """
    width, height = slide.dimensions
    thumbnail = slide.read_at_downsample(max(width, height) / max_size)
    return render_overlay(
        thumbnail,
        slide.dimensions,
        tissue_mask=None if tissue_mask is None else load_mask(tissue_mask, slide.dimensions),
        artifact_mask=None if artifact_mask is None else load_mask(artifact_mask, slide.dimensions),
        coords=coords,
        tile_size=tile_size,
        mask_alpha=mask_alpha,
        tile_thickness=tile_thickness,
        tile_shape=tile_shape,
        tile_color=tile_color,
    )


def visualize_result(
    result: TilingResult,
    *,
    tissue: bool = True,
    artifacts: bool = True,
    tiles: bool = True,
    slide: SlideReader | None = None,
    max_size: int | None = None,
    mask_alpha: float = 0.35,
    tile_thickness: int = 1,
    tile_shape: TileShape | None = None,
    tile_color: Color | None = None,
    ground_truth: tuple[list[Polygon], Sequence[int]] | None = None,
    label_colors: Mapping[int, Color] | None = None,
) -> Image.Image:
    """Overlay a :class:`TilingResult`'s masks and selected tiles on its thumbnail.

    Args:
        result: Output of :meth:`TilingPipeline.run`.
        tissue: Draw the tissue mask (after artifact removal) in green.
        artifacts: Draw removed artifacts in red.
        tiles: Draw the selected tiles.
        slide: If given, read a fresh thumbnail of ``max_size`` (default 2048)
            instead of upscaling the low-resolution detection thumbnail.
        max_size: Longer side of the output in pixels; defaults to the size of
            the detection thumbnail.
        mask_alpha: Opacity of the tissue tint.
        tile_thickness: Box outline thickness in pixels.
        tile_shape: ``"square"`` or ``"hex"`` outlines; defaults to the
            result's grid type.
        tile_color: Outline colour; blue for squares, black for hexagons.
        ground_truth: ``(polygons, labels)`` to draw, e.g. from a
            :class:`SegmentationTileGenerator`.
        label_colors: RGB colour per label; :func:`make_palette` by default.
    """
    thumbnail = result.thumbnail
    if slide is not None:
        width, height = slide.dimensions
        thumbnail = slide.read_at_downsample(max(width, height) / (max_size or 2048))
    return render_overlay(
        thumbnail,
        result.slide_dimensions,
        tissue_mask=result.tissue_mask if tissue else None,
        artifact_mask=result.artifact_mask if artifacts else None,
        coords=result.coords if tiles else None,
        tile_size=result.level0_tile_size,
        max_size=max_size,
        mask_alpha=mask_alpha,
        tile_thickness=tile_thickness,
        tile_shape=tile_shape or result.config.grid,
        tile_color=tile_color,
        ground_truth=ground_truth,
        label_colors=label_colors,
    )


def tiles_to_geojson(
    generator: TileGenerator,
    path: str | Path | None = None,
    color: Color = GEOJSON_TILE_COLOR,
    name: str = "tile",
) -> dict:
    """Tile bounding boxes as a GeoJSON FeatureCollection for QuPath.

    Boxes are in level-0 (full resolution) pixels, the coordinates QuPath
    uses, so the file can be dragged onto the slide in QuPath to view them.
    Each tile is a QuPath detection classified as ``name`` with line colour
    ``color``, and carries its index and level-0 ``x``/``y`` as measurements.

    Args:
        generator: A :class:`TileGenerator` (or subclass); runs its pipeline
            if needed.
        path: If given, also write the GeoJSON there.
        color: ``(r, g, b)`` outline colour, black by default.
        name: QuPath classification name of the tiles.
    """
    result = generator.result
    boxes = _tile_polygons(result.coords.astype(float), result.level0_tile_size, "square")
    features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                # GeoJSON rings are closed: the first vertex is repeated last.
                "coordinates": [np.vstack([box, box[:1]]).round(2).tolist()],
            },
            "properties": {
                # Detections render and load faster than annotations for thousands of tiles.
                "objectType": "detection",
                "classification": {"name": name, "color": list(color)},
                "color": list(color),
                "measurements": {"tile": i, "x": float(x), "y": float(y)},
            },
        }
        for i, (box, (x, y)) in enumerate(zip(boxes, result.coords))
    ]
    data = {"type": "FeatureCollection", "features": features}
    if path is not None:
        Path(path).write_text(json.dumps(data))
    return data


def _fit(image: np.ndarray, max_size: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = max_size / max(height, width)
    if scale == 1:
        return image
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(image, size, interpolation=interpolation)
