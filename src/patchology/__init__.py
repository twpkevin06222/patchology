"""patchology: whole slide image tissue detection, artifact removal and tiling."""

from patchology.color import (
    REFERENCE_STAINS,
    ColorConfig,
    Stains,
    load_stains,
    macenko_normalize,
)
from patchology.log import configure_logging
from patchology.reader import Backend, SlideOpenError, SlideReader, open_slide
from patchology.tiling import (
    DetectionTileGenerator,
    SegmentationTileGenerator,
    TileGenerator,
    TilingConfig,
    TilingPipeline,
    TilingResult,
    tile_slide,
)
from patchology.tissue import (
    ArtifactConfig,
    SlideMask,
    TissueConfig,
    detect_pen_marks,
    detect_tissue,
    load_mask,
    load_roi,
)
from patchology.visualize import (
    draw_mask,
    draw_tiles,
    tiles_to_geojson,
    visualize_result,
    visualize_slide,
)

configure_logging("patchology")

__all__ = [
    "REFERENCE_STAINS",
    "ArtifactConfig",
    "Backend",
    "ColorConfig",
    "DetectionTileGenerator",
    "SegmentationTileGenerator",
    "SlideMask",
    "SlideOpenError",
    "SlideReader",
    "Stains",
    "TileGenerator",
    "TilingConfig",
    "TilingPipeline",
    "TilingResult",
    "TissueConfig",
    "detect_pen_marks",
    "detect_tissue",
    "draw_mask",
    "draw_tiles",
    "load_mask",
    "load_roi",
    "load_stains",
    "macenko_normalize",
    "open_slide",
    "tile_slide",
    "tiles_to_geojson",
    "visualize_result",
    "visualize_slide",
]
