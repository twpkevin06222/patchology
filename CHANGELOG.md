# Changelog

All notable changes to patchology are listed here. Versions follow
[Semantic Versioning](https://semver.org/): before 1.0.0, a minor release
(0.2.0) may contain breaking changes and a patch release (0.1.1) contains
fixes only.

## 0.1.1

### Dependencies

- Require FastSlide 0.8.3 or newer (was 0.8.0). FastSlide now lays out BIF
  slides as OpenSlide does, reads the MRXS background colour correctly, matches
  the Philips SDK pixels for iSyntax slides, and uses its tile cache for TIFF
  slides. The FastSlide backend therefore gives the same tiles as the
  OpenSlide backend on more slide formats.

## 0.1.0 — initial release

First public release. Requires Python 3.12+; tested on Python 3.12 and 3.13
on Linux.

### Slide reading

- `open_slide` opens slides with OpenSlide or FastSlide behind one
  OpenSlide-compatible `SlideReader` interface. Both backends install with
  patchology. If the preferred backend fails to open a slide, the other one is
  tried, and `SlideOpenError` lists each backend's error.
- Helpers for microns per pixel, slide bounds, scaled reads and reads at any
  downsample.

### Tissue detection and artifact removal

- Threshold-based tissue detection on a low-resolution thumbnail (`triangle`,
  `otsu`, `otsu_gray` or `fixed`), with sizes in microns (`TissueConfig`).
- Grey-object rules drop dark markers, labels and slide edges, and light grey
  straight bands along glass and coverslip edges. Their thresholds are set in
  `GreyObjectConfig`, and `tissue_regions` reports what each rule measured.
- Pen ink and blur detection (`ArtifactConfig`).
- Custom tissue masks and GeoJSON regions of interest (QuPath style), with
  annotation classes that can be excluded.

### Tiling

- Square or hexagonal tile grids at a chosen tile size and microns per pixel,
  with overlap and a minimum tissue fraction per tile (`TilingConfig`).
- `TilingPipeline`, `TileGenerator` and `tile_slide`, with multi-threaded tile
  reads that keep tile order.
- `SegmentationTileGenerator` yields tiles with label masks, and
  `DetectionTileGenerator` yields tiles with COCO-style boxes, both from
  GeoJSON ground truth.
- `TilingResult.save` writes tile coordinates, metadata, the tissue mask and an
  overlay.

### Colour

- Optional ICC calibration of tiles to sRGB (`ColorConfig(icc=True)`, off by
  default). Tissue detection always runs on the stored pixels.
- Macenko stain normalisation to the classic reference stains, a reference
  image, or a `.json` file of saved stains.

### Visualisation

- Overlays of tissue, artifacts, tiles and ground truth on the slide
  thumbnail, and GeoJSON export of tiles for QuPath.

### Utilities

- `estimate_stains`: a cohort's reference stains as the median over slides,
  written to JSON, with the slides furthest from it listed.
- `inspect_tissue`: overlays and a CSV of what the grey-object rules removed
  and nearly removed, for recalibrating `GreyObjectConfig`.
- `xml_to_geojson`: ASAP XML annotations to GeoJSON.
- `benchmark_backends`, `plot_color` and `plot_overlays` for the README
  benchmarks and figures.
