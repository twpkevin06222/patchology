"""Tissue mask, hex grid and square grid overlays for one slide, side by side.

Writes one figure without tile overlap and one with 20% overlap. Needs
matplotlib, which is not a patchology dependency::

    uv run --with matplotlib python -m patchology.utilities.plot_overlays slide.svs [out_dir]
"""

import sys
from pathlib import Path

from patchology import TilingConfig, TilingPipeline, open_slide

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
TILE, MPP, OVERLAP = 512, 0.5, 0.1
TILE_OVERLAPS = {"overlays.png": 0.0, "overlays_overlap20.png": 0.2}


def plot(slide, out_path: Path, tile_overlap: float) -> None:
    import matplotlib.pyplot as plt

    results = {
        grid: TilingPipeline(
            tiling=TilingConfig(
                tile_size=TILE,
                tile_mpp=MPP,
                tile_overlap=tile_overlap,
                grid=grid,
                tissue_overlap=OVERLAP,
            )
        ).run(slide)
        for grid in ("hex", "square")
    }
    hex_result, square_result = results["hex"], results["square"]
    tissue = hex_result.tissue_mask.area_fraction()
    panels = [
        (
            hex_result.overlay(tiles=False, slide=slide, max_size=2048),
            f"(a) Tissue mask\n{tissue:.1%} of slide is tissue",
        ),
        (
            hex_result.overlay(tissue=False, artifacts=False, slide=slide, max_size=2048),
            f"(b) Hexagonal grid\n{len(hex_result.coords)} tiles",
        ),
        (
            square_result.overlay(tissue=False, artifacts=False, slide=slide, max_size=2048),
            f"(c) Rectangular grid\n{len(square_result.coords)} tiles",
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), dpi=200)
    for ax, (image, caption) in zip(axes, panels):
        ax.imshow(image)
        ax.set_axis_off()
        ax.set_title(caption, fontsize=13, y=-0.05, va="top")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(out_path)


def main(slide_path: Path, out_dir: Path) -> None:
    with open_slide(slide_path, backend="openslide") as slide:
        for name, tile_overlap in TILE_OVERLAPS.items():
            plot(slide, out_dir / name, tile_overlap)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else FIGURES_DIR)
