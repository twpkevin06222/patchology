"""Tiles as stored, after ICC calibration, and after Macenko stain normalisation
with and without ICC calibration first.

Rows are the tiles with the most tissue, spread over the slide. Needs
matplotlib, which is not a patchology dependency::

    uv run --with matplotlib python -m patchology.utilities.plot_color slide.svs [reference.png] [out_dir]

Without a reference image the tiles are normalised to ``REFERENCE_STAINS``.
"""

import sys
from pathlib import Path

import numpy as np

from patchology import REFERENCE_STAINS, ColorConfig, TilingConfig, TilingPipeline, open_slide

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
TILE, MPP, ROWS = 512, 0.5, 3


def plot(slide_path: Path, reference: Path | None, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    tiling = TilingConfig(tile_size=TILE, tile_mpp=MPP, tissue_overlap=0.9)
    target_stains = reference or REFERENCE_STAINS
    with open_slide(slide_path, backend="openslide") as slide:
        result = TilingPipeline(
            tiling=tiling, color=ColorConfig(stain_normalization="macenko", stain_target=target_stains)
        ).run(slide)
        # Stains re-estimated on uncalibrated colours, for Macenko without ICC.
        no_icc = TilingPipeline(
            tiling=tiling,
            color=ColorConfig(icc=False, stain_normalization="macenko", stain_target=target_stains),
        ).run(slide)
        # Most-tissue tiles first, then pick ROWS spread over that top tenth.
        best = np.argsort(-result.tissue_fraction)[: max(ROWS, len(result) // 10)]
        rows = best[np.linspace(0, len(best) - 1, ROWS).round().astype(int)]
        panels = []
        for i in rows:
            x, y = (int(v) for v in result.coords[i])
            raw = slide.read_region_scaled((x, y), result.downsample, (TILE, TILE))
            icc = slide.apply_icc(raw)
            panels.append(
                ((x, y), [raw, icc, result.read_tile(slide, int(i)), no_icc.read_tile_at(slide, x, y)])
            )
        has_icc = slide.icc_profile is not None

    target = reference.name if reference else "REFERENCE_STAINS"
    titles = [
        "(a) Original",
        "(b) ICC calibrated" if has_icc else "(b) ICC calibrated (no profile: unchanged)",
        f"(c) ICC + Macenko\n(target: {target})",
        f"(d) Original + Macenko\n(target: {target})",
    ]
    fig, axes = plt.subplots(ROWS, 4, figsize=(16, 4 * ROWS), dpi=150, squeeze=False)
    for r, ((x, y), images) in enumerate(panels):
        for c, image in enumerate(images):
            ax = axes[r, c]
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=12)
        axes[r, 0].set_ylabel(f"tile ({x}, {y})", fontsize=10)
    fig.suptitle(f"{TILE} px tiles at {MPP} mpp", fontsize=13)  # no slide name: anonymised
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    args = [Path(a) for a in sys.argv[1:]]
    reference = args[1] if len(args) > 1 else None
    out_dir = args[2] if len(args) > 2 else FIGURES_DIR
    plot(args[0], reference, out_dir / "color_comparison.png")
