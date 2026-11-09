"""Estimate a cohort's reference H&E stains from a set of slides.

Each slide's stains are estimated with Macenko, exactly as tiling does with
``ColorConfig(stain_normalization="macenko")``; the reference is the
per-component median over slides, which outlier slides cannot pull. Directories
are searched recursively for slide files; pass files to choose them yourself::

    uv run python -m patchology.utilities.estimate_stains out.json slides_dir/ [more ...] \
        [--max-slides 100] [--backend fastslide] [--icc]

Then normalise to it with ``ColorConfig(stain_normalization="macenko",
stain_target="out.json")``. Use H&E slides only: Macenko assumes two stains, so
IHC slides give wrong vectors. Tile with the same ``icc`` setting as here.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from patchology import ColorConfig, Stains, TilingConfig, TilingPipeline, open_slide

# No .dcm: a DICOM slide is many files, so pass those yourself.
SLIDE_SUFFIXES = {
    ".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn", ".vms", ".vmu", ".svslide", ".bif",
    ".isyntax", ".czi",
}  # fmt: skip


def find_slides(sources: list[Path]) -> list[Path]:
    slides = set()
    for source in sources:
        if source.is_dir():
            slides.update(p for p in source.rglob("*") if p.suffix.lower() in SLIDE_SUFFIXES)
        else:
            slides.add(source)
    return sorted(slides)


def median_stains(stains: list[Stains]) -> Stains:
    """Per-component median of slide stains, with unit stain vectors."""
    matrix = np.median([s.matrix for s in stains], axis=0)
    max_concentrations = np.median([s.max_concentrations for s in stains], axis=0)
    return Stains(matrix / np.linalg.norm(matrix, axis=0), max_concentrations)


def angles_deg(stains: Stains, reference: Stains) -> np.ndarray:
    """Angle in degrees between each stain vector and the reference's (H, E)."""
    cosines = (stains.matrix * reference.matrix).sum(axis=0).clip(-1, 1)
    return np.degrees(np.arccos(cosines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", type=Path, help="output .json")
    parser.add_argument("sources", type=Path, nargs="+", help="slide files or directories")
    parser.add_argument("--max-slides", type=int, default=100, help="evenly sampled (default 100)")
    parser.add_argument("--backend", default="openslide", choices=["openslide", "fastslide"])
    parser.add_argument("--icc", action="store_true", help="estimate on ICC-calibrated tiles")
    args = parser.parse_args()
    if args.out.suffix.lower() != ".json":
        parser.error("out must be a .json file")

    slides = find_slides(args.sources)
    if not slides:
        parser.error("no slides found")
    if len(slides) > args.max_slides:
        slides = [slides[i] for i in np.linspace(0, len(slides) - 1, args.max_slides).astype(int)]
    pipeline = TilingPipeline(
        tiling=TilingConfig(),
        color=ColorConfig(icc=args.icc, stain_normalization="macenko"),
    )

    estimates: dict[Path, Stains] = {}
    for i, path in enumerate(slides, 1):
        try:
            with open_slide(path, args.backend) as slide:
                estimates[path] = pipeline.run(slide).source_stains
        except (OSError, RuntimeError, ValueError) as e:  # one bad slide must not stop the run
            print(f"[{i}/{len(slides)}] skipped {path}: {e}")
            continue
        print(f"[{i}/{len(slides)}] {path}")
    if not estimates:
        raise SystemExit("no slide gave a stain estimate")

    reference = median_stains(list(estimates.values()))
    print(f"\nreference from {len(estimates)} of {len(slides)} slides: {reference.to_dict()}")
    print("slides furthest from it (H / E angle, degrees; large = check stain type or quality):")
    deviation = {path: angles_deg(s, reference) for path, s in estimates.items()}
    for path, (h, e) in sorted(deviation.items(), key=lambda kv: -kv[1].max())[:10]:
        print(f"  {h:5.1f} / {e:5.1f}  {path}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({**reference.to_dict(), "slides": len(estimates)}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
