"""Show what the grey-object rules of tissue detection removed and nearly removed.

For calibrating :class:`patchology.GreyObjectConfig` on new scanners, stains or
formats. For each slide it measures every region the rules looked at, writes
them all to ``regions.csv`` and an overlay per slide, and prints the removed
regions and near misses::

    uv run python -m patchology.utilities.inspect_tissue out_dir/ slides_dir/ [more ...] \
        [--max-slides 20] [--backend fastslide] [--set band_max_chroma=24 ...]

Overlay: final tissue tinted green; removed dark regions in red, removed light
bands in orange, near misses in yellow, numbered as in the CSV. A near miss is
a dark grey candidate that was kept (grainy and not touching the border or an
artifact), or a tissue region that failed exactly one light-band test. Try a
value with ``--set`` and compare the output before changing a default.
"""

import argparse
import csv
import math
from dataclasses import asdict, fields, replace
from pathlib import Path

import cv2
import numpy as np

from patchology import GreyObjectConfig, TilingPipeline, TissueConfig, open_slide
from patchology.tissue import GreyRegion, tissue_regions
from patchology.utilities.estimate_stains import find_slides

COLORS = {"dark": (220, 30, 30), "band": (255, 140, 0), "near_miss": (240, 220, 0)}
TISSUE_TINT = np.array([0, 170, 0])


def status(region: GreyRegion, config: GreyObjectConfig) -> str:
    """``"removed"``, ``"near_miss"`` or ``"kept"``."""
    if region.removed:
        return "removed"
    if region.rule == "dark":
        return "near_miss"
    passed = (
        region.chroma < config.band_max_chroma,
        region.value >= config.max_value,
        region.straight_fraction >= config.straight_min_fraction,
    )
    return "near_miss" if sum(passed) == 2 else "kept"


def parse_overrides(pairs: list[str]) -> GreyObjectConfig:
    types = {f.name: type(f.default) for f in fields(GreyObjectConfig)}
    overrides = {}
    for pair in pairs:
        name, _, value = pair.partition("=")
        if name not in types:
            raise SystemExit(f"--set {pair}: unknown field, known: {', '.join(types)}")
        overrides[name] = types[name](value)
    return replace(GreyObjectConfig(), **overrides)


def overlay(thumbnail: np.ndarray, mask: np.ndarray, rows: list[dict]) -> np.ndarray:
    out = thumbnail.copy()
    out[mask] = (out[mask] * 0.65 + TISSUE_TINT * 0.35).astype(np.uint8)
    thick = max(2, out.shape[1] // 700)  # readable on large thumbnails
    for row in rows:
        if row["status"] == "kept":
            continue
        color = COLORS["near_miss" if row["status"] == "near_miss" else row["rule"]]
        x, y, w, h = row["x"], row["y"], row["w"], row["h"]
        cv2.rectangle(out, (x, y), (x + w - 1, y + h - 1), color, thick)
        cv2.putText(out, str(row["index"]), (x + thick, y + 12 * thick), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4 * thick, color, thick // 2 + 1, cv2.LINE_AA)  # fmt: skip
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("sources", type=Path, nargs="+", help="slide files or directories")
    parser.add_argument("--max-slides", type=int, default=20, help="evenly sampled (default 20)")
    parser.add_argument("--backend", default="openslide", choices=["openslide", "fastslide"])
    parser.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                        help="override a GreyObjectConfig field; repeatable")  # fmt: skip
    args = parser.parse_args()
    try:
        grey = parse_overrides(args.set)
    except ValueError as e:
        raise SystemExit(f"--set: {e}") from e
    print(grey)

    slides = find_slides(args.sources)
    if not slides:
        parser.error("no slides found")
    if len(slides) > args.max_slides:
        slides = [slides[i] for i in np.linspace(0, len(slides) - 1, args.max_slides).astype(int)]
    tissue = TissueConfig(grey_objects=grey)
    pipeline = TilingPipeline(tissue=tissue)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for n, path in enumerate(slides, 1):
        try:
            with open_slide(path, args.backend) as slide:
                # The same thumbnail tiling detects on.
                thumbnail = pipeline.detect(slide)[0]
                mpp = pipeline.slide_mpp(slide) * slide.dimensions[0] / thumbnail.shape[1]
        except (OSError, RuntimeError, ValueError) as e:  # one bad slide must not stop the run
            print(f"\n[{n}/{len(slides)}] skipped {path}: {e}")
            continue
        mask, regions = tissue_regions(thumbnail, mpp, tissue)
        rows = []
        for i, region in enumerate(regions):
            row = {"slide": str(path), "index": i, "status": status(region, grey)}
            row |= {k: v for k, v in asdict(region).items() if k != "bbox"}
            row |= dict(zip("xywh", region.bbox, strict=True))
            rows.append(row)
        image = overlay(thumbnail, mask, rows)
        cv2.imwrite(str(args.out_dir / f"{n:03d}_{path.stem}.png"), image[..., ::-1])
        all_rows += rows

        shown = [r for r in rows if r["status"] != "kept"]
        print(f"\n[{n}/{len(slides)}] {path}: tissue {mask.mean():.1%}, "
              f"{len(shown)} removed or near miss of {len(rows)} regions")  # fmt: skip
        for r in shown:
            extra = (f"texture {r['texture']:.1f}, touching {r['touching']}" if r["rule"] == "dark"
                     else f"straight {r['straight_fraction']:.2f}")  # fmt: skip
            print(f"  #{r['index']:<3} {r['rule']:4} {r['status']:9} {r['area_um2'] / 1e6:7.3f} mm2  "
                  f"value {r['value']:5.1f}  chroma {r['chroma']:5.1f}  {extra}")  # fmt: skip

    if all_rows:
        with open(args.out_dir / "regions.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(
                {k: ("" if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()}
                for r in all_rows
            )
    print(f"\nwrote {args.out_dir}")


if __name__ == "__main__":
    main()
