"""Time OpenSlide against FastSlide on the same slides.

For each slide, backend and thread count: open the slide, run tissue detection
and tiling, then read every tile; peak memory is the run's resident high-water
mark. Each run is a fresh process after a warm-up read of the file, so both
backends start from the same OS file cache. Colour processing is
off so only the backend is timed. Slides are printed as A, B, ... so names
never reach the output::

    uv run python -m patchology.utilities.benchmark_backends slide1.svs [slide2.svs ...]
"""

import json
import multiprocessing as mp
import queue as queue_module
import statistics
import string
import sys
import time
from pathlib import Path

from patchology import ColorConfig, TileGenerator, TilingConfig
from patchology.tiling.tiler import default_num_threads

BACKENDS = ("openslide", "fastslide")
TILING = TilingConfig(tile_size=224, tile_mpp=0.5)
REPEATS = 3
THREADS = (1, default_num_threads())


def _run(path: str, backend: str, num_threads: int, queue: mp.Queue) -> None:
    t0 = time.perf_counter()
    gen = TileGenerator(
        path, backend=backend, tiling=TILING, color=ColorConfig(icc=False), num_threads=num_threads
    )
    t1 = time.perf_counter()
    n = len(gen)
    t2 = time.perf_counter()
    for _ in gen:
        pass
    t3 = time.perf_counter()
    queue.put(
        {
            "backend": gen.backend,
            "tiles": n,
            "dimensions": gen.dimensions,
            "open": t1 - t0,
            "tiling": t2 - t1,
            "read": t3 - t2,
            "peak_mb": _peak_rss_mb(),
        }
    )
    gen.close()


def _peak_rss_mb() -> int:
    # VmHWM starts afresh in each process; ru_maxrss would carry the parent's over exec.
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) // 1024
    return -1


def _warm_cache(path: Path) -> None:
    with path.open("rb") as f:
        while f.read(16 << 20):
            pass


def time_backend(path: Path, backend: str, num_threads: int) -> dict:
    """Median of ``REPEATS`` fresh-process runs."""
    runs = []
    for _ in range(REPEATS):
        queue = mp.get_context("spawn").Queue()
        proc = mp.get_context("spawn").Process(
            target=_run, args=(str(path), backend, num_threads, queue)
        )
        proc.start()
        while True:
            try:
                runs.append(queue.get(timeout=1))
                break
            except queue_module.Empty:
                # A crashed run never puts a result; don't wait for it forever.
                if not proc.is_alive():
                    raise RuntimeError(f"{backend} run exited with code {proc.exitcode}")
        proc.join()
    # The reader falls back to the other backend if one fails; never report that as a win.
    if any(r["backend"] != backend for r in runs):
        raise RuntimeError(f"{backend} could not open the slide (fell back to another backend)")
    keys = ("open", "tiling", "read")
    medians = {k: statistics.median(r[k] for r in runs) for k in keys}
    return {
        **runs[0],
        **medians,
        "threads": num_threads,
        "peak_mb": max(r["peak_mb"] for r in runs),
    }


def main(paths: list[Path]) -> None:
    for label, path in zip(string.ascii_uppercase, paths):
        _warm_cache(path)  # once for both backends
        for backend in BACKENDS:
            for num_threads in THREADS:
                try:
                    r = time_backend(path, backend, num_threads)
                except RuntimeError as e:
                    r = {"slide": label, "backend": backend, "error": str(e)}
                    print(json.dumps(r), flush=True)
                    break  # more threads will not fix a crash
                r["slide"], r["size_mb"] = label, round(path.stat().st_size / 1e6)
                r["tiles_per_s"] = round(r["tiles"] / r["read"])
                print(json.dumps(r), flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main([Path(p) for p in sys.argv[1:]])
