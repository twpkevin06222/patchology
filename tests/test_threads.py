import threading
import time

import numpy as np
import pytest

from patchology import ColorConfig, TileGenerator, TilingConfig
from patchology.tiling import tiler
from patchology.tiling.tiler import default_num_threads, ordered_map

from conftest import ArrayReader

CONFIG = TilingConfig(tile_size=128, tile_mpp=1.0)


def test_ordered_map_keeps_order_and_bounds_memory():
    started, lock = [], threading.Lock()

    def work(i):
        with lock:
            started.append(i)
        time.sleep(0.001 * (i % 3))  # finish out of order
        return i * i

    consumed = 0
    for i, value in enumerate(ordered_map(work, range(200), num_threads=4)):
        assert value == i * i
        consumed += 1
        # Never more than 2 * num_threads results ahead of the consumer.
        assert len(started) <= consumed + 2 * 4
    assert consumed == 200


def test_ordered_map_early_exit_and_errors():
    started = []
    gen = ordered_map(lambda i: started.append(i) or i, range(1000), num_threads=4)
    assert [next(gen) for _ in range(3)] == [0, 1, 2]
    gen.close()
    assert len(started) <= 3 + 2 * 4

    def fail_at_5(i):
        if i == 5:
            raise RuntimeError("boom")
        return i

    out = []
    with pytest.raises(RuntimeError, match="boom"):
        out.extend(ordered_map(fail_at_5, range(20), num_threads=3))  # appends until the error
    assert out == [0, 1, 2, 3, 4]


def test_threaded_iteration_matches_sequential(slide: ArrayReader, tmp_path):
    color = ColorConfig(stain_normalization="macenko", stain_sample_tiles=8)
    one = TileGenerator(slide, tiling=CONFIG, color=color, num_threads=1)
    many = TileGenerator(slide, tiling=CONFIG, color=color, num_threads=4)
    a, b = list(one), list(many)
    assert len(a) == len(b) == len(one) > 8
    for ((xa, ya), ta), ((xb, yb), tb) in zip(a, b):
        assert (xa, ya) == (xb, yb)
        np.testing.assert_array_equal(ta, tb)
    paths = many.save_tiles(tmp_path, fmt="png")
    assert [p.name for p in paths] == [f"{x}_{y}.png" for (x, y), _ in a]


def test_num_threads_validation(slide: ArrayReader):
    with pytest.raises(ValueError, match="num_threads"):
        TileGenerator(slide, num_threads=0)


@pytest.mark.parametrize(("cpus", "expected"), [(1, 1), (2, 1), (4, 3), (32, 8)])
def test_default_num_threads_leaves_one_cpu(monkeypatch, cpus, expected):
    monkeypatch.delenv("PATCHOLOGY_NUM_THREADS", raising=False)
    monkeypatch.setattr(tiler.os, "sched_getaffinity", lambda pid: set(range(cpus)), raising=False)
    assert default_num_threads() == expected


def test_num_threads_env_override(monkeypatch, slide: ArrayReader):
    monkeypatch.setenv("PATCHOLOGY_NUM_THREADS", "3")
    assert default_num_threads() == 3
    assert TileGenerator(slide).num_threads == 3
    assert TileGenerator(slide, num_threads=1).num_threads == 1  # explicit wins
    for bad in ("0", "-2", "many"):
        monkeypatch.setenv("PATCHOLOGY_NUM_THREADS", bad)
        with pytest.raises(ValueError, match="PATCHOLOGY_NUM_THREADS"):
            default_num_threads()
