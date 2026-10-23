import math

import numpy as np
import pytest

from patchology.tiling.grid import hex_grid, make_grid, square_grid


def test_square_grid_non_overlapping():
    coords = square_grid((0, 0, 1000, 500), tile_size=100)
    assert len(coords) == 10 * 5
    assert coords[:, 0].max() + 100 <= 1000
    assert coords[:, 1].max() + 100 <= 500


def test_square_grid_overlapping_and_offset():
    coords = square_grid((50, 20, 1000, 500), tile_size=100, tile_overlap=0.5)
    assert len(coords) == 19 * 9
    assert coords[:, 0].min() == 50
    assert coords[:, 1].min() == 20


def test_square_grid_region_smaller_than_tile():
    assert len(square_grid((0, 0, 50, 500), tile_size=100)) == 0


def test_hex_grid_lattice():
    distance = 80  # 100 * (1 - 0.2)
    coords = hex_grid((0, 0, 2000, 2000), tile_size=100, tile_overlap=0.2)
    rows = np.unique(coords[:, 1])
    np.testing.assert_allclose(np.diff(rows), distance * math.sqrt(3) / 2)
    # Odd rows are shifted by half the neighbour distance.
    assert coords[coords[:, 1] == rows[0], 0].min() == 0
    assert coords[coords[:, 1] == rows[1], 0].min() == distance / 2
    # Every interior tile has six neighbours at the same distance.
    centre = coords[np.argmin(np.hypot(*(coords - 1000).T))]
    dist = np.hypot(*(coords - centre).T)
    np.testing.assert_allclose(np.sort(dist)[1:7], distance)
    assert np.sort(dist)[7] > distance * 1.5


def test_hex_grid_stays_inside_region():
    coords = hex_grid((10, 10, 1000, 700), tile_size=128)
    assert coords[:, 0].min() >= 10
    assert coords[:, 1].min() >= 10
    assert (coords[:, 0] + 128).max() <= 1010
    assert (coords[:, 1] + 128).max() <= 710


def test_make_grid_validation():
    with pytest.raises(ValueError, match="unknown grid"):
        make_grid("triangle", (0, 0, 100, 100), 10, 10)
    with pytest.raises(ValueError, match="positive"):
        make_grid("square", (0, 0, 100, 100), 0)
    with pytest.raises(ValueError, match="tile_overlap"):
        make_grid("square", (0, 0, 100, 100), 10, 1.0)
