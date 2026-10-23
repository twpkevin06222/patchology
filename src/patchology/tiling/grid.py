"""Tile layouts: square and hexagonal grids of tile positions."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

GridType = Literal["square", "hex"]


def square_grid(
    region: tuple[float, float, float, float], tile_size: float, tile_overlap: float = 0.0
) -> np.ndarray:
    """Top-left corners of tiles on a square lattice.

    Args:
        region: ``(x, y, width, height)`` area to cover. Only tiles that lie
            fully inside it are returned.
        tile_size: Tile side length.
        tile_overlap: Fraction of ``tile_size`` by which neighbouring tiles
            overlap along each axis, in ``[0, 1)``; 0.2 is 20%. Tiles are
            ``tile_size * (1 - tile_overlap)`` apart.

    Returns:
        ``(N, 2)`` float array of ``(x, y)`` corners, row-major order.
    """
    x0, y0, width, height = region
    step = _step(tile_size, tile_overlap)
    xs = _positions(x0, width, tile_size, step)
    ys = _positions(y0, height, tile_size, step)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)


def hex_grid(
    region: tuple[float, float, float, float], tile_size: float, tile_overlap: float = 0.0
) -> np.ndarray:
    """Top-left corners of tiles centred on a hexagonal lattice.

    Each tile centre has six neighbours at distance
    ``d = tile_size * (1 - tile_overlap)``: rows are ``d * sqrt(3) / 2`` apart
    and every other row is shifted right by ``d / 2``. ``tile_overlap`` is
    thus the overlap between horizontal neighbours; with 0 they touch and
    consecutive rows still overlap by about 13% of the tile height.

    Arguments and return value are as for :func:`square_grid`.
    """
    x0, y0, width, height = region
    step = _step(tile_size, tile_overlap)
    row_step = step * math.sqrt(3) / 2
    ys = _positions(y0, height, tile_size, row_step)
    rows = []
    for i, y in enumerate(ys):
        offset = step / 2 if i % 2 else 0.0
        xs = _positions(x0 + offset, width - offset, tile_size, step)
        rows.append(np.stack([xs, np.full_like(xs, y)], axis=1))
    if not rows:
        return np.empty((0, 2))
    return np.concatenate(rows)


def make_grid(
    grid: GridType,
    region: tuple[float, float, float, float],
    tile_size: float,
    tile_overlap: float = 0.0,
) -> np.ndarray:
    if grid == "square":
        return square_grid(region, tile_size, tile_overlap)
    if grid == "hex":
        return hex_grid(region, tile_size, tile_overlap)
    raise ValueError(f"unknown grid {grid!r}, expected 'square' or 'hex'")


def _step(tile_size: float, tile_overlap: float) -> float:
    """Distance between neighbouring tiles."""
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    if not 0.0 <= tile_overlap < 1.0:
        raise ValueError(f"tile_overlap must be in [0, 1), got {tile_overlap}")
    return tile_size * (1 - tile_overlap)


def _positions(start: float, length: float, tile_size: float, step: float) -> np.ndarray:
    if length < tile_size:
        return np.empty(0)
    count = math.floor((length - tile_size) / step + 1e-9) + 1
    return start + np.arange(count) * step
