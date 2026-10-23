import json

import numpy as np

from patchology import TileGenerator, TilingConfig, tiles_to_geojson
from patchology.tissue import load_polygons

from conftest import ArrayReader

CONFIG = TilingConfig(tile_size=128, tile_mpp=1.0, tissue_overlap=0.9)


def test_tiles_to_geojson_boxes_at_level0(slide: ArrayReader, tmp_path):
    gen = TileGenerator(slide, tiling=CONFIG)
    path = tmp_path / "tiles.geojson"
    data = tiles_to_geojson(gen, path)
    assert data == json.loads(path.read_text())
    assert len(data["features"]) == len(gen) > 0
    props = data["features"][0]["properties"]
    assert props["color"] == [0, 0, 0] and props["classification"]["color"] == [0, 0, 0]

    # Round-trips through our own reader: each box spans its tile's level-0 footprint.
    for (x, y), rings in zip(gen.coords, load_polygons(path)):
        ring = rings[0]
        np.testing.assert_allclose(ring.min(axis=0), [x, y])
        np.testing.assert_allclose(ring.max(axis=0), [x + 256, y + 256])
        np.testing.assert_array_equal(ring[0], ring[-1])

    red = tiles_to_geojson(gen, color=(255, 0, 0))
    assert red["features"][0]["properties"]["color"] == [255, 0, 0]
