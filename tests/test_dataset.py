import pickle

import numpy as np
import pytest

from patchology import SegmentationTileGenerator, TileGenerator, TilingConfig
from patchology.tiling import generator as generator_module

from conftest import ArrayReader, make_slide_image

CONFIG = TilingConfig(tile_size=128, tile_mpp=2.0)


@pytest.fixture
def reopens(monkeypatch):
    """Make reopening a slide (in a copy or another process) build a fresh ArrayReader."""
    opened = []

    def fake_open(path, backend="openslide"):
        opened.append(path)
        return ArrayReader(make_slide_image(), path=path)

    monkeypatch.setattr(generator_module, "open_slide", fake_open)
    return opened


def test_get_tile_by_index_and_coordinates(slide: ArrayReader):
    gen = TileGenerator(slide, tiling=CONFIG)
    (x, y), tile = gen.get_tile(1)
    assert gen[1][0] == (x, y) and gen.get_tile(-1)[0] == tuple(gen.coords[-1])
    np.testing.assert_array_equal(gen.get_tile_at(x, y), tile)
    assert gen.index_of(x, y) == 1
    # Off-grid coordinates can be read too, but have no index.
    assert gen.get_tile_at(x + 3, y).shape == tile.shape
    with pytest.raises(KeyError):
        gen.index_of(x + 3, y)
    with pytest.raises(IndexError):
        gen.get_tile(len(gen))


def test_segmentation_get_tile_at_returns_mask(slide: ArrayReader, tmp_path):
    gt = tmp_path / "synthetic_gt.geojson"
    ring = [[1024, 512], [3072, 512], [3072, 2048], [1024, 2048], [1024, 512]]
    gt.write_text(f'{{"type": "Polygon", "coordinates": [{ring}]}}')
    gen = SegmentationTileGenerator(slide, gt, object_mask_overlap=0.5, tiling=CONFIG)
    (x, y), tile, mask = gen[0]
    tile_at, mask_at = gen.get_tile_at(x, y)
    np.testing.assert_array_equal(tile_at, tile)
    np.testing.assert_array_equal(mask_at, mask)


def test_pickled_copy_reopens_slide_and_keeps_result(slide: ArrayReader, reopens):
    gen = TileGenerator(slide, tiling=CONFIG)
    copy = pickle.loads(pickle.dumps(gen))
    # The parent ran tiling once; the copy reuses it and opens its own slide on first read.
    assert "result" in copy.__dict__ and not reopens
    np.testing.assert_array_equal(copy.coords, gen.coords)
    np.testing.assert_array_equal(copy[0][1], gen[0][1])
    assert reopens == [slide.path] and copy.slide is not slide
    copy.close()


# Fork so workers inherit the stubbed open_slide; spawn was checked on a real slide.
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded")
def test_torch_dataloader_with_workers(slide: ArrayReader, reopens):
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader, Dataset

    class Tiles(Dataset):
        def __init__(self, gen):
            self.gen = gen

        def __len__(self):
            return len(self.gen)

        def __getitem__(self, i):
            (x, y), tile = self.gen[i]
            return torch.from_numpy(tile).permute(2, 0, 1), torch.tensor([x, y])

    gen = TileGenerator(slide, tiling=CONFIG)
    loader = DataLoader(Tiles(gen), batch_size=4, num_workers=2, multiprocessing_context="fork")
    tiles, coords = zip(*loader)
    assert torch.cat(tiles).shape == (len(gen), 3, 128, 128)
    np.testing.assert_array_equal(torch.cat(coords).numpy(), gen.coords)
    np.testing.assert_array_equal(torch.cat(tiles)[0].permute(1, 2, 0).numpy(), gen[0][1])
