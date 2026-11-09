import json

import numpy as np
import pytest

from patchology import REFERENCE_STAINS, ColorConfig, Stains
from patchology.color import load_stains
from patchology.utilities.estimate_stains import angles_deg, find_slides, median_stains

_OTHER = np.array([[0.45, 0.30], [0.80, 0.85], [0.40, 0.43]])
OTHER_STAINS = Stains(_OTHER / np.linalg.norm(_OTHER, axis=0), np.array([1.5, 1.2]))


def test_json_stain_target_round_trips(tmp_path):
    path = tmp_path / "stains.json"
    path.write_text(json.dumps({**OTHER_STAINS.to_dict(), "slides": 3}))
    loaded = ColorConfig(stain_target=path).target_stains
    np.testing.assert_allclose(loaded.matrix, OTHER_STAINS.matrix, atol=1e-6)
    np.testing.assert_allclose(loaded.max_concentrations, OTHER_STAINS.max_concentrations)


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        json.dumps({"matrix": [[1, 0], [0, 1], [0, 0]]}),  # no max_concentrations
        json.dumps({"matrix": [[1, 0, 0], [0, 1, 0]], "max_concentrations": [1, 1]}),
        json.dumps({"matrix": [[1, 0], [0, 0], [0, 0]], "max_concentrations": [1, 1]}),
        json.dumps({"matrix": [[1, 0], [0, 1], [0, 0]], "max_concentrations": [1, -1]}),
        json.dumps([1, 2]),
    ],
)
def test_malformed_json_stains_raise(tmp_path, content):
    path = tmp_path / "stains.json"
    path.write_text(content)
    with pytest.raises(ValueError):
        load_stains(path)


def test_median_stains_ignores_an_outlier():
    outlier = Stains(np.array([[0, 1], [1, 0], [0, 0.0]]), np.array([9.0, 9.0]))
    reference = median_stains([OTHER_STAINS, OTHER_STAINS, outlier])
    np.testing.assert_allclose(reference.matrix, OTHER_STAINS.matrix, atol=1e-9)
    np.testing.assert_allclose(reference.max_concentrations, OTHER_STAINS.max_concentrations)
    np.testing.assert_allclose(angles_deg(OTHER_STAINS, reference), [0, 0], atol=1e-5)
    assert (angles_deg(REFERENCE_STAINS, reference) > 1).all()


def test_find_slides_searches_directories(tmp_path):
    for name in ("a.svs", "sub/b.NDPI", "sub/notes.txt", "c.json"):
        (tmp_path / name).parent.mkdir(exist_ok=True)
        (tmp_path / name).touch()
    explicit = tmp_path / "c.json"
    assert find_slides([tmp_path, explicit]) == sorted(
        [tmp_path / "a.svs", tmp_path / "sub/b.NDPI", explicit]
    )
