import pytest

from patchology import GreyObjectConfig
from patchology.tissue import GreyRegion
from patchology.utilities.inspect_tissue import parse_overrides, status

NAN = float("nan")


def _region(rule, removed=False, value=200.0, chroma=10.0, straight=0.6):
    return GreyRegion(rule, (0, 0, 1, 1), 1.0, value, chroma, NAN, straight, False, removed)


@pytest.mark.parametrize(
    "region, expected",
    [
        (_region("band", removed=True), "removed"),
        (_region("dark", value=100.0, straight=NAN), "near_miss"),  # dark candidate kept
        (_region("band", straight=0.3), "near_miss"),  # only too curved
        (_region("band", chroma=22.0), "near_miss"),  # only too stained
        (_region("band", chroma=60.0, straight=0.1), "kept"),  # ordinary tissue
    ],
)
def test_status(region, expected):
    assert status(region, GreyObjectConfig()) == expected


def test_parse_overrides():
    config = parse_overrides(["band_max_chroma=24", "max_texture=9.5"])
    assert config.band_max_chroma == 24 and config.max_texture == 9.5
    assert config.max_value == GreyObjectConfig().max_value
    with pytest.raises(SystemExit):
        parse_overrides(["nope=1"])
    with pytest.raises(ValueError):
        parse_overrides(["max_chroma=-1"])
