import numpy as np
import pytest

from patchology.tissue.roi import load_classified_polygons
from patchology.utilities.xml_to_geojson import convert

ASAP = """<?xml version="1.0" ?>
<ASAP_Annotations><Annotations>
  <Annotation Name="a" Type="Polygon" PartOfGroup="invasive tumor">
    <Coordinates>
      <Coordinate Order="1" X="10" Y="0"/><Coordinate Order="0" X="0" Y="0"/>
      <Coordinate Order="2" X="10" Y="10"/>
    </Coordinates>
  </Annotation>
  <Annotation Name="b" Type="Rectangle" PartOfGroup="None">
    <Coordinates>
      <Coordinate Order="0" X="0" Y="0"/><Coordinate Order="1" X="5" Y="0"/>
      <Coordinate Order="2" X="5" Y="5"/><Coordinate Order="3" X="0" Y="5"/>
    </Coordinates>
  </Annotation>
  <Annotation Name="c" Type="Dot" PartOfGroup="lymphocytes">
    <Coordinates><Coordinate Order="0" X="3" Y="4"/></Coordinates>
  </Annotation>
  <Annotation Name="d" Type="Polygon" PartOfGroup="rest">
    <Coordinates><Coordinate Order="0" X="1" Y="1"/></Coordinates>
  </Annotation>
</Annotations></ASAP_Annotations>"""


def test_asap_xml_round_trips_through_loader(tmp_path):
    xml = tmp_path / "slide.xml"
    xml.write_text(ASAP)
    out = convert(xml, tmp_path / "out")
    assert out.name == "slide.geojson"
    polygons, classes = load_classified_polygons(out)
    # Dot skipped by the loader, 1-vertex polygon dropped by the converter.
    assert classes == ["invasive tumor", None]
    ring = polygons[0][0]
    np.testing.assert_array_equal(ring, [[0, 0], [10, 0], [10, 10], [0, 0]])  # ordered, closed


def test_rejects_non_asap_xml(tmp_path):
    xml = tmp_path / "other.xml"
    xml.write_text("<Annotations/>")
    with pytest.raises(ValueError, match="not ASAP XML"):
        convert(xml)
