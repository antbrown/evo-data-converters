import pytest
import pandas as pd
from evo.data_converters.common.objects.downhole_collection import DownholeCollection


@pytest.fixture
def valid_collars():
    """Create a valid collars DataFrame."""
    return pd.DataFrame(
        {
            "hole_index": [1, 2, 3],
            "hole_id": ["DH-001", "DH-002", "DH-003"],
            "x": [100.0, 200.0, 300.0],
            "y": [500.0, 600.0, 700.0],
            "z": [50.0, 55.0, 60.0],
            "final_depth": [150.0, 200.0, 180.0],
        }
    )


@pytest.fixture
def distance_measurements():
    """Create distance-based measurements DataFrame."""
    return pd.DataFrame(
        {
            "hole_index": [1, 1, 2],
            "depth": [10.0, 20.0, 15.0],
            "value": [1.5, 2.3, 1.8],
        }
    )


@pytest.fixture
def interval_measurements():
    """Create interval-based measurements DataFrame."""
    return pd.DataFrame(
        {
            "hole_index": [1, 1, 2],
            "from": [0.0, 10.0, 0.0],
            "to": [10.0, 20.0, 15.0],
            "value": [1.5, 2.3, 1.8],
        }
    )


class TestIsCollarsValid:
    def test_valid_collars(self, valid_collars):
        dc = DownholeCollection(
            collars=valid_collars,
            measurements=pd.DataFrame(),
            name="Test",
            epsg_code=4326,
        )
        assert dc.is_collars_valid() is True

    def test_not_dataframe(self):
        dc = DownholeCollection(
            collars={"not": "dataframe"},
            measurements=pd.DataFrame(),
            name="Test",
            epsg_code=4326,
        )
        assert dc.is_collars_valid() is False

    def test_missing_columns(self, valid_collars):
        invalid_collars = valid_collars.drop(columns=["hole_id"])
        dc = DownholeCollection(
            collars=invalid_collars,
            measurements=pd.DataFrame(),
            name="Test",
            epsg_code=4326,
        )
        assert dc.is_collars_valid() is False

    def test_wrong_dtype(self, valid_collars):
        invalid_collars = valid_collars.copy()
        invalid_collars["hole_index"] = invalid_collars["hole_index"].astype(str)
        dc = DownholeCollection(
            collars=invalid_collars,
            measurements=pd.DataFrame(),
            name="Test",
            epsg_code=4326,
        )
        assert dc.is_collars_valid() is False


class TestMeasurementType:
    def test_is_distance_with_depth(self, valid_collars, distance_measurements):
        dc = DownholeCollection(
            collars=valid_collars,
            measurements=distance_measurements,
            name="Test",
            epsg_code=4326,
        )
        assert dc.is_distance() is True
        assert dc.is_interval() is False
        assert dc.measurement_type() == "distance"

    def test_is_distance_with_penetrationlength(self, valid_collars):
        measurements = pd.DataFrame(
            {
                "hole_index": [1, 2],
                "PenetrationLength": [10.0, 20.0],
                "value": [1.5, 2.3],
            }
        )
        dc = DownholeCollection(
            collars=valid_collars,
            measurements=measurements,
            name="Test",
            epsg_code=4326,
        )
        assert dc.is_distance() is True
        assert dc.measurement_type() == "distance"

    def test_is_interval(self, valid_collars, interval_measurements):
        dc = DownholeCollection(
            collars=valid_collars,
            measurements=interval_measurements,
            name="Test",
            epsg_code=4326,
        )
        assert dc.is_distance() is False
        assert dc.is_interval() is True
        assert dc.measurement_type() == "interval"

    def test_empty_measurements(self, valid_collars):
        dc = DownholeCollection(
            collars=valid_collars,
            measurements=pd.DataFrame(),
            name="Test",
            epsg_code=4326,
        )
        assert dc.is_distance() is False
        assert dc.is_interval() is False
        assert dc.measurement_type() == "unknown"

    def test_unknown_format(self, valid_collars):
        measurements = pd.DataFrame(
            {
                "hole_index": [1, 2],
                "other_column": [10.0, 20.0],
            }
        )
        dc = DownholeCollection(
            collars=valid_collars,
            measurements=measurements,
            name="Test",
            epsg_code=4326,
        )
        assert dc.measurement_type() == "unknown"


class TestBoundingBox:
    def test_bounding_box(self, valid_collars):
        dc = DownholeCollection(
            collars=valid_collars,
            measurements=pd.DataFrame(),
            name="Test",
            epsg_code=4326,
        )
        bbox = dc.bounding_box()

        assert bbox["min_x"] == 100.0
        assert bbox["max_x"] == 300.0
        assert bbox["min_y"] == 500.0
        assert bbox["max_y"] == 700.0
        assert bbox["min_z"] == 50.0
        assert bbox["max_z"] == 60.0

    def test_bounding_box_single_point(self):
        collars = pd.DataFrame(
            {
                "hole_index": [1],
                "hole_id": ["DH-001"],
                "x": [100.0],
                "y": [500.0],
                "z": [50.0],
                "final_depth": [150.0],
            }
        )
        dc = DownholeCollection(
            collars=collars,
            measurements=pd.DataFrame(),
            name="Test",
            epsg_code=4326,
        )
        bbox = dc.bounding_box()

        assert bbox["min_x"] == bbox["max_x"] == 100.0
        assert bbox["min_y"] == bbox["max_y"] == 500.0
        assert bbox["min_z"] == bbox["max_z"] == 50.0


class TestIsSchemaValid:
    def test_valid_schema(self, valid_collars):
        dc = DownholeCollection(
            collars=valid_collars,
            measurements=pd.DataFrame(),
            name="Test",
            epsg_code=4326,
        )
        schema = {
            "hole_index": "int",
            "x": "float",
            "hole_id": "str",
        }
        assert dc.is_schema_valid(valid_collars, schema) is True

    def test_invalid_int_type(self, valid_collars):
        dc = DownholeCollection(
            collars=valid_collars,
            measurements=pd.DataFrame(),
            name="Test",
            epsg_code=4326,
        )
        schema = {"hole_id": "int"}  # hole_id is actually str
        assert dc.is_schema_valid(valid_collars, schema) is False

    def test_invalid_float_type(self, valid_collars):
        dc = DownholeCollection(
            collars=valid_collars,
            measurements=pd.DataFrame(),
            name="Test",
            epsg_code=4326,
        )
        schema = {"hole_index": "float"}  # hole_index is actually int
        assert dc.is_schema_valid(valid_collars, schema) is False
