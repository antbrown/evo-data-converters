import pytest
import pandas as pd
import pyarrow as pa
from unittest.mock import Mock, MagicMock
from evo.data_converters.common.objects.downhole_collection import DownholeCollection
from evo.data_converters.common.objects.downhole_collection_to_geoscience_object import (
    DownholeCollectionToGeoscienceObject,
)


@pytest.fixture
def mock_data_client():
    """Create a mock ObjectDataClient that returns valid schemas for different table types."""
    client = Mock()

    def save_table_side_effect(table):
        """Return appropriate schema based on table structure."""
        num_rows = table.num_rows
        column_names = set(table.column_names)

        # LookupTable: has 'key' and 'value' columns
        if column_names == {"key", "value"}:
            return {
                "data": None,
                "length": num_rows,
                "keys_data_type": "int32",
                "values_data_type": "string",
            }

        if column_names == {"data"} and table.schema.field("data").type in (pa.int32(), pa.int64()):
            dtype_str = "int32" if table.schema.field("data").type == pa.int32() else "int64"
            return {
                "data": None,
                "length": num_rows,
                "data_type": dtype_str,
            }

        # Default for array types (FloatArray1, FloatArray3, IntegerArray1, etc.)
        return {
            "data": None,
            "length": num_rows,
        }

    client.save_table = MagicMock(side_effect=save_table_side_effect)
    return client


@pytest.fixture
def sample_dhc():
    """Create a sample DownholeCollection for testing."""
    collars = pd.DataFrame(
        {
            "hole_index": [1, 2],
            "hole_id": ["DH-001", "DH-002"],
            "x": [100.0, 200.0],
            "y": [500.0, 600.0],
            "z": [50.0, 55.0],
            "final_depth": [100.0, 150.0],
        }
    )

    measurements = pd.DataFrame(
        {
            "hole_index": [1, 1, 1, 2, 2],
            "penetrationLength": [10.0, 20.0, 30.0, 15.0, 25.0],
            "density": [2.5, 2.6, 2.7, 2.4, 2.5],
            "porosity": [0.15, 0.18, 0.20, 0.12, 0.16],
        }
    )

    return DownholeCollection(
        collars=collars,
        measurements=measurements,
        name="Test Collection",
        epsg_code=32633,
    )


@pytest.fixture
def converter(sample_dhc, mock_data_client):
    """Create a converter instance."""
    return DownholeCollectionToGeoscienceObject(
        dhc=sample_dhc,
        data_client=mock_data_client,
    )


class TestConvert:
    def test_convert_creates_geoscience_object(self, converter):
        """Test that convert() creates a DownholeCollectionGo object."""
        result = converter.convert()

        assert result is not None
        assert result.name == "Test Collection"
        assert result.coordinate_reference_system.epsg_code == 32633
        assert result.bounding_box is not None

    def test_convert_calls_data_client(self, converter, mock_data_client):
        """Test that convert() saves data via the data client."""
        converter.convert()

        # Should save multiple tables (coordinates, distances, holes, etc.)
        assert mock_data_client.save_table.call_count >= 5


class TestBoundingBox:
    def test_create_bounding_box(self, converter):
        """Test bounding box creation from collar data."""
        bbox = converter.create_bounding_box()

        assert bbox.min_x == 100.0
        assert bbox.max_x == 200.0
        assert bbox.min_y == 500.0
        assert bbox.max_y == 600.0
        assert bbox.min_z == 50.0
        assert bbox.max_z == 55.0


class TestCoordinatesTables:
    def test_coordinates_table_structure(self, converter):
        """Test coordinates table has correct schema and data."""
        table = converter.coordinates_table()

        assert isinstance(table, pa.Table)
        assert table.num_rows == 2
        assert table.num_columns == 3
        assert set(table.column_names) == {"x", "y", "z"}

        # Check data types
        assert table.schema.field("x").type == pa.float64()
        assert table.schema.field("y").type == pa.float64()
        assert table.schema.field("z").type == pa.float64()

    def test_coordinates_table_values(self, converter):
        """Test coordinates table contains correct values."""
        table = converter.coordinates_table()

        x_values = table.column("x").to_pylist()
        y_values = table.column("y").to_pylist()
        z_values = table.column("z").to_pylist()

        assert x_values == [100.0, 200.0]
        assert y_values == [500.0, 600.0]
        assert z_values == [50.0, 55.0]


class TestDistancesTables:
    def test_distances_table_structure(self, converter):
        """Test distances table has correct schema."""
        table = converter.distances_table()

        assert isinstance(table, pa.Table)
        assert table.num_rows == 2
        assert set(table.column_names) == {"final", "target", "current"}

    def test_distances_table_uses_final_depth(self, converter):
        """Test that all distance columns use final_depth values."""
        table = converter.distances_table()

        final_values = table.column("final").to_pylist()
        target_values = table.column("target").to_pylist()
        current_values = table.column("current").to_pylist()

        assert final_values == [100.0, 150.0]
        assert target_values == [100.0, 150.0]
        assert current_values == [100.0, 150.0]


class TestHolesTables:
    def test_holes_table_structure(self, converter):
        """Test holes table has correct schema and indexing."""
        table = converter.holes_table()

        assert isinstance(table, pa.Table)
        assert table.num_rows == 2
        assert set(table.column_names) == {"hole_index", "offset", "count"}

        # Check data types
        assert table.schema.field("hole_index").type == pa.int32()
        assert table.schema.field("offset").type == pa.uint64()
        assert table.schema.field("count").type == pa.uint64()

    def test_holes_table_counts_measurements(self, converter):
        """Test that holes table correctly counts measurements per hole."""
        table = converter.holes_table()

        hole_indices = table.column("hole_index").to_pylist()
        counts = table.column("count").to_pylist()
        offsets = table.column("offset").to_pylist()

        assert hole_indices == [1, 2]
        assert counts == [3, 2]  # 3 measurements for hole 1, 2 for hole 2
        assert offsets == [0, 3]  # hole 1 starts at 0, hole 2 starts at 3


class TestHoleIdTables:
    def test_hole_id_tables_structure(self, converter):
        """Test hole id lookup and index tables."""
        lookup_table, integer_array_table = converter.hole_id_tables()

        # Check lookup table
        assert isinstance(lookup_table, pa.Table)
        assert set(lookup_table.column_names) == {"key", "value"}
        assert lookup_table.schema.field("key").type == pa.int32()
        assert lookup_table.schema.field("value").type == pa.string()

        # Check integer array table
        assert isinstance(integer_array_table, pa.Table)
        assert set(integer_array_table.column_names) == {"data"}

    def test_hole_id_mapping(self, converter):
        """Test that hole IDs are correctly mapped to indices."""
        lookup_table, _ = converter.hole_id_tables()

        keys = lookup_table.column("key").to_pylist()
        values = lookup_table.column("value").to_pylist()

        assert keys == [1, 2]
        assert values == ["DH-001", "DH-002"]


class TestPathTable:
    def test_path_table_structure(self, converter):
        """Test path table has correct schema."""
        table = converter.path_table()

        assert isinstance(table, pa.Table)
        assert table.num_rows == 5  # Total measurements
        assert set(table.column_names) == {"distance", "azimuth", "dip"}

    def test_path_table_assumes_vertical(self, converter):
        """Test that path table assumes vertical holes."""
        table = converter.path_table()

        azimuth_values = table.column("azimuth").to_pylist()
        dip_values = table.column("dip").to_pylist()

        # All holes assumed vertical
        assert all(az == 0.0 for az in azimuth_values)
        assert all(dip == 90.0 for dip in dip_values)

    def test_path_table_uses_penetration_length(self, converter):
        """Test that distance values come from penetrationLength."""
        table = converter.path_table()

        distances = table.column("distance").to_pylist()
        expected = [10.0, 20.0, 30.0, 15.0, 25.0]

        assert distances == expected


class TestCollectionAttributes:
    def test_collection_distances_table(self, converter):
        """Test collection distances table creation."""
        table = converter.collection_distances_table()

        assert isinstance(table, pa.Table)
        assert "values" in table.column_names

        values = table.column("values").to_pylist()
        expected = [10.0, 20.0, 30.0, 15.0, 25.0]
        assert values == expected

    def test_collection_attribute_tables(self, converter):
        """Test attribute tables are created for each measurement column."""
        attribute_tables = converter.collection_attribute_tables()

        # Should have tables for density and porosity
        assert "density" in attribute_tables
        assert "porosity" in attribute_tables
        assert len(attribute_tables) == 2

        # Check density values
        density_table = attribute_tables["density"]
        density_values = density_table.column("data").to_pylist()
        assert density_values == [2.5, 2.6, 2.7, 2.4, 2.5]

        # Check porosity values
        porosity_table = attribute_tables["porosity"]
        porosity_values = porosity_table.column("data").to_pylist()
        assert porosity_values == [0.15, 0.18, 0.20, 0.12, 0.16]


class TestLocationCreation:
    def test_create_dhc_location(self, converter):
        """Test that DHC location is created with all required components."""
        location = converter.create_dhc_location()

        assert location is not None
        assert location.coordinates is not None
        assert location.distances is not None
        assert location.holes is not None
        assert location.hole_id is not None
        assert location.path is not None


class TestCollectionsCreation:
    def test_create_dhc_collections(self, converter):
        """Test that DHC collections are created."""
        collections = converter.create_dhc_collections()

        assert isinstance(collections, list)
        assert len(collections) > 0
        assert collections[0].name == "distances"

    def test_create_dhc_collection_distance(self, converter):
        """Test distance collection creation includes attributes."""
        distance = converter.create_dhc_collection_distance()

        assert distance is not None
        assert distance.values is not None
        assert len(distance.attributes) == 2  # density and porosity


class TestEdgeCases:
    def test_single_hole_single_measurement(self, mock_data_client):
        """Test conversion with minimal data."""
        collars = pd.DataFrame(
            {
                "hole_index": [1],
                "hole_id": ["DH-001"],
                "x": [100.0],
                "y": [500.0],
                "z": [50.0],
                "final_depth": [100.0],
            }
        )

        measurements = pd.DataFrame(
            {
                "hole_index": [1],
                "penetrationLength": [10.0],
                "density": [2.5],
            }
        )

        dhc = DownholeCollection(
            collars=collars,
            measurements=measurements,
            name="Minimal",
            epsg_code=4326,
        )

        converter = DownholeCollectionToGeoscienceObject(dhc, mock_data_client)
        result = converter.convert()

        assert result is not None
        assert result.name == "Minimal"

    def test_multiple_holes_unequal_measurements(self, mock_data_client):
        """Test conversion with varying measurement counts per hole."""
        collars = pd.DataFrame(
            {
                "hole_index": [1, 2, 3],
                "hole_id": ["DH-001", "DH-002", "DH-003"],
                "x": [100.0, 200.0, 300.0],
                "y": [500.0, 600.0, 700.0],
                "z": [50.0, 55.0, 60.0],
                "final_depth": [100.0, 150.0, 120.0],
            }
        )

        measurements = pd.DataFrame(
            {
                "hole_index": [1, 2, 2, 2, 3],
                "penetrationLength": [10.0, 15.0, 25.0, 35.0, 20.0],
                "density": [2.5, 2.4, 2.5, 2.6, 2.3],
            }
        )

        dhc = DownholeCollection(
            collars=collars,
            measurements=measurements,
            name="Unequal",
            epsg_code=4326,
        )

        converter = DownholeCollectionToGeoscienceObject(dhc, mock_data_client)
        holes_table = converter.holes_table()

        counts = holes_table.column("count").to_pylist()
        offsets = holes_table.column("offset").to_pylist()

        assert counts == [1, 3, 1]
        assert offsets == [0, 1, 4]
