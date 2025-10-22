import pytest
import polars as pl
from unittest.mock import Mock, patch
from evo.data_converters.common.objects.downhole_collection import DownholeCollection
from evo.data_converters.gef.importer.gef_to_downhole_collection import (
    _calculate_final_depth,
    _extract_epsg_code,
    _validate_location_attributes,
    create_from_parsed_gef_cpts,
    get_collection_name_from_collars,
)


class TestGetCollectionNameFromCollars:
    """Tests for get_collection_name_from_collars function."""

    def test_empty_list_returns_empty_string(self) -> None:
        assert get_collection_name_from_collars([]) == ""

    def test_single_collar_returns_hole_id(self) -> None:
        collars = [{"hole_id": "CPT-001"}]
        assert get_collection_name_from_collars(collars) == "CPT-001"

    def test_multiple_collars_returns_range_format(self) -> None:
        collars = [{"hole_id": "CPT-001"}, {"hole_id": "CPT-002"}, {"hole_id": "CPT-003"}]
        assert get_collection_name_from_collars(collars) == "CPT-001...CPT-003"

    def test_two_collars_returns_range_format(self) -> None:
        collars = [{"hole_id": "HOLE-A"}, {"hole_id": "HOLE-Z"}]
        assert get_collection_name_from_collars(collars) == "HOLE-A...HOLE-Z"

    def test_special_characters_in_hole_id(self) -> None:
        collars = [{"hole_id": "CPT-2024/01_Site-A"}]
        assert get_collection_name_from_collars(collars) == "CPT-2024/01_Site-A"


class TestExtractEpsgCode:
    """Tests for _extract_epsg_code helper function."""

    def test_valid_epsg_code_extraction(self) -> None:
        mock_cpt = Mock()
        mock_cpt.delivered_location.srs_name = "EPSG:28992"

        result = _extract_epsg_code(mock_cpt, "TEST-001")

        assert result == 28992

    def test_different_epsg_format(self) -> None:
        mock_cpt = Mock()
        mock_cpt.delivered_location.srs_name = "urn:ogc:def:crs:EPSG::4326"

        result = _extract_epsg_code(mock_cpt, "TEST-001")

        assert result == 4326

    def test_missing_srs_name_attribute_raises_error(self) -> None:
        mock_cpt = Mock()
        mock_cpt.delivered_location = Mock(spec=[])  # No srs_name attribute

        with pytest.raises(ValueError, match="missing delivered_location.srs_name"):
            _ = _extract_epsg_code(mock_cpt, "TEST-001")

    def test_malformed_srs_name_no_colon_raises_error(self) -> None:
        mock_cpt = Mock()
        mock_cpt.delivered_location.srs_name = "INVALID_FORMAT"

        with pytest.raises(ValueError, match="malformed SRS name"):
            _extract_epsg_code(mock_cpt, "TEST-001")

    def test_non_numeric_epsg_raises_error(self):
        mock_cpt = Mock()
        mock_cpt.delivered_location.srs_name = "EPSG:ABC"

        with pytest.raises(ValueError, match="invalid EPSG code"):
            _extract_epsg_code(mock_cpt, "TEST-001")


class TestValidateLocationAttributes:
    """Test suite for _validate_location_attributes helper function."""

    def test_valid_location_attributes_pass(self) -> None:
        mock_cpt = Mock()
        mock_cpt.delivered_location.x = 100000.0
        mock_cpt.delivered_location.y = 500000.0

        # Should not raise any exception
        _validate_location_attributes(mock_cpt, "TEST-001")

    def test_missing_x_attribute_raises_error(self) -> None:
        mock_cpt = Mock()
        mock_cpt.delivered_location = Mock(spec=["y"])
        mock_cpt.delivered_location.y = 500000.0

        with pytest.raises(ValueError, match="missing required location attribute"):
            _validate_location_attributes(mock_cpt, "TEST-001")

    def test_missing_y_attribute_raises_error(self) -> None:
        mock_cpt = Mock()
        mock_cpt.delivered_location = Mock(spec=["x"])
        mock_cpt.delivered_location.x = 100000.0

        with pytest.raises(ValueError, match="missing required location attribute"):
            _validate_location_attributes(mock_cpt, "TEST-001")


class TestCalculateFinalDepth:
    """Test suite for _calculate_final_depth helper function."""

    def test_uses_final_depth_when_available(self) -> None:
        mock_cpt = Mock()
        mock_cpt.final_depth = 15.5
        mock_cpt.data = pl.DataFrame({"penetrationLength": [0, 1, 2]})

        result = _calculate_final_depth(mock_cpt, "TEST-001")

        assert result == 15.5

    def test_calculates_from_penetration_length_when_final_depth_zero(self) -> None:
        mock_cpt = Mock()
        mock_cpt.final_depth = 0.0
        mock_cpt.data = pl.DataFrame({"penetrationLength": [0.0, 2.5, 5.0, 7.8]})

        result = _calculate_final_depth(mock_cpt, "TEST-001")

        assert result == 7.8

    def test_missing_penetration_length_column_raises_error(self) -> None:
        mock_cpt = Mock()
        mock_cpt.final_depth = 0.0
        mock_cpt.data = pl.DataFrame({"coneResistance": [1, 2, 3]})

        with pytest.raises(ValueError, match="missing 'penetrationLength' column"):
            _ = _calculate_final_depth(mock_cpt, "TEST-001")

    def test_empty_penetration_length_raises_error(self) -> None:
        mock_cpt = Mock()
        mock_cpt.final_depth = 0.0
        mock_cpt.data = pl.DataFrame({"penetrationLength": []})

        with pytest.raises(ValueError, match="empty penetrationLength column"):
            _calculate_final_depth(mock_cpt, "TEST-001")

    def test_negative_penetration_length_values(self):
        mock_cpt = Mock()
        mock_cpt.final_depth = 0.0
        mock_cpt.data = pl.DataFrame({"penetrationLength": [-1.0, 0.0, 5.0]})

        result = _calculate_final_depth(mock_cpt, "TEST-001")

        assert result == 5.0


class TestCreateFromParsedGefCpts:
    """Test suite for create_from_parsed_gef_cpts main function."""

    @pytest.fixture
    def mock_cpt_data(self) -> Mock:
        """Create a mock CPTData object with required attributes."""
        mock = Mock(
            spec=[
                "delivered_location",
                "delivered_vertical_position_offset",
                "final_depth",
                "data",
                "column_void_mapping",
            ]
        )

        # Mock location
        mock.delivered_location = Mock()
        mock.delivered_location.srs_name = "EPSG:28992"
        mock.delivered_location.x = 100000.0
        mock.delivered_location.y = 500000.0

        mock.delivered_vertical_position_offset = 1.5
        mock.final_depth = 10.0

        # Mock data as polars DataFrame
        mock.data = pl.DataFrame(
            {
                "penetrationLength": [0.0, 1.0, 2.0, 3.0],
                "coneResistance": [1.0, 2.0, 3.0, 4.0],
                "friction": [0.01, 0.02, 0.03, 0.04],
            }
        )

        mock.column_void_mapping = {
            "penetrationLength": 9999.0,
            "coneResistance": 9999.0,
            "friction": 9999.0,
        }

        return mock

    def test_empty_dict_raises_error(self):
        with pytest.raises(ValueError, match="No CPT files provided"):
            _ = create_from_parsed_gef_cpts({})

    def test_single_cpt_file_creates_valid_collection(self, mock_cpt_data) -> None:
        parsed_files = {"CPT-001": mock_cpt_data}

        result = create_from_parsed_gef_cpts(parsed_files)

        assert isinstance(result, DownholeCollection)
        assert result.name == "CPT-001"
        assert result.epsg_code == 28992
        assert len(result.collars) == 1
        assert result.collars.iloc[0]["hole_id"] == "CPT-001"
        assert result.collars.iloc[0]["x"] == 100000.0
        assert result.collars.iloc[0]["y"] == 500000.0
        assert result.collars.iloc[0]["z"] == 1.5
        assert result.collars.iloc[0]["final_depth"] == 10.0

    def test_multiple_cpt_files_creates_combined_collection(self, mock_cpt_data) -> None:
        mock_cpt_2 = Mock(
            spec=[
                "delivered_location",
                "delivered_vertical_position_offset",
                "final_depth",
                "data",
                "column_void_mapping",
            ]
        )
        mock_cpt_2.delivered_location = Mock()
        mock_cpt_2.delivered_location.srs_name = "EPSG:28992"
        mock_cpt_2.delivered_location.x = 100100.0
        mock_cpt_2.delivered_location.y = 500100.0
        mock_cpt_2.delivered_vertical_position_offset = 2.0
        mock_cpt_2.final_depth = 12.0
        mock_cpt_2.data = pl.DataFrame(
            {"penetrationLength": [0.0, 1.5, 3.0], "coneResistance": [2.0, 3.0, 4.0], "friction": [0.02, 0.03, 0.04]}
        )
        mock_cpt_2.column_void_mapping = {
            "penetrationLength": 9999.0,
            "coneResistance": 9999.0,
            "friction": 9999.0,
        }

        parsed_files = {"CPT-001": mock_cpt_data, "CPT-002": mock_cpt_2}

        result = create_from_parsed_gef_cpts(parsed_files)

        assert result.name == "CPT-001...CPT-002"
        assert len(result.collars) == 2
        assert len(result.measurements) == 7  # 4 + 3 measurements
        assert result.measurements["hole_index"].nunique() == 2

    def test_inconsistent_epsg_codes_raises_error(self, mock_cpt_data) -> None:
        mock_cpt_2 = Mock(
            spec=[
                "delivered_location",
                "delivered_vertical_position_offset",
                "final_depth",
                "data",
                "column_void_mapping",
            ]
        )
        mock_cpt_2.delivered_location = Mock()
        mock_cpt_2.delivered_location.srs_name = "EPSG:4326"  # Different EPSG!
        mock_cpt_2.delivered_location.x = 5.0
        mock_cpt_2.delivered_location.y = 52.0
        mock_cpt_2.delivered_vertical_position_offset = 0.0
        mock_cpt_2.final_depth = 8.0
        mock_cpt_2.data = pl.DataFrame({"penetrationLength": [0.0, 1.0], "coneResistance": [1.0, 2.0]})
        mock_cpt_2.column_void_mapping = {
            "penetrationLength": 9999.0,
            "coneResistance": 9999.0,
            "friction": 9999.0,
        }

        parsed_files = {"CPT-001": mock_cpt_data, "CPT-002": mock_cpt_2}

        with pytest.raises(ValueError, match="Inconsistent EPSG codes"):
            create_from_parsed_gef_cpts(parsed_files)

    def test_missing_vertical_offset_defaults_to_zero(self, mock_cpt_data) -> None:
        mock_cpt_data.delivered_vertical_position_offset = None
        parsed_files = {"CPT-001": mock_cpt_data}

        result = create_from_parsed_gef_cpts(parsed_files)

        assert result.collars.iloc[0]["z"] == 0.0

    def test_missing_final_depth_uses_max_penetration(self, mock_cpt_data) -> None:
        mock_cpt_data.final_depth = 0.0
        parsed_files = {"CPT-001": mock_cpt_data}

        result = create_from_parsed_gef_cpts(parsed_files)

        assert result.collars.iloc[0]["final_depth"] == 3.0  # max of [0,1,2,3]

    def test_measurements_have_hole_index_column(self, mock_cpt_data) -> None:
        parsed_files = {"CPT-001": mock_cpt_data}

        result = create_from_parsed_gef_cpts(parsed_files)

        assert "hole_index" in result.measurements.columns
        assert all(result.measurements["hole_index"] == 1)
        # Verify hole_index is the first column
        assert result.measurements.columns[0] == "hole_index"

    def test_hole_index_increments_correctly(self, mock_cpt_data) -> None:
        mock_cpt_2 = Mock(
            spec=[
                "delivered_location",
                "delivered_vertical_position_offset",
                "final_depth",
                "data",
                "column_void_mapping",
            ]
        )
        mock_cpt_2.delivered_location = Mock()
        mock_cpt_2.delivered_location.srs_name = "EPSG:28992"
        mock_cpt_2.delivered_location.x = 100100.0
        mock_cpt_2.delivered_location.y = 500100.0
        mock_cpt_2.delivered_vertical_position_offset = 0.0
        mock_cpt_2.final_depth = 8.0
        mock_cpt_2.data = pl.DataFrame(
            {"penetrationLength": [0.0, 2.0], "coneResistance": [1.0, 2.0], "friction": [0.01, 0.02]}
        )
        mock_cpt_2.column_void_mapping = {
            "penetrationLength": 9999.0,
            "coneResistance": 9999.0,
            "friction": 9999.0,
        }

        parsed_files = {"CPT-001": mock_cpt_data, "CPT-002": mock_cpt_2}

        result = create_from_parsed_gef_cpts(parsed_files)

        hole_1_measurements = result.measurements[result.measurements["hole_index"] == 1]
        hole_2_measurements = result.measurements[result.measurements["hole_index"] == 2]

        assert len(hole_1_measurements) == 4
        assert len(hole_2_measurements) == 2

    def test_collar_datatypes_are_correct(self, mock_cpt_data) -> None:
        parsed_files = {"CPT-001": mock_cpt_data}

        result = create_from_parsed_gef_cpts(parsed_files)

        assert result.collars["hole_index"].dtype == "int32"
        assert result.collars["hole_id"].dtype == "string"
        assert result.collars["x"].dtype == "float64"
        assert result.collars["y"].dtype == "float64"
        assert result.collars["z"].dtype == "float64"
        assert result.collars["final_depth"].dtype == "float64"

    def test_preserves_original_column_order(self, mock_cpt_data) -> None:
        parsed_files = {"CPT-001": mock_cpt_data}

        result = create_from_parsed_gef_cpts(parsed_files)

        # hole_index should be first, then original columns in order
        expected_columns = ["hole_index", "penetrationLength", "coneResistance", "friction"]
        assert list(result.measurements.columns) == expected_columns

    def test_no_column_duplication_if_hole_index_exists(self, mock_cpt_data) -> None:
        # Add hole_index to the original data
        mock_cpt_data.data = mock_cpt_data.data.with_columns(pl.lit(99).cast(pl.Int32).alias("hole_index"))

        parsed_files = {"CPT-001": mock_cpt_data}
        result = create_from_parsed_gef_cpts(parsed_files)

        # Should only have one hole_index column
        hole_index_count = sum(1 for col in result.measurements.columns if col == "hole_index")
        assert hole_index_count == 1
        # And it should be the correct value (1, not 99)
        assert all(result.measurements["hole_index"] == 1)

    @patch("evo.data_converters.gef.importer.gef_to_downhole_collection.logger")
    def test_logging_for_successful_processing(self, mock_logger, mock_cpt_data) -> None:
        parsed_files = {"CPT-001": mock_cpt_data}

        create_from_parsed_gef_cpts(parsed_files)

        # Verify logging calls were made
        assert mock_logger.info.called
        assert mock_logger.debug.called

    def test_three_cpt_files_collection_name(self, mock_cpt_data) -> None:
        mock_cpt_2 = Mock(
            spec=[
                "delivered_location",
                "delivered_vertical_position_offset",
                "final_depth",
                "data",
                "column_void_mapping",
            ]
        )
        mock_cpt_2.delivered_location = Mock()
        mock_cpt_2.delivered_location.srs_name = "EPSG:28992"
        mock_cpt_2.delivered_location.x = 100100.0
        mock_cpt_2.delivered_location.y = 500100.0
        mock_cpt_2.delivered_vertical_position_offset = 0.0
        mock_cpt_2.final_depth = 8.0
        mock_cpt_2.data = pl.DataFrame({"penetrationLength": [0.0], "coneResistance": [0.0], "friction": [0.0]})
        mock_cpt_2.column_void_mapping = {
            "penetrationLength": 9999.0,
            "coneResistance": 9999.0,
            "friction": 9999.0,
        }

        mock_cpt_3 = Mock(
            spec=[
                "delivered_location",
                "delivered_vertical_position_offset",
                "final_depth",
                "data",
                "column_void_mapping",
            ]
        )
        mock_cpt_3.delivered_location = Mock()
        mock_cpt_3.delivered_location.srs_name = "EPSG:28992"
        mock_cpt_3.delivered_location.x = 100200.0
        mock_cpt_3.delivered_location.y = 500200.0
        mock_cpt_3.delivered_vertical_position_offset = 0.0
        mock_cpt_3.final_depth = 9.0
        mock_cpt_3.data = pl.DataFrame({"penetrationLength": [0.0], "coneResistance": [0.0], "friction": [0.0]})
        mock_cpt_3.column_void_mapping = {
            "penetrationLength": 9999.0,
            "coneResistance": 9999.0,
            "friction": 9999.0,
        }

        parsed_files = {"CPT-001": mock_cpt_data, "CPT-002": mock_cpt_2, "CPT-003": mock_cpt_3}

        result = create_from_parsed_gef_cpts(parsed_files)

        assert result.name == "CPT-001...CPT-003"

    def test_handles_zero_vertical_offset_explicitly(self, mock_cpt_data) -> None:
        mock_cpt_data.delivered_vertical_position_offset = 0.0
        parsed_files = {"CPT-001": mock_cpt_data}

        result = create_from_parsed_gef_cpts(parsed_files)

        assert result.collars.iloc[0]["z"] == 0.0

    def test_handles_negative_vertical_offset(self, mock_cpt_data) -> None:
        mock_cpt_data.delivered_vertical_position_offset = -2.5
        parsed_files = {"CPT-001": mock_cpt_data}

        result = create_from_parsed_gef_cpts(parsed_files)

        assert result.collars.iloc[0]["z"] == -2.5


class TestIntegration:
    """Integration test with dummy data"""

    def test_realistic_multi_cpt_scenario(self) -> None:
        """Test with a set of dummy CPT data"""
        cpts = {}

        for i in range(1, 6):
            mock = Mock(
                spec=[
                    "delivered_location",
                    "delivered_vertical_position_offset",
                    "final_depth",
                    "data",
                    "column_void_mapping",
                ]
            )
            mock.delivered_location = Mock()
            mock.delivered_location.srs_name = "EPSG:28992"
            mock.delivered_location.x = 100000.0 + (i * 50)
            mock.delivered_location.y = 500000.0 + (i * 50)
            mock.delivered_vertical_position_offset = 0.5
            mock.final_depth = 10.0 + i
            mock.data = pl.DataFrame(
                {
                    "penetrationLength": [j * 0.5 for j in range(20)],
                    "coneResistance": [j * 0.1 + i for j in range(20)],
                    "friction": [j * 0.01 for j in range(20)],
                }
            )
            mock.column_void_mapping = {
                "penetrationLength": 9999.0,
                "coneResistance": 9999.0,
                "friction": 9999.0,
            }

            cpts[f"CPT-{i:03d}"] = mock

        result = create_from_parsed_gef_cpts(cpts)

        assert result.name == "CPT-001...CPT-005"
        assert len(result.collars) == 5
        assert len(result.measurements) == 100  # 5 CPTs * 20 measurements
        assert result.epsg_code == 28992
        assert set(result.measurements["hole_index"].unique()) == {1, 2, 3, 4, 5}
