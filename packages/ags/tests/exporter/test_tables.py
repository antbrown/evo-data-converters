#  Copyright © 2025 Bentley Systems, Incorporated
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from unittest.mock import MagicMock

import pandas as pd

from evo.data_converters.ags.exporter.tables import (
    _build_hole_id_lookup,
    _build_loca_table,
    _build_proj_table,
    _build_tran_table,
    _dataframe_to_ags_strings,
    _derive_headings,
    _detect_ags_table_name,
    _split_hole_id_and_tesn,
)


class TestSplitHoleIdAndTesn:
    """Tests for _split_hole_id_and_tesn function."""

    def test_no_colon_returns_original_id_empty_tesn(self):
        base, tesn = _split_hole_id_and_tesn("HKZ1-CPT04")
        assert base == "HKZ1-CPT04"
        assert tesn == ""

    def test_colon_with_numeric_suffix_splits_correctly(self):
        base, tesn = _split_hole_id_and_tesn("HKZ1-CPT04:1")
        assert base == "HKZ1-CPT04"
        assert tesn == "1"

    def test_colon_with_multi_digit_suffix(self):
        base, tesn = _split_hole_id_and_tesn("HKZ1-CPT04:123")
        assert base == "HKZ1-CPT04"
        assert tesn == "123"

    def test_colon_with_non_numeric_suffix_returns_original(self):
        base, tesn = _split_hole_id_and_tesn("HKZ1-CPT04:abc")
        assert base == "HKZ1-CPT04:abc"
        assert tesn == ""

    def test_multiple_colons_splits_on_last(self):
        base, tesn = _split_hole_id_and_tesn("HKZ1:CPT04:2")
        assert base == "HKZ1:CPT04"
        assert tesn == "2"

    def test_colon_with_zero_suffix(self):
        base, tesn = _split_hole_id_and_tesn("HKZ1-CPT04:0")
        assert base == "HKZ1-CPT04"
        assert tesn == "0"


class TestBuildHoleIdLookup:
    """Tests for _build_hole_id_lookup function."""

    def test_builds_lookup_with_simple_ids(self):
        collars = pd.DataFrame(
            {
                "hole_index": [1, 2, 3],
                "hole_id": ["BH01", "BH02", "BH03"],
            }
        )
        lookup = _build_hole_id_lookup(collars)

        assert lookup.loc[1, "loca_id"] == "BH01"
        assert lookup.loc[2, "loca_id"] == "BH02"
        assert lookup.loc[3, "loca_id"] == "BH03"
        assert lookup.loc[1, "tesn"] == ""
        assert lookup.loc[2, "tesn"] == ""
        assert lookup.loc[3, "tesn"] == ""

    def test_builds_lookup_with_tesn_suffixes(self):
        collars = pd.DataFrame(
            {
                "hole_index": [1, 2],
                "hole_id": ["BH01:1", "BH01:2"],
            }
        )
        lookup = _build_hole_id_lookup(collars)

        assert lookup.loc[1, "loca_id"] == "BH01"
        assert lookup.loc[1, "tesn"] == "1"
        assert lookup.loc[2, "loca_id"] == "BH01"
        assert lookup.loc[2, "tesn"] == "2"

    def test_lookup_indexed_by_hole_index(self):
        collars = pd.DataFrame(
            {
                "hole_index": [10, 20, 30],
                "hole_id": ["BH01", "BH02", "BH03"],
            }
        )
        lookup = _build_hole_id_lookup(collars)

        assert list(lookup.index) == [10, 20, 30]


class TestDataframeToAgsStrings:
    """Tests for _dataframe_to_ags_strings function."""

    def test_converts_all_values_to_strings(self):
        df = pd.DataFrame({"col": [1, 2.5, True]})
        result = _dataframe_to_ags_strings(df)

        assert result["col"].tolist() == ["1", "2.5", "True"]

    def test_replaces_nan_string_with_empty(self):
        df = pd.DataFrame({"col": ["value", "nan", "other"]})
        result = _dataframe_to_ags_strings(df)

        assert result["col"].tolist() == ["value", "", "other"]

    def test_replaces_nat_string_with_empty(self):
        df = pd.DataFrame({"col": ["value", "NaT", "other"]})
        result = _dataframe_to_ags_strings(df)

        assert result["col"].tolist() == ["value", "", "other"]

    def test_handles_actual_nan_values(self):
        df = pd.DataFrame({"col": [1.0, float("nan"), 3.0]})
        result = _dataframe_to_ags_strings(df)

        # float nan becomes string "nan" then replaced with ""
        assert result["col"].tolist() == ["1.0", "", "3.0"]


class TestDeriveHeadings:
    """Tests for _derive_headings function."""

    def test_extracts_column_names_as_lists(self):
        tables = {
            "PROJ": pd.DataFrame({"PROJ_ID": [], "PROJ_NAME": []}),
            "LOCA": pd.DataFrame({"LOCA_ID": [], "LOCA_NATE": [], "LOCA_NATN": []}),
        }
        headings = _derive_headings(tables)

        assert headings["PROJ"] == ["PROJ_ID", "PROJ_NAME"]
        assert headings["LOCA"] == ["LOCA_ID", "LOCA_NATE", "LOCA_NATN"]

    def test_empty_tables_dict(self):
        headings = _derive_headings({})
        assert headings == {}


class TestBuildProjTable:
    """Tests for _build_proj_table function."""

    def test_uses_tags_when_available(self):
        dhc = MagicMock()
        dhc.tags = {
            "AGS:PROJ:PROJ_ID": "PROJECT123",
            "AGS:PROJ:PROJ_NAME": "Test Project",
        }
        dhc.uuid = "uuid-123"
        dhc.name = "DHC Name"
        dhc.description = "Test description"

        result = _build_proj_table(dhc)

        assert result.loc[0, "PROJ_ID"] == "PROJECT123"
        assert result.loc[0, "PROJ_NAME"] == "Test Project"
        assert result.loc[0, "PROJ_MEMO"] == "Test description"

    def test_uses_defaults_when_tags_missing(self):
        dhc = MagicMock()
        dhc.tags = {}
        dhc.uuid = "uuid-123"
        dhc.name = "DHC Name"
        dhc.description = None

        result = _build_proj_table(dhc)

        assert result.loc[0, "PROJ_ID"] == "uuid-123"
        assert result.loc[0, "PROJ_NAME"] == "DHC Name"
        assert "Exported from Seequent Evo Data Converters" in result.loc[0, "PROJ_MEMO"]

    def test_handles_none_tags(self):
        dhc = MagicMock()
        dhc.tags = None
        dhc.uuid = "uuid-123"
        dhc.name = "DHC Name"
        dhc.description = "Description"

        result = _build_proj_table(dhc)

        assert result.loc[0, "PROJ_ID"] == "uuid-123"


class TestBuildTranTable:
    """Tests for _build_tran_table function."""

    def test_returns_expected_columns(self):
        result = _build_tran_table()

        expected_columns = [
            "TRAN_ISNO",
            "TRAN_DATE",
            "TRAN_PROD",
            "TRAN_STAT",
            "TRAN_DESC",
            "TRAN_AGS",
            "TRAN_RECV",
        ]
        assert list(result.columns) == expected_columns

    def test_has_single_row(self):
        result = _build_tran_table()
        assert len(result) == 1

    def test_has_expected_static_values(self):
        result = _build_tran_table()

        assert result.loc[0, "TRAN_ISNO"] == "1"
        assert result.loc[0, "TRAN_PROD"] == "Evo Data Converters"
        assert result.loc[0, "TRAN_STAT"] == "Final"
        assert result.loc[0, "TRAN_AGS"] == "4.1"


class TestBuildLocaTable:
    """Tests for _build_loca_table function."""

    def test_builds_basic_loca_table(self):
        collars_df = pd.DataFrame(
            {
                "hole_index": [1, 2],
                "hole_id": ["BH01", "BH02"],
                "x": [100.0, 200.0],
                "y": [1000.0, 2000.0],
                "z": [50.0, 60.0],
            }
        )
        collars_mock = MagicMock()
        collars_mock.df = collars_df
        collars_mock.get_attribute_column_names.return_value = []

        dhc = MagicMock()
        dhc.collars = collars_mock

        result = _build_loca_table(dhc)

        assert list(result.columns) == ["LOCA_ID", "LOCA_NATE", "LOCA_NATN", "LOCA_GL"]
        assert result.loc[0, "LOCA_ID"] == "BH01"
        assert result.loc[0, "LOCA_NATE"] == 100.0
        assert result.loc[0, "LOCA_NATN"] == 1000.0
        assert result.loc[0, "LOCA_GL"] == 50.0

    def test_includes_additional_loca_attributes(self):
        collars_df = pd.DataFrame(
            {
                "hole_index": [1],
                "hole_id": ["BH01"],
                "x": [100.0],
                "y": [1000.0],
                "z": [50.0],
                "LOCA_TYPE": ["CP"],
                "LOCA_STAT": ["Active"],
                "OTHER_COL": ["ignored"],
            }
        )
        collars_mock = MagicMock()
        collars_mock.df = collars_df
        collars_mock.get_attribute_column_names.return_value = ["LOCA_TYPE", "LOCA_STAT", "OTHER_COL"]

        dhc = MagicMock()
        dhc.collars = collars_mock

        result = _build_loca_table(dhc)

        assert "LOCA_TYPE" in result.columns
        assert "LOCA_STAT" in result.columns
        assert "OTHER_COL" not in result.columns

    def test_splits_tesn_from_hole_id(self):
        collars_df = pd.DataFrame(
            {
                "hole_index": [1],
                "hole_id": ["BH01:1"],
                "x": [100.0],
                "y": [1000.0],
                "z": [50.0],
            }
        )
        collars_mock = MagicMock()
        collars_mock.df = collars_df
        collars_mock.get_attribute_column_names.return_value = []

        dhc = MagicMock()
        dhc.collars = collars_mock

        result = _build_loca_table(dhc)

        # LOCA_ID should have the base without TESN suffix
        assert result.loc[0, "LOCA_ID"] == "BH01"


class TestDetectAgsTableName:
    """Tests for _detect_ags_table_name function."""

    def test_detects_scpt_from_column_prefixes(self):
        table = MagicMock()
        table.get_attribute_columns.return_value = ["SCPT_RES", "SCPT_FRES", "SCPT_PWP1"]

        result = _detect_ags_table_name(table)
        assert result == "SCPT"

    def test_detects_geol_from_column_prefixes(self):
        table = MagicMock()
        table.get_attribute_columns.return_value = ["GEOL_DESC", "GEOL_LEG"]

        result = _detect_ags_table_name(table)
        assert result == "GEOL"

    def test_ignores_foreign_key_columns(self):
        table = MagicMock()
        table.get_attribute_columns.return_value = ["LOCA_ID", "FILE_FSET", "SCPT_RES"]

        result = _detect_ags_table_name(table)
        # LOCA_ID and FILE_FSET are foreign keys, should be ignored
        assert result == "SCPT"

    def test_returns_most_common_prefix(self):
        table = MagicMock()
        table.get_attribute_columns.return_value = ["SCPT_RES", "SCPT_FRES", "OTHER_COL"]

        result = _detect_ags_table_name(table)
        assert result == "SCPT"

    def test_returns_none_for_no_prefixed_columns(self):
        table = MagicMock()
        table.get_attribute_columns.return_value = ["nounderscore", "another"]

        result = _detect_ags_table_name(table)
        assert result is None

    def test_returns_none_for_empty_columns(self):
        table = MagicMock()
        table.get_attribute_columns.return_value = []

        result = _detect_ags_table_name(table)
        assert result is None

    def test_returns_none_for_only_foreign_keys(self):
        table = MagicMock()
        table.get_attribute_columns.return_value = ["LOCA_ID", "FILE_FSET"]

        result = _detect_ags_table_name(table)
        assert result is None
