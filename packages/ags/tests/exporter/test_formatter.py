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

import pandas as pd

from evo.data_converters.ags.exporter.formatting import (
    _format_decimal_places,
    _format_scientific_notation,
    _format_significant_figures,
    _format_yes_no,
    add_ags_prefix_rows,
    apply_type_formatting,
    format_column_by_type,
    lookup_column_unit_and_type,
)


class TestFormatDecimalPlaces:
    """Tests for _format_decimal_places function."""

    def test_formats_integer_to_decimal_places(self):
        series = pd.Series(["10", "20", "30"])
        result = _format_decimal_places(series, 2)
        assert result.tolist() == ["10.00", "20.00", "30.00"]

    def test_formats_float_to_fewer_decimal_places(self):
        series = pd.Series(["10.12345", "20.98765"])
        result = _format_decimal_places(series, 2)
        assert result.tolist() == ["10.12", "20.99"]

    def test_formats_float_to_more_decimal_places(self):
        series = pd.Series(["10.1", "20.9"])
        result = _format_decimal_places(series, 4)
        assert result.tolist() == ["10.1000", "20.9000"]

    def test_zero_decimal_places(self):
        series = pd.Series(["10.6", "20.4", "30.5"])
        result = _format_decimal_places(series, 0)
        assert result.tolist() == ["11", "20", "30"]

    def test_empty_values_pass_through(self):
        series = pd.Series(["", "nan", "None", "10.5"])
        result = _format_decimal_places(series, 2)
        assert result.tolist() == ["", "", "", "10.50"]

    def test_non_numeric_values_pass_through(self):
        series = pd.Series(["abc", "10.5", "xyz"])
        result = _format_decimal_places(series, 2)
        assert result.tolist() == ["abc", "10.50", "xyz"]

    def test_negative_numbers(self):
        series = pd.Series(["-10.123", "-0.5"])
        result = _format_decimal_places(series, 2)
        assert result.tolist() == ["-10.12", "-0.50"]


class TestFormatSignificantFigures:
    """Tests for _format_significant_figures function."""

    def test_formats_to_significant_figures(self):
        series = pd.Series(["123.456", "0.00789"])
        result = _format_significant_figures(series, 3)
        assert result.tolist() == ["123", "0.00789"]

    def test_one_significant_figure(self):
        series = pd.Series(["123", "456", "789"])
        result = _format_significant_figures(series, 1)
        assert result.tolist() == ["100", "500", "800"]

    def test_zero_returns_zero(self):
        series = pd.Series(["0", "0.0", "0.000"])
        result = _format_significant_figures(series, 3)
        assert result.tolist() == ["0", "0", "0"]

    def test_empty_values_pass_through(self):
        series = pd.Series(["", "nan", "None", "123"])
        result = _format_significant_figures(series, 2)
        assert result.tolist() == ["", "", "", "120"]

    def test_non_numeric_values_pass_through(self):
        series = pd.Series(["abc", "123", "xyz"])
        result = _format_significant_figures(series, 2)
        assert result.tolist() == ["abc", "120", "xyz"]

    def test_negative_numbers(self):
        series = pd.Series(["-123.456"])
        result = _format_significant_figures(series, 2)
        assert result.tolist() == ["-120"]


class TestFormatScientificNotation:
    """Tests for _format_scientific_notation function."""

    def test_formats_to_scientific_notation(self):
        series = pd.Series(["1234.5", "0.00123"])
        result = _format_scientific_notation(series, 2)
        assert result.tolist() == ["1.23E+03", "1.23E-03"]

    def test_zero_decimal_places(self):
        series = pd.Series(["1234.5"])
        result = _format_scientific_notation(series, 0)
        assert result.tolist() == ["1E+03"]

    def test_empty_values_pass_through(self):
        series = pd.Series(["", "nan", "None", "100"])
        result = _format_scientific_notation(series, 1)
        assert result.tolist() == ["", "", "", "1.0E+02"]

    def test_non_numeric_values_pass_through(self):
        series = pd.Series(["abc", "100", "xyz"])
        result = _format_scientific_notation(series, 1)
        assert result.tolist() == ["abc", "1.0E+02", "xyz"]

    def test_negative_numbers(self):
        series = pd.Series(["-1234.5"])
        result = _format_scientific_notation(series, 2)
        assert result.tolist() == ["-1.23E+03"]


class TestFormatYesNo:
    """Tests for _format_yes_no function."""

    def test_converts_true_false_to_y_n(self):
        series = pd.Series(["True", "False", "True"])
        result = _format_yes_no(series)
        assert result.tolist() == ["Y", "N", "Y"]

    def test_handles_lowercase(self):
        series = pd.Series(["true", "false"])
        result = _format_yes_no(series)
        assert result.tolist() == ["Y", "N"]

    def test_other_values_unchanged(self):
        series = pd.Series(["Y", "N", "other", ""])
        result = _format_yes_no(series)
        assert result.tolist() == ["Y", "N", "other", ""]


class TestFormatColumnByType:
    """Tests for format_column_by_type function."""

    def test_decimal_places_type_codes(self):
        series = pd.Series(["10.12345"])
        assert format_column_by_type(series, "0DP").tolist() == ["10"]
        assert format_column_by_type(series, "1DP").tolist() == ["10.1"]
        assert format_column_by_type(series, "2DP").tolist() == ["10.12"]
        assert format_column_by_type(series, "3DP").tolist() == ["10.123"]
        assert format_column_by_type(series, "4DP").tolist() == ["10.1235"]

    def test_significant_figures_type_codes(self):
        series = pd.Series(["12345"])
        assert format_column_by_type(series, "1SF").tolist() == ["10000"]
        assert format_column_by_type(series, "2SF").tolist() == ["12000"]
        assert format_column_by_type(series, "3SF").tolist() == ["12300"]

    def test_scientific_notation_type_codes(self):
        series = pd.Series(["12345"])
        assert format_column_by_type(series, "0SCI").tolist() == ["1E+04"]
        assert format_column_by_type(series, "1SCI").tolist() == ["1.2E+04"]
        assert format_column_by_type(series, "2SCI").tolist() == ["1.23E+04"]

    def test_yes_no_type_code(self):
        series = pd.Series(["True", "False"])
        result = format_column_by_type(series, "YN")
        assert result.tolist() == ["Y", "N"]

    def test_unknown_type_code_passes_through(self):
        series = pd.Series(["abc", "123"])
        result = format_column_by_type(series, "UNKNOWN")
        assert result.tolist() == ["abc", "123"]

    def test_empty_type_code_passes_through(self):
        series = pd.Series(["abc", "123"])
        result = format_column_by_type(series, "")
        assert result.tolist() == ["abc", "123"]

    def test_none_type_code_passes_through(self):
        series = pd.Series(["abc", "123"])
        result = format_column_by_type(series, None)
        assert result.tolist() == ["abc", "123"]


class TestApplyTypeFormatting:
    """Tests for apply_type_formatting function."""

    def test_formats_data_rows_according_to_type_row(self):
        df = pd.DataFrame(
            {
                "HEADING": ["UNIT", "TYPE", "DATA", "DATA"],
                "COL_A": ["m", "2DP", "10.12345", "20.98765"],
                "COL_B": ["", "1SF", "12345", "67890"],
            }
        )
        result = apply_type_formatting(df)

        assert result.loc[2, "COL_A"] == "10.12"
        assert result.loc[3, "COL_A"] == "20.99"
        assert result.loc[2, "COL_B"] == "10000"
        assert result.loc[3, "COL_B"] == "70000"

    def test_preserves_unit_and_type_rows(self):
        df = pd.DataFrame(
            {
                "HEADING": ["UNIT", "TYPE", "DATA"],
                "COL_A": ["m", "2DP", "10.12345"],
            }
        )
        result = apply_type_formatting(df)

        assert result.loc[0, "COL_A"] == "m"
        assert result.loc[1, "COL_A"] == "2DP"

    def test_returns_unchanged_if_fewer_than_three_rows(self):
        df = pd.DataFrame(
            {
                "HEADING": ["UNIT", "TYPE"],
                "COL_A": ["m", "2DP"],
            }
        )
        result = apply_type_formatting(df)
        pd.testing.assert_frame_equal(result, df)

    def test_handles_empty_type_codes(self):
        df = pd.DataFrame(
            {
                "HEADING": ["UNIT", "TYPE", "DATA"],
                "COL_A": ["", "", "unchanged"],
            }
        )
        result = apply_type_formatting(df)
        assert result.loc[2, "COL_A"] == "unchanged"


class TestLookupColumnUnitAndType:
    """Tests for lookup_column_unit_and_type function."""

    def test_returns_defaults_when_dict_table_missing(self):
        unit, dtype = lookup_column_unit_and_type("SCPT", "SCPT_RES", {})
        assert unit == ""
        assert dtype == "X"

    def test_returns_defaults_when_dict_table_empty(self):
        static_tables = {"DICT": pd.DataFrame()}
        unit, dtype = lookup_column_unit_and_type("SCPT", "SCPT_RES", static_tables)
        assert unit == ""
        assert dtype == "X"

    def test_returns_defaults_when_required_columns_missing(self):
        static_tables = {"DICT": pd.DataFrame({"WRONG_COL": ["value"]})}
        unit, dtype = lookup_column_unit_and_type("SCPT", "SCPT_RES", static_tables)
        assert unit == ""
        assert dtype == "X"

    def test_finds_matching_entry(self):
        static_tables = {
            "DICT": pd.DataFrame(
                {
                    "DICT_GRP": ["SCPT", "SCPT", "LOCA"],
                    "DICT_HDNG": ["SCPT_RES", "SCPT_DPTH", "LOCA_ID"],
                    "DICT_UNIT": ["MPa", "m", ""],
                    "DICT_DTYP": ["2DP", "3DP", "ID"],
                }
            )
        }
        unit, dtype = lookup_column_unit_and_type("SCPT", "SCPT_RES", static_tables)
        assert unit == "MPa"
        assert dtype == "2DP"

    def test_returns_defaults_when_no_match(self):
        static_tables = {
            "DICT": pd.DataFrame(
                {
                    "DICT_GRP": ["SCPT"],
                    "DICT_HDNG": ["SCPT_RES"],
                    "DICT_UNIT": ["MPa"],
                    "DICT_DTYP": ["2DP"],
                }
            )
        }
        unit, dtype = lookup_column_unit_and_type("UNKNOWN", "UNKNOWN_COL", static_tables)
        assert unit == ""
        assert dtype == "X"


class TestAddAgsPrefixRows:
    """Tests for add_ags_prefix_rows function."""

    def test_adds_heading_column(self):
        df = pd.DataFrame({"COL_A": ["val1", "val2"]})
        result = add_ags_prefix_rows(df, "TEST")

        assert "HEADING" in result.columns
        assert result.columns[0] == "HEADING"

    def test_adds_unit_and_type_rows(self):
        df = pd.DataFrame({"COL_A": ["val1", "val2"]})
        result = add_ags_prefix_rows(df, "TEST")

        assert result.loc[0, "HEADING"] == "UNIT"
        assert result.loc[1, "HEADING"] == "TYPE"
        assert result.loc[2, "HEADING"] == "DATA"
        assert result.loc[3, "HEADING"] == "DATA"

    def test_result_has_correct_row_count(self):
        df = pd.DataFrame({"COL_A": ["val1", "val2", "val3"]})
        result = add_ags_prefix_rows(df, "TEST")

        # 2 prefix rows + 3 data rows
        assert len(result) == 5

    def test_uses_static_tables_for_lookup(self):
        df = pd.DataFrame({"SCPT_RES": ["10.12345"]})
        static_tables = {
            "DICT": pd.DataFrame(
                {
                    "DICT_GRP": ["SCPT"],
                    "DICT_HDNG": ["SCPT_RES"],
                    "DICT_UNIT": ["MPa"],
                    "DICT_DTYP": ["2DP"],
                }
            )
        }
        result = add_ags_prefix_rows(df, "SCPT", static_tables)

        assert result.loc[0, "SCPT_RES"] == "MPa"
        assert result.loc[1, "SCPT_RES"] == "2DP"
        # Data should be formatted according to 2DP
        assert result.loc[2, "SCPT_RES"] == "10.12"

    def test_handles_none_static_tables(self):
        df = pd.DataFrame({"COL_A": ["val1"]})
        result = add_ags_prefix_rows(df, "TEST", None)

        # Should not raise, uses defaults
        assert len(result) == 3
