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

from __future__ import annotations

import re
from importlib.resources import as_file, files

import numpy as np
import pandas as pd
from python_ags4 import AGS4

import evo.logging

logger = evo.logging.getLogger("data_converters")

# Regex patterns for AGS type codes
_DP_PATTERN = re.compile(r"^(\d+)DP$")  # Decimal places: 0DP, 1DP, 2DP, etc.
_SF_PATTERN = re.compile(r"^(\d+)SF$")  # Significant figures: 1SF, 2SF, etc.
_SCI_PATTERN = re.compile(r"^(\d+)SCI$")  # Scientific notation: 0SCI, 1SCI, etc.


def _format_decimal_places(series: pd.Series, decimals: int) -> pd.Series:
    """
    Format numeric values to a fixed number of decimal places.

    Non-numeric and empty values are passed through unchanged.
    """

    def format_value(val: str) -> str:
        if not val or val in ("", "nan", "None"):
            return ""
        try:
            return f"{float(val):.{decimals}f}"
        except (ValueError, TypeError):
            return val

    return series.apply(format_value)


def _format_significant_figures(series: pd.Series, sig_figs: int) -> pd.Series:
    """
    Format numeric values to a specific number of significant figures.

    Non-numeric and empty values are passed through unchanged.
    """

    def format_value(val: str) -> str:
        if not val or val in ("", "nan", "None"):
            return ""
        try:
            num = float(val)
            if num == 0:
                return "0"
            # Use numpy for significant figure rounding
            rounded = float(
                np.format_float_positional(
                    num,
                    precision=sig_figs,
                    unique=False,
                    fractional=False,
                    trim="k",
                )
            )
            # Remove trailing zeros after decimal point if not needed
            return f"{rounded:g}"
        except (ValueError, TypeError):
            return val

    return series.apply(format_value)


def _format_scientific_notation(series: pd.Series, decimals: int) -> pd.Series:
    """
    Format numeric values in scientific notation.

    Non-numeric and empty values are passed through unchanged.
    """

    def format_value(val: str) -> str:
        if not val or val in ("", "nan", "None"):
            return ""
        try:
            return f"{float(val):.{decimals}E}"
        except (ValueError, TypeError):
            return val

    return series.apply(format_value)


def _format_yes_no(series: pd.Series) -> pd.Series:
    """
    Format boolean values as Y/N.

    Converts "True"/"False" strings to "Y"/"N".
    """
    return series.replace({"True": "Y", "False": "N", "true": "Y", "false": "N"})


def format_column_by_type(series: pd.Series, type_code: str) -> pd.Series:
    """
    Format a column's values according to its AGS type code.

    Currently implemented type codes:
    - ``nDP``: Fixed decimal places (0DP, 1DP, 2DP, 3DP, 4DP)
    - ``nSF``: Significant figures (1SF, 2SF, 3SF, 4SF)
    - ``nSCI``: Scientific notation (0SCI, 1SCI, 2SCI, 3SCI, 4SCI)

    Unrecognised type codes pass values through unchanged.

    :param series: Column values to format.
    :param type_code: AGS type code (e.g., ``"2DP"``, ``"3SF"``).
    :returns: Formatted series.
    """
    if not type_code:
        return series

    # Decimal places: 0DP, 1DP, 2DP, etc.
    if match := _DP_PATTERN.match(type_code):
        decimals = int(match.group(1))
        return _format_decimal_places(series, decimals)

    # Significant figures: 1SF, 2SF, etc.
    if match := _SF_PATTERN.match(type_code):
        sig_figs = int(match.group(1))
        return _format_significant_figures(series, sig_figs)

    # Scientific notation: 0SCI, 1SCI, etc.
    if match := _SCI_PATTERN.match(type_code):
        decimals = int(match.group(1))
        return _format_scientific_notation(series, decimals)

    # Yes/No: YN
    if type_code == "YN":
        return _format_yes_no(series)

    # TODO: Implement other type codes as needed:
    # - DMS: Degrees:Minutes:Seconds
    # - DT: DateTime in ISO format
    # - MC: Moisture content (special rounding rules)
    # - T: Elapsed time

    return series


def apply_type_formatting(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply AGS type formatting to a DataFrame that has UNIT/TYPE prefix rows.

    Expects the DataFrame to have:
    - A ``HEADING`` column with values ``UNIT``, ``TYPE``, ``DATA``
    - Row 0: UNIT row
    - Row 1: TYPE row
    - Rows 2+: DATA rows

    :param df: DataFrame with AGS prefix rows.
    :returns: DataFrame with DATA rows formatted according to TYPE specifications.
    """
    if len(df) < 3:
        return df

    # Extract type codes from row 1 (TYPE row)
    type_row = df.iloc[1]

    result = df.copy()

    for col in df.columns:
        if col == "HEADING":
            continue

        type_code = str(type_row[col]) if pd.notna(type_row[col]) else ""

        # Only format DATA rows (index 2 onwards)
        data_values = result.loc[2:, col].copy()
        formatted_values = format_column_by_type(data_values, type_code)
        result.loc[2:, col] = formatted_values

    return result


def load_static_tables() -> dict[str, pd.DataFrame]:
    """
    Load static reference tables from the AGS standard dictionary.

    The standard dictionary contains definitions for units, data types,
    abbreviations, and column specifications used across AGS files.

    :returns: DataFrames for each static group (ABBR, DICT, UNIT, TYPE).
        Returns an empty dict if the dictionary cannot be loaded.
    """
    try:
        dict_file = files("python_ags4") / "Standard_dictionary_v4_1_1.ags"
        with as_file(dict_file) as path:
            tables, _ = AGS4.AGS4_to_dataframe(path)
        logger.info(f"Loaded {len(tables)} static tables from dictionary")
        return tables
    except (FileNotFoundError, OSError) as e:
        logger.warning(f"Could not load standard AGS dictionary: {e}")
        return {}


def lookup_column_unit_and_type(
    table_name: str,
    column_name: str,
    static_tables: dict[str, pd.DataFrame],
) -> tuple[str, str]:
    """
    Look up AGS unit and data type for a column from the standard dictionary.

    :param table_name: The AGS group name (e.g., ``SCPT``, ``LOCA``).
    :param column_name: The column/heading name (e.g., ``SCPT_RES``, ``LOCA_ID``).
    :param static_tables: Static tables loaded from the AGS dictionary.
    :returns: A tuple of ``(unit, data_type)``. Returns ``("", "X")`` if not found.
    """
    default_unit = ""
    default_type = "X"

    dict_table = static_tables.get("DICT")
    if dict_table is None or dict_table.empty:
        return default_unit, default_type

    required_cols = {"DICT_GRP", "DICT_HDNG", "DICT_UNIT", "DICT_DTYP"}
    if not required_cols.issubset(dict_table.columns):
        return default_unit, default_type

    try:
        mask = (dict_table["DICT_GRP"] == table_name) & (dict_table["DICT_HDNG"] == column_name)
        matching_rows = dict_table.loc[mask]

        if not matching_rows.empty:
            unit = str(matching_rows["DICT_UNIT"].iloc[0])
            dtype = str(matching_rows["DICT_DTYP"].iloc[0])
            return unit, dtype
    except (KeyError, IndexError):
        pass

    return default_unit, default_type


def add_ags_prefix_rows(
    df: pd.DataFrame,
    table_name: str,
    static_tables: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """
    Add AGS-required prefix rows (UNIT, TYPE) and HEADING column.

    AGS 4.x files require each data group to have:

    1. A HEADING column identifying row types (UNIT, TYPE, DATA)
    2. UNIT row specifying units for each column
    3. TYPE row specifying data types for each column

    :param df: The data table to format.
    :param table_name: The AGS group name (e.g., ``SCPT``, ``LOCA``).
    :param static_tables: Static tables for unit/type lookups.
        If ``None``, defaults are used.
    :returns: The DataFrame with UNIT/TYPE rows prepended and HEADING column added.
    """
    static_tables = static_tables or {}

    # Build UNIT and TYPE values for each column
    unit_values = []
    type_values = []

    for col in df.columns:
        col_unit, col_type = lookup_column_unit_and_type(table_name, str(col), static_tables)
        unit_values.append(col_unit)
        type_values.append(col_type)

    # Create prefix rows
    prefix_data = [
        dict(zip(df.columns, unit_values)),
        dict(zip(df.columns, type_values)),
    ]
    prefix_df = pd.DataFrame(prefix_data)

    # Concatenate prefix rows with data
    result = pd.concat([prefix_df, df], ignore_index=True)

    # Add HEADING column at the start
    num_data_rows = len(df)
    heading_values = ["UNIT", "TYPE"] + ["DATA"] * num_data_rows
    result.insert(0, "HEADING", heading_values)

    # Apply type formatting to data rows
    result = apply_type_formatting(result)

    return result
