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

from collections import Counter
from datetime import date
from typing import TYPE_CHECKING

import pandas as pd

import evo.logging

from .formatting import add_ags_prefix_rows, load_static_tables

if TYPE_CHECKING:
    from evo.data_converters.common.objects.downhole_collection import (
        DistanceTable,
        DownholeCollection,
        IntervalTable,
        MeasurementTableAdapter,
    )

    from .types import AgsTablesResult


logger = evo.logging.getLogger("data_converters")

# Tables that require a test number (TESN) field
_TABLES_WITH_TESN = frozenset({"SCPP", "SCPT"})

# Columns that are foreign keys, not table-specific attributes
_FOREIGN_KEY_COLUMNS = frozenset({"LOCA_ID", "FILE_FSET"})

# Static reference tables to include from the AGS dictionary
_STATIC_TABLE_KEYS = ("ABBR", "DICT", "UNIT", "TYPE")

# Null values that have come through as literals, to be replaced with an empty string
_NULL_VALUE_STRINGS = ["nan", "NaT"]


def _derive_headings(tables: dict[str, pd.DataFrame]) -> dict[str, list[str]]:
    """
    Derive the headings dictionary from table columns.

    :param tables: AGS tables keyed by group name.
    :returns: Column names for each table, as required by
        ``AGS4.dataframe_to_AGS4()``.
    """
    return {name: df.columns.tolist() for name, df in tables.items()}


def _split_hole_id_and_tesn(hole_id: str) -> tuple[str, str]:
    """
    Split a hole_id into the base LOCA_ID and optional TESN (test number).

    The LOCA_ID stored in Evo from AGS import may contain a TESN suffix,
    which we split out for storage as a separate field.

    :param hole_id: The full hole identifier (e.g., ``"HKZ1-CPT04.000:1"``).
    :returns: A tuple of ``(base_loca_id, tesn)`` where tesn is an empty
        string if no valid suffix is present.
    """
    if ":" not in hole_id:
        return hole_id, ""

    base, suffix = hole_id.rsplit(":", 1)
    if suffix.isdigit():
        return base, suffix
    return hole_id, ""


def _build_hole_id_lookup(collars: pd.DataFrame) -> pd.DataFrame:
    """
    Build a lookup table mapping hole_index to LOCA_ID and TESN.

    :param collars: Collar DataFrame with hole_index and hole_id columns.
    :returns: DataFrame indexed by hole_index with loca_id and tesn columns.
    """
    split_data = [_split_hole_id_and_tesn(hid) for hid in collars["hole_id"]]
    return pd.DataFrame(
        {"loca_id": [s[0] for s in split_data], "tesn": [s[1] for s in split_data]},
        index=collars["hole_index"],
    )


def _dataframe_to_ags_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a DataFrame to string values suitable for AGS output.

    The AGS format expects empty fields rather than NaN markers.

    :param df: The DataFrame to convert.
    :returns: DataFrame with all string values, NaNs replaced with ``""``.
    """
    return df.astype(str).replace(_NULL_VALUE_STRINGS, "")


def _build_proj_table(dhc: DownholeCollection) -> pd.DataFrame:
    """Build the PROJ (project) table."""
    tags = dhc.tags or {}
    default_memo = f"Exported from Seequent Evo Data Converters - {date.today()}"

    return pd.DataFrame(
        {
            "PROJ_ID": [tags.get("AGS:PROJ:PROJ_ID", dhc.uuid or "")],
            "PROJ_NAME": [tags.get("AGS:PROJ:PROJ_NAME", dhc.name)],
            "PROJ_MEMO": [dhc.description or default_memo],
        }
    )


def _build_tran_table() -> pd.DataFrame:
    """Build the TRAN (data transfer) table."""
    return pd.DataFrame(
        {
            "TRAN_ISNO": ["1"],
            "TRAN_DATE": [str(date.today())],
            "TRAN_PROD": ["Evo Data Converters"],
            "TRAN_STAT": ["Final"],
            "TRAN_DESC": ["Export of Downhole Collection from Evo to AGS 4.1 file"],
            "TRAN_AGS": ["4.1"],
            "TRAN_RECV": ["Unknown"],
        }
    )


def _build_loca_table(dhc: DownholeCollection) -> pd.DataFrame:
    """Build the LOCA (location) table from collar information."""
    collars = dhc.collars.df
    lookup = _build_hole_id_lookup(collars)

    result = pd.DataFrame(
        {
            "LOCA_ID": lookup["loca_id"].values,
            "LOCA_NATE": collars["x"].values,
            "LOCA_NATN": collars["y"].values,
            "LOCA_GL": collars["z"].values,
        }
    )

    # Add any additional LOCA_ attributes from collars
    attr_columns = dhc.collars.get_attribute_column_names()
    loca_columns = [col for col in attr_columns if col.startswith("LOCA_") and col not in result.columns]

    for col in loca_columns:
        result[col] = collars[col].values

    return result


def _build_scpg_table(dhc: DownholeCollection) -> pd.DataFrame | None:
    """
    Build SCPG (CPT general) table from collar attributes.

    Extracts collar-level attributes that belong to the SCPG table
    (columns starting with ``SCPG_``).
    """
    collars = dhc.collars.df
    attr_columns = dhc.collars.get_attribute_column_names()
    scpg_columns = [col for col in attr_columns if col.startswith("SCPG_")]

    if not scpg_columns:
        return None

    lookup = _build_hole_id_lookup(collars)

    result = pd.DataFrame({"LOCA_ID": lookup["loca_id"].values})

    # Add SCPG_TESN if any test numbers exist
    if lookup["tesn"].any():
        result["SCPG_TESN"] = lookup["tesn"].values

    for col in scpg_columns:
        result[col] = collars[col].values

    return result


def _build_distance_table(
    dhc: DownholeCollection,
    table: DistanceTable,
    table_name: str,
) -> pd.DataFrame | None:
    """
    Build a distance-based AGS table.

    :param dhc: The downhole collection (for hole_id lookup).
    :param table: Distance measurement table.
    :param table_name: The AGS table name (e.g., ``SCPT``).
    :returns: DataFrame, or ``None`` if no data exists.
    """
    df = table.df
    if df.empty:
        return None

    lookup = _build_hole_id_lookup(dhc.collars.df)
    hole_index_col = table.get_hole_index_column()
    depth_col = table.get_depth_column()

    # Filter to relevant attribute columns
    attr_columns = table.get_attribute_columns()

    # TODO: The import process seems to be importing the HEADING column, once that is fixed the following can be removed:
    if "HEADING" in attr_columns:
        attr_columns.remove("HEADING")

    # Join lookup data to measurements
    result = df[[hole_index_col, depth_col, *attr_columns]].copy()
    result = result.merge(lookup, left_on=hole_index_col, right_index=True, how="left")

    # Check for unmatched holes
    unmatched = result["loca_id"].isna()
    if unmatched.any():
        unmatched_indices = result.loc[unmatched, hole_index_col].unique()
        logger.warning(f"No hole_id found for hole_index values: {unmatched_indices.tolist()}")
        result = result[~unmatched]

    if result.empty:
        return None

    # Build output columns
    output = pd.DataFrame({"LOCA_ID": result["loca_id"]})

    if table_name in _TABLES_WITH_TESN:
        output[f"{table_name}_TESN"] = result["tesn"]

    output[f"{table_name}_DPTH"] = result[depth_col]

    for col in attr_columns:
        output[col] = result[col]

    logger.info(f"Built {table_name}: {len(output)} rows")
    return output.reset_index(drop=True)


def _build_interval_table(
    dhc: DownholeCollection,
    table: IntervalTable,
    table_name: str,
) -> pd.DataFrame | None:
    """
    Build an interval-based AGS table.

    :param dhc: The downhole collection (for hole_id lookup).
    :param table: Interval measurement table.
    :param table_name: The AGS table name (e.g., ``GEOL``, ``SCPP``).
    :returns: DataFrame, or ``None`` if no data exists.
    """
    df = table.df
    if df.empty:
        return None

    lookup = _build_hole_id_lookup(dhc.collars.df)
    hole_index_col = table.get_hole_index_column()
    from_col = table.get_from_column()
    to_col = table.get_to_column()

    # Filter to relevant attribute columns
    attr_columns = [
        col
        for col in table.get_attribute_columns()
        if col not in _FOREIGN_KEY_COLUMNS and col.startswith(f"{table_name}_")
    ]

    # Join lookup data to measurements
    result = df[[hole_index_col, from_col, to_col, *attr_columns]].copy()
    result = result.merge(lookup, left_on=hole_index_col, right_index=True, how="left")

    # Check for unmatched holes
    unmatched = result["loca_id"].isna()
    if unmatched.any():
        unmatched_indices = result.loc[unmatched, hole_index_col].unique()
        logger.warning(f"No hole_id found for hole_index values: {unmatched_indices.tolist()}")
        result = result[~unmatched]

    if result.empty:
        return None

    # Build output columns
    output = pd.DataFrame({"LOCA_ID": result["loca_id"]})

    if table_name in _TABLES_WITH_TESN:
        output[f"{table_name}_TESN"] = result["tesn"]

    output[f"{table_name}_TOP"] = result[from_col]
    output[f"{table_name}_BASE"] = result[to_col]

    for col in attr_columns:
        output[col] = result[col]

    logger.info(f"Built {table_name}: {len(output)} rows")
    return output.reset_index(drop=True)


def _detect_ags_table_name(table: MeasurementTableAdapter) -> str | None:
    """
    Detect the AGS table name from attribute column prefixes.

    AGS columns follow the pattern TABLE_FIELD (e.g., SCPT_RES, GEOL_DESC).
    We extract the most common prefix to determine the table name.

    :param table: Measurement table to inspect.
    :returns: The detected AGS table name, or ``None`` if not detected.
    """
    prefixes = [
        col.split("_", 1)[0] for col in table.get_attribute_columns() if "_" in col and col not in _FOREIGN_KEY_COLUMNS
    ]

    if not prefixes:
        return None

    # Return the most common prefix
    prefix_counts = Counter(prefixes)
    return prefix_counts.most_common(1)[0][0]


def build_ags_tables(dhc: DownholeCollection) -> AgsTablesResult:
    """
    Build all AGS tables from a downhole collection.

    :param dhc: The downhole collection to export.
    :returns: A tuple of ``(tables, headings)`` where:

        - ``tables``: Dict of AGS group name to DataFrame (string values)
        - ``headings``: Dict of AGS group name to column name list
    """
    from evo.data_converters.common.objects.downhole_collection import DistanceTable, IntervalTable

    logger.info(f"Building AGS tables for '{dhc.name}'")

    static_tables = load_static_tables()

    # Build core tables
    tables_to_format: list[tuple[str, pd.DataFrame | None]] = [
        ("PROJ", _build_proj_table(dhc)),
        ("TRAN", _build_tran_table()),
        ("LOCA", _build_loca_table(dhc)),
        ("SCPG", _build_scpg_table(dhc)),
    ]

    # Build measurement tables
    for measurement in dhc.measurements:
        table_name = _detect_ags_table_name(measurement)
        if table_name is None:
            logger.warning(
                f"Could not detect AGS table type for measurement columns: {measurement.get_attribute_columns()}"
            )
            continue

        if isinstance(measurement, DistanceTable):
            table = _build_distance_table(dhc, measurement, table_name)
        elif isinstance(measurement, IntervalTable):
            table = _build_interval_table(dhc, measurement, table_name)
        else:
            logger.warning(f"Unknown measurement type: {type(measurement).__name__}")
            continue

        if table is not None:
            tables_to_format.append((table_name, table))

    # Format tables with prefix rows
    tables: dict[str, pd.DataFrame] = {}
    for name, df in tables_to_format:
        if df is not None and not df.empty:
            formatted = add_ags_prefix_rows(_dataframe_to_ags_strings(df), name, static_tables)
            tables[name] = formatted
            logger.info(f"  {name}: {len(df)} data rows, {len(df.columns)} columns")

    # Add static reference tables
    for key in _STATIC_TABLE_KEYS:
        if key in static_tables and not static_tables[key].empty:
            tables[key] = _dataframe_to_ags_strings(static_tables[key])

    logger.info(f"Successfully built {len(tables)} AGS tables")

    return tables, _derive_headings(tables)
