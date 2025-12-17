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

import asyncio

import nest_asyncio

from evo.data_converters.common import (
    EvoObjectMetadata,
    EvoWorkspaceMetadata,
    create_evo_object_service_and_data_client,
)
from evo.objects.client import ObjectAPIClient
from evo.objects.data import ObjectSchema
from evo.objects.utils.data import ObjectDataClient
from evo_schemas import schema_lookup
from evo_schemas.objects import DownholeCollection_V1_3_1

from python_ags4 import AGS4
from datetime import datetime
from importlib.abc import Traversable
from importlib.resources import as_file, files
from typing import TYPE_CHECKING, Optional, Iterable
from uuid import UUID
import pandas as pd
import pyarrow.compute as pc

import evo.logging

if TYPE_CHECKING:
    from evo.notebooks import ServiceManagerWidget


class AgsExporterException(Exception):
    """
    Raised for exporter exceptions
    """

    pass


class UnsupportedObjectException(Exception):
    """
    Raised if the object to export is not supported
    """

    pass


logger = evo.logging.getLogger("data_converters")


def _fetch_downhole_attrs(
    data_client: ObjectDataClient, object_id: UUID, object_version: Optional[str], attrs: Iterable[object]
) -> dict[str, object]:
    fetched_vals = asyncio.run(
        asyncio.gather(*[data_client.download_table(object_id, object_version, a.values.as_dict()) for a in attrs])
    )
    fetched_attrs = dict(zip(map(lambda v: v.name, attrs), fetched_vals))
    return fetched_attrs


def _downhole_to_ags_groups(
    data_client: ObjectDataClient,
    object_id: UUID,
    object_version: Optional[str],
    dhc: DownholeCollection_V1_3_1,
) -> (pd.DataFrame, pd.DataFrame):
    holes = asyncio.run(data_client.download_table(object_id, object_version, dhc.location.hole_id.table.as_dict()))
    # Collect hole collar data for SCPG
    hole_collar_attrs = {
        k: v.column("data").to_pylist()
        for k, v in _fetch_downhole_attrs(data_client, object_id, object_version, dhc.location.attributes).items()
    }

    coords = asyncio.run(data_client.download_table(object_id, object_version, dhc.location.coordinates.as_dict()))

    # Match distance and interval data to relevent measurements and locations
    distance_collections = [c for c in dhc.collections if c.collection_type == "distance"]
    interval_collections = [c for c in dhc.collections if c.collection_type == "interval"]
    measurements = [
        (
            asyncio.run(data_client.download_table(object_id, object_version, m.holes.as_dict())).to_pandas(),
            asyncio.run(data_client.download_table(object_id, object_version, m.distance.values.as_dict())).to_pandas(),
            {
                k: v.to_pandas()
                for k, v in _fetch_downhole_attrs(data_client, object_id, object_version, m.distance.attributes).items()
            },
        )
        for m in distance_collections
    ]
    interval_measurements = [
        (
            asyncio.run(data_client.download_table(object_id, object_version, c.holes.as_dict())).to_pandas(),
            asyncio.run(
                data_client.download_table(object_id, object_version, c.from_to.intervals.start_and_end.as_dict())
            ).to_pandas(),
            {
                k: v.to_pandas()
                for k, v in _fetch_downhole_attrs(data_client, object_id, object_version, c.from_to.attributes).items()
            },
        )
        for c in interval_collections
    ]

    hole_idx = holes.column("key")
    # Strip hole index from hole id
    hole_id = pc.split_pattern(holes.column("value"), ":", max_splits=1, reverse=True)
    hole_id = pc.list_element(hole_id, 0)

    loca = pd.DataFrame(
        {
            "LOCA_ID": hole_id,
            "LOCA_NATE": coords.column("x"),
            "LOCA_NATN": coords.column("y"),
            "LOCA_GL": coords.column("z"),
        },
        index=hole_idx,
    )

    scpg = [
        pd.Series(
            {
                key.split(":")[0] if key == "LOCA_ID" else key: vals[hc]
                for key, vals in hole_collar_attrs.items()
                if key in ["LOCA_ID", "FILE_FSET"] or key.startswith("SCPG")
            }
        )
        for hc in range(len(hole_collar_attrs["LOCA_ID"]))
    ]

    geol = []
    scpp = []
    scpt = []

    # Fill hole data for each test number
    for holes, depth, data in measurements:
        for hole_idx in holes["hole_index"]:
            row = holes["hole_index"] == hole_idx
            offset = holes.loc[row, "offset"].item()
            for test_n in range(holes.loc[row, "count"].item()):
                entry_scpt = {
                    "LOCA_ID": hole_id.loc[row].item(),
                    "SCPG_TESN": test_n,
                    "SCPT_DPTH": depth.at[test_n + offset, "values"],
                }

                for title, col in data.items():
                    if title.startswith("SCPT") and title not in ["SCPT_DPTH"]:
                        entry_scpt[title] = col.at[test_n + offset, "data"]

                scpt.append(pd.Series(entry_scpt))

    # Fill hole data for each test interval
    for holes, from_to, data in interval_measurements:
        for hole_idx in holes["hole_index"]:
            row = holes["hole_index"] == hole_idx
            offset = holes.loc[row, "offset"].item()
            for test_n in range(holes.loc[row, "count"].item()):
                for i, row_data in from_to.iterrows():
                    entry_geol = {
                        "LOCA_ID": hole_id.at[hole_idx],
                        "GEOL_TOP": row_data.at["from"],
                        "GEOL_BASE": row_data.at["to"],
                    }
                    entry_scpp = {
                        "LOCA_ID": hole_id.at[hole_idx],
                        "SCPP_TESTN": test_n,
                        "SCPP_TOP": row_data.at["from"],
                        "SCPP_BASE": row_data.at["to"],
                    }

                    for title, col in data.items():
                        if title.startswith("GEOL") and title not in ["GEOL_TOP", "GEOL_BASE"]:
                            entry_geol[title] = col.at[test_n + offset, "data"]

                        if title.startswith("SCPP") and title not in ["SCPP_TOP", "SCPP_BASE", "SCPP_TESTN"]:
                            entry_geol[title] = col.at[test_n + offset, "data"]

                    geol.append(pd.Series(entry_geol))
                    scpp.append(pd.Series(entry_scpp))

    tran = pd.DataFrame(
        {
            "TRAN_ISNO": ["1"],
            "TRAN_DATE": [datetime.today().strftime("%Y-%m-%d")],
            "TRAN_PROD": ["Evo Data Converters"],
            "TRAN_STAT": ["Final"],
            "TRAN_DESC": ["Export of Downhole Collection from Evo to AGS 4.1 file"],
            "TRAN_AGS": ["4.1"],
            "TRAN_RECV": ["Unknown"],
        }
    )

    proj = pd.DataFrame(
        {
            "HEADING": [
                "UNIT",
                "TYPE",
                "DATA",
            ],
            "PROJ_ID": [
                "",
                "ID",
                dhc.tags.get("AGS:PROJ:PROJ_ID", dhc.uuid),
            ],
            "PROJ_NAME": [
                "",
                "X",
                dhc.tags.get("AGS:PROJ:PROJ_NAME", dhc.name),
            ],
            "PROJ_MEMO": [
                "",
                "X",
                dhc.description
                or f"Exported from Seequent Evo Data Converters - {pd.Timestamp.now().strftime('%Y-%m-%d')}",
            ],
        }
    )

    # Fetch static tables from standard dictionary
    static_tables = {}
    static_headings = {}

    try:
        dict_file: Traversable = files("python_ags4") / "Standard_dictionary_v4_1_1.ags"
        with as_file(dict_file) as path:
            static_tables, static_headings = AGS4.AGS4_to_dataframe(path)
    except (FileNotFoundError, OSError) as e:
        logger.warning(f"Could not load standard AGS dictionary: {e}")

    tables = {
        "PROJ": proj.map(str),
        "LOCA": loca.map(str),
        "TRAN": tran.map(str),
    }
    headings = {
        "PROJ": proj.columns.to_list(),
        "LOCA": loca.columns.to_list(),
        "TRAN": tran.columns.to_list(),
    }

    # These tables are lists, and need to become dataframe
    transpose_tables = [
        ("GEOL", geol),
        ("SCPG", scpg),
        ("SCPP", scpp),
        ("SCPT", scpt),
    ]
    for name, series in transpose_tables:
        if series:
            table = pd.concat(series, axis=1).transpose()
            tables[name] = table.map(str)
            headings[name] = table.columns.to_list()
        # else:
        #     tables[name] = series

    # Add header and prefix rows to some tables
    prefix_tables = [
        ("LOCA", loca),
        ("TRAN", tran),
        ("SCPG", tables["SCPG"]),
        # ("SCPP", tables["SCPP"]),
        ("SCPT", tables["SCPT"]),
    ]
    for name, series in prefix_tables:
        if not series.empty:
            tables[name] = _ags_prefix_dataframe(series, name, static_tables=static_tables)
            headings[name] = ["HEADING"] + headings[name]

    for key in ["ABBR", "DICT", "UNIT", "TYPE"]:
        if key in static_tables and not static_tables[key].empty:
            tables[key] = static_tables[key].map(str)
            headings[key] = static_headings[key]

    return (tables, headings)


def _ags_column_unit_type(
    table_name: str,
    column_name: str,
    static_tables: dict[str, pd.DataFrame],
) -> tuple[str, str]:
    """
    Provide an AGS4 unit and type (table, column) from AGS standard dictionary tables.

    Example: for table 'SCPT' column 'SCPT_RES':
      - Look in static DICT table
      - Find row where DICT_GRP == 'SCPT' and DICT_HDNG == 'SCPT_RES'
      - Return DICT_UNIT and DICT_DTYP values: MPa, 3DP
    """
    default_unit = ""
    default_type = "X"

    dict_table = static_tables.get("DICT")
    if dict_table is not None and not dict_table.empty:
        required_cols = {"DICT_GRP", "DICT_HDNG", "DICT_UNIT", "DICT_DTYP"}
        if required_cols.issubset(dict_table.columns):
            try:
                mask = (dict_table["DICT_GRP"] == table_name) & (dict_table["DICT_HDNG"] == column_name)
                dict_row = dict_table.loc[mask]

                if not dict_row.empty:
                    dict_unit = str(dict_row["DICT_UNIT"].iloc[0])
                    dict_type = str(dict_row["DICT_DTYP"].iloc[0])
                    return dict_unit, dict_type
            except Exception:
                pass  # Fall through to defaults
    return default_unit, default_type


def _ags_prefix_dataframe(
    df: pd.DataFrame,
    table_name: str,
    static_tables: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """
    Apply AGS4 required prefix rows (UNIT, TYPE) then HEADING column to a dataframe.

    UNIT/TYPE values are looked up from `static_tables[table_name]` if provided.
    """
    static_tables = static_tables or {}

    # Build UNIT/TYPE rows per column (looked up from standard dictionary)
    unit_values: list[str] = []
    type_values: list[str] = []
    for col in df.columns:
        col_unit, col_type = _ags_column_unit_type(table_name, str(col), static_tables=static_tables)
        unit_values.append(col_unit or "X")
        type_values.append(col_type)

    unit_row = pd.DataFrame([unit_values], columns=df.columns)
    type_row = pd.DataFrame([type_values], columns=df.columns)

    df = pd.concat(
        [
            unit_row,
            type_row,
            df,
        ],
        ignore_index=True,
    )
    num_rows = df.shape[0]

    # Prepend HEADING column before data
    column_values = ["UNIT", "TYPE"] + ["DATA"] * (num_rows - 2)
    df.insert(0, "HEADING", column_values)

    return df


def _export_obj(
    obj_meta: EvoObjectMetadata,
    service_client: ObjectAPIClient,
    data_client: ObjectDataClient,
) -> (pd.DataFrame, pd.DataFrame):
    evo_dict = asyncio.run(service_client.download_object_by_id(obj_meta.object_id, obj_meta.version_id)).as_dict()
    schema = ObjectSchema.from_id(evo_dict["schema"])
    object_class = schema_lookup.get(str(schema))

    if not object_class:
        raise UnsupportedObjectException(f"Unknown Geoscience Object schema '{schema.sub_classification}'")

    evo_object = object_class.from_dict(evo_dict)

    match schema.sub_classification:
        case "downhole-collection":
            return _downhole_to_ags_groups(data_client, obj_meta.object_id, obj_meta.version_id, evo_object)
        case _:
            raise UnsupportedObjectException(f"Cannot export {object_class} to AGS")


def export_ags(
    filepath: str,
    objects: list[EvoObjectMetadata],
    evo_workspace_metadata: Optional[EvoWorkspaceMetadata] = None,
    service_manager_widget: Optional["ServiceManagerWidget"] = None,
):
    service_client, data_client = create_evo_object_service_and_data_client(
        evo_workspace_metadata, service_manager_widget
    )

    nest_asyncio.apply()

    tables, headings = _export_obj(objects[0], service_client, data_client)

    AGS4.dataframe_to_AGS4(tables, headings, filepath)
