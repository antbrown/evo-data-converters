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
from typing import TYPE_CHECKING, Optional
from uuid import UUID
import pandas as pd

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


def _downhole_to_ags_groups(
    data_client: ObjectDataClient,
    object_id: UUID,
    object_version: Optional[str],
    dhc: DownholeCollection_V1_3_1,
) -> (pd.DataFrame, pd.DataFrame):
    holes = asyncio.run(data_client.download_table(object_id, object_version, dhc.location.hole_id.table.as_dict()))
    coords = asyncio.run(data_client.download_table(object_id, object_version, dhc.location.coordinates.as_dict()))
    distance_collections = [c for c in dhc.collections if c.collection_type == "distance"]
    interval_collections = [c for c in dhc.collections if c.collection_type == "interval"]
    measurements = [
        (
            asyncio.run(data_client.download_table(object_id, object_version, m.holes.as_dict())).to_pandas(),
            asyncio.run(data_client.download_table(object_id, object_version, m.distance.values.as_dict())).to_pandas(),
            {
                attr.name: asyncio.run(
                    data_client.download_table(object_id, object_version, attr.values.as_dict())
                ).to_pandas()
                for attr in m.distance.attributes
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
                attr.name: asyncio.run(
                    data_client.download_table(object_id, object_version, attr.values.as_dict())
                ).to_pandas()
                for attr in c.from_to.attributes
            },
        )
        for c in interval_collections
    ]

    hole_idx = holes.column("key")
    hole_id = holes.column("value")

    loca = pd.DataFrame(
        {
            "LOCA_ID": hole_id,
            "LOCA_NATE": coords.column("x"),
            "LOCA_NATN": coords.column("y"),
            "LOCA_GL": coords.column("z"),
        },
        index=hole_idx,
    )

    # Needs to be applied per table somehow
    # unit = pd.DataFrame(
    #     {
    #         "UNIT_UNIT": [dhc.distance_unit or "m"],
    #         "UNIT_DESC": [dhc.distance_unit or "Metre"],
    #     }
    # )

    hole_id = hole_id.to_pandas()
    geol = []
    scpg = []
    scpp = []
    scpt = []

    for holes, depth, data in measurements:
        for hole_idx in holes["hole_index"]:
            row = holes["hole_index"] == hole_idx
            offset = holes.loc[row, "offset"].item()
            for test_n in range(holes.loc[row, "count"].item()):
                entry_scpg = {"LOCA_ID": hole_id.loc[row].item(), "SCPG_TESN": test_n}
                entry_scpt = {
                    "LOCA_ID": hole_id.loc[row].item(),
                    "SCPG_TESN": test_n,
                    "SCPT_DPTH": depth.at[test_n + offset, "values"],
                }

                for title, col in data.items():
                    if title.startswith("SCPG") and title not in ["SCPG_TESN"]:
                        entry_scpg[title] = col.at[test_n + offset, "data"]
                    elif title.startswith("SCPT") and title not in ["SCPT_DPTH"]:
                        entry_scpt[title] = col.at[test_n + offset, "data"]

                scpg.append(pd.Series(entry_scpg))
                scpt.append(pd.Series(entry_scpt))

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
                dhc.tags["AGS:PROJ:PROJ_ID"] or dhc.uuid,
            ],
            "PROJ_NAME": [
                "",
                "X",
                dhc.tags["AGS:PROJ:PROJ_NAME"] or dhc.name,
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

    # Add header and prefix rows to these tables.
    prefix_tables = [
        ("LOCA", loca),
        # ("TRAN", tran),
        # ("SCPG", scpg),
        # ("SCPP", scpp),
        # ("SCPT", scpt),
    ]
    for name, series in prefix_tables:
        tables[name] = _ags_prefix_dataframe(series, name)

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

    for key in ["ABBR", "UNIT", "TYPE"]:
        if key in static_tables and not static_tables[key].empty:
            tables[key] = static_tables[key].map(str)
            headings[key] = static_headings[key]

    return (tables, headings)


def _ags_prefix_dataframe(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Apply AGS4 required prefix rows (unit, type) then columns (heading) to a dataframe

    @TODO Work out how to obtain appropriate UNIT and TYPE values for each row.
    """
    unit_row = pd.DataFrame([[""] * df.shape[1]], columns=df.columns)
    type_row = pd.DataFrame([["X"] * df.shape[1]], columns=df.columns)
    df = pd.concat(
        [
            unit_row,
            type_row,
            df,
        ],
        ignore_index=True,
    )
    num_rows = df.shape[0]
    column_values = ["UNIT", "TYPE"] + ["DATA"] * (num_rows - 2)
    df.insert(0, "HEADING", column_values)

    # If I pprint.pp(df) here, the construction seems right.
    # But AGS4 exports without those
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
