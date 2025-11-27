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
from typing import TYPE_CHECKING, Optional
from uuid import UUID

import nest_asyncio
from python_ags4 import AGS4

import evo_schemas

from evo_schemas.objects import DownholeCollection_V1_3_1

from ..common.ags_context import AgsFileInvalidException
from evo.data_converters.common import (
    EvoObjectMetadata,
    EvoWorkspaceMetadata,
    create_evo_object_service_and_data_client,
)
from evo.objects.client import ObjectAPIClient
from evo.objects.data import ObjectSchema
from evo.objects.utils.data import ObjectDataClient
import pandas as pd

if TYPE_CHECKING:
    from evo.notebooks import ServiceManagerWidget


def _downhole_to_ags_groups(
    data_client: ObjectDataClient, object_id: UUID, object_version: Optional[str], dhc: DownholeCollection_V1_3_1
) -> (pd.DataFrame, pd.DataFrame):
    holes = asyncio.run(data_client.download_table(object_id, object_version, dhc.location.hole_id.table.as_dict()))
    coords = asyncio.run(data_client.download_table(object_id, object_version, dhc.location.coordinates.as_dict()))
    distance_collections = [c for c in dhc.collections if c.collection_type == "distance"]
    measurments = [
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

    hole_id = hole_id.to_pandas()
    scpg = []
    scpt = []

    for holes, depth, data in measurments:
        for hole_idx in range(hole_id.size):
            for test_n in range(holes.at[hole_idx, "count"]):
                entry_scpg = {"LOCA_ID": hole_id.at[hole_idx], "SCPG_TESN": test_n}
                entry_scpt = {
                    "LOCA_ID": hole_id.at[hole_idx],
                    "SCPG_TESN": test_n,
                    "SCPT_DPTH": depth.at[test_n, "values"],
                }

                for title, col in data.items():
                    if title.startswith("SCPG") and title not in ["SCPG_TESN"]:
                        entry_scpg[title] = col.at[test_n, "data"]
                    elif title.startswith("SCPT") and title not in ["SCPT_DPTH"]:
                        entry_scpt[title] = col.at[test_n, "data"]

                scpg.append(pd.Series(entry_scpg))
                scpt.append(pd.Series(entry_scpt))

    scpg = pd.concat(scpg, axis=1).transpose()
    scpt = pd.concat(scpt, axis=1).transpose()
    tables = {"LOCA": loca.map(str), "SCPT": scpt.map(str), "SCPG": scpg.map(str)}
    headings = {"LOCA": loca.columns.to_list(), "SCPT": scpt.columns.to_list(), "SCPG": scpg.columns.to_list()}

    return (tables, headings)


def _export_obj(
    obj_meta: EvoObjectMetadata,
    service_client: ObjectAPIClient,
    data_client: ObjectDataClient,
) -> (pd.DataFrame, pd.DataFrame):
    evo_object = asyncio.run(service_client.download_object_by_id(obj_meta.object_id, obj_meta.version_id)).as_dict()
    schema = str(ObjectSchema.from_id(evo_object["schema"]))
    object_class = evo_schemas.schema_lookup.get(schema)

    if not object_class:
        raise AgsFileInvalidException(f"Unknown Geoscience Object schema '{schema}'")

    evo_object = object_class.from_dict(evo_object)

    match schema:
        case "/objects/downhole-collection/1.3.1/downhole-collection.schema.json":
            return _downhole_to_ags_groups(data_client, obj_meta.object_id, obj_meta.version_id, evo_object)
        case _:
            raise AgsFileInvalidException(f"Cannot export {object_class} to AGS")


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

    tables, heading = _export_obj(objects[0], service_client, data_client)

    AGS4.dataframe_to_AGS4(tables, heading, filepath)
