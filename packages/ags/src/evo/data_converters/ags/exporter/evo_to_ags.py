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
    holes = asyncio.run(data_client.download_table(object_id, object_version, dhc.location.hole_id.as_dict()))
    coords = asyncio.run(data_client.download_table(object_id, object_version, dhc.location.coordinates.as_dict()))
    measurments = asyncio.run(
        data_client.download_table(object_id, object_version, dhc.collections.as_dict())
    ).to_pandas()

    # hole_idx = holes.column("key")[0]
    hole_id = holes.column("value")[0]

    loca = pd.DataFrame(
        {
            "LOCA_ID": hole_id,
            "LOCA_NATE": coords.column("x")[0],
            "LOCA_NATN": coords.column("y")[0],
            "LOCA_GL": coords.column("z")[0],
        }
    )

    scpt = {"LOCA_ID": hole_id}
    for col in measurments.columns:
        if col.startswith("SCPT"):
            scpt[col] = measurments.get(col)

    scpg = {"LOCA_ID": hole_id}
    for col in measurments.columns:
        if col.startswith("SCPG"):
            scpt[col] = measurments.get(col)

    tables = {"LOCA": loca}
    headings = {"LOCA": loca.columns.to_list()}
    for n, t in {"SCPT": scpt, "SCPG": scpg}:
        if len(t) > 1:
            tables[n] = t
            headings[n] = t.columns.to_list()

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

    objs = [_export_obj(obj, service_client, data_client) for obj in objects]

    AGS4.dataframe_to_AGS4(objs[0][0], objs[0][1], filepath)
