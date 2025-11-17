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
import dataclasses
from typing import TYPE_CHECKING, Any, Optional
from uuid import UUID

import nest_asyncio
import ags
from ags import AGSMetadata
from python_ags4 import AGS4

from evo_schemas import schema_lookup

import evo.logging
from evo.data_converters.common.objects.downhole_collection import DownholeCollection, ColumnMapping, HoleCollars
from evo.data_converters.common.objects.downhole_collection.tables import DistanceTable
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


def _downhole_to_ags_groups(dhc: DownholeCollection) -> dict[DataFrame]:
    collars_df = dhc.collars.df
    
    for measurement in dhc.get_measurement_tables(filter=[DistanceTable]):
        pass


def _export_obj(
    obj_meta: EvoObjectMetadata,
    service_client: ObjectAPIClient,
    data_client: ObjectDataClient,
) -> Dataframe:
    evo_object = asyncio.run(service_client.download_object_by_id(obj_meta.object_id, obj_meta.version_id)).as_dict()
    object_class = schema_lookup.get(str(ObjectSchema.from_id(evo_object["schema"])))

    if not object_class:
        raise UnsupportedObjectError(f"Unknown Geoscience Object schema '{schema}'")

    evo_object = object_class.from_dict(evo_object)

    match object_class:
        case DownholeCollection():
            return _downhole_to_ags_groups(evo_object)
        case _:
            raise AgsFileInvalidException(f"Cannot export {obj_type} to AGS")

def export_ags(
    filepath: str,
    objects: list[EvoObjectMetadata],
    evo_workspace_metadata: Optional[EvoWorkspaceMetadata] = None,
    service_manager_widget: Optional["ServiceManagerWidget"] = None,
) -> [Dataframe]:

    service_client, data_client = create_evo_object_service_and_data_client(
        evo_workspace_metadata, service_manager_widget
    )

    nest_asyncio.apply()

    objs = [_export_obj(obj, service_client, data_client) for obj in objects]

    return dataframe_to_AGS4(objs, {}, filepath)
