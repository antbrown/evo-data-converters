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
from evo_schemas.objects import (
    LineSegments_V2_0_0,
    LineSegments_V2_1_0,
    Pointset_V1_1_0,
    Pointset_V1_2_0,
    TriangleMesh_V2_0_0,
    TriangleMesh_V2_1_0,
)

import evo.logging
from evo.data_converters.common import (
    EvoObjectMetadata,
    EvoWorkspaceMetadata,
    create_evo_object_service_and_data_client,
)
from evo.objects.client import ObjectAPIClient
from evo.objects.data import ObjectSchema
from evo.objects.utils.data import ObjectDataClient
from pandas import Dataframe

if TYPE_CHECKING:
    from evo.notebooks import ServiceManagerWidget


def _map_to_ags_groups():
    pass

def _export_element(
    object_metadata: EvoObjectMetadata,
    service_client: ObjectAPIClient,
    data_client: ObjectDataClient,
) -> tuple[Dataframe, ObjectSchema]:
    pass

def export_ags(
    filepath: str,
    objects: list[EvoObjectMetadata],
    ags_metadata: Optional[AGSMetadata] = None,
    evo_workspace_metadata: Optional[EvoWorkspaceMetadata] = None,
    service_manager_widget: Optional["ServiceManagerWidget"] = None,
) -> None:

    service_client, data_client = create_evo_object_service_and_data_client(
        evo_workspace_metadata, service_manager_widget
    )

    nest_asyncio.apply()

    ags_metadata = dataclasses.replace(ags_metadata) if ags_metadata else AGSMetadata()
    
