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

from uuid import UUID
from typing import Optional

from evo_schemas.components import BaseSpatialDataProperties_V1_0_1
import ifcopenshell

from evo.objects.utils.data import ObjectDataClient


def export_ifc_spatial_data(
    object_id: UUID,
    version_id: Optional[str],
    spatial_go: BaseSpatialDataProperties_V1_0_1,
    data_client: ObjectDataClient,
    ifc_file: ifcopenshell.file,
):
    """Convert EVO spatial data into IFC entities then add them to the IFC file"""
    pass
