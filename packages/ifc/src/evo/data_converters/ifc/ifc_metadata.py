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

from dataclasses import dataclass
from typing import Optional

import ifcopenshell
from ifcopenshell.api.project import create_file
from ifcopenshell.api.root import create_entity


@dataclass
class IFCMetadata:
    name: Optional[str] = ""
    global_id: Optional[str] = ""
    description: Optional[str] = ""
    version: ifcopenshell.util.schema.IFC_SCHEMA = "IFC4"

    def to_file(self) -> tuple[ifcopenshell.file, ifcopenshell.entity_instance]:
        """Create an IFC file with an IfcProject entity using the metadata.
        The file will then be used to aggregate products to construct a complete file.
        """
        model = create_file(version=self.version)
        project = create_entity(model, ifc_class="IfcProject", name=self.name)
        project.Description = self.description
        if self.global_id:
            project.GlobalId = self.global_id

        return model, project
