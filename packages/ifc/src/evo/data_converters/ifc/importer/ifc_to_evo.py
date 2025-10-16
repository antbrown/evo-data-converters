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

from typing import TYPE_CHECKING, Optional

from evo.data_converters.common import (
    EvoWorkspaceMetadata,
    create_evo_object_service_and_data_client,
    publish_geoscience_objects,
)
from evo.objects.data import ObjectMetadata

from . import parse_ifc_files

if TYPE_CHECKING:
    from evo.notebooks import ServiceManagerWidget


def convert_ifc(
    filepath: str,
    epsg_code: int,
    evo_workspace_metadata: Optional[EvoWorkspaceMetadata] = None,
    service_manager_widget: Optional["ServiceManagerWidget"] = None,
    upload_path: str = "",
) -> list[ObjectMetadata]:
    """Converts an IFC file into Geoscience Objects.

    :param filepath: Path to the IFC file.
    :param epsg_code: The EPSG code to use when creating a Coordinate Reference System object.
    :param evo_workspace_metadata: (Optional) Evo workspace metadata.
    :param service_manager_widget: (Optional) Service Manager Widget for use in jupyter notebooks.
    :param upload_path: (Optional) Path objects will be published under.

    One of evo_workspace_metadata or service_manager_widget is required.

    Converted objects will be published if either of the following is true:
    - evo_workspace_metadata.hub_url is present, or
    - service_manager_widget was passed to this function.

    :return: List of Geoscience Objects and Block Models, or list of ObjectMetadata and Block Models if published.

    :raise MissingConnectionDetailsError: If no connections details could be derived.
    :raise ConflictingConnectionDetailsError: If both evo_workspace_metadata and service_manager_widget present.
    """
    geoscience_objects = []

    # create a service and data clients to handle upload to the Seequent Evo API
    object_service_client, data_client = create_evo_object_service_and_data_client(
        evo_workspace_metadata=evo_workspace_metadata,
        service_manager_widget=service_manager_widget,
    )

    # Read the IFC file and get the parsed IFC model
    model = parse_ifc_files([filepath])[0]

    # Loop through the elements found in the parsed model - converting any point set elements into to
    # Evo geoscience objects
    for element in model:
        geoscience_object = None
        if element.is_a() == "IfcPoint":
            # Call the specific pointset converter here
            # geoscience_object = _convert_pointset(element, reader, data_client, epsg_code)
            pass

        if geoscience_object is not None:
            geoscience_objects.append(geoscience_object)

    # Publish the found geoscience objects to Evo
    objects_metadata = publish_geoscience_objects(
        geoscience_objects,
        object_service_client,
        data_client,
        upload_path,
    )

    # Return the publishing response
    return objects_metadata
