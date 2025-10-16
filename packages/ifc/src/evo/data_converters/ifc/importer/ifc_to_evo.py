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

from .parse_ifc_files import parse_ifc_files
from .ifc_to_spatial_data import convert_spatial_data

if TYPE_CHECKING:
    from evo.notebooks import ServiceManagerWidget


def convert_ifc(
    filepaths: list[str],
    evo_workspace_metadata: Optional[EvoWorkspaceMetadata] = None,
    service_manager_widget: Optional["ServiceManagerWidget"] = None,
    tags: Optional[dict[str, str]] = None,
    upload_path: str = "",
) -> list[ObjectMetadata]:
    """Converts an IFC file into Geoscience Objects.

    :param filepath: Path to the IFC file.
    :param evo_workspace_metadata: (Optional) Evo workspace metadata.
    :param service_manager_widget: (Optional) Service Manager Widget for use in jupyter notebooks.
    :param tags: (Optional) Dict of tags to add to the Geoscience Object.
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

    # Read the IFC files and get the parsed IFC models
    models = parse_ifc_files(filepaths)

    # Loop through the elements found in the parsed models - converting any IfcProduct into
    # Evo geoscience objects
    for model in models:
        for product in model.by_type("IfcProduct"):
            # Call the spatial converter
            converted_geoscience_objects = convert_spatial_data(product)
            for geoscience_object in converted_geoscience_objects:
                if geoscience_object.tags is None:
                    geoscience_object.tags = {}
                geoscience_object.tags["Source"] = "IFC files (via Evo Data Converters)"
                geoscience_object.tags["Stage"] = "Experimental"
                geoscience_object.tags["InputType"] = "IFC"

            if converted_geoscience_objects:
                geoscience_objects.extend(converted_geoscience_objects)

    # Publish the found geoscience objects to Evo
    objects_metadata = publish_geoscience_objects(
        geoscience_objects,
        object_service_client,
        data_client,
        upload_path,
    )

    # Return the publishing response
    return objects_metadata
