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
from typing import TYPE_CHECKING, Optional, Any
from uuid import UUID

import nest_asyncio
from evo_schemas import schema_lookup
from evo_schemas.components import BaseSpatialDataProperties_V1_0_1

import evo.logging
from evo.data_converters.common import (
    EvoObjectMetadata,
    EvoWorkspaceMetadata,
    create_evo_object_service_and_data_client,
)
from evo.objects.client import ObjectAPIClient
from evo.objects.data import ObjectSchema
from evo.objects.utils.data import ObjectDataClient
import ifcopenshell

from ..ifc_metadata import IFCMetadata
from .spatial_data_to_ifc import export_ifc_spatial_data


logger = evo.logging.getLogger("data_converters")

if TYPE_CHECKING:
    from evo.notebooks import ServiceManagerWidget


class IFCExporterException(Exception):
    pass


class UnsupportedObjectError(IFCExporterException):
    pass


def _download_evo_object_by_id(
    service_client: ObjectAPIClient,
    object_id: UUID,
    version_id: Optional[str] = None,
) -> dict[str, Any]:
    downloaded_object = asyncio.run(service_client.download_object_by_id(object_id, version_id))
    result: dict[str, Any] = downloaded_object.as_dict()
    return result


def _export_element(
    object_metadata: EvoObjectMetadata,
    service_client: ObjectAPIClient,
    data_client: ObjectDataClient,
    ifc_file: ifcopenshell.file,
) -> ObjectSchema:
    object_id = object_metadata.object_id
    version_id = object_metadata.version_id

    # Download object
    geoscience_object_dict = _download_evo_object_by_id(service_client, object_id, version_id)

    # Check if this is a known geoscience object schema type
    schema = ObjectSchema.from_id(geoscience_object_dict["schema"])
    object_class = schema_lookup.get(str(schema))
    if not object_class:
        raise UnsupportedObjectError(f"Unknown Geoscience Object schema '{schema}'")

    geoscience_object = object_class.from_dict(geoscience_object_dict)

    # Convert to IFC entities and add to IFC file
    if issubclass(geoscience_object.__class__, BaseSpatialDataProperties_V1_0_1):
        export_ifc_spatial_data(object_id, version_id, geoscience_object, data_client, ifc_file)
    else:
        raise UnsupportedObjectError(
            f"Exporting {geoscience_object.__class__.__name__} Geoscience Objects to IFC is not supported"
        )

    return schema


def export_ifc(
    filepath: str,
    objects: list[EvoObjectMetadata],
    ifc_metadata: Optional[IFCMetadata] = None,
    evo_workspace_metadata: Optional[EvoWorkspaceMetadata] = None,
    service_manager_widget: Optional["ServiceManagerWidget"] = None,
) -> None:
    """Export an Evo Geoscience Object to an IFC file.

    :param filepath: Path of the IFC file to create.
    :param objects: List of EvoObjectMetadata objects containing the UUID and version of the Evo objects to export.
    :param ifc_metadata: Optional project metadata to embed in the IFC file.
    :param evo_workspace_metadata: Optional Evo Workspace metadata.
    :param service_manager_widget: Optional ServiceManagerWidget for use in notebooks.

    One of evo_workspace_metadata or service_manager_widget is required.

    :raise UnsupportedObjectError: If the type of object is not supported.
    :raise MissingConnectionDetailsError: If no connections details could be derived.
    :raise ConflictingConnectionDetailsError: If both evo_workspace_metadata and service_manager_widget present.
    """

    service_client, data_client = create_evo_object_service_and_data_client(
        evo_workspace_metadata, service_manager_widget
    )

    nest_asyncio.apply()

    ifc_metadata = dataclasses.replace(ifc_metadata) if ifc_metadata else IFCMetadata()
    ifc_file, ifc_project = ifc_metadata.to_file()

    ifc_project.Name = ifc_project.Name or "EvoObjects"
    if len(objects) == 1:
        object_metadata = objects[0]
        schema = _export_element(object_metadata, service_client, data_client, ifc_file)
        ifc_project.Description = (
            ifc_project.Description
            or f"{schema.sub_classification.capitalize()} object with ID {object_metadata.object_id}"
        )
    else:
        for object_metadata in objects:
            _ = _export_element(object_metadata, service_client, data_client, ifc_file)
        ifc_project.Description = ifc_project.Description or "Objects with IDs " + ", ".join(
            str(object_metadata.object_id) for object_metadata in objects
        )

    # Write file
    ifc_file.write(filepath)
