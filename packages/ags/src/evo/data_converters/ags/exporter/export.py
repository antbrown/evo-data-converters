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

"""
AGS export.

This module provides the main entry point for exporting Evo objects to AGS.
It coordinates the fetching of data, building tables, formatting and finally
writing to a file.

    Example::

        await export_ags(
            filepath="output.ags",
            objects=[obj_metadata],
            evo_workspace_metadata=workspace_meta,
        )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from python_ags4 import AGS4

import evo.logging
from evo.data_converters.common import (
    EvoObjectMetadata,
    EvoWorkspaceMetadata,
    create_evo_object_service_and_data_client,
)
from evo.data_converters.common.objects.downhole_collection_from_evo import create_downhole_collection_from_evo
from evo.objects.data import ObjectSchema

from .exceptions import UnsupportedObjectException
from .tables import build_ags_tables

if TYPE_CHECKING:
    from evo.notebooks import ServiceManagerWidget
    from evo.objects.client import ObjectAPIClient

    from .types import AgsTablesResult


logger = evo.logging.getLogger("data_converters")


async def _export_single_object(
    obj_meta: EvoObjectMetadata,
    service_client: ObjectAPIClient,
) -> AgsTablesResult:
    """
    Download a single Evo object and return tables, headings that can be used in an AGS export.
    """
    downloaded_obj = await service_client.download_object_by_id(obj_meta.object_id, obj_meta.version_id)

    schema = ObjectSchema.from_id(downloaded_obj.as_dict()["schema"])
    if schema.sub_classification != "downhole-collection":
        raise UnsupportedObjectException(
            f"Cannot export {schema.sub_classification} to AGS. Only downhole-collection is supported."
        )

    logger.info(f"Object type: {schema.sub_classification}")

    # Convert to intermediary DownholeCollection object
    downhole_collection = await create_downhole_collection_from_evo(downloaded_obj)

    # Build and return tables, headings
    return build_ags_tables(downhole_collection)


async def export_ags(
    filepath: str,
    objects: list[EvoObjectMetadata],
    evo_workspace_metadata: EvoWorkspaceMetadata | None = None,
    service_manager_widget: ServiceManagerWidget | None = None,
) -> None:
    """
    Export Evo objects to an AGS 4.1 file.
    """
    logger.info(f"=== Starting AGS export to {filepath} ===")
    logger.info(f"Exporting {len(objects)} object(s)")

    service_client, _ = create_evo_object_service_and_data_client(evo_workspace_metadata, service_manager_widget)

    # TODO: Support multiple objects in a single AGS file
    tables, headings = await _export_single_object(objects[0], service_client)

    AGS4.dataframe_to_AGS4(tables, headings, filepath)

    logger.info("=== AGS export completed successfully ===")
