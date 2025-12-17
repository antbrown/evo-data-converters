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

import os
import tempfile
from os import path
from unittest import TestCase
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pandas as pd
import pyarrow as pa
from python_ags4 import AGS4
from evo_schemas.objects import DownholeCollection_V1_3_1

from evo.data_converters.ags.exporter import export_ags
from evo.data_converters.ags.importer import convert_ags
from evo.data_converters.common import (
    EvoObjectMetadata,
    EvoWorkspaceMetadata,
    create_evo_object_service_and_data_client,
)


class MockTable:
    """Mocks the table object returned by ObjectDataClient.download_table"""

    def __init__(self, df: pd.DataFrame):
        self.table = pa.Table.from_pandas(df)

    def to_pandas(self) -> pd.DataFrame:
        return self.table.to_pandas()

    def column(self, name: str):
        return self.table[name]


class TestEvoToAGSExporter(TestCase):
    def setUp(self) -> None:
        self.cache_root_dir = tempfile.TemporaryDirectory()
        # Note: No hub_url needed for local conversion, which prevents importer from trying to publish
        self.workspace_metadata = EvoWorkspaceMetadata(workspace_id=str(uuid4()), cache_root=self.cache_root_dir.name)

        # Create clients just to populate cache via importer (we won't use these for export test directly)
        _, self.data_client = create_evo_object_service_and_data_client(self.workspace_metadata)

        # Convert an AGS file to Evo to populate the cache with tables for the exporter to read
        filepaths = [path.join(path.dirname(__file__), "../importer/data/valid_ags.ags")]
        self.evo_object = convert_ags(
            filepaths=filepaths,
            evo_workspace_metadata=self.workspace_metadata,
        )
        self.assertIsInstance(self.evo_object, DownholeCollection_V1_3_1)

    def tearDown(self) -> None:
        self.cache_root_dir.cleanup()

    def find_parquet_in_cache(self, data_id: str) -> str | None:
        """Helper to find a parquet file by its data ID (filename) in the cache directory."""
        for root, _, files in os.walk(self.cache_root_dir.name):
            if data_id in files:
                return path.join(root, data_id)
        return None

    @patch("evo.data_converters.ags.exporter.evo_to_ags.create_evo_object_service_and_data_client")
    def test_should_create_expected_ags_file(self, mock_create_clients: MagicMock) -> None:
        from pprint import pp

        temp_ags_file = tempfile.NamedTemporaryFile(suffix=".ags", delete=False)
        temp_ags_file.close()

        object_id = uuid4()
        version_id = "any version"
        obj_metadata = EvoObjectMetadata(object_id=object_id, version_id=version_id)

        mock_service_client = MagicMock()

        async def async_return_object(oid, vid):
            return self.evo_object

        mock_service_client.download_object_by_id.side_effect = async_return_object

        mock_data_client = MagicMock()

        async def async_download_table(oid, vid, table_info: dict):
            # The exporter passes the table info dict, which contains the 'data' key (the file ID/Hash)
            data_id = table_info["data"]

            # Find the actual parquet file created by the importer in the cache
            parquet_path = self.find_parquet_in_cache(data_id)
            if not parquet_path:
                raise FileNotFoundError(f"Could not find cached parquet for data ID: {data_id}")

            # Read it into a DataFrame
            df = pd.read_parquet(parquet_path)
            return MockTable(df)

        mock_data_client.download_table.side_effect = async_download_table
        mock_create_clients.return_value = (mock_service_client, mock_data_client)

        export_ags(
            temp_ags_file.name,
            objects=[obj_metadata],
            evo_workspace_metadata=self.workspace_metadata,
        )

        try:
            with open(temp_ags_file.name, "r") as file:
                file_content = file.read()
                print(file_content)
        except FileNotFoundError:
            print(f"Error: The file '{temp_ags_file.name}' was not found.")
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")

        # Verify exported file contents
        tables, headings = AGS4.AGS4_to_dataframe(temp_ags_file.name)

        pp(temp_ags_file.name)
        t, h = AGS4.AGS4_to_dict(temp_ags_file.name)

        pp(t)
        pp(h)

        pp(tables)
        pp(headings)

        self.assertIn("PROJ", tables)
        self.assertEqual(tables["PROJ"]["PROJ_MEMO"].iloc[0], "Example description")
        # # self.assertEqual(tables["PROJ"]["PROJ_NAME"].iloc[0], self.evo_object.name)
        # # self.assertIn("LOCA", tables)
        # # self.assertFalse(tables["LOCA"].empty)
        # self.assertIn("LOCA_ID", headings["LOCA"])
        # self.assertIn("LOCA_NATE", headings["LOCA"])
        # self.assertIn("LOCA_NATN", headings["LOCA"])
        # self.assertIn("LOCA_GL", headings["LOCA"])
        # self.assertIn("SCPG", tables)
        # self.assertIn("SCPT", tables)
        # self.assertFalse(tables["SCPT"].empty)
        # self.assertIn("SCPT_RES", headings["SCPT"])

        # Cleanup
        pp(temp_ags_file.name)
        # os.remove(temp_ags_file.name)
