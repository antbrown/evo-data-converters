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

"""Unit tests for AGS exporter export module and exceptions."""

from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock, patch

from evo.data_converters.ags.exporter.exceptions import (
    AgsExporterException,
    UnsupportedObjectException,
)
from evo.data_converters.ags.exporter.export import _export_single_object, export_ags


class TestExceptions(TestCase):
    """Tests for exception classes."""

    def test_ags_exporter_exception_is_exception(self) -> None:
        exc = AgsExporterException("test error")
        self.assertIsInstance(exc, Exception)
        self.assertEqual(str(exc), "test error")

    def test_unsupported_object_exception_inherits_from_base(self) -> None:
        exc = UnsupportedObjectException("unsupported type")
        self.assertIsInstance(exc, AgsExporterException)
        self.assertIsInstance(exc, Exception)
        self.assertEqual(str(exc), "unsupported type")


class TestExportSingleObject(IsolatedAsyncioTestCase):
    """Tests for _export_single_object function."""

    async def test_raises_for_non_downhole_collection(self) -> None:
        """Test that non-downhole-collection objects raise UnsupportedObjectException."""
        obj_meta = MagicMock()
        obj_meta.object_id = "test-id"
        obj_meta.version_id = "test-version"

        # Mock downloaded object with wrong schema type
        downloaded_obj = MagicMock()
        downloaded_obj.as_dict.return_value = {"schema": "/objects/new-object/1.0.0/new-object.schema.json"}

        service_client = AsyncMock()
        service_client.download_object_by_id.return_value = downloaded_obj

        with self.assertRaises(UnsupportedObjectException) as context:
            await _export_single_object(obj_meta, service_client)

        self.assertIn("Only downhole-collection is supported", str(context.exception))

    async def test_processes_downhole_collection(self) -> None:
        """Test that downhole-collection objects are processed correctly."""
        obj_meta = MagicMock()
        obj_meta.object_id = "test-id"
        obj_meta.version_id = "test-version"

        downloaded_obj = MagicMock()
        downloaded_obj.as_dict.return_value = {
            "schema": "/objects/downhole-collection/1.3.1/downhole-collection.schema.json"
        }

        service_client = AsyncMock()
        service_client.download_object_by_id.return_value = downloaded_obj

        mock_dhc = MagicMock()
        mock_tables = {"PROJ": MagicMock()}
        mock_headings = {"PROJ": ["PROJ_ID"]}

        with (
            patch(
                "evo.data_converters.ags.exporter.export.create_downhole_collection_from_evo",
                new_callable=AsyncMock,
                return_value=mock_dhc,
            ) as mock_create_dhc,
            patch(
                "evo.data_converters.ags.exporter.export.build_ags_tables",
                return_value=(mock_tables, mock_headings),
            ) as mock_build_tables,
        ):
            tables, headings = await _export_single_object(obj_meta, service_client)

            mock_create_dhc.assert_called_once_with(downloaded_obj)
            mock_build_tables.assert_called_once_with(mock_dhc)
            self.assertEqual(tables, mock_tables)
            self.assertEqual(headings, mock_headings)


class TestExportAgs(IsolatedAsyncioTestCase):
    """Tests for export_ags function."""

    async def test_exports_to_file(self) -> None:
        """Test that export_ags writes to the specified file."""
        obj_meta = MagicMock()
        obj_meta.object_id = "test-id"
        obj_meta.version_id = "test-version"

        mock_service_client = AsyncMock()
        mock_tables = {"PROJ": MagicMock()}
        mock_headings = {"PROJ": ["PROJ_ID"]}

        with (
            patch(
                "evo.data_converters.ags.exporter.export.create_evo_object_service_and_data_client",
                return_value=(mock_service_client, None),
            ),
            patch(
                "evo.data_converters.ags.exporter.export._export_single_object",
                new_callable=AsyncMock,
                return_value=(mock_tables, mock_headings),
            ),
            patch("evo.data_converters.ags.exporter.export.AGS4.dataframe_to_AGS4") as mock_ags4,
        ):
            await export_ags(
                filepath="output.ags",
                objects=[obj_meta],
                evo_workspace_metadata=MagicMock(),
            )

            mock_ags4.assert_called_once_with(mock_tables, mock_headings, "output.ags")

    async def test_only_processes_first_object(self) -> None:
        """Test that only the first object is processed (current limitation)."""
        obj_meta_1 = MagicMock()
        obj_meta_1.object_id = "test-id-1"
        obj_meta_2 = MagicMock()
        obj_meta_2.object_id = "test-id-2"

        mock_service_client = AsyncMock()
        mock_tables = {"PROJ": MagicMock()}
        mock_headings = {"PROJ": ["PROJ_ID"]}

        with (
            patch(
                "evo.data_converters.ags.exporter.export.create_evo_object_service_and_data_client",
                return_value=(mock_service_client, None),
            ),
            patch(
                "evo.data_converters.ags.exporter.export._export_single_object",
                new_callable=AsyncMock,
                return_value=(mock_tables, mock_headings),
            ) as mock_export_single,
            patch("evo.data_converters.ags.exporter.export.AGS4.dataframe_to_AGS4"),
        ):
            await export_ags(
                filepath="output.ags",
                objects=[obj_meta_1, obj_meta_2],
                evo_workspace_metadata=MagicMock(),
            )

            # Should only be called once with first object
            self.assertEqual(mock_export_single.call_count, 1)
            call_args = mock_export_single.call_args
            self.assertEqual(call_args[0][0], obj_meta_1)

    async def test_accepts_service_manager_widget(self) -> None:
        """Test that service_manager_widget is passed to client creation."""
        obj_meta = MagicMock()
        mock_widget = MagicMock()
        mock_service_client = AsyncMock()

        with (
            patch(
                "evo.data_converters.ags.exporter.export.create_evo_object_service_and_data_client",
                return_value=(mock_service_client, None),
            ) as mock_create_client,
            patch(
                "evo.data_converters.ags.exporter.export._export_single_object",
                new_callable=AsyncMock,
                return_value=({}, {}),
            ),
            patch("evo.data_converters.ags.exporter.export.AGS4.dataframe_to_AGS4"),
        ):
            await export_ags(
                filepath="output.ags",
                objects=[obj_meta],
                service_manager_widget=mock_widget,
            )

            mock_create_client.assert_called_once_with(None, mock_widget)
