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

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

from evo_schemas.components import BaseSpatialDataProperties_V1_0_1
from evo.data_converters.common import (
    EvoWorkspaceMetadata,
)
from evo.data_converters.ifc.importer import convert_ifc, parse_ifc_files
from evo.objects.data import ObjectMetadata


class TestConvertIfc:
    """Test the convert_ifc function behaves as intended."""

    test_data_dir = Path(__file__).parent / "data"

    @pytest.fixture
    def sample_filepaths(self):
        """Sample file paths for testing."""
        return [self.test_data_dir / "Ifc4_WallElementedCase.ifc"]

    @pytest.fixture
    def workspace_metadata(self):
        """Mock workspace metadata with hub_url."""
        hub_url = "https://example.org"
        cache_root = tempfile.TemporaryDirectory()
        return EvoWorkspaceMetadata(workspace_id=str(uuid4()), cache_root=cache_root.name, hub_url=hub_url)

    @pytest.fixture
    def mock_base_spatial_data(self):
        """Mock spatial data properties object."""
        properties = Mock(spec=BaseSpatialDataProperties_V1_0_1)
        properties.tags = {}
        return properties

    @pytest.fixture
    def mock_object_metadata(self):
        """Mock object metadata."""
        return Mock(spec=ObjectMetadata)

    @patch("evo.data_converters.ifc.importer.ifc_to_evo.publish_geoscience_objects")
    @patch("evo.data_converters.ifc.importer.ifc_to_evo.create_evo_object_service_and_data_client")
    @patch("evo.data_converters.ifc.importer.ifc_to_evo.convert_spatial_data")
    def test_convert_ifc_with_workspace_metadata_and_hub_url(
        self,
        mock_convert_spatial_data,
        mock_create_clients,
        mock_publish,
        sample_filepaths,
        workspace_metadata,
        mock_base_spatial_data,
        mock_object_metadata,
    ):
        """Test conversion with workspace metadata and hub_url - should publish."""
        mock_create_clients.return_value = (Mock(), Mock())
        mock_convert_spatial_data.return_value = [mock_base_spatial_data]
        mock_publish.return_value = mock_object_metadata

        result = convert_ifc(filepaths=sample_filepaths, evo_workspace_metadata=workspace_metadata)
        model = parse_ifc_files(sample_filepaths)[0]

        assert result == mock_object_metadata
        assert isinstance(result, ObjectMetadata)
        mock_publish.assert_called_once()
        assert mock_convert_spatial_data.call_count == len(model.by_type("IfcProduct"))

        # Check tags were added
        expected_tags = {
            "Source": "IFC files (via Evo Data Converters)",
            "Stage": "Experimental",
            "InputType": "IFC",
        }
        for key, value in expected_tags.items():
            assert mock_base_spatial_data.tags[key] == value
