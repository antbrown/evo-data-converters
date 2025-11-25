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
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

import ifcopenshell
from evo_schemas.components import BaseSpatialDataProperties_V1_0_1

from evo.data_converters.common import (
    EvoObjectMetadata,
    EvoWorkspaceMetadata,
)
from evo.data_converters.ifc import IFCMetadata
from evo.data_converters.ifc.exporter import UnsupportedObjectError, export_ifc


class TestEvoToIFCExporter:
    @pytest.fixture
    def ifc_metadata(self):
        return IFCMetadata(
            name="Test IFC", description="Test description", global_id="RandomGlobalID", version="IFC4X3"
        )

    @pytest.fixture
    def workspace_metadata(self):
        """Mock workspace metadata with hub_url."""
        hub_url = "https://example.org"
        cache_root = tempfile.TemporaryDirectory()
        return EvoWorkspaceMetadata(workspace_id=str(uuid4()), cache_root=cache_root.name, hub_url=hub_url)

    @pytest.fixture
    def mock_base_spatial_data(self):
        """Mock spatial data properties objects."""
        objects = []
        mock_subclasses = ["Regular-3d-grid", "Tensor-2d-grid"]
        mock_dicts = [
            dict(
                schema="/objects/regular-3d-grid/1.1.0/regular-3d-grid.schema.json",
                origin=[0.0, 0.0, 0.0],
                size=[10, 10, 10],
                cell_size=[1.0, 1.0, 1.0],
            ),
            dict(
                schema="/objects/tensor-2d-grid/1.2.0/tensor-2d-grid.schema.json",
                bounding_box={"min_y": 0, "max_y": 10, "min_x": 0, "max_x": 10, "min_z": 0, "max_z": 10.0},
                origin=[0.0, 0.0, 0.0],
                size=[10, 10, 10],
                grid_cells_2d={"cell_sizes_x": [1.0, 2.0], "cell_sizes_y": [1.0, 2.0]},
            ),
        ]
        for i, mock_object in enumerate(zip(mock_subclasses, mock_dicts)):
            mock_subclass, mock_dict = mock_object
            properties = Mock(spec=BaseSpatialDataProperties_V1_0_1)
            base_dict = dict(
                name=f"Spatial data {i}",
                uuid=uuid4(),
                bounding_box={"min_y": 0, "max_y": 10, "min_x": 0, "max_x": 10, "min_z": 0, "max_z": 10.0},
                coordinate_reference_system={"epsg_code": 1024},
            )
            properties.as_dict.return_value = {**base_dict, **mock_dict}
            objects.append(dict(name=mock_subclass, mock=properties))
        return objects

    @patch("evo.data_converters.ifc.exporter.evo_to_ifc._download_evo_object_by_id")
    @pytest.mark.parametrize("multiple_evo_objects", [False, True])
    def test_should_create_expected_ifc_file_with_metadata(
        self,
        mock_download_evo_object_by_id,
        ifc_metadata,
        workspace_metadata,
        mock_base_spatial_data,
        multiple_evo_objects,
    ):
        temp_ifc_file = tempfile.NamedTemporaryFile(suffix=".ifc", delete=False)

        if multiple_evo_objects:
            objs = [
                EvoObjectMetadata(object_id=uuid4(), version_id="any version")
                for _ in range(len(mock_base_spatial_data))
            ]
        else:
            objs = [EvoObjectMetadata(object_id=uuid4(), version_id="any version")]

        base_spatial_data_generator = iter(mock_base_spatial_data)
        mock_download_evo_object_by_id.return_value = next(base_spatial_data_generator)["mock"].as_dict()

        export_ifc(
            temp_ifc_file.name,
            objects=objs,
            ifc_metadata=ifc_metadata,
            evo_workspace_metadata=workspace_metadata,
        )

        model = ifcopenshell.open(temp_ifc_file.name)
        project = model.by_type("IfcProject")[0]

        assert project.Name == ifc_metadata.name
        assert project.GlobalId == ifc_metadata.global_id
        assert model.schema == ifc_metadata.version
        assert project.Description == ifc_metadata.description

    @patch("evo.data_converters.ifc.exporter.evo_to_ifc._download_evo_object_by_id")
    @pytest.mark.parametrize("multiple_evo_objects", [False, True])
    def test_should_create_expected_ifc_file_without_metadata(
        self,
        mock_download_evo_object_by_id,
        workspace_metadata,
        mock_base_spatial_data,
        multiple_evo_objects,
    ):
        temp_ifc_file = tempfile.NamedTemporaryFile(suffix=".ifc", delete=False)

        if multiple_evo_objects:
            objs = [
                EvoObjectMetadata(object_id=uuid4(), version_id="any version")
                for _ in range(len(mock_base_spatial_data))
            ]
        else:
            objs = [EvoObjectMetadata(object_id=uuid4(), version_id="any version")]

        base_spatial_data_generator = iter(mock_base_spatial_data)
        mock_download_evo_object_by_id.return_value = next(base_spatial_data_generator)["mock"].as_dict()

        export_ifc(
            temp_ifc_file.name,
            objects=objs,
            evo_workspace_metadata=workspace_metadata,
        )

        model = ifcopenshell.open(temp_ifc_file.name)
        project = model.by_type("IfcProject")[0]
        expected_description = f"{mock_base_spatial_data[0]['name']} object with ID {objs[0].object_id}"
        if multiple_evo_objects:
            expected_description = "Objects with IDs " + ", ".join(str(obj.object_id) for obj in objs)

        assert project.Name == "EvoObjects"
        assert model.schema == "IFC4"
        assert project.Description == expected_description

    @patch("evo.data_converters.ifc.exporter.evo_to_ifc._download_evo_object_by_id")
    def test_should_raise_expected_exception_for_unknown_object_schema(
        self,
        mock_download_evo_object_by_id,
        workspace_metadata,
    ):
        temp_ifc_file = tempfile.NamedTemporaryFile(suffix=".ifc", delete=False)

        evo_object_dict = dict(schema="/objects/unknown/1.0.0/unknown.schema.json")
        mock_download_evo_object_by_id.return_value = evo_object_dict

        with pytest.raises(
            UnsupportedObjectError, match=f"Unknown Geoscience Object schema '{evo_object_dict['schema']}'"
        ):
            export_ifc(
                temp_ifc_file.name,
                objects=[EvoObjectMetadata(object_id=uuid4())],
                evo_workspace_metadata=workspace_metadata,
            )
