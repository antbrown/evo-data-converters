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
from pathlib import Path
from evo.data_converters.ifc import IFCMetadata
from evo.data_converters.ifc.importer import parse_ifc_files


class TestIfcMetadata:
    """Test IFCMetadata class collects the correct metadata."""

    test_data_dir = Path(__file__).parent / "data"

    def test_valid_ifc_metadata(self):
        """IFCMetadata generates an IFC file with correct version
        and an IfcProject with correct metadata
        """
        sample = self.test_data_dir / "Ifc4_WallElementedCase.ifc"
        model = parse_ifc_files([sample])[0]
        project = model.by_type("IfcProject")[0]
        metadata = IFCMetadata(
            name=project.Name,
            description=project.Description,
            global_id=project.GlobalId,
            version=model.schema,
        )
        new_model, new_project = metadata.to_file()
        assert new_model.schema == model.schema
        for attr in ["Name", "Description", "GlobalId"]:
            assert getattr(new_project, attr) == getattr(project, attr)
