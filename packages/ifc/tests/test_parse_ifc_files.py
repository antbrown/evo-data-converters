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
import pytest
from evo.data_converters.ifc.importer import parse_ifc_files


class TestParseIfcFiles:
    """Test the parse_ifc_files function behaves as intended."""

    test_data_dir = Path(__file__).parent / "data"

    def test_parse_valid_ifc_file(self):
        """Parsing a known sample IFC file returns an ifcopenshell file and has expected entities."""
        sample = self.test_data_dir / "Ifc4_WallElementedCase.ifc"
        model = parse_ifc_files([sample])[0]
        # model should be an ifcopenshell.file with some products
        assert hasattr(model, "by_type")
        walls = model.by_type("IfcWall")
        products = model.by_type("IfcProduct")
        assert isinstance(walls, list)
        assert isinstance(products, list)
        # Expect at least one product in a valid sample file
        assert len(products) > 0

    def test_missing_file_raises(self):
        """Passing a non-existent path raises FileNotFoundError."""
        missing = Path("/tmp/this_file_should_not_exist.ifc")
        with pytest.raises(FileNotFoundError):
            parse_ifc_files([missing])
