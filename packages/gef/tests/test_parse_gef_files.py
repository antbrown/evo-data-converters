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

import pytest
from pathlib import Path
from pygef.cpt import CPTData
from evo.data_converters.gef.importer.parse_gef_files import parse_gef_files

import os


class TestParseGefFiles:
    """Test the parse_gef_files function behaves as intended."""

    test_data_dir = Path(os.path.dirname(__file__)) / "data"

    def test_parse_valid_cpt_file(self) -> None:
        cpt_file = self.test_data_dir / "cpt.gef"
        result = parse_gef_files([cpt_file])
        assert isinstance(result, dict)
        assert len(result) == 1
        hole_id = cpt_file.stem
        assert hole_id in result
        assert isinstance(result[hole_id], CPTData)

    def test_parse_multiple_valid_cpt_files(self) -> None:
        files = [self.test_data_dir / "cpt.gef", self.test_data_dir / "cpt2.gef"]
        result = parse_gef_files(files)
        assert len(result) == 2
        for f in files:
            assert f.stem in result
            assert isinstance(result[f.stem], CPTData)

    def test_file_not_found(self) -> None:
        missing_file = self.test_data_dir / "does_not_exist.gef"
        with pytest.raises(RuntimeError) as exc:
            parse_gef_files([missing_file])
        assert "File not found" in str(exc.value)

    def test_not_cpt_type(self) -> None:
        bore_file = self.test_data_dir / "bore.gef"
        with pytest.raises(RuntimeError) as exc:
            parse_gef_files([bore_file])
        assert "is not a CPT GEF file" in str(exc.value)
