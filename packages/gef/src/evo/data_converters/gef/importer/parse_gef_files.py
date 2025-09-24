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

from pygef import read_cpt
from pygef.cpt import CPTData
from pygef.gef.gef import _Gef


def parse_gef_files(filepaths: list[str | Path]) -> dict[str, CPTData]:
    """
    Parse a list of GEF files and return a dictionary of CPTData objects keyed by filename.

    Only files identified as CPT (Cone Penetration Test) are read and included.

    Args:
        filepaths (list[str | Path]): List of file paths to GEF files to parse.

    Returns:
        dict[str, CPTData]: Dictionary mapping each CPT file's filename to its CPTData object.
    """
    data: dict[str, CPTData] = {}

    for filepath in filepaths:
        try:
            if not Path(filepath).exists():
                raise FileNotFoundError(f"File not found: {filepath}")
            gef = _Gef(filepath)
            if gef.type != "cpt":
                raise ValueError(f"File '{filepath}' is not a CPT GEF file (type: {gef.type})")

            cpt_data = read_cpt(filepath)

            # Required columns taken from https://bedrock.engineer/reference/formats/gef/gef-cpt/#column-quantities
            required_columns = ["penetrationLength", "coneResistance"]
            if hasattr(cpt_data, "data"):
                missing = [col for col in required_columns if col not in cpt_data.data.columns]
                if missing:
                    raise ValueError(f"File '{filepath}' is missing required columns: {missing}")

            filename = Path(filepath).stem
            data[filename] = cpt_data
        except Exception as e:
            raise RuntimeError(f"Error processing file '{filepath}': {e}") from e

    return data
