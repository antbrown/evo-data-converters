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

import evo.logging

from pygef.broxml.parse_cpt import read_cpt as read_cpt_xml
from pygef import read_cpt
from pygef.cpt import CPTData

logger = evo.logging.getLogger("data_converters")


def parse_gef_files(filepaths: list[str | Path]) -> dict[str, CPTData]:
    """
    Parse a list of GEF & CPT XML files and return a dictionary of CPTData objects keyed by filename.

    Only files identified as CPT (Cone Penetration Test) are read and included.

    Args:
        filepaths (list[str | Path]): List of file paths to parse.

    Returns:
        dict[str, CPTData]: Dictionary mapping each CPT file's filename to its CPTData object.
    """
    data: dict[str, CPTData] = {}

    for filepath in filepaths:
        try:
            if not Path(filepath).exists():
                raise FileNotFoundError(f"File not found: {filepath}")
            ext = Path(filepath).suffix.lower()
            hole_id_cpt_pairs = []
            if ext == ".xml":
                # No method in pygef to detect type, so just try to read as CPT.
                # XML files can contain multiple CPT entries.
                try:
                    multiple_cpt_data = read_cpt_xml(filepath)
                except Exception:
                    raise ValueError(f"File '{filepath}' is not a CPT XML file or could not be parsed as CPT data.")

                for cpt_data in multiple_cpt_data:
                    hole_id = getattr(cpt_data, "bro_id", None)
                    if not hole_id:
                        hole_id = Path(filepath).stem
                    check_for_required_columns(cpt_data, filepath)
                    hole_id_cpt_pairs.append((hole_id, cpt_data))
            else:
                cpt_data = read_cpt(filepath)
                hole_id = getattr(cpt_data, "alias", None)
                if not hole_id:
                    hole_id = Path(filepath).stem
                check_for_required_columns(cpt_data, filepath)
                hole_id_cpt_pairs.append((hole_id, cpt_data))

            # Add CPT data to output dict, ensuring unique hole_id keys
            for hole_id, cpt_data in hole_id_cpt_pairs:
                if hole_id in data:
                    raise ValueError(
                        f"Duplicate hole_id '{hole_id}' encountered. Each hole_id (from test_id, bro_id, or filename) must be unique across all input files."
                    )
                data[hole_id] = cpt_data
        except Exception as e:
            raise RuntimeError(f"Error processing file '{filepath}': {e}") from e

    logger.info(f"Parsed {len(data)} CPT files from {len(filepaths)} input files.")
    return data


def check_for_required_columns(cpt_data: CPTData, filepath: str) -> None:
    """Check that the CPTData object has the required columns.

    Required columns taken from https://bedrock.engineer/reference/formats/gef/gef-cpt/#column-quantities

    Args:
        cpt_data (CPTData): The CPTData object to check.
        filepath (str): The file path of the GEF file being processed.

    Raises:
        ValueError: If any required columns are missing.
    """
    required_columns = ["penetrationLength", "coneResistance"]
    if hasattr(cpt_data, "data"):
        missing = [col for col in required_columns if col not in cpt_data.data.columns]
        if missing:
            raise ValueError(f"File '{filepath}' is missing required columns: {missing}")
