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

import ifcopenshell
from pathlib import Path


def parse_ifc_files(filepaths: list[str | Path]) -> list[ifcopenshell.file]:
    """
    Parse multiple IFC files and return the parsed IFC models as a list of ifcopenshell.file objects.

    Args:
        filepaths (list[str | Path]): List of paths to the IFC files.
    Returns:
        list[ifcopenshell.file]: List of parsed IFC model objects.
    Raises:
        FileNotFoundError: If a file does not exist.
        RuntimeError: If a file cannot be parsed.
    """
    models = []
    for filepath in filepaths:
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"IFC file not found: {filepath}")
        try:
            model = ifcopenshell.open(str(path))
        except Exception as e:
            raise RuntimeError(f"Failed to open IFC file: {e}")
        models.append(model)
    return models
