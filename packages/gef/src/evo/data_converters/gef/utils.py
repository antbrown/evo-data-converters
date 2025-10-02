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

from evo.data_converters.gef.importer.parse_gef_files import parse_gef_file


def is_gef_cpt(filepath: str) -> bool:
    """
    Returns `True` if the file appears to be a valid GEF-CPT file

    Args:
        filepath (str): Path to the file to check.

    Returns:
        bool: `True` if the file appears to be a valid GEF-CPT file, `False` otherwise.
    """
    try:
        if parse_gef_file(filepath):
            return True
        else:
            return False
    except ValueError:
        return False
    except FileNotFoundError:
        return False
