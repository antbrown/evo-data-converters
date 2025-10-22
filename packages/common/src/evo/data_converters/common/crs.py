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

from pyproj import CRS
from pyproj.exceptions import CRSError

from evo_schemas.components import Crs_V1_0_1 as Crs
from evo_schemas.components import Crs_V1_0_1_EpsgCode as Crs_EpsgCode
from evo_schemas.components import Crs_V1_0_1_OgcWkt as Crs_OgcWkt


def crs_from_epsg_code(epsg_code: int | str) -> Crs_EpsgCode:
    """
    Parse and validate an EPSG code. If valid, return the Crs geoscience
    object with the integer EPSG code.
    """
    try:
        # Accept formats 2193, '2193' and 'EPSG:2193'; Normalise to authority format
        code_str = str(epsg_code).strip().upper()
        if not code_str.startswith("EPSG:"):
            code_str = f"EPSG:{code_str}"

        crs = CRS.from_user_input(code_str)

        # Extract authority and code to validate
        authority = crs.to_authority()
        if not authority or authority[0] != "EPSG":
            raise ValueError(f"Input '{epsg_code}' is not a valid EPSG CRS")

        epsg_code = int(authority[1])
        return Crs_EpsgCode(epsg_code=epsg_code)

    except (CRSError, ValueError, TypeError) as e:
        raise ValueError(f"Invalid or unrecognized EPSG code '{epsg_code}': {e}")


def crs_from_ogc_wkt(wkt_string: str) -> Crs_OgcWkt:
    """
    Parse an OGC WKT string. If valid, return the Crs geoscience
    object with normalized WKT2 string.
    Notes: We explicitly reject ESRI WKT in this implementation. Converting
    instead to OGC would however be easy if that is required in future.
    """
    try:
        # Check if it looks like an ESRI WKT instead of OGC
        if "ESRI" in wkt_string.upper():
            raise ValueError("ESRI WKT detected, expecting OGC WKT")

        crs = CRS.from_wkt(wkt_string)

        # Return Crs with canonical WKT2 format
        return Crs_OgcWkt(ogc_wkt=crs.to_wkt(version="WKT2_2019"))

    except (CRSError, ValueError, TypeError) as e:
        raise ValueError(f"Invalid or unrecognized WKT string: {e}")


def crs_unspecified() -> Crs:
    """
    When the Crs is not specified, the goescience Crs object is
    returned as a simple string constant
    """
    return "unspecified"
