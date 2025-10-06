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

from dataclasses import dataclass
import pandas as pd
import typing


@dataclass
class DownholeCollection:
    """
    Data structure for a Downhole Collection.

    Separates collar information (stored once per hole) from measurement data (stored once per measurement).
    """

    # Data
    collars: pd.DataFrame
    """
    Hole collar information (one row per hole)

    Required columns:
        - hole_index: int - Unique identifier for each row, 1-based
        - hole_id: str - Unique identifier for each survey
        - x: float - Easting coordinate
        - y: float - Northing coordinate
        - z: float - Elevation at collar
        - final_depth: float - ?
    """

    measurements: pd.DataFrame
    """
    Downhole survey data (one row per measurement point across all holes)

    Columns:
        1: hole_index: int - Links to collars.hole_index
    For a distance table:
        2: penetrationLength: float - depth of measurement
    For an interval table:
        2: from: float - ?
        3: to: float - ?
    Measurements, e.g.
        n: coneResistance: float - cone resistance at measurement depth
    """

    # Metadata
    name: str
    epsg_code: int

    uuid: str | None = None
    tags: dict[str, str] | None = None
    description: str | None = None

    distance_unit: str | None = None
    desurvey_method: str | None = None

    extensions: dict[str, typing.Any] | None = None

    def is_distance(self) -> bool:
        """Check if measurements use a distance-based format."""
        if self.measurements.empty or len(self.measurements.columns) < 2:
            return False
        second_col = str(self.measurements.columns[1])
        return second_col.lower() in ("penetrationlength", "depth")

    def is_interval(self) -> bool:
        """Check if measurements use an interval-based format."""
        if self.measurements.empty or len(self.measurements.columns) < 3:
            return False
        second_col = str(self.measurements.columns[1])
        third_col = str(self.measurements.columns[2])
        return second_col.lower() == "from" and third_col.lower() == "to"

    def measurement_type(self) -> str:
        """Get the type of measurement used"""
        if self.is_distance():
            return "distance"
        elif self.is_interval():
            return "interval"
        return "unknown"

    def bounding_box(self) -> dict[str, float]:
        # Validate we have x, y, z columns and they contain floats and not NaN's
        return {
            "min_x": float(self.collars["x"].min()),
            "max_x": float(self.collars["x"].max()),
            "min_y": float(self.collars["y"].min()),
            "max_y": float(self.collars["y"].max()),
            "min_z": float(self.collars["z"].min()),
            "max_z": float(self.collars["z"].max()),
        }
