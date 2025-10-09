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

from dataclasses import dataclass, field
import pandas as pd
import typing


@dataclass
class DownholeCollection:
    """
    Data structure for a Downhole Collection.

    Separates collar information (stored once per hole) from measurement data (stored once per measurement).
    """

    # Hole collar information (one row per hole)
    collars: pd.DataFrame

    # Downhole survey data (one row per measurement point across all holes)
    measurements: pd.DataFrame

    # Metadata
    name: str
    epsg_code: int

    uuid: str | None = None
    tags: dict[str, str] | None = None
    description: str | None = None

    distance_unit: str | None = None
    desurvey_method: str | None = None

    extensions: dict[str, typing.Any] | None = None

    nan_values_by_attribute: dict = field(default_factory=dict)
    """
    A dict keyed on 'attribute_name' with items which are a list of unique
    column void values that represent a NaN value which can be present
    for that attribute, across all of the holes in the dataset
    """

    def is_collars_valid(self) -> bool:
        # Check collars is a pandas dataframe
        if not isinstance(self.collars, pd.DataFrame):
            return False

        collars_schema: dict[str, str] = {
            # Unique identifier for each row, 1-based
            "hole_index": "int",
            # Unique identifier for each survey
            "hole_id": "str",
            # Easting coordinate
            "x": "float",
            # Northing coordinate
            "y": "float",
            # Elevation at collar
            "z": "float",
            # Depth of final measurement
            "final_depth": "float",
        }

        # Check columns exist
        if not all(col in self.collars.columns for col in collars_schema.keys()):
            return False

        # Check data is of expected type
        if not self.is_schema_valid(self.collars, collars_schema):
            return False

        return True

    def is_schema_valid(self, df: pd.DataFrame, schema: dict[str, str]) -> bool:
        for col, expected_type in schema.items():
            actual_dtype = df[col].dtype

            if expected_type == "int" and not pd.api.types.is_integer_dtype(actual_dtype):
                return False
            elif expected_type == "float" and not pd.api.types.is_float_dtype(actual_dtype):
                return False
            elif expected_type == "str" and not pd.api.types.is_string_dtype(actual_dtype) and actual_dtype != "object":
                return False

        return True

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
