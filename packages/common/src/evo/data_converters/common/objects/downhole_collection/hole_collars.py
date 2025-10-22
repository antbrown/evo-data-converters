import pandas as pd


HOLE_COLLARS_SCHEMA: dict[str, str] = {
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


class HoleCollars:
    def __init__(self, df: pd.DataFrame) -> None:
        # Hole collar information (one row per hole)
        self.df: pd.DataFrame = df
        self._validate()

    def _validate(self) -> None:
        if not all(col in self.df.columns for col in HOLE_COLLARS_SCHEMA.keys()):
            raise ValueError("Could not find all required columns")

        if not self.is_schema_valid(self.df, HOLE_COLLARS_SCHEMA):
            raise ValueError("Data is of incorrect type in collars table")

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
