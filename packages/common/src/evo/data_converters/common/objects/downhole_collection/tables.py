import pandas as pd
import sys
from abc import ABC, abstractmethod

from .column_mapping import ColumnMapping

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


class MeasurementTableAdapter(ABC):
    """Base class for different measurement table types"""

    def __init__(self, df: pd.DataFrame, column_mapping: ColumnMapping) -> None:
        self.df: pd.DataFrame = df
        self.mapping: ColumnMapping = column_mapping
        self._validate()

    @abstractmethod
    def _validate(self) -> None:
        """Ensure required columns are present"""
        pass

    def get_hole_index_column(self) -> str:
        """Get the column that relates each measurement to a downhole"""
        return "hole_index"

    @abstractmethod
    def get_primary_column(self) -> str:
        """Return the name of the primary measurement column"""
        pass

    def _find_column(self, possible_names: list[str]) -> str | None:
        """Find first matching column name in dataframe"""
        df_columns_lower: dict[str, str] = {col.lower(): col for col in self.df.columns}
        for name in possible_names:
            if name.lower() in df_columns_lower:
                return df_columns_lower[name.lower()]
        return None

    def get_attribute_columns(self) -> list[str]:
        """Return all columns except the primary measurement column(s)"""
        primary: list[str] = [self.get_hole_index_column()] + self.get_primary_columns()
        return [col for col in self.df.columns if col not in primary]

    @abstractmethod
    def get_primary_columns(self) -> list[str]:
        """Return list of primary measurement columns"""
        pass


class DistanceTable(MeasurementTableAdapter):
    """For measurements at specific depths/distances"""

    @override
    def _validate(self) -> None:
        if not self._find_column(self.mapping.DEPTH_COLUMNS):
            raise ValueError(f"No depth column found. Expected one of: {self.mapping.DEPTH_COLUMNS}")

    def get_depth_column(self) -> str:
        """Get the actual depth column name"""
        col: str | None = self._find_column(self.mapping.DEPTH_COLUMNS)
        if not col:
            raise ValueError("No depth column found")
        return col

    @override
    def get_primary_column(self) -> str:
        return self.get_depth_column()

    @override
    def get_primary_columns(self) -> list[str]:
        return [self.get_depth_column()]

    def get_depth_values(self) -> pd.Series:
        """Convenience method to get depth values"""
        return self.df[self.get_depth_column()]


class IntervalTable(MeasurementTableAdapter):
    """For measurements over depth intervals"""

    @override
    def _validate(self) -> None:
        from_col: str | None = self._find_column(self.mapping.FROM_COLUMNS)
        to_col: str | None = self._find_column(self.mapping.TO_COLUMNS)

        if not from_col or not to_col:
            raise ValueError(
                f"Missing interval columns. Expected: FROM: {self.mapping.FROM_COLUMNS}, TO: {self.mapping.TO_COLUMNS}"
            )

    def get_from_column(self) -> str:
        col: str | None = self._find_column(self.mapping.FROM_COLUMNS)
        if not col:
            raise ValueError("No 'from' column found.")
        return col

    def get_to_column(self) -> str:
        col: str | None = self._find_column(self.mapping.TO_COLUMNS)
        if not col:
            raise ValueError("No 'to' column found.")
        return col

    @override
    def get_primary_column(self) -> str:
        return self.get_from_column()

    @override
    def get_primary_columns(self) -> list[str]:
        return [self.get_from_column(), self.get_to_column()]

    def get_intervals(self) -> pd.DataFrame:
        """Get dataframe with standardized interval columns"""
        return self.df[[self.get_from_column(), self.get_to_column()]]


class MeasurementTableFactory:
    """Factory to detect and create appropriate measurement table adapter"""

    @staticmethod
    def create(df: pd.DataFrame, override_column_mapping: ColumnMapping | None = None) -> MeasurementTableAdapter:
        column_mapping: ColumnMapping = (
            override_column_mapping if override_column_mapping is not None else ColumnMapping()
        )

        df_columns_lower: set[str] = set(col.lower() for col in df.columns)

        # Check for interval measurement
        has_from: bool = any(col.lower() in df_columns_lower for col in column_mapping.FROM_COLUMNS)
        has_to: bool = any(col.lower() in df_columns_lower for col in column_mapping.TO_COLUMNS)

        if has_from and has_to:
            return IntervalTable(df, column_mapping)

        # Check for distance measurement
        has_depth = any(col.lower() in df_columns_lower for col in column_mapping.DEPTH_COLUMNS)

        if has_depth:
            return DistanceTable(df, column_mapping)

        raise ValueError(
            f"Cannot determine measurement type. Expected either depth column {column_mapping.DEPTH_COLUMNS} or interval columns {column_mapping.FROM_COLUMNS}/{column_mapping.TO_COLUMNS}"
        )
