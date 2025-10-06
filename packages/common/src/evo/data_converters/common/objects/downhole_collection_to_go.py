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
import evo.logging
from evo.objects.utils.data import ObjectDataClient
from evo_schemas.components import (
    BoundingBox_V1_0_1 as BoundingBox,
    CategoryData_V1_0_1 as CategoryData,
    Crs_V1_0_1_EpsgCode as Crs_EpsgCode,
    DownholeAttributes_V1_0_0_Item as DownholeAttributes,
    DownholeAttributes_V1_0_0_Item_DistanceTable as DownholeAttributes_Item_DistanceTable,
    DownholeDirectionVector_V1_0_0 as DownholeDirectionVector,
    HoleChunks_V1_0_0 as HoleChunks,
    DistanceTable_V1_2_0_Distance as Distance,
    ContinuousAttribute_V1_1_0 as ContinuousAttribute,
    NanContinuous_V1_0_1 as NanContinuous,
)
from evo_schemas.elements import (
    FloatArray1_V1_0_1 as FloatArray1,
    FloatArray3_V1_0_1 as FloatArray3,
    IntegerArray1_V1_0_1 as IntegerArray1,
    LookupTable_V1_0_1 as LookupTable,
    UnitLength_V1_0_1_UnitCategories as UnitLength_UnitCategories,
)
from evo_schemas.objects import (
    DownholeCollection_V1_3_1 as DownholeCollectionGo,
    DownholeCollection_V1_3_1_Location as DownholeCollection_Location,
)
import pyarrow as pa

from .downhole_collection import DownholeCollection

AZIMUTH: float = 0.0  # Assume vertical
DIP: float = 90.0  # Positive dip = down

logger = evo.logging.getLogger("data_converters")


@dataclass
class DownholeCollectionToGeoscienceObject:
    dc: DownholeCollection
    data_client: ObjectDataClient

    def convert(self) -> DownholeCollectionGo:
        """
        Create a Downhole Collection geoscience object.
        """
        logger.debug("Converting to Geoscience Object.")

        coordinate_reference_system = Crs_EpsgCode(epsg_code=self.dc.epsg_code)
        bounding_box = self.create_bounding_box()

        distance_unit = UnitLength_UnitCategories("m")

        dc_location = self.create_dc_location()
        dc_collections = self.create_dc_collections()

        dc = DownholeCollectionGo(
            # Base Object
            name=self.dc.name,
            uuid=None,
            # Base Spatial Data
            bounding_box=bounding_box,
            coordinate_reference_system=coordinate_reference_system,
            # Downhole Collection
            distance_unit=distance_unit,
            location=dc_location,
            collections=dc_collections,
        )

        logger.debug(f"Created: {dc}")

        return dc

    def create_bounding_box(self) -> BoundingBox:
        """
        Calculate the 3D bounding box encompassing all CPT file locations.

        Determines the minimum and maximum x, y, and z coordinates across all
        provided CPT files to create a bounding box that contains all data points.

        Args:
            parsed_cpt_files: Dictionary mapping CPT file identifiers to CPTData objects.
                Each CPTData must have delivered_location with x and y attributes, and
                optionally delivered_vertical_position_offset for z values.

        Returns:
            A BoundingBox object with min_x, max_x, min_y, max_y, min_z, and max_z
            values representing the extent of all CPT locations.

        Note:
            If delivered_vertical_position_offset is None, defaults to 0.0 for z-coordinate.
        """
        bounding_box = self.dc.bounding_box()

        return BoundingBox(
            min_x=bounding_box["min_x"],
            max_x=bounding_box["max_x"],
            min_y=bounding_box["min_y"],
            max_y=bounding_box["max_y"],
            min_z=bounding_box["min_z"],
            max_z=bounding_box["max_z"],
        )

    def create_dc_location(self) -> DownholeCollection_Location:
        """
        Create a DownholeCollection_Location object from CPT files.

        Aggregates coordinates, distances, hole chunk mapping, hole IDs, and path data
        for all CPT files into a single location object.

        Args:
            parsed_cpt_files: Dictionary mapping CPT file identifiers to CPTData objects.
            data_client: Client for handling object data storage.

        Returns:
            A DownholeCollection_Location object containing coordinates, distances,
            hole chunks, hole IDs, and path information for all CPT files.
        """
        return DownholeCollection_Location(
            # Locations
            coordinates=self.create_dc_location_coordinates(),
            # Hole Collars
            distances=self.create_dc_location_distances(),
            holes=self.create_dc_hole_chunks(),
            # DC Location
            hole_id=self.create_dc_location_hole_id(),
            path=self.create_dc_location_path(),
        )

    def create_dc_location_coordinates(self) -> FloatArray3:
        """
        Create a 3D coordinate array from CPT file locations.

        Extracts x, y, and z coordinates from all CPT files and creates a FloatArray3 with those values.

        Args:
            parsed_cpt_files: Dictionary mapping CPT file identifiers to CPTData objects.
                Each must have delivered_location.x, delivered_location.y, and
                delivered_vertical_position_offset attributes.
            data_client: Client for handling object data storage.

        Returns:
            A FloatArray3 object representing the 3D coordinates of all CPT file locations.

        Note:
            If delivered_vertical_position_offset is None, 0.0 will be used instead.
        """
        coordinates_table = self.coordinates_table()
        coordinates_args = self.data_client.save_table(coordinates_table)
        return FloatArray3.from_dict(coordinates_args)

    def create_dc_location_distances(self) -> FloatArray3:
        """
        Create distance measurements for each CPT hole.

        Generates a table with final, target, and current distances for each hole.

        Args:
            parsed_cpt_files: Dictionary mapping CPT file identifiers to CPTData objects.
                Each must have final_depth and predrilled_depth attributes.
            data_client: Client for handling object data storage.

        Returns:
            A FloatArray3 object containing final, target, and current distances
            for each CPT hole.

        Notes:
            - Final and target distances are currently identical (both use final_depth?)
            - Current distance uses the predrilled depth property (is this correct?)
        """
        distances_table = self.distances_table()
        distances_args = self.data_client.save_table(distances_table)
        return FloatArray3.from_dict(distances_args)

    def create_dc_hole_chunks(self) -> HoleChunks:
        """
        Create hole chunk metadata describing data organisation for each CPT hole.

        Generates indexing information that describes where each hole's data begins
        (offset) and how many data points it contains (count) in a flattened data array.

        Args:
            parsed_cpt_files: Dictionary mapping CPT file identifiers to CPTData objects.
                Each CPTData must have a 'data' attribute with measurable length.
            data_client: Client for handling object data storage.

        Returns:
            A HoleChunks object containing hole_index (1-based), offset, and count
            for each CPT hole.

        Note:
            - Hole indices are 1-based
            - Offsets are cumulative, starting at 0
            - Count represents the number of data points (rows) in each CPT file
        """
        holes_table = self.holes_table()
        holes_args = self.data_client.save_table(holes_table)
        return HoleChunks.from_dict(holes_args)

    def create_dc_location_hole_id(self) -> CategoryData:
        """
        Create categorical hole ID mappings for CPT files.

        Generates a lookup table that maps integer keys (1-based indices) to CPT hole
        identifiers, along with an integer array containing the indices.

        Args:
            parsed_cpt_files: Dictionary mapping CPT file identifiers (strings) to
                CPTData objects. Keys are used as the hole ID values.
            data_client: Client for handling object data storage.

        Returns:
            A CategoryData object containing a lookup table mapping integer keys to
            hole ID strings, and an integer array with sequential indices.

        Note:
            - Keys are 1-based integers (1, 2, 3, ...)
            - Values are the string keys from parsed_cpt_files dictionary
            - Order follows the dictionary's iteration order
        """
        lookup_table, integer_array_table = self.hole_id_tables()

        lookup_table_args = self.data_client.save_table(lookup_table)
        lookup_table_go = LookupTable.from_dict(lookup_table_args)

        integer_array_args = self.data_client.save_table(integer_array_table)
        integer_array_go = IntegerArray1.from_dict(integer_array_args)

        return CategoryData(table=lookup_table_go, values=integer_array_go)

    def create_dc_location_path(self) -> DownholeDirectionVector:
        """
        Create directional path data for CPT holes.

        Generates a downhole direction vector containing distance, azimuth, and dip
        information for each measurement point in all CPT files. Currently assumes
        all holes are vertical.

        Args:
            parsed_cpt_files: Dictionary mapping CPT file identifiers to CPTData objects.
                Each CPTData.data must contain a 'penetrationLength' column.
            data_client: Client used to save the path table and generate storage arguments.

        Returns:
            A DownholeDirectionVector object containing distance, azimuth, and dip
            values for all measurement points across all CPT files.

        Note:
            - Currently assumes all holes are vertical (azimuth=0.0, dip=90.0)
            - Positive dip indicates downward direction
            - Distance values are taken from penetrationLength in each CPT data point
        """
        path_table = self.path_table()
        path_args = self.data_client.save_table(path_table)
        return DownholeDirectionVector.from_dict(path_args)

    def create_dc_collections(self) -> list[DownholeAttributes]:
        distance_go = self.create_dc_collection_distance()

        distance_table_go = DownholeAttributes_Item_DistanceTable(
            name="distances", holes=self.create_dc_hole_chunks(), distance=distance_go
        )
        return [distance_table_go]

    def create_dc_collection_distance(self) -> Distance:
        distances_table = self.collection_distances_table()
        distances_args = self.data_client.save_table(distances_table)
        distances_go = FloatArray1.from_dict(distances_args)

        # Distance table attributes
        attributes = []

        attribute_tables = self.collection_attribute_tables()
        for attribute_name, attribute_table in attribute_tables.items():
            attribute_go = self.create_continuous_attribute_component(
                attribute_name,
                attribute_table,
            )
            attributes.append(attribute_go)

        distances_unit = UnitLength_UnitCategories("m")

        distance_go = Distance(
            attributes=attributes,
            unit=distances_unit,
            values=distances_go,
        )

        # distance_table_go = DistanceTable(name="hole ids?", distance=distance_go)
        return distance_go

    def create_continuous_attribute_component(self, name: str, attribute_table: pa.Table) -> ContinuousAttribute:
        float_array_args = self.data_client.save_table(attribute_table)
        float_array_go = FloatArray1.from_dict(float_array_args)
        return ContinuousAttribute(
            key=name,
            name=name,
            nan_description=NanContinuous(values=[]),
            values=float_array_go,
        )

    def coordinates_table(self) -> pa.Table:
        coordinates_df = self.dc.collars[["x", "y", "z"]]
        coordinates_schema = pa.schema(
            [
                pa.field("x", pa.float64()),
                pa.field("y", pa.float64()),
                pa.field("z", pa.float64()),
            ]
        )
        return pa.Table.from_pandas(coordinates_df, schema=coordinates_schema)

    def distances_table(self) -> pa.Table:
        """Still unsure why final depth is used for all values"""
        distances_schema = pa.schema(
            [
                pa.field("final", pa.float64()),
                pa.field("target", pa.float64()),
                pa.field("current", pa.float64()),
            ]
        )
        arrays = [
            pa.array(self.dc.collars["final_depth"], type=pa.float64()),
            pa.array(self.dc.collars["final_depth"], type=pa.float64()),
            pa.array(self.dc.collars["final_depth"], type=pa.float64()),
        ]
        return pa.Table.from_arrays(arrays, schema=distances_schema)

    def holes_table(self) -> pa.Table:
        grouped = (
            self.dc.measurements.groupby("hole_index", sort=False).size().reset_index().rename(columns={0: "count"})
        )
        grouped["offset"] = grouped["count"].shift(1, fill_value=0).cumsum()
        holes_schema = pa.schema(
            [
                pa.field("hole_index", pa.int32()),
                pa.field("offset", pa.uint64()),
                pa.field("count", pa.uint64()),
            ]
        )
        arrays = [
            pa.array(grouped["hole_index"], type=pa.int32()),
            pa.array(grouped["offset"], type=pa.uint64()),
            pa.array(grouped["count"], type=pa.uint64()),
        ]
        return pa.Table.from_arrays(arrays, schema=holes_schema)

    def hole_id_tables(self) -> tuple[pa.Table, pa.Table]:
        lookup_table = pa.table(
            {"key": self.dc.collars["hole_index"], "value": self.dc.collars["hole_id"]},
            schema=pa.schema(
                [
                    pa.field("key", pa.int32()),
                    pa.field("value", pa.string()),
                ]
            ),
        )
        integer_array_table = pa.table(
            {"data": self.dc.collars["hole_index"]}, schema=pa.schema([pa.field("data", pa.int32())])
        )
        return (lookup_table, integer_array_table)

    def path_table(self) -> pa.Table:
        path_schema = pa.schema(
            [
                pa.field("distance", pa.float64()),
                pa.field("azimuth", pa.float64()),
                pa.field("dip", pa.float64()),
            ]
        )
        num_measurements = len(self.dc.measurements)

        arrays = [
            pa.array(self.dc.measurements["penetrationLength"], type=pa.float64()),
            pa.array([AZIMUTH] * num_measurements, type=pa.float64()),
            pa.array([DIP] * num_measurements, type=pa.float64()),
        ]

        return pa.Table.from_arrays(arrays, schema=path_schema)

    def collection_distances_table(self) -> pa.Table:
        # only applicable to distance based measurements? can depth be used if penetration length is missing?
        distances_df = self.dc.measurements[["penetrationLength"]].rename(columns={"penetrationLength": "values"})
        distances_schema = pa.schema([pa.field("values", pa.float64())])
        return pa.Table.from_pandas(distances_df, schema=distances_schema)

    def collection_attribute_tables(self) -> dict[str, pa.Table]:
        # Add handling for interval tables?
        attribute_tables = {}
        attribute_schema = pa.schema([("data", pa.float64())])
        for attribute_name in self.dc.measurements[2:]:
            attribute_df = self.dc.measurements[[attribute_name]].rename(columns={attribute_name: "data"})
            attribute_tables[attribute_name] = pa.Table.from_pandas(attribute_df, schema=attribute_schema)
        return attribute_tables
