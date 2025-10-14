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
    DownholeCollection_V1_3_1 as DownholeCollectionGeoscienceObject,
    DownholeCollection_V1_3_1_Location as DownholeCollection_Location,
)
import pyarrow as pa

from .downhole_collection import DownholeCollection

AZIMUTH: float = 0.0  # Assume vertical
DIP: float = 90.0  # Positive dip = down

logger = evo.logging.getLogger("data_converters")


class DownholeCollectionToGeoscienceObject:
    dhc: DownholeCollection
    data_client: ObjectDataClient

    def __init__(self, dhc: DownholeCollection, data_client: ObjectDataClient) -> None:
        self.dhc = dhc
        self.data_client = data_client

    def convert(self) -> DownholeCollectionGeoscienceObject:
        """Converts the downhole collection into a geoscience object"""
        logger.debug("Converting to Geoscience Object.")

        coordinate_reference_system = Crs_EpsgCode(epsg_code=self.dhc.epsg_code)
        bounding_box = self.create_bounding_box()

        distance_unit = UnitLength_UnitCategories("m")

        dhc_location = self.create_dhc_location()
        dhc_collections = self.create_dhc_collections()

        dhc_go = DownholeCollectionGeoscienceObject(
            # Base Object
            name=self.dhc.name,
            uuid=None,
            # Base Spatial Data
            bounding_box=bounding_box,
            coordinate_reference_system=coordinate_reference_system,
            # Downhole Collection
            distance_unit=distance_unit,
            location=dhc_location,
            collections=dhc_collections,
        )

        logger.debug(f"Created: {dhc_go}")

        return dhc_go

    def create_bounding_box(self) -> BoundingBox:
        """Create a Bounding Box object"""
        bounding_box: dict[str, float] = self.dhc.bounding_box()

        return BoundingBox(
            min_x=bounding_box["min_x"],
            max_x=bounding_box["max_x"],
            min_y=bounding_box["min_y"],
            max_y=bounding_box["max_y"],
            min_z=bounding_box["min_z"],
            max_z=bounding_box["max_z"],
        )

    def create_dhc_location(self) -> DownholeCollection_Location:
        """Create a downhole collection location object"""
        return DownholeCollection_Location(
            # Locations
            coordinates=self.create_dhc_location_coordinates(),
            # Hole Collars
            distances=self.create_dhc_location_distances(),
            holes=self.create_dhc_hole_chunks(),
            # DC Location
            hole_id=self.create_dhc_location_hole_id(),
            path=self.create_dhc_location_path(),
        )

    def create_dhc_location_coordinates(self) -> FloatArray3:
        """Create a 3D coordinate array from downhole locations"""
        coordinates_table = self.coordinates_table()
        coordinates_args = self.data_client.save_table(coordinates_table)
        return FloatArray3.from_dict(coordinates_args)

    def create_dhc_location_distances(self) -> FloatArray3:
        """Create distance measurements for each downhole"""
        distances_table = self.distances_table()
        distances_args = self.data_client.save_table(distances_table)
        return FloatArray3.from_dict(distances_args)

    def create_dhc_hole_chunks(self) -> HoleChunks:
        """Create a hole chunks object"""
        holes_table = self.holes_table()
        holes_args = self.data_client.save_table(holes_table)
        return HoleChunks.from_dict(holes_args)

    def create_dhc_location_hole_id(self) -> CategoryData:
        """Create a hole id category object"""
        lookup_table, integer_array_table = self.hole_id_tables()

        lookup_table_args = self.data_client.save_table(lookup_table)
        lookup_table_go = LookupTable.from_dict(lookup_table_args)

        integer_array_args = self.data_client.save_table(integer_array_table)
        integer_array_go = IntegerArray1.from_dict(integer_array_args)

        return CategoryData(table=lookup_table_go, values=integer_array_go)

    def create_dhc_location_path(self) -> DownholeDirectionVector:
        """Create a downhole direction vector for the downholes"""
        path_table = self.path_table()
        path_args = self.data_client.save_table(path_table)
        return DownholeDirectionVector.from_dict(path_args)

    def create_dhc_collections(self) -> list[DownholeAttributes]:
        """Create collections of data associated with the downholes"""
        distance_go = self.create_dhc_collection_distance()

        distance_table_go = DownholeAttributes_Item_DistanceTable(
            name="distances", holes=self.create_dhc_hole_chunks(), distance=distance_go
        )
        return [distance_table_go]

    def create_dhc_collection_distance(self) -> Distance:
        """Create a distance based attribute collection"""
        distances_table = self.collection_distances_table()
        distances_args = self.data_client.save_table(distances_table)
        distances_go = FloatArray1.from_dict(distances_args)

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

        return distance_go

    def create_continuous_attribute_component(self, name: str, attribute_table: pa.Table) -> ContinuousAttribute:
        """Create a continuous attribute"""
        float_array_args = self.data_client.save_table(attribute_table)
        float_array_go = FloatArray1.from_dict(float_array_args)
        nan_values = self.dhc.nan_values_by_attribute.get(name, [])
        return ContinuousAttribute(
            key=name,
            name=name,
            nan_description=NanContinuous(values=nan_values),
            values=float_array_go,
        )

    def coordinates_table(self) -> pa.Table:
        """Create a table of 3D coordinates from collar information"""
        coordinates_df = self.dhc.collars[["x", "y", "z"]]
        coordinates_schema = pa.schema(
            [
                pa.field("x", pa.float64()),
                pa.field("y", pa.float64()),
                pa.field("z", pa.float64()),
            ]
        )
        return pa.Table.from_pandas(coordinates_df, schema=coordinates_schema)

    def distances_table(self) -> pa.Table:
        """Create a distances table from final depth of each downhole"""
        distances_schema = pa.schema(
            [
                pa.field("final", pa.float64()),
                pa.field("target", pa.float64()),
                pa.field("current", pa.float64()),
            ]
        )
        arrays = [
            pa.array(self.dhc.collars["final_depth"], type=pa.float64()),
            pa.array(self.dhc.collars["final_depth"], type=pa.float64()),
            pa.array(self.dhc.collars["final_depth"], type=pa.float64()),
        ]
        return pa.Table.from_arrays(arrays, schema=distances_schema)

    def holes_table(self) -> pa.Table:
        """
        Create hole chunk metadata table describing data organisation for each downhole.

        Generates indexing information that describes where each hole's data begins
        (offset) and how many data points it contains (count) in a flattened data array.
        """
        grouped = (
            self.dhc.measurements.groupby("hole_index", sort=False).size().reset_index().rename(columns={0: "count"})
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
        """Create lookup and index table for hole ids"""
        lookup_table = pa.table(
            {"key": self.dhc.collars["hole_index"], "value": self.dhc.collars["hole_id"]},
            schema=pa.schema(
                [
                    pa.field("key", pa.int32()),
                    pa.field("value", pa.string()),
                ]
            ),
        )
        integer_array_table = pa.table(
            {"data": self.dhc.collars["hole_index"]}, schema=pa.schema([pa.field("data", pa.int32())])
        )
        return (lookup_table, integer_array_table)

    def path_table(self) -> pa.Table:
        """
        Create directional path table for downholes.

        - Currently assumes all holes are vertical (azimuth=0.0, dip=90.0)
        - Positive dip indicates downward direction
        - Distance values are taken from penetrationLength in each measurement point
        """
        path_schema = pa.schema(
            [
                pa.field("distance", pa.float64()),
                pa.field("azimuth", pa.float64()),
                pa.field("dip", pa.float64()),
            ]
        )
        num_measurements = len(self.dhc.measurements)

        arrays = [
            pa.array(self.dhc.measurements["penetrationLength"], type=pa.float64()),
            pa.array([AZIMUTH] * num_measurements, type=pa.float64()),
            pa.array([DIP] * num_measurements, type=pa.float64()),
        ]

        return pa.Table.from_arrays(arrays, schema=path_schema)

    def collection_distances_table(self) -> pa.Table:
        """
        Create table of all distances.

        Note:
        - Is this only applicable to distance based measurements?
        - Can depth column be used if penetration length is missing (will it ever be missing)?
        """
        distances_df = self.dhc.measurements[["penetrationLength"]].rename(columns={"penetrationLength": "values"})
        distances_schema = pa.schema([pa.field("values", pa.float64())])
        return pa.Table.from_pandas(distances_df, schema=distances_schema)

    def collection_attribute_tables(self) -> dict[str, pa.Table]:
        """Create a table for each attribute and their measurements"""
        attribute_tables = {}
        attribute_schema = pa.schema([("data", pa.float64())])
        for attribute_name in self.dhc.measurements.columns[2:]:
            attribute_df = self.dhc.measurements[[attribute_name]].rename(columns={attribute_name: "data"})
            attribute_tables[attribute_name] = pa.Table.from_pandas(attribute_df, schema=attribute_schema)
        return attribute_tables
