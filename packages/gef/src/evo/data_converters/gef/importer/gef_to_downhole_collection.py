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
    DownholeCollection_V1_3_1 as DownholeCollection,
    DownholeCollection_V1_3_1_Location as DownholeCollection_Location,
)
import pyarrow as pa
from pygef.cpt import CPTData

AZIMUTH: float = 0.0  # Assume vertical
DIP: float = 90.0  # Positive dip = down

logger = evo.logging.getLogger("data_converters")


def create_downhole_collection(
    parsed_cpt_files: dict[str, CPTData], data_client: ObjectDataClient
) -> DownholeCollection:
    """
    Create a Downhole Collection object from parsed GEF-CPT files.

    This function aggregates multiple CPT files into a single DownholeCollection object,
    extracting CRS, Bounding Box, Location and Collections data from the
    parsed CPT files.

    Args:
        parsed_cpt_files: Dictionary mapping CPT hole identifiers (str) to their
            corresponding parsed CPTData objects. Must contain at least one entry.
        data_client: Client for handling object data storage.

    Returns:
        A DownholeCollection object containing the aggregated CPT data.

    Raises:
        ValueError: If parsed_cpt_files is empty.

    Notes:
        - All CPT data are assumed to share the same coordinate reference system
        - Distance unit is currently hardcoded to meters ('m')
    """
    try:
        first_cpt_data = next(iter(parsed_cpt_files.values()))
    except StopIteration:
        raise ValueError("Parsed CPT files is empty but it is expected to contain at least one item")

    logger.debug(f"Converting {len(parsed_cpt_files.items())} GEF-CPT files to Downhole Collection.")

    coordinate_reference_system = create_coordinate_reference_system(first_cpt_data)

    bounding_box = create_bounding_box(parsed_cpt_files)

    distance_unit = UnitLength_UnitCategories("m")

    dc_location = create_dc_location(parsed_cpt_files, data_client)
    dc_collections = create_dc_collections(parsed_cpt_files, data_client)

    dc = DownholeCollection(
        # Base Object
        name="GEF-CPT Data Converter",
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


def create_coordinate_reference_system(cpt_data: CPTData) -> Crs_EpsgCode:
    """
    Extract and create a Coordinate Reference System object from CPT data.

    Parses the EPSG code from the SRS (Spatial Reference System) name in the
    CPT data's delivered location. Expected format is "prefixes:code" where code
    is an EPSG code.

    Args:
        cpt_data: CPTData object containing delivered location information with
            an SRS name in the format "prefixes:epsg_code".

    Returns:
        A Crs_EpsgCode object using the extracted EPSG code.

    Raises:
        ValueError: If the EPSG code cannot be derived from delivered_location,
            typically when srs_name is missing or malformed.
    """
    try:
        epsg_code = int(cpt_data.delivered_location.srs_name.split(":")[-1])
    except AttributeError:
        raise ValueError(f"Could not derive EPSG code from delivered_location: {cpt_data.delivered_location.srs_name}")

    return Crs_EpsgCode(epsg_code=epsg_code)


def create_bounding_box(parsed_cpt_files: dict[str, CPTData]) -> BoundingBox:
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
    x_values = [cpt_data.delivered_location.x for cpt_data in parsed_cpt_files.values()]
    y_values = [cpt_data.delivered_location.y for cpt_data in parsed_cpt_files.values()]
    z_values = [cpt_data.delivered_vertical_position_offset or 0.0 for cpt_data in parsed_cpt_files.values()]

    return BoundingBox(
        min_x=min(x_values),
        max_x=max(x_values),
        min_y=min(y_values),
        max_y=max(y_values),
        min_z=min(z_values),
        max_z=max(z_values),
    )


def create_dc_location(
    parsed_cpt_files: dict[str, CPTData], data_client: ObjectDataClient
) -> DownholeCollection_Location:
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
        coordinates=create_dc_location_coordinates(parsed_cpt_files, data_client),
        # Hole Collars
        distances=create_dc_location_distances(parsed_cpt_files, data_client),
        holes=create_dc_hole_chunks(parsed_cpt_files, data_client),
        # DC Location
        hole_id=create_dc_location_hole_id(parsed_cpt_files, data_client),
        path=create_dc_location_path(parsed_cpt_files, data_client),
    )


def create_dc_location_coordinates(parsed_cpt_files: dict[str, CPTData], data_client: ObjectDataClient) -> FloatArray3:
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
    coordinates_schema = pa.schema(
        [
            pa.field("x", pa.float64()),
            pa.field("y", pa.float64()),
            pa.field("z", pa.float64()),
        ]
    )

    arrays = [
        pa.array([cpt_data.delivered_location.x for cpt_data in parsed_cpt_files.values()], type=pa.float64()),
        pa.array([cpt_data.delivered_location.y for cpt_data in parsed_cpt_files.values()], type=pa.float64()),
        pa.array(
            [cpt_data.delivered_vertical_position_offset or 0.0 for cpt_data in parsed_cpt_files.values()],
            type=pa.float64(),
        ),
    ]

    coordinates_table = pa.Table.from_arrays(arrays, schema=coordinates_schema)
    coordinates_args = data_client.save_table(coordinates_table)

    return FloatArray3.from_dict(coordinates_args)


def create_dc_location_distances(parsed_cpt_files: dict[str, CPTData], data_client: ObjectDataClient) -> FloatArray3:
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
    distances_schema = pa.schema(
        [
            pa.field("final", pa.float64()),
            pa.field("target", pa.float64()),
            pa.field("current", pa.float64()),
        ]
    )

    arrays = [
        pa.array([cpt.final_depth for cpt in parsed_cpt_files.values()], type=pa.float64()),
        pa.array([cpt.final_depth for cpt in parsed_cpt_files.values()], type=pa.float64()),
        pa.array([cpt.predrilled_depth for cpt in parsed_cpt_files.values()], type=pa.float64()),
    ]

    distances_table = pa.Table.from_arrays(arrays, schema=distances_schema)
    distances_args = data_client.save_table(distances_table)

    return FloatArray3.from_dict(distances_args)


def create_dc_hole_chunks(parsed_cpt_files: dict[str, CPTData], data_client: ObjectDataClient) -> HoleChunks:
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
    hole_indices = []
    offsets = []
    counts = []

    current_offset = 0

    for hole_index, cpt_data in enumerate(parsed_cpt_files.values(), start=1):
        count = len(cpt_data.data)

        hole_indices.append(hole_index)
        offsets.append(current_offset)
        counts.append(count)

        current_offset += count

    holes_schema = pa.schema(
        [
            pa.field("hole_index", pa.int32()),
            pa.field("offset", pa.uint64()),
            pa.field("count", pa.uint64()),
        ]
    )

    arrays = [
        pa.array(hole_indices, type=pa.int32()),
        pa.array(offsets, type=pa.uint64()),
        pa.array(counts, type=pa.uint64()),
    ]

    holes_table = pa.Table.from_arrays(arrays, schema=holes_schema)
    holes_args = data_client.save_table(holes_table)

    return HoleChunks.from_dict(holes_args)


def create_dc_location_hole_id(parsed_cpt_files: dict[str, CPTData], data_client: ObjectDataClient) -> CategoryData:
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
    lookup_table_schema = pa.schema(
        [
            pa.field("key", pa.int32()),
            pa.field("value", pa.string()),
        ]
    )
    lookup_table = pa.table(
        {"key": range(1, len(parsed_cpt_files) + 1), "value": list(parsed_cpt_files.keys())}, schema=lookup_table_schema
    )
    lookup_table_args = data_client.save_table(lookup_table)
    lookup_table_go = LookupTable.from_dict(lookup_table_args)

    integer_array_table = pa.table(
        {"data": range(1, len(parsed_cpt_files) + 1)}, schema=pa.schema([pa.field("data", pa.int32())])
    )
    integer_array_args = data_client.save_table(integer_array_table)
    integer_array_go = IntegerArray1.from_dict(integer_array_args)

    return CategoryData(table=lookup_table_go, values=integer_array_go)


def create_dc_location_path(
    parsed_cpt_files: dict[str, CPTData], data_client: ObjectDataClient
) -> DownholeDirectionVector:
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
    distances = []
    azimuths = []
    dips = []

    for cpt_data in parsed_cpt_files.values():
        penetration_lengths = cpt_data.data["penetrationLength"].to_list()
        num_points = len(penetration_lengths)

        distances.extend(penetration_lengths)
        azimuths.extend([AZIMUTH] * num_points)
        dips.extend([DIP] * num_points)

    path_schema = pa.schema(
        [
            pa.field("distance", pa.float64()),
            pa.field("azimuth", pa.float64()),
            pa.field("dip", pa.float64()),
        ]
    )

    arrays = [
        pa.array(distances, type=pa.float64()),
        pa.array(azimuths, type=pa.float64()),
        pa.array(dips, type=pa.float64()),
    ]

    path_table = pa.Table.from_arrays(arrays, schema=path_schema)
    path_args = data_client.save_table(path_table)

    return DownholeDirectionVector.from_dict(path_args)


def create_dc_collections(
    parsed_cpt_files: dict[str, CPTData], data_client: ObjectDataClient
) -> list[DownholeAttributes]:
    distance_go = create_dc_collection_distance(parsed_cpt_files, data_client)

    distance_table = DownholeAttributes_Item_DistanceTable(
        name="distances", holes=create_dc_hole_chunks(parsed_cpt_files, data_client), distance=distance_go
    )
    return [distance_table]


def create_dc_collection_distance(parsed_cpt_files: dict[str, CPTData], data_client: ObjectDataClient) -> Distance:
    distances_schema = pa.schema([pa.field("values", pa.float64())])
    distances = []
    for cpt_data in parsed_cpt_files.values():
        penetration_lengths = cpt_data.data["penetrationLength"].to_list()
        distances.extend(penetration_lengths)

    distances_table = pa.Table.from_arrays([pa.array(distances, type=pa.float64())], schema=distances_schema)
    distances_args = data_client.save_table(distances_table)
    distances_go = FloatArray1.from_dict(distances_args)

    # Distance table attributes
    attributes = []

    first_cpt_data = next(iter(parsed_cpt_files.values()))
    columns = first_cpt_data.data.columns
    _ = columns.pop(0)

    for column_name in columns:
        attribute_go = create_continuous_attribute_component(
            column_name,
            column_name,
            parsed_cpt_files,
            data_client,
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


def create_continuous_attribute_component(
    key: str, name: str, parsed_cpt_files: dict[str, CPTData], data_client: ObjectDataClient
) -> ContinuousAttribute:
    attribute_schema = pa.schema([("data", pa.float64())])

    attribute_data = []
    for cpt_data in parsed_cpt_files.values():
        attribute_data.extend(cpt_data.data[key].to_list())

    attribute_table = pa.Table.from_arrays([pa.array(attribute_data, type=pa.float64())], schema=attribute_schema)
    float_array_args = data_client.save_table(attribute_table)
    float_array_go = FloatArray1.from_dict(float_array_args)

    return ContinuousAttribute(
        key=name,
        name=name,
        nan_description=NanContinuous(values=[]),
        values=float_array_go,
    )
