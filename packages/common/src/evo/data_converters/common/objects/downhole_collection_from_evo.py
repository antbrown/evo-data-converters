import pandas as pd

import evo.logging

from evo.objects import DownloadedObject
from evo_schemas.objects import DownholeCollection_V1_3_1
from evo_schemas.components.downhole_attributes import DownholeAttributes_V1_0_0_Item
from .downhole_collection import (
    DownholeCollection,
    HoleCollars,
    ColumnMapping,
    MeasurementTableAdapter,
    MeasurementTableFactory,
)
from .attributes import AttributeFactory, UnsupportedAttributeType


evo_export_column_mapping = ColumnMapping(
    HOLE_INDEX_COLUMNS=["hole_index"],
    DEPTH_COLUMNS=["penetration_length"],
    FROM_COLUMNS=["from_depth"],
    TO_COLUMNS=["to_depth"],
)

logger = evo.logging.getLogger("data_converters")


async def create_downhole_collection_from_evo(obj: DownloadedObject) -> DownholeCollection:
    """
    Creates a local `DownholeCollection` with Pandas Dataframe copies of a remote Evo Downhole Collection.

    :param obj: A resolved Evo Geoscience object that points to a Downhole Collection
    :return: An instance of `DownholeCollection` with data populated
    """
    remote_dhc = DownholeCollection_V1_3_1.from_dict(obj.as_dict())
    if not isinstance(remote_dhc, DownholeCollection_V1_3_1):
        raise ValueError("Provided object is not a DownholeCollection_V1_3_1")

    # While the SDK advertises download_* work with TableInfo objects, they actually require dicts.
    collars_df = (await obj.download_dataframe(remote_dhc.location.hole_id.table.as_dict())).rename(
        columns={"key": "hole_index", "value": "hole_id"}
    )
    coordinates_df = await obj.download_dataframe(remote_dhc.location.coordinates.as_dict())
    distances_df = await obj.download_dataframe(remote_dhc.location.distances.as_dict())

    collar_attrs = []
    collar_nan_mappings = {}

    for attribute in remote_dhc.location.attributes or []:
        try:
            attr = await AttributeFactory.create_from_evo(obj, attribute)
        except UnsupportedAttributeType:
            logger.warn(f"Could not interpret collar attribute {attribute.name}, skipping.")
            continue
        collar_nan_mappings[attribute.name] = attr.attrs["nan_values"]
        assert len(attr) == len(collars_df), f"Collar attribute {attribute['key']} has the wrong number of rows"
        collar_attrs.append(attr)

    collars_df = pd.concat(
        [collars_df, coordinates_df, distances_df["final"].rename("final_depth"), *collar_attrs], axis=1
    )

    collars = HoleCollars(df=collars_df, nan_values_by_column=collar_nan_mappings)

    dhc = DownholeCollection(
        collars=collars,
        coordinate_reference_system=remote_dhc.coordinate_reference_system.epsg_code,
        uuid=str(obj.metadata.id),
        name=obj.metadata.name,
        description=remote_dhc.description,
        extensions=remote_dhc.extensions,
        tags=remote_dhc.tags,
    )

    for collection in remote_dhc.collections:
        if collection.collection_type in ("distance", "interval"):
            measurement_table = await create_measurement_table(obj=obj, collection=collection)
            dhc.add_measurement_table(input=measurement_table, column_mapping=evo_export_column_mapping)

    return dhc


async def create_measurement_table(
    obj: DownloadedObject, collection: DownholeAttributes_V1_0_0_Item
) -> MeasurementTableAdapter:
    if collection.collection_type not in ("distance", "interval"):
        raise ValueError(f"Unsupported collection_type {collection.collection_type}")

    lookup_df = await obj.download_dataframe(collection.holes.as_dict())
    """
    Table of holes and how many measurements are in each.

        hole_index  offset  count
    0           1       0    501
    1           2     501    501
    2           3    1002    501
    3           4    1503    501
    4           5    2004    501
    """

    # Pad a flat list repeating the hole_index according to the count column, to be used to
    # correlate each measurement to a collar.
    hole_indexes = lookup_df["hole_index"].repeat(lookup_df["count"]).values

    remote_attributes = []
    if collection.collection_type == "distance":
        distances_df = await obj.download_dataframe(collection.distance.values.as_dict())
        """
        This is an array of distances as floats in "values" column with no other columns.
        """

        # Combine the array of values with an index and the hole it belongs to, based on the lookup table.
        measurements_df = pd.DataFrame(
            {
                # The repeated indices above join up with each measurement to correlate it to a collar.
                "hole_index": hole_indexes,
                # The actual distance measurements
                evo_export_column_mapping.DEPTH_COLUMNS[0]: distances_df["values"].values,
            }
        )
        remote_attributes = collection.distance.attributes or []
    elif collection.collection_type == "interval":
        intervals_df = await obj.download_dataframe(collection.from_to.intervals.start_and_end.as_dict())
        """
        This is an indexed table with "from" and "to" columns (in that order) but no actual attributes
        """
        # Standardise the naming of the from/to columns, which are in that order
        intervals_df = intervals_df.rename(
            columns={
                intervals_df.columns[0]: evo_export_column_mapping.FROM_COLUMNS[0],
                intervals_df.columns[1]: evo_export_column_mapping.TO_COLUMNS[0],
            }
        )

        # Combine the array of values with an index and the hole it belongs to, based on the lookup table.
        measurements_df = pd.concat(
            [
                pd.DataFrame(
                    {
                        # The repeated indices above join up with each measurement to correlate it to a collar.
                        "hole_index": hole_indexes,
                    }
                ),
                intervals_df,
            ],
            axis=1,
        )
        remote_attributes = collection.from_to.attributes or []

    # Fold in other related data series from the collection
    attributes = []
    nan_mappings = {}
    for attribute in remote_attributes:
        try:
            attr = await AttributeFactory.create_from_evo(obj, attribute)
        except UnsupportedAttributeType:
            logger.warn(f"Could not interpret measurement attribute {attribute.name}, skipping.")
            continue
        nan_mappings[attribute.name] = attr.attrs["nan_values"]
        assert len(attr) == len(measurements_df), f"Attribute {attribute['key']} has the wrong number of rows"
        attributes.append(attr)

    measurements_df = pd.concat([measurements_df, *attributes], axis=1)

    return MeasurementTableFactory.create(
        df=measurements_df, column_mapping=evo_export_column_mapping, nan_values_by_column=nan_mappings
    )
