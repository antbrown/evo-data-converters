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

from typing import Optional

import pyarrow as pa
from evo.notebooks import ServiceManagerWidget
from evo.objects import ObjectAPIClient
from evo_schemas.objects import DownholeCollection_V1_3_0 as DownholeCollection
from pygef.cpt import CPTData

from evo.data_converters.common import EvoWorkspaceMetadata


def save_vector(array, pa_type, data_client) -> dict:
    """
    Save a 1D vector (single column) to cache and return a packed reference dict (width=1)
    """
    col = pa.array(array, type=pa_type)
    table = pa.table({"c0": col})  # name the single column
    ref = data_client.save_table(table)
    ref["width"] = 1
    return ref


def save_matrix(columns: list[pa.Array], width: int, data_client) -> dict:
    """
    Save a 2D matrix (multi-column) to cache and return a packed reference dict with width set
    """
    names = [f"c{i}" for i in range(width)]
    table = pa.table(columns, names=names)  # provide names to avoid ValueError
    ref = data_client.save_table(table)
    ref["width"] = width
    return ref


def assert_packed_refs(obj, path="root"):
    """
    Sanity check that data is a reference string.
    """
    if isinstance(obj, dict):
        if (
            "length" in obj
            and "width" in obj
            and ("data_type" in obj or "keys_data_type" in obj or "values_data_type" in obj)
        ):
            assert isinstance(obj.get("data"), str), f"'data' must be a string ref at {path}"
        for k, v in obj.items():
            assert_packed_refs(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            assert_packed_refs(v, f"{path}[{i}]")


def create_downhole_collection(
    cpt_dict: dict[str, CPTData],
    name: str = "Unnamed collection",
    # epsg_code: int,
    evo_workspace_metadata: Optional[EvoWorkspaceMetadata] = None,
    service_manager_widget: Optional["ServiceManagerWidget"] = None,
    tags: Optional[dict[str, str]] = None,
    # combine_objects_in_layers: bool = False,
    upload_path: str = "",
) -> dict | DownholeCollection | None:
    """
    Create a Downhole Collection object from a dictionary of CPT data.

    Q: Should this be named "convert_gef_cpt" similar to DUF?
    Q: How to combine CPT data into a single Downhole Collection?
    """
    environment = service_manager_widget.get_environment()
    connector = service_manager_widget.get_connector()

    object_client = ObjectAPIClient(environment, connector)
    data_client = object_client.get_data_client(service_manager_widget.cache)

    downhole_collection = {
        "name": name,
        "uuid": None,
        "schema": "/objects/downhole-collection/1.3.1/downhole-collection.schema.json",
        "type": "downhole",
        "distance_unit": "m",
        "desurvey": "balanced_tangent",
        "bounding_box": {
            "min_x": 0.0,
            "max_x": 1.0,
            "min_y": 0.0,
            "max_y": 1.0,
            "min_z": 0.0,
            "max_z": 0.0,
        },
        "coordinate_reference_system": {"epsg_code": 32650},
        "location": {
            "hole_id": {
                "values": {
                    "data_type": "int32",
                    "length": 2,
                    "width": 1,
                    "data": [1, 2],
                },
            },
            # path: [distance, azimuth, dip]
            "path": {
                "data_type": "float64",
                "length": 2,  # one row per hole
                "width": 3,
                "data": [0.0, 0.0, 90.0, 0.0, 90.0, 90.0],
                "attributes": [],
            },
            # holes rows [hole_index, offset, count]
            "holes": {
                "data_type": "int32/uint64/uint64",
                "length": 2,
                "width": 3,
                "data": [1, 0, 1, 2, 1, 1],
            },
            "attributes": [],
        },
        "collections": [
            {
                "name": "Minimal distance collection",
                "collection_type": "distance",
                "distance": {
                    "values": {
                        "data_type": "float64",
                        "length": 2,
                        "width": 1,
                        "data": [1.0, 2.0],
                    },
                    "attributes": [],
                    "unit": "m",
                },
                "holes": {
                    "data_type": "int32/uint64/uint64",
                    "length": 2,
                    "width": 3,
                    "data": [1, 0, 1, 2, 1, 1],
                },
            }
        ],
    }
    # The following properties are populated with parquet tables below, using data_client.save_table(), save_vector() and save_matrix().
    # collections
    # location.coordinates
    # location.distances
    # location.hole_id.table
    # location.holes
    # location.path

    # location.hole_id.values (1D, int32)
    downhole_collection["location"]["hole_id"]["values"] = save_vector(
        array=[1, 2], pa_type=pa.int32(), data_client=data_client
    )

    # location.hole_id.table (2 columns: keys int32, values string), keys then values.
    hole_id_keys = pa.array([1, 2], type=pa.int32())
    hole_id_vals = pa.array(["ABC-001", "ABC-002"], type=pa.string())
    hole_id_table = pa.table({"keys": hole_id_keys, "values": hole_id_vals})
    downhole_collection["location"]["hole_id"]["table"] = data_client.save_table(table=hole_id_table)

    # location.coordinates (2 rows x 3 cols float64) in row-major order
    coords_x = pa.array([1000.0, 1010.0], type=pa.float64())
    coords_y = pa.array([2000.0, 2010.0], type=pa.float64())
    coords_z = pa.array([50.0, 48.0], type=pa.float64())
    downhole_collection["location"]["coordinates"] = save_matrix(
        columns=[coords_x, coords_y, coords_z], width=3, data_client=data_client
    )

    # location.distances (columns: final, target, current), one row per hole
    dist_final = pa.array([10.0, 12.0], type=pa.float64())
    dist_target = pa.array([10.0, 12.0], type=pa.float64())
    dist_current = pa.array([10.0, 12.0], type=pa.float64())
    downhole_collection["location"]["distances"] = save_matrix(
        columns=[dist_final, dist_target, dist_current], width=3, data_client=data_client
    )

    # location.path (2 rows x 3 cols float64): distance, azimuth, dip
    path_distance = pa.array([0.0, 0.0], type=pa.float64())
    path_azimuth = pa.array([0.0, 90.0], type=pa.float64())
    path_dip = pa.array([90.0, 90.0], type=pa.float64())
    downhole_collection["location"]["path"] = save_matrix(
        columns=[path_distance, path_azimuth, path_dip], width=3, data_client=data_client
    )

    # location.holes (2 rows x 3 cols: int32, uint64, uint64) rows = [hole_index, offset, count]
    holes_idx = pa.array([1, 2], type=pa.int32())
    holes_offset = pa.array([0, 1], type=pa.uint64())
    holes_count = pa.array([1, 1], type=pa.uint64())
    downhole_collection["location"]["holes"] = save_matrix(
        columns=[holes_idx, holes_offset, holes_count], width=3, data_client=data_client
    )

    # location.collections[0].distance.values (1D float64)
    downhole_collection["collections"][0]["distance"]["values"] = save_vector(
        array=[1.0, 2.0], pa_type=pa.float64(), data_client=data_client
    )

    # location.collections[0].holes (same structure as location.holes)
    downhole_collection["collections"][0]["holes"] = save_matrix(
        columns=[holes_idx, holes_offset, holes_count], width=3, data_client=data_client
    )
    downhole_collection_path = upload_path

    return {
        "path": downhole_collection_path,
        "data": downhole_collection,
        "cpt_data": cpt_dict,
    }
    # # Upload the referenced blobs saved to the cache.
    # await data_client.upload_referenced_data(downhole_collection, fb=FeedbackWidget("Uploading data"))
    #
    # # Create our Downhole Collection object.
    # new_downhole_collection = await object_client.create_geoscience_object(
    #     downhole_collection_path, downhole_collection
    # )
    #
    # new_downhole_collection = await object_client.create_geoscience_object(
    #     downhole_collection_path, downhole_collection
    # )
    #
    # print(f"{new_downhole_collection.path}: <{new_downhole_collection.schema_id}>")
    # print(f"\tCreated at: {new_downhole_collection.created_at}")
    #
    # return new_downhole_collection
