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

import asyncio
import pytest
import numpy as np
import pandas as pd
import pyarrow as pa
from unittest.mock import Mock
from datetime import datetime, timedelta
from typing import Sequence, Iterator, TypeVar, Any
from typing_extensions import override
from uuid import UUID, uuid4
from pydantic import TypeAdapter
from evo.data_converters.common.crs import crs_from_epsg_code
from evo.data_converters.common.objects.downhole_collection import (
    IntervalTable as IntervalMeasurementTable,
    DistanceTable as DistanceMeasurementTable,
)
from evo_schemas.components import (
    BoundingBox_V1_0_1,
)
from evo_schemas.objects import (
    DownholeCollection_V1_3_1,
    DownholeCollection_V1_3_1_Location,
)
from evo.objects.parquet import TableInfo
from evo.common.interfaces import IFeedback
from evo_schemas.elements.unit_length import UnitLength_V1_0_1_UnitCategories
from evo_schemas.components.downhole_attributes import (
    DownholeAttributes_V1_0_0_Item_DistanceTable,
    DownholeAttributes_V1_0_0_Item_IntervalTable,
)
from evo_schemas.components import NanContinuous_V1_0_1, NanCategorical_V1_0_1
from evo_schemas.elements.string_array import StringArray_V1_0_1
from evo_schemas.components import (
    StringAttribute_V1_1_0,
    BoolAttribute_V1_1_0,
    ContinuousAttribute_V1_1_0,
    DateTimeAttribute_V1_1_0,
    CategoryAttribute_V1_1_0,
)
from evo_schemas.components.hole_chunks import HoleChunks_V1_0_0
from evo_schemas.components.category_data import CategoryData_V1_0_1
from evo_schemas.components.distance_table import DistanceTable_V1_2_0_Distance
from evo.objects.io import ObjectDataDownload
from evo_schemas.elements.integer_array_1 import IntegerArray1_V1_0_1
from evo_schemas.components.interval_table import IntervalTable_V1_2_0_FromTo
from evo_schemas.elements.float_array_3 import FloatArray3_V1_0_1
from evo_schemas.elements.lookup_table import LookupTable_V1_0_1
from evo_schemas.elements.float_array_2 import FloatArray2_V1_0_1
from evo_schemas.components.intervals import Intervals_V1_0_1
from evo_schemas.elements.date_time_array import DateTimeArray_V1_0_1
from evo_schemas.elements.bool_array_1 import BoolArray1_V1_0_1
from evo_schemas.elements.float_array_1 import FloatArray1_V1_0_1
from evo.common.utils.feedback import NoFeedback
from evo.objects.data import ObjectSchema
from evo_schemas.components.downhole_direction_vector import DownholeDirectionVector_V1_0_0
from evo.objects import DownloadedObject
from evo.data_converters.common.objects.downhole_collection_from_evo import (
    create_downhole_collection_from_evo,
)

T_TABLE_CLASS = TypeVar("T_TABLE_CLASS", bound=type)


def table_with_data(table_class: T_TABLE_CLASS, table_args: dict, table: pa.Table) -> T_TABLE_CLASS:
    """
    Wraps what would otherwise be a geoscience table object with a pyarrow
    table directly attached to it. This allows a "download" of this table to
    be mocked, and keeps data in the DHC where it's easy to access/mutate.
    """

    class DataAttachedTable(table_class):
        pa_table: pa.Table

    id = uuid4()
    instance = DataAttachedTable(**table_args, data=id, length=len(table))

    # attach the data
    instance.pa_table = table

    return instance


class FakeDownloadedObject(DownloadedObject):
    def __init__(self, dhc: DownholeCollection_V1_3_1):
        super().__init__(
            object_=dhc,
            metadata=Mock(),
            urls_by_name={},
            connector=Mock(),
            cache=None,
        )

    @override
    def as_dict(self) -> dict:
        # Normally DownloadedObject uses pydantic to deserialize the DHC. We'll hack around this here.
        ta = TypeAdapter(DownholeCollection_V1_3_1)
        return dict(ta.dump_python(self._object, by_alias=True))

    @property
    def schema(self) -> ObjectSchema:
        return ObjectSchema.from_id(self._object.schema)

    @override
    def prepare_data_download(self, data_identifiers: Sequence[str | UUID]) -> Iterator[ObjectDataDownload]:
        raise NotImplementedError("Don't call this")

    @override
    async def download_table(self, table_info: TableInfo | str | dict, fb: IFeedback = NoFeedback) -> pa.Table:
        if isinstance(table_info, str):
            # Convert JMESPath string to a dict lookup
            # The string paths like "location.coordinates" need to resolve to the actual data UUID
            # We'll use the object's search method to find it
            search_result = self._object.search(table_info)
            if search_result and hasattr(search_result, "data"):
                table_info = {"data": search_result.data}
            else:
                raise ValueError(f"Could not find table at path: {table_info}")
        elif isinstance(table_info, dict):
            # Use the data key to find the original table data that we attached to the DHC
            def find_by_data_key(obj: Any, key: str) -> Any | None:
                if getattr(obj, "data", None) == key:
                    # found it
                    return obj
                elif isinstance(obj, list):
                    for item in obj:
                        found = find_by_data_key(item, key)
                        if found:
                            return found
                elif isinstance(obj, dict):
                    # never going to be in a dict
                    return None
                elif not isinstance(obj, (str, int, float, complex, bytes, type(None), UUID, pa.Table)):
                    for attr_val in obj.__dict__.values():
                        found = find_by_data_key(attr_val, key)
                        if found:
                            return found
                return None

            table_obj = find_by_data_key(self._object, dict(table_info)["data"])
            if not table_obj:
                # it appears the real thing also raises ValueError for unknown tables, which is a bit unusual
                raise ValueError(f"{table_info} is not defined")

            # we stashed the table against the table object, so simply return it now
            return table_obj.pa_table

    @override
    async def download_dataframe(self, table_info: TableInfo | str, fb: IFeedback = NoFeedback) -> pd.DataFrame:
        return (await self.download_table(table_info, fb)).to_pandas()

    @override
    async def download_array(self, table_info: TableInfo | str, fb: IFeedback = NoFeedback) -> np.ndarray:
        return np.array((await self.download_table(table_info, fb)))


def blank_dhc() -> DownholeCollection_V1_3_1:
    return DownholeCollection_V1_3_1(
        name="Test DHC",
        uuid=UUID("00000000-0000-0000-0000-000000000000"),
        description="Test Downhole",
        extensions={"Foo": "Bar"},
        tags={"Test tag": "Test tag value"},
        bounding_box=BoundingBox_V1_0_1(
            min_x=166.509, max_x=178.517, min_y=-46.641, max_y=-34.450, min_z=-10.0, max_z=1.0
        ),
        coordinate_reference_system=crs_from_epsg_code(4326),
        location=DownholeCollection_V1_3_1_Location(
            attributes=[],
            coordinates=table_with_data(
                FloatArray3_V1_0_1,
                dict(width=3),
                pa.Table.from_pylist(
                    [
                        {"x": 174.7762, "y": -41.2897, "z": 0.2},
                        {"x": 168.7358, "y": -45.0241, "z": 0.3},
                        {"x": 175.2793, "y": -37.7749, "z": 0.1},
                        {"x": 172.6405, "y": -43.5402, "z": 0.4},
                        {"x": 174.0702, "y": -39.1508, "z": 0.5},
                    ],
                    schema=pa.schema({"x": pa.float64(), "y": pa.float64(), "z": pa.float64()}),
                ),
            ),
            distances=table_with_data(
                FloatArray3_V1_0_1,
                dict(width=3),
                pa.Table.from_pylist(
                    [
                        {"final": 9, "target": 9, "current": 9},
                        {"final": 6, "target": 6, "current": 6},
                        {"final": 7, "target": 7, "current": 7},
                        {"final": 5, "target": 5, "current": 5},
                        {"final": 15, "target": 15, "current": 15},
                    ],
                    schema=pa.schema({"final": pa.float64(), "target": pa.float64(), "current": pa.float64()}),
                ),
            ),
            holes=table_with_data(
                HoleChunks_V1_0_0,
                dict(),
                pa.Table.from_pylist(
                    [
                        {"hole_index": 1, "offset": 0, "count": 10},
                        {"hole_index": 2, "offset": 10, "count": 10},
                        {"hole_index": 3, "offset": 20, "count": 10},
                        {"hole_index": 4, "offset": 30, "count": 5},
                        {"hole_index": 5, "offset": 35, "count": 10},
                    ],
                    schema=pa.schema({"hole_index": pa.int32(), "offset": pa.int32(), "count": pa.int32()}),
                ),
            ),
            hole_id=CategoryData_V1_0_1(
                table=table_with_data(
                    LookupTable_V1_0_1,
                    dict(keys_data_type="int32"),
                    pa.Table.from_pylist(
                        [
                            {"key": 1, "value": "HOLE-1"},
                            {"key": 2, "value": "HOLE-2"},
                            {"key": 3, "value": "HOLE-3"},
                            {"key": 4, "value": "HOLE-4"},
                            {"key": 5, "value": "HOLE-5"},
                        ],
                        schema=pa.schema({"key": pa.int32(), "value": pa.string()}),
                    ),
                ),
                # note this one isn't used
                values=table_with_data(
                    IntegerArray1_V1_0_1,
                    dict(data_type="int32"),
                    pa.Table.from_pydict({"data": [1, 2, 3, 4, 5]}, schema=pa.schema({"data": "int32"})),
                ),
            ),
            # not providing this one as we don't read it, though this is technically invalid
            path=DownholeDirectionVector_V1_0_0(data=None, attributes=[], length=0),
        ),
        collections=[],
        lineage=None,
        distance_unit=UnitLength_V1_0_1_UnitCategories.Unit_m,
    )


@pytest.fixture
def omnibus_dhc() -> DownholeCollection_V1_3_1:
    dhc = blank_dhc()

    dhc.location.hole_id.table.length = 5
    dhc.location.distances.length = 5
    dhc.location.holes.length = 5
    dhc.location.path.length = 5

    dhc.location.attributes.append(
        ContinuousAttribute_V1_1_0(
            name="TEST_ATTR",
            key="TEST_ATTR",
            nan_description=NanContinuous_V1_0_1(values=[5.123]),
            values=table_with_data(
                FloatArray1_V1_0_1,
                dict(),
                pa.Table.from_pydict(
                    {
                        "data": [
                            1.23,
                            3.45,
                            5.67,
                            np.nan,  # real NaN
                            5.123,  # NaN marker
                        ]
                    },
                    schema=pa.schema({"data": pa.float64()}),
                ),
            ),
        )
    )

    dhc.collections.append(
        DownholeAttributes_V1_0_0_Item_DistanceTable(
            name="Test distance 1",
            distance=DistanceTable_V1_2_0_Distance(
                attributes=[
                    ContinuousAttribute_V1_1_0(
                        name="TEST_DIST_ATTR",
                        key="TEST_DIST_ATTR",
                        nan_description=NanContinuous_V1_0_1(values=[6.321]),
                        values=table_with_data(
                            FloatArray1_V1_0_1,
                            dict(),
                            pa.Table.from_pydict(
                                {
                                    "data": (
                                        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 6.321, 1.0]
                                        + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 6.321, 1.0]
                                        + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 6.321, 1.0]
                                        + [0.4, 0.5, 0.6, 0.7, 0.8]
                                        + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 6.321, 1.0]
                                    )
                                },
                                schema=pa.schema({"data": pa.float64()}),
                            ),
                        ),
                    ),
                    StringAttribute_V1_1_0(
                        name="TEST_STR_ATTR",
                        key="TEST_STR_ATTR",
                        values=table_with_data(
                            StringArray_V1_0_1,
                            dict(),
                            pa.Table.from_pydict(
                                {"data": [f"string{i}" for i in range(1, 45)] + [""]},
                                schema=pa.schema({"data": pa.string()}),
                            ),
                        ),
                    ),
                    BoolAttribute_V1_1_0(
                        name="TEST_BOOL_ATTR",
                        key="TEST_BOOL_ATTR",
                        values=table_with_data(
                            BoolArray1_V1_0_1,
                            dict(),
                            pa.Table.from_pydict(
                                {
                                    # Based on position except the last which is "null"
                                    "data": [True if i % 2 == 0 else False for i in range(1, 45)] + [None],
                                },
                                schema=pa.schema({"data": pa.bool_()}),
                            ),
                        ),
                    ),
                    DateTimeAttribute_V1_1_0(
                        name="TEST_DT_ATTR",
                        key="TEST_DT_ATTR",
                        values=table_with_data(
                            DateTimeArray_V1_0_1,
                            dict(),
                            pa.Table.from_pydict(
                                {
                                    "data": [
                                        (datetime.fromisoformat("2025-01-01T00:00:00+00:00") + timedelta(minutes=i))
                                        for i in range(1, 45)
                                    ]
                                    + [-1]  # last one is a NaN
                                },
                                schema=pa.schema({"data": pa.timestamp(unit="us")}),
                            ),
                        ),
                        nan_description=NanCategorical_V1_0_1(values=[-1]),
                    ),
                    CategoryAttribute_V1_1_0(
                        name="TEST_CAT_ATTR",
                        key="TEST_CAT_ATTR",
                        table=table_with_data(
                            LookupTable_V1_0_1,
                            dict(keys_data_type="int32"),
                            pa.Table.from_pylist(
                                [
                                    {"key": 1, "value": "cat1"},
                                    {"key": 2, "value": "cat2"},
                                    {"key": 3, "value": "cat3"},
                                ],
                                schema=pa.schema({"key": pa.int32(), "value": pa.string()}),
                            ),
                        ),
                        values=table_with_data(
                            IntegerArray1_V1_0_1,
                            dict(data_type="int32"),
                            pa.Table.from_pydict(
                                # all except the last one are categories determined by index, the last is a NaN marker
                                {"data": [(i % 3) + 1 for i in range(1, 45)] + [-1]},
                                schema=pa.schema({"data": pa.int32()}),
                            ),
                        ),
                        nan_description=NanCategorical_V1_0_1(values=[-1]),
                    ),
                ],
                unit=UnitLength_V1_0_1_UnitCategories.Unit_m,
                values=table_with_data(
                    FloatArray1_V1_0_1,
                    dict(),
                    pa.Table.from_pydict(
                        {
                            # The holes are to different target depths and different numbers of readings.
                            # Use np.linspace() to automatically pad out the many depth readings.
                            "values": np.concatenate(
                                [
                                    np.linspace(0, 9, 10),
                                    np.linspace(0, 6, 10),
                                    np.linspace(0, 7, 10),
                                    np.linspace(0, 5, 5),
                                    np.linspace(0, 15, 10),
                                ]
                            )
                        },
                        schema=pa.schema({"values": pa.float64()}),
                    ),
                ),
            ),
            holes=table_with_data(
                HoleChunks_V1_0_0,
                dict(),
                pa.Table.from_pylist(
                    [
                        {"hole_index": 1, "offset": 0, "count": 10},
                        {"hole_index": 2, "offset": 10, "count": 10},
                        {"hole_index": 3, "offset": 20, "count": 10},
                        {"hole_index": 4, "offset": 30, "count": 5},
                        {"hole_index": 5, "offset": 35, "count": 10},
                    ],
                    schema=pa.schema({"hole_index": pa.int32(), "offset": pa.int32(), "count": pa.int32()}),
                ),
            ),
        )
    )
    dhc.collections.append(
        DownholeAttributes_V1_0_0_Item_IntervalTable(
            name="Test interval 1",
            from_to=IntervalTable_V1_2_0_FromTo(
                attributes=[
                    ContinuousAttribute_V1_1_0(
                        name="TEST_INTERVAL_ATTR",
                        key="TEST_INTERVAL_ATTR",
                        nan_description=NanContinuous_V1_0_1(values=[7.111]),
                        values=table_with_data(
                            FloatArray1_V1_0_1,
                            dict(),
                            pa.Table.from_pydict(
                                {
                                    "data": (
                                        [0.1, 0.5, 0.9]
                                        + [0.2, 0.6, 0.4]
                                        + [0.9, 0.4, 0.5]
                                        + [0.7, 0.4, 0.6, 7.111, 1.2]  # note the NaN marker
                                        + [0.1, 0.2, 0.1]
                                    )
                                },
                                schema=pa.schema({"data": pa.float64()}),
                            ),
                        ),
                    )
                ],
                intervals=Intervals_V1_0_1(
                    start_and_end=table_with_data(
                        FloatArray2_V1_0_1,
                        dict(),
                        pa.Table.from_pylist(
                            [
                                {"from": f, "to": t}
                                for (f, t) in [
                                    # Hole 1
                                    (0, 0.5),
                                    (1.2, 3.5),
                                    (4.1, 6.6),
                                    # Hole 2
                                    (0, 0.6),
                                    (1.1, 3.3),
                                    (4.0, 5.7),
                                    # Hole 3
                                    (0.2, 1.1),
                                    (1.5, 3.2),
                                    (4.5, 6.1),
                                    # Hole 4 (has 5 readings)
                                    (1.1, 1.5),
                                    (3.1, 3.7),
                                    (3.9, 4.0),
                                    (4.1, 4.3),
                                    (4.5, 4.9),
                                    # Hole 5
                                    (0.5, 2.1),
                                    (4.4, 6.6),
                                    (14.2, 15.0),
                                ]
                            ],
                            schema=pa.schema({"from": pa.float64(), "to": pa.float64()}),
                        ),
                    )
                ),
                unit=UnitLength_V1_0_1_UnitCategories.Unit_m,
            ),
            holes=table_with_data(
                HoleChunks_V1_0_0,
                dict(),
                pa.Table.from_pylist(
                    [
                        {"hole_index": 1, "offset": 0, "count": 3},
                        {"hole_index": 2, "offset": 3, "count": 3},
                        {"hole_index": 3, "offset": 6, "count": 3},
                        {"hole_index": 4, "offset": 9, "count": 5},
                        {"hole_index": 5, "offset": 15, "count": 3},
                    ],
                    schema=pa.schema({"hole_index": pa.int32(), "offset": pa.int32(), "count": pa.int32()}),
                ),
            ),
        )
    )

    return dhc


@pytest.fixture
def omnibus_dhc_obj(omnibus_dhc: DownholeCollection_V1_3_1) -> DownloadedObject:
    return FakeDownloadedObject(omnibus_dhc)


class TestOmnibusDHCConversion:
    def test_basic_conversion(self, omnibus_dhc: DownholeCollection_V1_3_1, omnibus_dhc_obj: DownloadedObject) -> None:
        print(omnibus_dhc_obj.as_dict())
        dhc = asyncio.run(create_downhole_collection_from_evo(omnibus_dhc_obj))

        assert dhc is not None

        assert omnibus_dhc.name == "Test DHC"
        assert len(omnibus_dhc.collections) == 2
        assert len(dhc.collars.df) == 5
        assert len(dhc.measurements) == 2

        num_distance_measurements = len(omnibus_dhc.collections[0].distance.values.pa_table)
        assert num_distance_measurements == 10 + 10 + 10 + 5 + 10
        assert isinstance(dhc.measurements[0], DistanceMeasurementTable)
        assert len(dhc.measurements[0].df) == num_distance_measurements

        num_interval_measurements = len(omnibus_dhc.collections[1].from_to.attributes[0].values.pa_table)
        assert num_interval_measurements == 3 + 3 + 3 + 5 + 3
        assert isinstance(dhc.measurements[1], IntervalMeasurementTable)
        assert len(dhc.measurements[1].df) == num_interval_measurements

    def test_collar_parsing(self, omnibus_dhc_obj: DownloadedObject) -> None:
        dhc = asyncio.run(create_downhole_collection_from_evo(omnibus_dhc_obj))

        pd.testing.assert_frame_equal(
            dhc.collars.df,
            pd.DataFrame(
                {
                    "hole_index": [1, 2, 3, 4, 5],
                    "hole_id": ["HOLE-1", "HOLE-2", "HOLE-3", "HOLE-4", "HOLE-5"],
                    "x": [174.7762, 168.7358, 175.2793, 172.6405, 174.0702],
                    "y": [-41.2897, -45.0241, -37.7749, -43.5402, -39.1508],
                    "z": [0.2, 0.3, 0.1, 0.4, 0.5],
                    "final_depth": [9.0, 6.0, 7.0, 5.0, 15.0],
                    "TEST_ATTR": [1.23, 3.45, 5.67, np.nan, np.nan],  # None the two types of NaN on the end
                }
            ).astype({"hole_index": "int32"}),
        )

    def test_distance_table_parsing(self, omnibus_dhc_obj: DownloadedObject) -> None:
        dhc = asyncio.run(create_downhole_collection_from_evo(omnibus_dhc_obj))

        assert isinstance(dhc.measurements[0], DistanceMeasurementTable)

        base_dt = datetime.fromisoformat("2025-01-01T00:00:00+00:00")
        pd.testing.assert_frame_equal(
            dhc.measurements[0].df,
            pd.DataFrame(
                columns=[
                    "hole_index",
                    "penetration_length",
                    "TEST_DIST_ATTR",
                    "TEST_STR_ATTR",
                    "TEST_BOOL_ATTR",
                    "TEST_DT_ATTR",
                    "TEST_CAT_ATTR",
                ],
                data=[
                    (1, 0.000000, 0.1, "string1", False, base_dt + timedelta(minutes=1), "cat2"),
                    (1, 1.000000, 0.2, "string2", True, base_dt + timedelta(minutes=2), "cat3"),
                    (1, 2.000000, 0.3, "string3", False, base_dt + timedelta(minutes=3), "cat1"),
                    (1, 3.000000, 0.4, "string4", True, base_dt + timedelta(minutes=4), "cat2"),
                    (1, 4.000000, 0.5, "string5", False, base_dt + timedelta(minutes=5), "cat3"),
                    (1, 5.000000, 0.6, "string6", True, base_dt + timedelta(minutes=6), "cat1"),
                    (1, 6.000000, 0.7, "string7", False, base_dt + timedelta(minutes=7), "cat2"),
                    (1, 7.000000, 0.8, "string8", True, base_dt + timedelta(minutes=8), "cat3"),
                    (1, 8.000000, np.nan, "string9", False, base_dt + timedelta(minutes=9), "cat1"),
                    (1, 9.000000, 1.0, "string10", True, base_dt + timedelta(minutes=10), "cat2"),
                    (2, 0.000000, 0.1, "string11", False, base_dt + timedelta(minutes=11), "cat3"),
                    (2, 0.666667, 0.2, "string12", True, base_dt + timedelta(minutes=12), "cat1"),
                    (2, 1.333333, 0.3, "string13", False, base_dt + timedelta(minutes=13), "cat2"),
                    (2, 2.000000, 0.4, "string14", True, base_dt + timedelta(minutes=14), "cat3"),
                    (2, 2.666667, 0.5, "string15", False, base_dt + timedelta(minutes=15), "cat1"),
                    (2, 3.333333, 0.6, "string16", True, base_dt + timedelta(minutes=16), "cat2"),
                    (2, 4.000000, 0.7, "string17", False, base_dt + timedelta(minutes=17), "cat3"),
                    (2, 4.666667, 0.8, "string18", True, base_dt + timedelta(minutes=18), "cat1"),
                    (2, 5.333333, np.nan, "string19", False, base_dt + timedelta(minutes=19), "cat2"),
                    (2, 6.000000, 1.0, "string20", True, base_dt + timedelta(minutes=20), "cat3"),
                    (3, 0.000000, 0.1, "string21", False, base_dt + timedelta(minutes=21), "cat1"),
                    (3, 0.777778, 0.2, "string22", True, base_dt + timedelta(minutes=22), "cat2"),
                    (3, 1.555556, 0.3, "string23", False, base_dt + timedelta(minutes=23), "cat3"),
                    (3, 2.333333, 0.4, "string24", True, base_dt + timedelta(minutes=24), "cat1"),
                    (3, 3.111111, 0.5, "string25", False, base_dt + timedelta(minutes=25), "cat2"),
                    (3, 3.888889, 0.6, "string26", True, base_dt + timedelta(minutes=26), "cat3"),
                    (3, 4.666667, 0.7, "string27", False, base_dt + timedelta(minutes=27), "cat1"),
                    (3, 5.444444, 0.8, "string28", True, base_dt + timedelta(minutes=28), "cat2"),
                    (3, 6.222222, np.nan, "string29", False, base_dt + timedelta(minutes=29), "cat3"),
                    (3, 7.000000, 1.0, "string30", True, base_dt + timedelta(minutes=30), "cat1"),
                    (4, 0.000000, 0.4, "string31", False, base_dt + timedelta(minutes=31), "cat2"),
                    (4, 1.250000, 0.5, "string32", True, base_dt + timedelta(minutes=32), "cat3"),
                    (4, 2.500000, 0.6, "string33", False, base_dt + timedelta(minutes=33), "cat1"),
                    (4, 3.750000, 0.7, "string34", True, base_dt + timedelta(minutes=34), "cat2"),
                    (4, 5.000000, 0.8, "string35", False, base_dt + timedelta(minutes=35), "cat3"),
                    (5, 0.000000, 0.1, "string36", True, base_dt + timedelta(minutes=36), "cat1"),
                    (5, 1.666667, 0.2, "string37", False, base_dt + timedelta(minutes=37), "cat2"),
                    (5, 3.333333, 0.3, "string38", True, base_dt + timedelta(minutes=38), "cat3"),
                    (5, 5.000000, 0.4, "string39", False, base_dt + timedelta(minutes=39), "cat1"),
                    (5, 6.666667, 0.5, "string40", True, base_dt + timedelta(minutes=40), "cat2"),
                    (5, 8.333333, 0.6, "string41", False, base_dt + timedelta(minutes=41), "cat3"),
                    (5, 10.000000, 0.7, "string42", True, base_dt + timedelta(minutes=42), "cat1"),
                    (5, 11.666667, 0.8, "string43", False, base_dt + timedelta(minutes=43), "cat2"),
                    (5, 13.333333, np.nan, "string44", True, base_dt + timedelta(minutes=44), "cat3"),
                    # Last row is NaNs
                    (5, 15.000000, 1.0, "", None, None, pd.NA),
                ],
            ).astype({"hole_index": "int32", "TEST_DT_ATTR": "datetime64[us, UTC]", "TEST_CAT_ATTR": "category"}),
            by_blocks=True,
        )

    def test_interval_table_parsing(self, omnibus_dhc_obj: DownloadedObject) -> None:
        dhc = asyncio.run(create_downhole_collection_from_evo(omnibus_dhc_obj))

        assert isinstance(dhc.measurements[1], IntervalMeasurementTable)
        pd.testing.assert_frame_equal(
            dhc.measurements[1].df,
            pd.DataFrame(
                columns=["hole_index", "from_depth", "to_depth", "TEST_INTERVAL_ATTR"],
                data=[
                    (1, 0.0, 0.5, 0.1),
                    (1, 1.2, 3.5, 0.5),
                    (1, 4.1, 6.6, 0.9),
                    (2, 0.0, 0.6, 0.2),
                    (2, 1.1, 3.3, 0.6),
                    (2, 4.0, 5.7, 0.4),
                    (3, 0.2, 1.1, 0.9),
                    (3, 1.5, 3.2, 0.4),
                    (3, 4.5, 6.1, 0.5),
                    (4, 1.1, 1.5, 0.7),
                    (4, 3.1, 3.7, 0.4),
                    (4, 3.9, 4.0, 0.6),
                    (4, 4.1, 4.3, np.nan),
                    (4, 4.5, 4.9, 1.2),
                    (5, 0.5, 2.1, 0.1),
                    (5, 4.4, 6.6, 0.2),
                    (5, 14.2, 15.0, 0.1),
                ],
            ).astype({"hole_index": "int32"}),
        )

    def test_nan_substitution(self, omnibus_dhc_obj: DownloadedObject) -> None:
        dhc = asyncio.run(create_downhole_collection_from_evo(omnibus_dhc_obj))

        assert np.isnan(dhc.collars.df["TEST_ATTR"][4])
        assert dhc.collars.get_nan_values("TEST_ATTR") == [5.123]

        assert np.isnan(dhc.measurements[0].df["TEST_DIST_ATTR"][8])
        assert dhc.measurements[0].get_nan_values("TEST_DIST_ATTR") == [6.321]

        assert np.isnan(dhc.measurements[1].df["TEST_INTERVAL_ATTR"][12])
        assert dhc.measurements[1].get_nan_values("TEST_INTERVAL_ATTR") == [7.111]

    def test_out_of_order_chunks(self, omnibus_dhc: DownholeCollection_V1_3_1) -> None:
        # Modify the dhc to have the hole chunks out of order on the distance table
        omnibus_dhc.collections[0].holes.pa_table = omnibus_dhc.collections[0].holes.pa_table.take([0, 2, 3, 1, 4])

        # Check the rows are in the order we asked
        assert omnibus_dhc.collections[0].holes.pa_table["hole_index"][3].as_py() == 2

        obj = FakeDownloadedObject(omnibus_dhc)
        dhc = asyncio.run(create_downhole_collection_from_evo(obj))

        # Check that the row 3 hole distance is still correct
        assert np.isclose(dhc.measurements[0].df["penetration_length"][22], 1.555556, 0.000001)
