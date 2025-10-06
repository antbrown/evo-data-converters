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

from evo.data_converters.common.objects.downhole_collection import DownholeCollection
from pygef.cpt import CPTData
import pandas as pd
import polars as pl
import typing


def create_from_parsed_gef_cpts(parsed_cpt_files: dict[str, CPTData]) -> DownholeCollection:
    epsg_code: int = 0
    collar_rows: list[dict[str, typing.Any]] = []
    measurement_dfs: list[pl.DataFrame] = []

    for hole_index, (hole_id, cpt_data) in enumerate(parsed_cpt_files.items(), start=1):
        if epsg_code == 0:
            try:
                epsg_code = int(cpt_data.delivered_location.srs_name.split(":")[-1])
            except AttributeError:
                raise ValueError(
                    f"Could not derive EPSG code from delivered_location: {cpt_data.delivered_location.srs_name}"
                )

        collar_rows.append(
            {
                "hole_index": hole_index,
                "hole_id": hole_id,
                "x": cpt_data.delivered_location.x,
                "y": cpt_data.delivered_location.y,
                "z": cpt_data.delivered_vertical_position_offset or 0.0,
                "final_depth": cpt_data.final_depth or max(cpt_data.data["penetrationLength"]),
            }
        )

        measurements = cpt_data.data.with_columns(
            [
                pl.lit(hole_index).cast(pl.Int32).alias("hole_index"),
            ]
        ).select(["hole_index"] + [col for col in cpt_data.data.columns])

        measurement_dfs.append(measurements)

    collars = pd.DataFrame(collar_rows).astype(
        {
            "hole_index": "int32",
            "hole_id": "string",
            "x": "float64",
            "y": "float64",
            "z": "float64",
            "final_depth": "float64",
        }
    )

    if measurement_dfs:
        measurements_pl = pl.concat(measurement_dfs, how="vertical")
        measurements = measurements_pl.to_pandas(use_pyarrow_extension_array=True)
    else:
        measurements = pd.DataFrame()

    return DownholeCollection(name="madeupname", collars=collars, measurements=measurements, epsg_code=epsg_code)
