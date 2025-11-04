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

from evo.data_converters.common.objects.downhole_collection import (
    ColumnMapping,
    DownholeCollection,
    HoleCollars,
    MeasurementTableFactory,
)
from pygef.cpt import CPTData
import pandas as pd
import polars as pl
import typing
import evo.logging

from collections import defaultdict

logger = evo.logging.getLogger("data_converters")


def create_from_parsed_gef_cpts(parsed_cpt_files: dict[str, CPTData]) -> DownholeCollection:
    """
    Create a DownholeCollection from parsed GEF CPT files.

    Args:
        parsed_cpt_files: Dictionary mapping hole IDs to CPTData objects

    Returns:
        DownholeCollection containing collar and measurement data

    Raises:
        ValueError: If no CPT files provided, EPSG codes are inconsistent,
                    or required data is missing/malformed
    """
    if not parsed_cpt_files:
        raise ValueError("No CPT files provided - parsed_cpt_files dictionary is empty")

    epsg_code: int | str | None = None
    collar_rows: list[dict[str, typing.Any]] = []
    measurement_dfs: list[pl.DataFrame] = []
    nan_values_by_attribute: dict[str, list] = defaultdict(list)

    for hole_index, (hole_id, cpt_data) in enumerate(parsed_cpt_files.items(), start=1):
        # Extract and validate EPSG code
        current_epsg = _extract_epsg_code(cpt_data, hole_id)

        if epsg_code is None:
            epsg_code = current_epsg
            logger.info(f"Using EPSG code {epsg_code} from first CPT file")
        elif epsg_code != current_epsg:
            raise ValueError(
                f"Inconsistent EPSG codes detected: {hole_id} has EPSG:{current_epsg}, but expected EPSG:{epsg_code}"
            )

        # Validate required location attributes
        _validate_location_attributes(cpt_data, hole_id)

        # Calculate final depth
        final_depth = _calculate_final_depth(cpt_data, hole_id)

        collar_rows.append(
            {
                "hole_index": hole_index,
                "hole_id": hole_id,
                "x": cpt_data.delivered_location.x,
                "y": cpt_data.delivered_location.y,
                "z": cpt_data.delivered_vertical_position_offset or 0.0,
                "final_depth": final_depth,
            }
        )

        # Add hole_index to measurements
        measurements = cpt_data.data.with_columns(pl.lit(hole_index).cast(pl.Int32).alias("hole_index"))

        # Reorder columns to put hole_index first
        other_cols = [col for col in cpt_data.data.columns if col != "hole_index"]
        measurements = measurements.select(["hole_index"] + other_cols)

        measurement_dfs.append(measurements)
        logger.debug(f"Processed {hole_id}: {len(measurements)} measurements")

        # update the nan values from this hole by attribute name
        for attribute_name, value in cpt_data.column_void_mapping.items():
            nan_values_by_attribute[attribute_name].append(value)

    if epsg_code is None:
        raise ValueError("Could not find valid epsg code in CPT files")

    collars_df = pd.DataFrame(collar_rows).astype(
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
        measurements = measurements_pl.to_pandas()
        logger.info(f"Creating collection with {len(measurements)} total measurements")
    else:
        # Create empty DataFrame with proper schema
        measurements = pd.DataFrame(columns=["hole_index"])
        logger.warning("No measurement data found in CPT files")

    collection_name = get_collection_name_from_collars(collar_rows)

    column_mapping = ColumnMapping(DEPTH_COLUMNS=["penetrationLength"])
    distance_measurements = MeasurementTableFactory.create(df=measurements, column_mapping=column_mapping)
    collars = HoleCollars(df=collars_df)

    return DownholeCollection(
        name=collection_name,
        collars=collars,
        measurements=[distance_measurements],
        coordinate_reference_system=epsg_code,
        nan_values_by_attribute=nan_values_by_attribute,
    )


def _extract_epsg_code(cpt_data: CPTData, hole_id: str) -> int | str:
    """Extract EPSG code from CPTData object"""

    try:
        srs_name = cpt_data.delivered_location.srs_name
    except AttributeError:
        raise ValueError(f"CPT file '{hole_id}' is missing delivered_location.srs_name attribute")

    if ":" not in srs_name:
        raise ValueError(f"CPT file '{hole_id}' has malformed SRS name: '{srs_name}'. Expected format: 'urn:123'")

    try:
        epsg_code = int(srs_name.split(":")[-1])
    except (ValueError, IndexError) as e:
        raise ValueError(f"CPT file '{hole_id}' has invalid EPSG code in SRS name: '{srs_name}'. Error: {e}")

    if epsg_code == 404000:
        epsg_code = "unspecified"

    return epsg_code


def _validate_location_attributes(cpt_data: CPTData, hole_id: str) -> None:
    """Validate that required x, y location attributes exist"""

    try:
        _ = cpt_data.delivered_location.x
        _ = cpt_data.delivered_location.y
    except AttributeError as e:
        raise ValueError(f"CPT file '{hole_id}' is missing required location attribute (x or y): {e}")


def _calculate_final_depth(cpt_data: CPTData, hole_id: str) -> float:
    """Calculate final depth from CPTData"""

    if cpt_data.final_depth != 0.0:
        # Can a final depth of 0 be valid?
        return cpt_data.final_depth

    # Try to calculate from penetrationLength column
    if "penetrationLength" not in cpt_data.data.columns:
        raise ValueError(f"CPT file '{hole_id}' is missing 'penetrationLength' column.")

    penetration_lengths = cpt_data.data["penetrationLength"]

    if len(penetration_lengths) == 0:
        raise ValueError(f"CPT file '{hole_id}' has empty penetrationLength column.")

    try:
        return float(penetration_lengths.max())
    except Exception as e:
        raise ValueError(f"CPT file '{hole_id}' has invalid penetrationLength data: {e}")


def get_collection_name_from_collars(collar_rows: list[dict[str, typing.Any]]) -> str:
    """Generate a collection name from collar data"""

    if not collar_rows:
        return ""
    elif len(collar_rows) == 1:
        return collar_rows[0]["hole_id"]
    else:
        return f"{collar_rows[0]['hole_id']}...{collar_rows[-1]['hole_id']}"
