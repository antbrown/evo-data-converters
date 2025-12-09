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

# It should be noted here that the following column names are those provided
# by AGS v4.0.3 thru v4.1.1 noting that the group 'SCDG' is not present
# in v4.1.1.
# This table is used to provide units for measurement data via a simple lookup
# based on the column name. AGS does support a unit dictionary via the 'UNIT'
# group, but this is not used here. Future versions may consider using this.
# This list of column names is not exhaustive. Columns which are dimensionless
# are omitted.

MEASUREMENT_UNITS: dict[str, str] = {
    "SCPT_DPTH": "m",
    "SCPT_RES": "MPa",
    "SCPT_FRES": "MPa",
    "SCPT_PWP1": "MPa",
    "SCPT_PWP2": "MPa",
    "SCPT_PWP3": "MPa",
    "SCPT_CON": "uS / cm",
    "SCPT_TEMP": "celsius",
    "SCPT_SLP1": "degrees",
    "SCPT_SLP2": "degrees",
    "SCPT_REDX": "mV",
    "SCPT_MAGT": "nT",
    "SCPT_MAGX": "nT",
    "SCPT_MAGY": "nT",
    "SCPT_MAGZ": "nT",
    "SCPT_NGAM": "Hz",
    "SCPT_QT": "MPa",
    "SCPT_FT": "MPa",
    "SCPT_QE": "MPa",
    "SCPT_BDEN": "Mg / m ** 3",
    "SCPT_CPO": "kPa",
    "SCPT_CPOD": "kPa",
    "SCPT_QNET": "MPa",
    "SCPT_EXPP": "MPa",
    "SCPT_ISPP": "MPa",
    "SCPP_TOP": "m",
    "SCPP_BASE": "m",
    "SCPP_CSU": "kPa",
    "SCPP_CPHI": "degrees",
    "SCDG_DPTH": "m",
    "SCDG_PWPI": "MPa",
    "SCDG_PWPE": "MPa",
    "SCDG_T": "s",
    "SCDG_CV": "m ** 2 / year",
    "SCDG_CH": "m ** 2 / year",
}
