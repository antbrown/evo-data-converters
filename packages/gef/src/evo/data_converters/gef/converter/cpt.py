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

from __future__ import annotations
import io
import os
import dataclasses
from pathlib import Path
from typing import Any, Optional

import evo.logging

from pygef.broxml.parse_cpt import read_cpt as base_read_cpt_xml
from pygef.cpt import CPTData as _BaseCPTData
from pygef.gef.parse_cpt import _GefCpt
from pygef.shim import gef_cpt_to_cpt_data

logger = evo.logging.getLogger("data_converters")


class CPTData(_BaseCPTData):
    def __init__(
        self,
        *args: Any,
        gef_headers: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        # Bypass frozen dataclass __setattr__
        object.__setattr__(self, "_gef_headers", gef_headers)

    @property
    def gef_headers(self) -> Optional[dict[str, Any]]:
        return getattr(self, "_gef_headers", None)

    @gef_headers.setter
    def gef_headers(self, value: Optional[dict[str, Any]]) -> None:
        if value is not None and not isinstance(value, dict):
            raise TypeError("gef_headers must be a dict or None")
        object.__setattr__(self, "_gef_headers", value)

    @classmethod
    def from_base(
        cls,
        base: _BaseCPTData,
        *,
        gef_headers: Optional[dict[str, Any]] = None,
    ) -> "CPTData":
        # Build kwargs from dataclass fields of the base instance
        field_names = [f.name for f in dataclasses.fields(base)]
        kwargs = {name: getattr(base, name) for name in field_names}
        return cls(**kwargs, gef_headers=gef_headers)


def wrap_gef_cpt_to_cpt_data(gef_cpt: Any) -> CPTData:
    """
    Wrapper to convert a GEF CPT object into our extended CPTData subclass.

    This wraps the third-party gef_cpt_to_cpt_data() which returns a frozen
    _BaseCPTData instance, and re-instantiates it as our CPTData subclass
    with any additional fields (e.g. gef_headers).
    """
    base_obj: _BaseCPTData = gef_cpt_to_cpt_data(gef_cpt)
    base_kwargs = dataclasses.asdict(base_obj)
    return CPTData(**base_kwargs, gef_headers=getattr(gef_cpt, "_headers", None))


def read_cpt(
    file: io.BytesIO | Path | str,
    replace_column_voids=True,
) -> CPTData:
    """
    Parse the cpt file. Can either be BytesIO, Path or str

    :param file: bore file
    :param replace_column_voids: if true replace void values with nulls or interpolate; else retain value.
        Please note that auto engine checks if the files starts with `#GEFID`.
    """
    cpt_data = {}
    if isinstance(file, io.BytesIO):
        gef_cpt = _GefCpt(string=file.read().decode(), replace_column_voids=replace_column_voids)
    elif os.path.exists(file):
        gef_cpt = _GefCpt(path=file, replace_column_voids=replace_column_voids)
    else:
        gef_cpt = _GefCpt(string=file, replace_column_voids=replace_column_voids)

    if gef_cpt:
        cpt_data = wrap_gef_cpt_to_cpt_data(gef_cpt)

    return cpt_data


def read_cpt_xml(file: io.BytesIO | Path | str) -> list[CPTData]:
    base_cpt_data = base_read_cpt_xml(file)
    if base_cpt_data:
        return [CPTData.from_base(cpt_data) for cpt_data in base_cpt_data]
    return []
