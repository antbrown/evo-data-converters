from dataclasses import dataclass, field


@dataclass
class ColumnMapping:
    DEPTH_COLUMNS: list[str] = field(default_factory=lambda: ["penetrationLength", "SCPT_DPTH"])
    CONE_RESISTANCE_COLUMNS: list[str] = field(default_factory=lambda: ["coneResistance", "SCPT_RES"])

    FROM_COLUMNS: list[str] = field(default_factory=lambda: ["SCPP_TOP", "GEOL_TOP"])
    TO_COLUMNS: list[str] = field(default_factory=lambda: ["SCPP_BASE", "GEOL_BASE"])
