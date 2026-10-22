"""Validated chart description shared by the planner and renderer."""

from dataclasses import dataclass, field
import hashlib
import json


@dataclass
class ChartSpec:
    kind: str
    title: str
    x: str | None = None
    y: str | None = None
    color: str | None = None
    agg: str = "sum"
    score: int = 0
    rationale: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        payload = {"kind": self.kind, "x": self.x, "y": self.y,
                   "color": self.color, "agg": self.agg, "extra": self.extra}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()