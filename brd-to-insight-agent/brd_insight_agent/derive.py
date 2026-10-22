"""Columns the user asks for that are not in the sheet, computed locally from approved columns."""

from __future__ import annotations

import numpy as np
import pandas as pd

MAX_DERIVED = 4
OPS = ("difference", "total", "ratio", "percent_change", "percent_of")


def apply_derived(frame: pd.DataFrame, derived, profiles=None) -> pd.DataFrame:
    """Return a frame with the requested derived columns added; the model never supplies values."""
    if not derived:
        return frame
    if not isinstance(derived, list) or len(derived) > MAX_DERIVED:
        raise ValueError(f"Use at most {MAX_DERIVED} derived columns.")
    out = frame
    for item in derived:
        if not isinstance(item, dict) or set(item) - {"name", "op", "left", "right"}:
            raise ValueError("A derived column needs a name, op, left and right column.")
        name, op = str(item.get("name", "")).strip(), str(item.get("op", "")).strip()
        left, right = item.get("left"), item.get("right")
        if not name or len(name) > 60:
            raise ValueError("Derived column names must be 1-60 characters.")
        if name in out.columns:
            raise ValueError(f"'{name}' already exists in this sheet; choose another name.")
        if op not in OPS:
            raise ValueError(f"'{op}' is not a supported calculation. Use: {', '.join(OPS)}.")
        columns = []
        for side in (left, right):
            if side not in frame.columns:
                raise ValueError(f"Column '{side}' is not in this sheet.")
            if not pd.api.types.is_numeric_dtype(frame[side]) or pd.api.types.is_bool_dtype(frame[side]):
                raise ValueError(f"'{side}' is not numeric, so it cannot be used in a calculation.")
            if profiles is not None and profiles[side].is_identifier:
                raise ValueError(f"'{side}' is an identifier, not a measure.")
            columns.append(frame[side].to_numpy(dtype=float, na_value=np.nan))
        first, second = columns
        with np.errstate(divide="ignore", invalid="ignore"):
            if op == "difference":
                values = first - second
            elif op == "total":
                values = first + second
            elif op == "ratio":
                values = np.where(second == 0, np.nan, first / second)
            elif op == "percent_change":
                values = np.where(second == 0, np.nan, (first - second) / second * 100)
            else:
                values = np.where(second == 0, np.nan, first / second * 100)
        if not np.isfinite(values).any():
            raise ValueError(f"'{name}' has no usable values; check the columns you asked me to combine.")
        out = out.assign(**{name: np.where(np.isfinite(values), values, np.nan)})
    return out


def describe(derived) -> str:
    labels = {"difference": "minus", "total": "plus", "ratio": "divided by",
              "percent_change": "percent change from", "percent_of": "as a percent of"}
    return "; ".join(f"{item['name']} = {item['left']} {labels[item['op']]} {item['right']}"
                     for item in derived or [])
