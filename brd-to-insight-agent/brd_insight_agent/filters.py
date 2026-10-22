"""Filter choices offered from the real sheet, emitted as the existing filter contract."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .intent import apply_filters, parse_filters

MAX_CATEGORY_CHOICES = 200
MAX_COLUMNS = 12
MAX_SCOPE_FILTERS = 10


def filter_offers(frame, profiles):
    """Columns this sheet can actually be filtered on, with their real values."""
    offers = []
    for name in frame.columns:
        profile = profiles[name]
        values = frame[name].dropna()
        if values.empty:
            continue
        if profile.role == "datetime":
            stamps = pd.DatetimeIndex(values)
            offers.append({"column": name, "kind": "date", "periods": _periods(stamps),
                           "min": stamps.min().date().isoformat(), "max": stamps.max().date().isoformat()})
        elif profile.role == "boolean":
            offers.append({"column": name, "kind": "boolean"})
        elif profile.role == "categorical" and not profile.is_identifier and profile.unique <= MAX_CATEGORY_CHOICES:
            offers.append({"column": name, "kind": "category", "values": sorted(values.astype(str).unique())})
        elif profile.role == "numeric" and not profile.is_identifier and not pd.api.types.is_complex_dtype(values):
            finite = values.to_numpy(dtype=float, na_value=np.nan)
            finite = finite[np.isfinite(finite)]
            if finite.size and float(finite.min()) < float(finite.max()):
                offers.append({"column": name, "kind": "number",
                               "min": float(finite.min()), "max": float(finite.max())})
    return offers[:MAX_COLUMNS]


def _periods(stamps):
    """Calendar years and quarters present in the data, each labelled with its exact dates."""
    periods = []
    for year in sorted({int(value) for value in stamps.year}):
        periods.append({"label": f"{year} (full year)", "start": f"{year}-01-01", "end": f"{year}-12-31"})
        for quarter in sorted({int(value) for value in stamps[stamps.year == year].quarter}):
            start = pd.Timestamp(year=year, month=quarter * 3 - 2, day=1)
            end = start + pd.offsets.QuarterEnd(0)
            periods.append({"label": f"{year} Q{quarter} ({start.date()} to {end.date()})",
                            "start": start.date().isoformat(), "end": end.date().isoformat()})
    return periods


def date_filter(column, start, end):
    return {"column": column, "op": "between", "value": [str(start), str(end)]}


def category_filter(column, values):
    return {"column": column, "op": "in", "value": [str(value) for value in values]}


def number_filter(column, low, high):
    return {"column": column, "op": "between", "value": [str(float(low)), str(float(high))]}


def boolean_filter(column, value):
    return {"column": column, "op": "eq", "value": ["true" if value else "false"]}


def describe(frame, profiles, scope):
    """Local summary and matched row count for the active scope."""
    if not scope:
        return "All rows in the selected sheet", len(frame)
    parsed = parse_filters(scope, frame, profiles)
    return "; ".join(item.describe() for item in parsed), len(apply_filters(frame, parsed))
