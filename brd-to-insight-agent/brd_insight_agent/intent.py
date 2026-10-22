"""The contract between a natural-language request and a rendered chart.

The model may only emit this shape, and every field is re-checked against the real
sheet before anything runs. Numbers on the chart always come from pandas, never
from the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import math
import re

import pandas as pd

from .charts import AGGS, CHART_KINDS, ORIENTATIONS, PALETTES, _validate_spec
from .profiler import ColumnProfile, profile_frame
from .spec import ChartSpec

OPS = ["between", "in", "not_in", "gte", "lte", "gt", "lt", "eq", "neq", "contains",
    "is_null", "not_null", "above_mean", "below_mean"]
GRAINS = ["auto", "D", "W", "M", "Q", "Y"]

NUMERIC_OPS = {"between", "in", "not_in", "gte", "lte", "gt", "lt", "eq", "neq",
               "is_null", "not_null", "above_mean", "below_mean"}
TEXT_OPS = {"in", "not_in", "eq", "neq", "contains", "is_null", "not_null"}
ARITY = {"between": 2, "is_null": 0, "not_null": 0, "above_mean": 0, "below_mean": 0}

# Filter values use strings for controlled generation.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "chart": {"type": "string", "enum": CHART_KINDS},
        "x": {"type": "string"},
        "y": {"type": "string", "nullable": True},
        "color": {"type": "string", "nullable": True},
        "agg": {"type": "string", "enum": AGGS},
        "grain": {"type": "string", "enum": GRAINS},
        "orientation": {"type": "string", "enum": ORIENTATIONS},
        "palette": {"type": "string", "enum": list(PALETTES)},
        "colors": {"type": "array", "items": {"type": "string"}},
        "category_colors": {"type": "array", "items": {
            "type": "object", "properties": {"value": {"type": "string"}, "color": {"type": "string"}},
            "required": ["value", "color"]}},
        "bins": {"type": "integer", "minimum": 5, "maximum": 100},
        "top_n": {"type": "integer", "minimum": 1, "maximum": 40},
        "filters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "op": {"type": "string", "enum": OPS},
                    "value": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["column", "op", "value"],
            },
        },
        "title": {"type": "string"},
        "columns": {"type": "array", "items": {"type": "string"}},
        "interpretation": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "clarifying_question": {"type": "string", "nullable": True},
    },
    "required": ["chart", "agg", "filters", "title", "interpretation", "confidence"],
}


class SpecError(ValueError):
    """The model produced something that does not fit the actual sheet."""


def _resolve_name(name: str, df: pd.DataFrame) -> str | None:
    """Match a loosely written column name; prefer a label column over a measure that repeats the word."""
    from .profiler import _tokens, singular

    generic = {"name", "code", "id", "number", "value", "column", "field", "total", "amount"}
    wanted = {singular(token) for token in _tokens(name)}
    if not wanted:
        return None
    columns = {column: {singular(token) for token in _tokens(str(column))} for column in df.columns}
    matches = [column for column, tokens in columns.items() if wanted <= tokens]
    if not matches and wanted - generic:
        matches = [column for column, tokens in columns.items() if (wanted - generic) <= tokens]
    if len(matches) > 1:
        labels = [column for column in matches if not pd.api.types.is_numeric_dtype(df[column])]
        matches = labels or matches
    if len(matches) > 1:
        matches = sorted(matches, key=lambda column: len(_tokens(str(column))))
        if len(_tokens(str(matches[0]))) == len(_tokens(str(matches[1]))):
            return None
    return matches[0] if matches else None


@dataclass
class Filter:
    column: str
    op: str
    value: list

    def describe(self) -> str:
        c = self.column
        if self.op == "between":
            return f"{c} from {_fmt(self.value[0])} to {_fmt(self.value[1])}"
        if self.op in ("in", "not_in"):
            joined = ", ".join(str(v) for v in self.value[:6])
            more = "" if len(self.value) <= 6 else f" (+{len(self.value) - 6} more)"
            return f"{c} {'is' if self.op == 'in' else 'is not'} {joined}{more}"
        if self.op in ("is_null", "not_null"):
            return f"{c} is {'blank' if self.op == 'is_null' else 'present'}"
        if self.op in ("above_mean", "below_mean"):
            direction = "above" if self.op == "above_mean" else "below"
            return f"{c} {direction} the sheet-wide mean (original input rows)"
        symbols = {"gte": ">=", "lte": "<=", "gt": ">", "lt": "<", "eq": "=", "neq": "!=", "contains": "contains"}
        return f"{c} {symbols[self.op]} {_fmt(self.value[0])}"


@dataclass
class ParsedRequest:
    spec: ChartSpec
    filters: list[Filter]
    interpretation: str
    confidence: str
    clarifying_question: str | None
    warnings: list[str]

    def summary(self) -> str:
        if not self.filters:
            return "No filter applied - using every row."
        return "Filtered to " + "; ".join(f.describe() for f in self.filters) + "."


def validate(raw: dict, df: pd.DataFrame, profiles: dict[str, ColumnProfile]) -> ParsedRequest:
    if not isinstance(raw, dict):
        raise SpecError("The chart request must be an object.")
    lookup = {str(c).strip().lower(): c for c in df.columns}
    if len(lookup) != len(df.columns):
        raise SpecError("Column names must be unique, including case-insensitive matches.")
    warnings: list[str] = []

    def column(name: str | None, label: str, required: bool) -> str | None:
        if name is not None and not isinstance(name, str):
            raise SpecError(f"The column for {label} must be a name.")
        if name is None or name.strip().lower() in ("none", "null", ""):
            if required:
                raise SpecError(f"The request did not identify a column for {label}.")
            return None
        resolved = lookup.get(str(name).strip().lower())
        if resolved is None:
            resolved = _resolve_name(str(name), df)
            if resolved is None:
                raise SpecError(f"Column '{name}' is not in this sheet.")
            warnings.append(f"Read '{name}' as the column '{resolved}'.")
        return resolved

    chart = str(raw.get("chart", "bar"))
    if chart not in CHART_KINDS:
        raise SpecError(f"'{chart}' is not a chart type this tool supports.")

    x = column(raw.get("x"), "the X axis", required=chart != "correlation")
    y = column(raw.get("y"), "the measure", required=False)
    color = column(raw.get("color"), "the split", required=False)

    agg = str(raw.get("agg", "sum"))
    if agg not in AGGS:
        raise SpecError(f"'{agg}' is not a supported aggregation.")

    if chart == "histogram":
        y = None

    grain = str(raw.get("grain", "auto"))
    if grain not in GRAINS:
        raise SpecError(f"'{grain}' is not a supported time grain.")
    extra = {"grain": None if grain == "auto" else grain,
             "orientation": raw.get("orientation", "auto"),
             "palette": raw.get("palette", "default"), "bins": raw.get("bins", 30)}
    if "top_n" in raw:
        extra["top_n"] = raw["top_n"]
    for option in ("colors", "category_colors"):
        if option in raw:
            extra[option] = raw[option]
    if chart == "correlation":
        if "columns" in raw:
            requested = raw["columns"]
            if not isinstance(requested, list):
                raise SpecError("Correlation columns must be a list of column names.")
            chosen = [column(c, "the correlation set", required=True) for c in requested]
        elif x is not None or y is not None:
            chosen = [c for c in (x, y) if c is not None]
        else:
            actual_profiles = profile_frame(df)
            chosen = [c for c in df.select_dtypes("number").columns
                      if not actual_profiles[c].is_identifier
                      and not (c in profiles and profiles[c].is_identifier)]
        extra["columns"] = list(dict.fromkeys(chosen))

    spec = ChartSpec(
        kind=chart,
        title=str(raw.get("title") or "Chart").strip(),
        x=x,
        y=y,
        color=color,
        agg=agg,
        rationale=str(raw.get("interpretation") or "").strip(),
        extra=extra,
    )
    try:
        _validate_spec(df, spec, profiles)
    except ValueError as exc:
        raise SpecError(str(exc)) from exc

    raw_filters = raw.get("filters", [])
    if not isinstance(raw_filters, list):
        raise SpecError("Filters must be a list.")
    filters = parse_filters(raw_filters, df, profiles, warnings)
    return ParsedRequest(
        spec=spec,
        filters=filters,
        interpretation=str(raw.get("interpretation") or "").strip(),
        confidence=str(raw.get("confidence") or "medium"),
        clarifying_question=(raw.get("clarifying_question") or None),
        warnings=warnings,
    )


def parse_filters(raw_filters, df: pd.DataFrame, profiles: dict[str, ColumnProfile],
                  warnings: list[str] | None = None) -> list[Filter]:
    """Check filter dictionaries against the real sheet before anything uses them."""
    if not isinstance(raw_filters, list):
        raise SpecError("Filters must be a list.")
    lookup = {str(c).strip().lower(): c for c in df.columns}
    if len(lookup) != len(df.columns):
        raise SpecError("Column names must be unique, including case-insensitive matches.")
    return [_filter(f, df, profiles, lookup, warnings if warnings is not None else []) for f in raw_filters]


def apply_filters(df: pd.DataFrame, filters: list[Filter]) -> pd.DataFrame:
    if not df.columns.is_unique:
        raise SpecError("Column names must be unique.")
    mask = pd.Series(True, index=df.index)
    for f in filters:
        if f.column not in df.columns:
            raise SpecError(f"Column '{f.column}' is not in this sheet.")
        checked = _typed_filter(f.column, f.op, f.value, df[f.column])
        mask &= _filter_mask(df[f.column], checked).fillna(False)
    return df.loc[mask]


def _filter(raw: dict, df, profiles, lookup, warnings) -> Filter:
    if not isinstance(raw, dict):
        raise SpecError("Each filter must be an object.")
    name = str(raw.get("column", "")).strip()
    col = lookup.get(name.lower())
    if col is None:
        raise SpecError(f"Filter refers to column '{name}', which is not in this sheet.")

    result = _typed_filter(col, str(raw.get("op", "")).strip(), raw.get("value", []), df[col])
    s = df[col]
    bounded = (pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
               or pd.api.types.is_datetime64_any_dtype(s))
    if bounded and result.value and s.notna().any() and not _filter_mask(s, result).any():
        warnings.append(f"Filter on '{col}' matches no rows in the sheet "
                        f"(observed range: {_fmt(s.min())} to {_fmt(s.max())}).")
    return result


def _typed_filter(col: str, op: str, values, series: pd.Series) -> Filter:
    if op not in OPS:
        raise SpecError(f"'{op}' is not a supported filter operator.")

    if pd.api.types.is_bool_dtype(series):
        role, allowed = "boolean", TEXT_OPS - {"contains"}
    elif pd.api.types.is_datetime64_any_dtype(series):
        role, allowed = "datetime", NUMERIC_OPS - {"above_mean", "below_mean"}
    elif pd.api.types.is_numeric_dtype(series):
        role, allowed = "numeric", NUMERIC_OPS
    else:
        role, allowed = "text", TEXT_OPS
    if op not in allowed:
        raise SpecError(f"Cannot apply '{op}' to {role} column '{col}'.")

    if not isinstance(values, list):
        raise SpecError(f"Filter values for '{col}' must be a list.")
    expected = ARITY.get(op, 1)
    if op in ("in", "not_in"):
        if not values:
            raise SpecError(f"Filter on '{col}' with '{op}' needs at least one value.")
    elif len(values) != expected:
        raise SpecError(f"Filter on '{col}' with '{op}' needs {expected} value(s), got {len(values)}.")

    coerced = [_coerce(v, series, col) for v in values]
    if role == "datetime" and op in ("between", "lte") and _date_only(values[-1]):
        try:
            coerced[-1] = coerced[-1].normalize() + pd.DateOffset(days=1) - pd.Timedelta(nanoseconds=1)
        except (ValueError, OverflowError) as exc:
            raise SpecError(f"Date upper bound for '{col}' is out of range.") from exc
    if op == "between" and coerced[0] > coerced[1]:
        raise SpecError(f"Filter on '{col}' needs two ordered between values.")
    return Filter(column=col, op=op, value=coerced)


def _filter_mask(series: pd.Series, f: Filter) -> pd.Series:
    if f.op == "is_null":
        return series.isna()
    if f.op == "not_null":
        return series.notna()
    text_values = not (pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)
                       or pd.api.types.is_datetime64_any_dtype(series))
    if text_values and f.op in ("in", "not_in", "eq", "neq"):
        # The model never sees values, so exact text matching ignores case only.
        folded = series.astype("string").str.casefold()
        wanted = [str(value).casefold() for value in f.value]
        mask = folded.isin(wanted)
        if f.op in ("not_in", "neq"):
            mask = ~mask
        return series.notna() & mask.fillna(False)
    if f.op in ("above_mean", "below_mean"):
        mean = series.mean()
        mask = series.gt(mean) if f.op == "above_mean" else series.lt(mean)
    elif f.op == "between":
        mask = series.between(f.value[0], f.value[1])
    elif f.op in ("in", "not_in"):
        mask = series.isin(f.value)
        if f.op == "not_in":
            mask = ~mask
    elif f.op == "contains":
        mask = series.astype("string").str.contains(f.value[0], case=False, regex=False, na=False)
    else:
        op = {"gte": "ge", "lte": "le", "gt": "gt", "lt": "lt", "eq": "eq", "neq": "ne"}[f.op]
        mask = getattr(series, op)(f.value[0])
    return series.notna() & mask.fillna(False)


def _date_only(value) -> bool:
    if isinstance(value, date) and not isinstance(value, datetime):
        return True
    if not isinstance(value, str):
        return False
    pattern = (r"\d{4}[/-]\d{1,2}[/-]\d{1,2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{8}|"
               r"[A-Za-z]+\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+[A-Za-z]+\s+\d{4}")
    return bool(re.fullmatch(pattern, value.strip()))


def _coerce(value, series: pd.Series, col: str):
    if not pd.api.types.is_scalar(value) or pd.isna(value):
        raise SpecError(f"Filter values for '{col}' must be non-null scalars.")
    if pd.api.types.is_datetime64_any_dtype(series):
        try:
            if not isinstance(value, (str, date, datetime, pd.Timestamp)):
                raise ValueError("Dates must be date strings or date objects.")
            parsed = pd.Timestamp(value)
            if pd.isna(parsed):
                raise ValueError("Missing date.")
            tz = series.dt.tz
            if tz is not None:
                return parsed.tz_localize(tz) if parsed.tzinfo is None else parsed.tz_convert(tz)
            if parsed.tzinfo is not None:
                raise ValueError("Timezone-aware value on a timezone-naive column.")
            return parsed
        except (ValueError, TypeError, OverflowError) as exc:
            raise SpecError(f"'{value}' is not a date that can be applied to '{col}'.") from exc
    if pd.api.types.is_bool_dtype(series):
        text = str(value).strip().lower()
        if text in ("true", "yes", "1"):
            return True
        if text in ("false", "no", "0"):
            return False
        raise SpecError(f"'{value}' is not a boolean that can be applied to '{col}'.")
    if pd.api.types.is_numeric_dtype(series):
        cleaned = str(value).replace(",", "").replace("$", "").strip()
        try:
            number = float(cleaned)
            if not math.isfinite(number):
                raise ValueError("Non-finite number.")
            return number
        except ValueError as exc:
            raise SpecError(f"'{value}' is not a number that can be applied to '{col}'.") from exc
    return str(value)


def _fmt(v) -> str:
    if isinstance(v, pd.Timestamp):
        return v.date().isoformat()
    if isinstance(v, float):
        return f"{v:,.2f}".rstrip("0").rstrip(".")
    return str(v)
