"""Classify each column so the recommender knows what it can plot."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
from pandas.api import types as ptypes

LOW_CARDINALITY = 25
GROUPABLE_CODES = 500
KEY_TOKENS = {"id", "ids", "code", "codes", "key", "uuid", "guid", "sku", "number", "num", "no", "ref"}
RATE_TOKENS = {"pct", "percent", "percentage", "rate", "rating", "ratio", "score",
               "margin", "avg", "average", "mean", "per"}


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    role: str  # numeric | datetime | categorical | boolean | freetext
    unique: int
    missing_pct: float
    is_identifier: bool


def profile_frame(df: pd.DataFrame) -> dict[str, ColumnProfile]:
    n = max(len(df), 1)
    profiles: dict[str, ColumnProfile] = {}

    for col in df.columns:
        s = df[col]
        unique = int(s.nunique(dropna=True))
        missing = float(s.isna().mean() * 100)

        if ptypes.is_bool_dtype(s):
            role = "boolean"
        elif ptypes.is_datetime64_any_dtype(s):
            role = "datetime"
        elif ptypes.is_numeric_dtype(s):
            role = "numeric"
        elif unique <= LOW_CARDINALITY or unique / n <= 0.3:
            role = "categorical"
        else:
            role = "freetext"

        # Uniqueness alone is not ID evidence: revenue and timestamps can be
        # unique too. Key names apply to strings as well as repeated numeric IDs.
        identifier = bool(
            s.notna().any()
            and role not in {"datetime", "boolean"}
            and _looks_like_key(col)
        )
        if role == "numeric" and identifier:
            role = "freetext" if unique > LOW_CARDINALITY else "categorical"
        # Codes such as FIPS or postal codes label rows, so they can group a chart when they repeat.
        if role == "freetext" and identifier and 1 < unique <= GROUPABLE_CODES and unique < n:
            role = "categorical"

        profiles[col] = ColumnProfile(col, role, unique, missing, identifier)

    return profiles


def singular(token: str) -> str:
    """Crude singular form so 'counties' and 'county' match the same column."""
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ses") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def by_role(profiles: dict[str, ColumnProfile], role: str) -> list[ColumnProfile]:
    return [p for p in profiles.values() if p.role == role]


def _name_tokens(name: str) -> list[str]:
    # Split camelCase and ACRONYMWord boundaries *before* lowercasing, otherwise
    # there are no capitals left to split on and every name becomes one token.
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(name))
    spaced = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", spaced)
    return [t for t in re.split(r"[^a-z0-9]+", spaced.lower()) if t]


def _tokens(name: str) -> set[str]:
    return set(_name_tokens(name))


def _looks_like_key(name: str) -> bool:
    # Token match, not substring: "Width" and "Paid" contain "id" but are not keys.
    tokens = [t for t in _name_tokens(name) if not t.isdigit()]
    number_tokens = {"number", "num", "no"}
    if set(tokens) & (KEY_TOKENS - number_tokens):
        return True
    # 'OrderNumber' is credible ID evidence; 'NumberOfOrders', 'number' and
    # 'RenumberedTotal' are not. Ignore generated numeric header suffixes.
    return len(tokens) > 1 and tokens[-1] in number_tokens


def is_rate_like(name: str) -> bool:
    """Name-only rate/score heuristic, not a guarantee that a mean is valid.

    Correct aggregation may require weights, denominators or domain context.
    """
    if "%" in str(name):
        return True
    return bool(_tokens(name) & RATE_TOKENS)
