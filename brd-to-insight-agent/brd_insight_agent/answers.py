"""Answer factual questions about the selected sheet with pandas, never with the model."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

CHART_WORDS = re.compile(r"\b(chart|plot|graph|image|histogram|scatter|pie|bar|line|box|visuali[sz]e|draw|by\b)")
# Only genuine questions are answered locally; instructions stay with the chart planner.
QUESTION = re.compile(r"^(what|which|when|who|how many|how much|how far|is there|are there|do we have|"
                      r"tell me|list the)\b")
MEASURES = {"largest": "max", "highest": "max", "maximum": "max", "max": "max", "biggest": "max", "smallest": "min",
            "lowest": "min", "minimum": "min", "min": "min", "average": "mean", "mean": "mean", "median": "median",
            "total": "sum", "sum": "sum"}


def _mentioned(question, frame, profiles=None):
    """Columns named in the question, longest name first so 'NetSales' wins over 'Sales'."""
    from .profiler import _tokens, singular

    text = question.casefold()
    found = [name for name in frame.columns if str(name).casefold() in text]
    if not found:
        asked = {singular(token) for token in _tokens(text)}
        found = [name for name in frame.columns
                 if {singular(token) for token in _tokens(str(name))} & asked]
        if len(found) > 1 and profiles is not None:
            labels = [name for name in found if profiles[name].role not in {"numeric", "datetime"}]
            found = labels or found
        if len(found) != 1:
            return []
    return sorted(found, key=lambda name: -len(str(name)))


def _format(value):
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value).date())
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):,.2f}".rstrip("0").rstrip(".")
    return str(value)


def _dates(question, frame, profiles):
    named = [name for name in _mentioned(question, frame) if profiles[name].role == "datetime"]
    columns = named or [name for name in frame.columns if profiles[name].role == "datetime"]
    answers = []
    for name in columns[:3]:
        values = frame[name].dropna()
        if values.empty:
            continue
        stamps = pd.DatetimeIndex(values)
        quarters = sorted({f"{stamp.year} Q{stamp.quarter}" for stamp in stamps})
        answers.append(f"{name} runs from {stamps.min().date()} to {stamps.max().date()} "
                       f"across {len(quarters)} quarter(s): {', '.join(quarters)}.")
    return answers


def data_answer(question, frame, profiles, scope_text=None):
    """Return an explain response when the question asks for a fact this sheet can state."""
    text = " ".join(question.strip().casefold().split()).rstrip("?.! ")
    if not text or len(text) > 200:
        return None
    if "?" not in question and not QUESTION.match(text):
        return None
    scope = f" This applies to the selected sheet; the current image covers {scope_text}." if scope_text else ""

    if re.search(r"\b(date range|time range|range of dates|which quarters|what quarters|which periods|"
                 r"what period|which years|what years|how far back|earliest|latest|oldest|newest)\b", text):
        answers = _dates(question, frame, profiles)
        if answers:
            return {"action": "explain", "message": " ".join(answers) + scope}
        return {"action": "explain", "message": "This sheet has no date column, so it has no date range."}

    if re.fullmatch(r"(?:how many|what(?:'s| is) the number of|count(?: the)?)\s+(?:rows|records|entries|data points)"
                    r"(?:\s+(?:are there|do we have|does (?:it|this) have|are|is)?\s*"
                    r"(?:in|on)?\s*(?:the |this )?(?:sheet|file|data|tab|workbook)?)?", text):
        return {"action": "explain", "message": f"The selected sheet has {len(frame):,} nonblank rows."}

    if re.search(r"\b(what columns|which columns|what fields|what data do (?:we|i) have|what can (?:i|we) chart)\b", text):
        described = [f"{name} ({profiles[name].role})" for name in frame.columns]
        return {"action": "explain",
                "message": f"The sheet has {len(frame):,} rows and these columns: {', '.join(described)}.",
                # The reply names every column, including ones withheld from sharing, so the
                # conversation context carries the count instead of the names.
                "context_text": f"Listed the selected sheet's {len(frame.columns)} column names and types for the user."}

    match = re.search(r"\b(?:how many|number of|count of|distinct|unique)\s+(?:distinct |unique |different )?(.+)", text)
    if match and not CHART_WORDS.search(text):
        for name in _mentioned(match[1], frame, profiles):
            if profiles[name].role != "datetime":
                return {"action": "explain",
                        "message": f"{name} has {frame[name].nunique():,} distinct values in {len(frame):,} rows.{scope}"}

    word = re.search(r"\b(largest|highest|maximum|max|biggest|smallest|lowest|minimum|min|average|mean|median|total|sum)\b", text)
    if word and not CHART_WORDS.search(text):
        for name in _mentioned(question, frame, profiles):
            profile = profiles[name]
            if profile.is_identifier or profile.role not in {"numeric", "datetime"}:
                continue
            how = MEASURES[word[1]]
            values = frame[name].dropna()
            if profile.role == "datetime" and how not in {"min", "max"}:
                continue
            if values.empty:
                return {"action": "explain", "message": f"{name} has no usable values in this sheet."}
            if how == "sum" and profile.role == "numeric" and _rate_like(name):
                return {"action": "clarify",
                        "message": f"Adding {name} would be misleading because it looks like a rate or score. "
                                   f"Shall I report its average instead?"}
            result = getattr(values, how)()
            return {"action": "explain",
                    "message": f"The {word[1]} {name} across {len(values):,} rows with a value is {_format(result)}.{scope}"}
    return None


def _rate_like(name):
    from .charts import is_rate_like

    return is_rate_like(name)
