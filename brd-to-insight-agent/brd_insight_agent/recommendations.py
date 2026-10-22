"""Local, dataset-specific starting options, checked by the real chart pipeline."""

from __future__ import annotations

import copy
import re

import numpy as np
import pandas as pd

from .workflow import build_artifact, fingerprint

MAX_OPTIONS_PER_TYPE = 2

# Each chart type with the data it needs before this sheet can offer it.
TYPES = {
    "bar": ("Bar chart", "grouping column with 2-40 categories"),
    "grouped_bar": ("Grouped bar chart", "second grouping column with at most 8 categories"),
    "histogram": ("Histogram", "numeric column with at least two finite values"),
    "line": ("Line chart", "date column with at least two distinct dates"),
    "scatter": ("Scatter plot", "two numeric columns that vary across at least two shared rows"),
    "pie": ("Pie chart", "grouping column with 2-8 categories"),
    "box": ("Box plot", "numeric column together with a grouping column"),
}
ORDINALS = ["first", "second", "third", "fourth", "fifth"]


def _usable_columns(frame, profiles):
    numeric, dates, categories, codes = [], [], [], []
    for name in frame.columns:
        profile = profiles[name]
        series = frame[name]
        if profile.role in {"categorical", "boolean"} and not profile.is_identifier and 2 <= profile.unique <= 40:
            categories.append(name)
            continue
        if profile.role == "categorical" and profile.is_identifier and 2 <= profile.unique <= 500:
            codes.append(name)
            continue
        if profile.is_identifier:
            continue
        if profile.role == "numeric" and not pd.api.types.is_complex_dtype(series):
            finite = np.isfinite(series.to_numpy(dtype=float, na_value=np.nan))
            if finite.sum() >= 2:
                numeric.append(name)
        elif profile.role == "datetime" and series.nunique() >= 2:
            dates.append(name)
    numeric.sort(key=lambda name: profiles[name].missing_pct)
    categories.sort(key=lambda name: (profiles[name].unique > 15, profiles[name].missing_pct, profiles[name].unique))
    return numeric, dates, categories, codes


def _candidates(frame, profiles, numeric, dates, categories, codes):
    plans = {kind: [] for kind in TYPES}
    if not categories and codes and numeric:
        group, measure = codes[0], numeric[0]
        plans["bar"].append((f"Top {measure} by {group}",
                             f"Bar chart of the largest {group} values; the display keeps the top 15.",
                             {"chart": "bar", "x": group, "y": measure, "agg": "sum", "top_n": 15}))
    if categories:
        group = categories[0]
        plans["bar"].append((f"Record counts by {group}",
                             "Bar chart comparing group sizes; no numeric measure required.",
                             {"chart": "bar", "x": group, "agg": "count"}))
        if numeric:
            plans["bar"].append((f"Median {numeric[0]} by {group}",
                                 "Bar chart comparing typical values per group; median, not a sum.",
                                 {"chart": "bar", "x": group, "y": numeric[0], "agg": "median"}))
            plans["box"].append((f"{numeric[0]} spread by {group}",
                                 "Box plot comparing distributions rather than totals.",
                                 {"chart": "box", "x": group, "y": numeric[0], "agg": "median"}))
        if profiles[group].unique <= 8:
            plans["pie"].append((f"Share of records by {group}",
                                 "Pie chart of record share; each slice is a count, not a total.",
                                 {"chart": "pie", "x": group, "agg": "count"}))
    if len(categories) >= 2 and profiles[categories[1]].unique <= 8:
        plans["grouped_bar"].append((f"Record counts by {categories[0]} split by {categories[1]}",
                                     "Grouped bar chart comparing counts across two grouping columns.",
                                     {"chart": "grouped_bar", "x": categories[0], "color": categories[1], "agg": "count"}))
    for measure in numeric[:MAX_OPTIONS_PER_TYPE]:
        plans["histogram"].append((f"Distribution of {measure}",
                                   f"Histogram showing the spread of finite {measure} values.",
                                   {"chart": "histogram", "x": measure, "y": None, "agg": "count"}))
    if len(numeric) >= 2:
        first, second = numeric[:2]
        pair = frame[[first, second]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(pair) >= 2 and pair[first].nunique() >= 2 and pair[second].nunique() >= 2:
            plans["scatter"].append((f"{second} versus {first}",
                                     "Scatter plot for a possible relationship; it does not establish causation.",
                                     {"chart": "scatter", "x": first, "y": second, "agg": "mean"}))
    if dates:
        date = dates[0]
        plans["line"].append((f"Record counts over {date}",
                              f"Line chart counting records by {date}; not an inferred event or sales date.",
                              {"chart": "line", "x": date, "agg": "count", "grain": "auto"}))
        if numeric:
            plans["line"].append((f"Median {numeric[0]} over {date}",
                                  f"Line chart of typical {numeric[0]} per period; median, not a running total.",
                                  {"chart": "line", "x": date, "y": numeric[0], "agg": "median", "grain": "auto"}))
    return plans


def _validated(frame, dataset_id, kind, entries):
    options = []
    for title, reason, chart in entries[:MAX_OPTIONS_PER_TYPE]:
        if len(title) > 180:
            title = f"{TYPES[kind][0]} of selected columns"
        response = {"action": "create", "message": reason, "chart": {**chart, "title": title}, "filters": []}
        plan = {"kind": "chart", "chart": response["chart"], "filters": []}
        try:
            artifact = build_artifact(frame, dataset_id, plan)
        except ValueError:
            continue
        facts = artifact.figure.layout.meta or {}
        notes = []
        eligible = facts.get("eligible_rows", len(frame))
        if eligible < len(frame):
            notes.append(f"{len(frame) - eligible:,} rows lack required usable values.")
        if facts.get("sampled"):
            notes.append(f"Display samples {facts['drawn_rows']:,} of {eligible:,} eligible rows.")
        if facts.get("omitted_categories"):
            notes.append(f"Display omits {len(facts['omitted_categories'])} categories at its default limit.")
        if facts.get("merged_categories"):
            notes.append(f"Display merges {len(facts['merged_categories'])} smaller categories into one slice.")
        if facts.get("grain"):
            notes.append(f"Time grouping: {facts['grain']}.")
        options.append({"id": fingerprint([dataset_id, plan]), "label": title, "reason": reason,
                        "notes": notes, "response": response})
    return options


def recommend_visuals(frame, profiles, dataset_id):
    """Chart types this sheet can actually produce, each with validated suggestions."""
    numeric, dates, categories, codes = _usable_columns(frame, profiles)
    candidates = _candidates(frame, profiles, numeric, dates, categories, codes)
    types, unavailable = [], []
    for kind, (label, requirement) in TYPES.items():
        options = _validated(frame, dataset_id, kind, candidates[kind])
        if options:
            types.append({"chart": kind, "label": label, "options": options})
        else:
            unavailable.append({"chart": kind, "label": label, "requirement": requirement})
    limitations = []
    if frame.isna().any().any():
        limitations.append("Missing values are excluded where a chart requires them; exclusions are disclosed per option.")
    if not types:
        limitations.append("Check the selected sheet/header and Correct column format before charting. No chart type could be validated.")
    return {"dataset_id": dataset_id, "types": types, "unavailable": unavailable,
            "limitations": limitations, "selected_type": None}


def group_for(bundle, chart_type):
    return next((group for group in bundle.get("types", []) if group["chart"] == chart_type), None)


def active_options(bundle):
    group = group_for(bundle, bundle.get("selected_type"))
    return group["options"] if group else []


def introduction(bundle):
    """First message: only the chart types this data supports, and why the rest are not offered."""
    types = bundle["types"]
    lines = ["I read the complete selected sheet locally. This data supports these chart types:" if types else
             "I read the complete selected sheet locally, but it cannot produce any supported chart type yet."]
    for group in types:
        lines.append(f"{group['label']}: {len(group['options'])} suggestion(s) built from this sheet.")
    for item in bundle["unavailable"]:
        lines.append(f"{item['label']} is unavailable: this sheet has no {item['requirement']}.")
    lines.extend(bundle["limitations"])
    if types:
        lines.append("Choose a chart type below to see suggestions for this data, or just ask for the chart you want.")
    context = ["Chart types this sheet supports: " + (", ".join(group["label"] for group in types) or "none") + "."]
    return {"role": "assistant", "text": "\n\n".join(lines), "context_text": "\n".join(context),
            "recommendation_types": bundle["dataset_id"]}


def type_message(bundle, chart_type, allowed_columns=None):
    group = group_for(bundle, chart_type)
    if group is None:
        return None
    lines = [f"{group['label']} suggestions validated against this sheet:"]
    context = [f"{group['label']} recommendations for the selected sheet:"]
    for number, option in enumerate(group["options"], 1):
        lines.append(f"{number}. {option['label']} - {option['reason']} " + " ".join(option["notes"]))
        chart = option["response"]["chart"]
        columns = {chart[field] for field in ("x", "y", "color") if chart.get(field)}
        if allowed_columns is None or columns.issubset(allowed_columns):
            context.append(f"Recommendation {number}: {option['label']}. {option['reason']}")
    lines.append('Select one below, say "the second one", or ask for a different chart. '
                 "Each selection starts a new chart using whatever scope the filter panel has set.")
    return {"role": "assistant", "text": "\n\n".join(lines), "context_text": "\n".join(context),
            "recommendations_for": bundle["dataset_id"], "recommendation_options": chart_type}


def select_recommendation(question, bundle, messages, pending=None):
    """Resolve only exact option references, leaving modified requests to normal chat."""
    text = question.strip().casefold().rstrip(".! ")
    explicit = re.fullmatch(r"(?:(?:show|create|use|choose|select|generate)\s+)?(?:recommendation|option|suggestion)\s+(\d+)", text)
    last_assistant = next((message for message in reversed(messages) if message["role"] == "assistant"), {})
    if explicit:
        number = int(explicit[1])
    elif not pending and last_assistant.get("recommendations_for") == bundle["dataset_id"]:
        ordinal = re.fullmatch(r"(?:(?:show|create|use|choose|select|generate)\s+)?(?:the\s+)?(first|second|third|fourth|fifth)(?:\s+(?:one|option|chart))?", text)
        if not ordinal:
            return None
        number = ORDINALS.index(ordinal[1]) + 1
    else:
        return None
    options = active_options(bundle)
    if not options:
        return {"action": "clarify", "message": "Choose a chart type first, then I can list suggestions you can select by number."}
    if not 1 <= number <= len(options):
        return {"action": "clarify", "message": f"Choose one of the {len(options)} listed recommendations, or describe the chart you want."}
    return copy.deepcopy(options[number - 1]["response"])