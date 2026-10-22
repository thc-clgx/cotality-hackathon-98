"""Bounded acceptance check of the current app flow, using approved test data."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

import app
from brd_insight_agent.export import figures_to_images
from brd_insight_agent.intent import apply_filters, validate
from brd_insight_agent.loader import clean_frame, load_sheet, workbook_sheet_names
from brd_insight_agent.privacy import PrivacyPolicy, default_allowed_columns
from brd_insight_agent.workflow import PLANNER_SCHEMA, PLANNER_SYSTEM, chart_specs, dataset_identity, fingerprint


def cases_for(frame, profiles):
    allowed = set(default_allowed_columns(profiles))
    numeric = [p.name for p in profiles.values() if p.role == "numeric" and p.name in allowed]
    categorical = [p.name for p in profiles.values() if p.role == "categorical" and p.name in allowed and p.unique > 1]
    if len(numeric) < 2 or not categorical:
        raise ValueError("This acceptance scenario needs two numeric measures and one non-ID category with multiple values.")
    measure, other, group = numeric[0], numeric[1], categorical[0]
    selected = str(frame[group].dropna().iloc[0])
    base = {"chart": "bar", "x": group, "y": measure, "agg": "mean", "title": f"Mean {measure} by {group}"}
    return measure, other, group, selected, [
        (f"Create a bar chart of average {measure} by {group} using all rows.",
         {"action": "create", "message": "Category comparison", "chart": base, "filters": []}),
        (f"Modify that chart: keep only rows where {group} equals {json.dumps(selected)}. Use #008080 for all bars; keep the measure and average unchanged.",
         {"action": "update", "message": "Filtered teal bars", "chart": {"colors": ["#008080"]},
          "filters": [{"column": group, "op": "eq", "value": [selected]}]}),
        (f"New image using all rows: show both a histogram of {measure} and a scatter plot of {other} against {measure}, as exactly two charts in one image.",
         {"action": "create", "message": "Distribution and relationship", "filters": [], "charts": [
             {"chart": "histogram", "x": measure, "agg": "count", "title": f"Distribution of {measure}"},
             {"chart": "scatter", "x": measure, "y": other, "agg": "mean", "title": f"{other} against {measure}"}]}),
    ]


def array_values(values):
    if isinstance(values, dict) and "bdata" in values:
        return np.frombuffer(base64.b64decode(values["bdata"]), dtype=values["dtype"]).tolist()
    return list(values)


def check_result(frame, artifact, index, measure, other, group, selected):
    specs = chart_specs(artifact.plan)
    expected = frame.loc[frame[group].astype(str).eq(selected)] if index == 1 else frame
    if index < 2:
        assert len(specs) == 1 and specs[0]["chart"] == "bar"
        assert specs[0]["x"] == group and specs[0]["y"] == measure and specs[0]["agg"] == "mean"
        parsed = validate({**{k: v for k, v in specs[0].items() if k not in {"theme", "show_legend"}}, "filters": artifact.plan["filters"]}, frame, app.profile_frame(frame))
        assert apply_filters(frame, parsed.filters).index.equals(expected.index), "Wrong filtered rows"
        expected_means = expected.groupby(group, observed=True)[measure].mean().dropna()
        trace = artifact.figure.data[0]
        labels = array_values(trace.y if trace.orientation == "h" else trace.x)
        values = array_values(trace.x if trace.orientation == "h" else trace.y)
        actual = dict(zip(map(str, labels), values))
        displayed = expected_means.loc[expected_means.abs().nlargest(specs[0].get("top_n", 15)).index]
        assert set(actual) == set(map(str, displayed.index)), "Displayed groups differ from the expected groups"
        for label, value in displayed.items():
            np.testing.assert_allclose(actual[str(label)], value)
        if index == 1:
            assert trace.marker.color.lower() in {"#008080", "teal", "rgb(0,128,128)"}, "Requested teal color was not applied"
    else:
        assert len(specs) == 2 and {s["chart"] for s in specs} == {"histogram", "scatter"}
        assert artifact.plan["filters"] == [], "New question retained old filters"
        histogram = next(s for s in specs if s["chart"] == "histogram")
        scatter = next(s for s in specs if s["chart"] == "scatter")
        assert histogram["x"] == measure
        assert {scatter["x"], scatter["y"]} == {measure, other}
    assert artifact.matched_rows == len(expected)


def evaluate(frame, identity, live=False, model=None, max_calls=3):
    if not 0 <= max_calls <= 3:
        raise ValueError("The live acceptance budget must be between zero and three calls.")
    profiles = app.profile_frame(frame)
    measure, other, group, selected, cases = cases_for(frame, profiles)
    data = {"id": identity, "frame": frame, "profiles": profiles}
    state = app.conversation({"datasets": {}}, identity)
    policy = PrivacyPolicy(tuple(default_allowed_columns(profiles)))
    chosen = model or (app.gemini.get_model() if live else "offline-mock")
    summary = {"mode": "live" if live else "offline-mock", "model": chosen,
               "contract": fingerprint([PLANNER_SCHEMA, PLANNER_SYSTEM]), "api_calls": 0, "steps": [], "complete": False}
    for index, (question, answer) in enumerate(cases):
        if live and summary["api_calls"] >= max_calls:
            summary["stop_reason"] = "Live-call budget reached; remaining cases were not evaluated."
            break
        if live:
            summary["api_calls"] += 1
            app.ask(state, data, question, True, policy, chosen)
        else:
            with patch.object(app.gemini, "request_json", return_value=answer):
                app.ask(state, data, question, True, policy, chosen)
        if state["messages"][-1].get("error") or state["pending"]:
            summary["stop_reason"] = state["messages"][-1]["text"]
            break
        artifact = app.current(state)
        try:
            check_result(frame, artifact, index, measure, other, group, selected)
        except (AssertionError, ValueError, KeyError) as exc:
            summary["stop_reason"] = f"Case {index + 1} failed: {exc}"
            break
        summary["steps"].append({"question": question, "matched_rows": artifact.matched_rows,
                                 "panel_count": len(chart_specs(artifact.plan))})
        print("PASS", question, flush=True)
    summary["complete"] = len(summary["steps"]) == len(cases)
    return summary, app.current(state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", default="testdata/retail_sales.xlsx")
    parser.add_argument("--sheet")
    parser.add_argument("--header", type=int, default=0)
    parser.add_argument("--live", action="store_true", help="Send approved synthetic/test metadata to the configured model.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-live-calls", type=int, choices=range(4), default=3)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    path = Path(args.workbook)
    if not path.is_absolute():
        path = root / path
    content = path.read_bytes()
    sheet = args.sheet or workbook_sheet_names(content, path.name)[0]
    frame = clean_frame(load_sheet(content, path.name, sheet, args.header))
    summary, artifact = evaluate(frame, dataset_identity(content, sheet, args.header), args.live, args.model, args.max_live_calls)
    output = root / ".validation"
    output.mkdir(exist_ok=True)
    if args.render and summary["complete"]:
        summary["exports"] = {}
        for fmt in ("png", "svg", "jpeg"):
            started = time.perf_counter()
            result = figures_to_images([("Acceptance image", artifact.figure)], fmt, 1280, 720, 1, 30)
            (output / f"current_workflow.{fmt}").write_bytes(result[0][1])
            summary["exports"][fmt] = {"seconds": round(time.perf_counter() - started, 2), "bytes": len(result[0][1])}
    (output / "current_workflow.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())