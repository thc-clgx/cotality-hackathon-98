"""Run scripted chat scenarios against the real planner and check results with pandas.

Every expectation is computed locally from the workbook, never from the model reply.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

import app
from brd_insight_agent import gemini
from brd_insight_agent.loader import clean_frame, detect_header_row, load_sheet
from brd_insight_agent.privacy import PrivacyPolicy, default_allowed_columns
from brd_insight_agent.profiler import profile_frame
from brd_insight_agent.workflow import chart_specs

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "testdata"
BRUSHFIRE = DATA / "OneDrive_2026-09-15" / "Brushfire Change Logs"
LOGS_2025 = BRUSHFIRE / "202504" / "BrushfireChangeLog" / "BrushfireChangeLog"
LOGS_2026 = BRUSHFIRE / "202604" / "BrushfireChangeLog"


def between(column, start, end):
    return lambda frame: frame[(frame[column] >= start) & (frame[column] <= f"{end} 23:59:59.999999999")]


def equals(column, value):
    return lambda frame: frame[frame[column].astype(str) == value]


SCENARIOS = [
    ("retail_sales.xlsx", "Sales", [
        ("Bar chart of total NetSales by Category.", {"kinds": ["bar"], "x": "Category", "y": "NetSales", "rows": None}),
        ("Now only the first quarter of 2025.", {"kinds": ["bar"], "rows": between("SaleDate", "2025-01-01", "2025-03-31")}),
        ("Make it horizontal.", {"kinds": ["bar"], "rows": between("SaleDate", "2025-01-01", "2025-03-31")}),
    ]),
    ("retail_sales.xlsx", "Sales", [
        ("Show total NetSales by month for 2025.", {"kinds": ["line", "bar"], "rows": between("SaleDate", "2025-01-01", "2025-12-31")}),
        ("What date range does this data cover?", {"action": "explain", "no_invented_dates": "SaleDate"}),
        ("Restrict it to the last quarter of 2025.", {"rows": between("SaleDate", "2025-10-01", "2025-12-31")}),
    ]),
    ("retail_sales.xlsx", "Sales", [
        ("Histogram of UnitPrice.", {"kinds": ["histogram"], "x": "UnitPrice", "rows": None}),
        ("Scatter of NetSales against UnitsSold.", {"kinds": ["scatter"], "x": "UnitsSold", "y": "NetSales"}),
        ("Predict NetSales for next year.", {"action": ["explain", "clarify"]}),
    ]),
    ("retail_sales.xlsx", "Sales", [
        ("Pie chart of record share by ReturnFlag.", {"kinds": ["pie", "donut"], "x": "ReturnFlag"}),
        ("Only returned items.", {"rows": equals("ReturnFlag", "True")}),
        ("Sum the CustomerRating by Store.", {"action": ["image", "clarify", "error"], "no_rate_sum": "CustomerRating"}),
    ]),
    ("healthcare_claims.xlsx", "Claims", [
        ("Line chart of average ClaimAmount by quarter.", {"kinds": ["line"], "rows": None}),
        ("Which quarters does this chart cover?", {"action": "explain", "no_invented_dates": "ServiceDate"}),
        ("Focus on April to July 2025.", {"rows": between("ServiceDate", "2025-04-01", "2025-07-31")}),
        ("Now show the same for 2024 instead.", {"rows": between("ServiceDate", "2024-01-01", "2024-12-31")}),
    ]),
    ("healthcare_claims.xlsx", "Claims", [
        ("Compare median ClaimAmount for in network versus out of network.", {"kinds": ["bar", "box"], "x": "InNetwork"}),
        ("Add the PaidAmount as a second chart in the same image.", {"panels": 2}),
        ("Use a dark theme.", {"panels": 2}),
    ]),
    ("healthcare_claims.xlsx", "Claims", [
        ("Box plot of ReimbursementRate by DiagnosisGroup.", {"kinds": ["box"], "y": "ReimbursementRate"}),
        ("Total the ReimbursementRate per provider.", {"action": "clarify", "offer": "average"}),
    ]),
    ("finance_ledger.xlsx", "Ledger", [
        ("Bar chart of total Amount by Account.", {"kinds": ["bar"], "x": "Account", "y": "Amount"}),
        ("Only the third quarter of 2024.", {"rows": between("PostingDate", "2024-07-01", "2024-09-30")}),
        ("Only debit rows.", {"rows": lambda frame: frame[(frame.PostingDate >= "2024-07-01") & (frame.PostingDate <= "2024-09-30 23:59:59.999999999") & (frame.DebitCredit.astype(str) == "Debit")]}),
    ]),
    ("finance_ledger.xlsx", "Ledger", [
        ("Show Amount by FiscalQuarter.", {"kinds": ["bar", "line", "box"], "x": "FiscalQuarter"}),
        ("What is the largest single Amount in the sheet?", {"action": "explain", "states_value": ("Amount", "max")}),
    ]),
    ("hr_employees.xlsx", "Employees", [
        ("Median Salary by Department.", {"kinds": ["bar", "box"], "x": "Department", "y": "Salary"}),
        ("Only employees hired in 2024.", {"rows": between("HireDate", "2024-01-01", "2024-12-31")}),
        ("Split it by JobLevel as well.", {"rows": between("HireDate", "2024-01-01", "2024-12-31")}),
        ("Remove all filters.", {"rows": None}),
    ]),
    ("hr_employees.xlsx", "Employees", [
        ("Show attrition rate by Location.", {"action": ["image", "clarify"]}),
        ("Histogram of TenureYears.", {"kinds": ["histogram"], "x": "TenureYears"}),
    ]),
    ("marketing_campaigns.xlsx", "Campaigns", [
        ("Bar chart of total Spend by Channel.", {"kinds": ["bar"], "x": "Channel", "y": "Spend"}),
        ("Sum the ConversionRate by Channel.", {"action": "clarify", "offer": "average"}),
        ("Average ClickThroughRate by Channel instead.", {"kinds": ["bar", "box", "line"], "x": "Channel"}),
        ("Only campaigns in the second half of 2025.", {"rows": between("CampaignDate", "2025-07-01", "2025-12-31")}),
    ]),
    ("property_listings.xlsx", "Listings", [
        ("Median ListPrice by PropertyType.", {"kinds": ["bar", "box"], "x": "PropertyType", "y": "ListPrice"}),
        ("Only listings above the average ListPrice.", {"rows": lambda frame: frame[frame.ListPrice > frame.ListPrice.mean()]}),
        ("Show me a beautiful image summarising this data for customers.", {"panels_min": 2}),
    ]),
    ("supply_chain.xlsx", "Shipments", [
        ("Average LeadTimeDays by Supplier.", {"kinds": ["bar", "box"], "x": "Supplier", "y": "LeadTimeDays"}),
        ("Only late shipments.", {"action": ["image", "clarify"]}),
        ("Only rows where OnTimeFlag is false.", {"rows": equals("OnTimeFlag", "False")}),
    ]),
    ("ecommerce_orders.xlsx", "Orders", [
        ("Line chart of order counts by week for 2025.", {"kinds": ["line"], "rows": between("OrderDate", "2025-01-01", "2025-12-31")}),
        ("Break it down by Channel.", {"rows": between("OrderDate", "2025-01-01", "2025-12-31")}),
        ("Join this with the Catalog sheet.", {"action": ["explain", "clarify"]}),
    ]),
    ("iot_sensors.xlsx", "Readings", [
        ("Average TemperatureC over time by SiteName.", {"kinds": ["line"], "y": "TemperatureC"}),
        ("Only readings from the third quarter of 2025.", {"rows": between("Timestamp", "2025-07-01", "2025-09-30")}),
        ("Only sensors with BatteryPct below 20.", {"rows": lambda frame: frame[(frame.Timestamp >= "2025-07-01") & (frame.Timestamp <= "2025-09-30 23:59:59.999999999") & (frame.BatteryPct < 20)]}),
    ]),
    ("messy_survey.xlsx", "Responses", [
        ("Bar chart of responses by column_3.", {"kinds": ["bar"], "x": "column_3"}),
        ("Show the average score.", {"action": ["clarify", "explain", "image"]}),
    ]),
    ("retail_sales.xlsx", "Sales", [
        ("Show NetSales by City with a blue palette.", {"kinds": ["bar", "line", "box"], "x": "City"}),
        ("Make the bars teal.", {"colors": ["teal", "#008080", "rgb(0,128,128)"]}),
        ("Top 5 cities only.", {"top_n": 5}),
    ]),
]


def load(name, sheet, header=None):
    path = name if isinstance(name, Path) else DATA / name
    content = path.read_bytes()
    if header is None:
        header = detect_header_row(content, path.name, sheet)
    frame = clean_frame(load_sheet(content, path.name, sheet, header))
    return frame, profile_frame(frame)


def risk(value):
    return lambda frame: frame[frame["Risk Value"] == value]


def where(column, op, value):
    return lambda frame: frame[getattr(frame[column], op)(value)]


NO_ERROR = ["image", "clarify", "explain"]

# Realistic questions the team asked during the review of these change logs.
BRUSHFIRE_SCENARIOS = [
    (LOGS_2026 / "AK_Change_Log.xlsx", "Sheet1", None, [
        ("Show me what this file contains.", {"action": NO_ERROR}),
        ("Bar chart of Acreage Change by County FIPS Code.", {"action": NO_ERROR}),
        ("Which columns do we have?", {"action": "explain"}),
        ("Chart the New Acres against Previous Acres.", {"action": NO_ERROR}),
        ("How many rows are in this sheet?", {"action": "explain"}),
    ]),
    (LOGS_2026 / "AK_Change_Log.xlsx", "Sheet1", 3, [
        ("Which columns do we have?", {"action": "explain"}),
        ("How many rows are in this sheet?", {"action": "explain"}),
        ("Bar chart of total Acreage Change by Risk Value.", {"kinds": ["bar"], "x": "Risk Value", "rows": None}),
        ("Scatter of New Acres against Previous Acres.", {"kinds": ["scatter"], "x": "Previous Acres", "y": "New Acres"}),
        ("Only risk value 3.", {"rows": risk(3)}),
        ("Histogram of Acreage Change.", {"kinds": ["histogram"], "x": "Acreage Change"}),
        ("What is the largest Acreage Change?", {"action": "explain", "states_value": ("Acreage Change", "max")}),
        ("What is the total Acreage Change?", {"action": "explain", "states_value": ("Acreage Change", "sum")}),
        ("Show counties where acreage decreased.", {"action": NO_ERROR}),
        ("Box plot of Net % Change in County by Risk Value.", {"kinds": ["box"], "x": "Risk Value"}),
        ("Compare New County % against Previous County % for each risk value.", {"action": NO_ERROR}),
        ("Show the difference between New Acres and Previous Acres by county.", {"action": NO_ERROR}),
    ]),
    (LOGS_2026 / "AZ_Change_Log.xlsx", "Sheet1", 3, [
        ("Bar chart of total New Acres by Risk Value.", {"kinds": ["bar"], "x": "Risk Value", "y": "New Acres"}),
        ("Now only rows where Acreage Change is positive.", {"rows": where("Acreage Change", "gt", 0)}),
        ("Which county had the biggest increase in acres?", {"action": NO_ERROR}),
        ("Top 10 counties by Acreage Change.", {"action": NO_ERROR}),
        ("Pie chart of New Acres share by Risk Value.", {"kinds": ["pie", "donut"]}),
        ("What is the average Net % Change in County?", {"action": "explain", "states_value": ("Net % Change in County", "mean")}),
        ("How many distinct counties are in this file?", {"action": "explain"}),
        ("Show risk value 1 and 2 only.", {"action": NO_ERROR}),
        ("Make it horizontal.", {"action": NO_ERROR}),
        ("Use a dark theme.", {"action": NO_ERROR}),
    ]),
    (LOGS_2025 / "AZ_Change_Log.xlsx", "Sheet1", 3, [
        ("Bar chart of total Acreage Change by Risk Value.", {"kinds": ["bar"], "x": "Risk Value"}),
        ("What is the total Acreage Change?", {"action": "explain", "states_value": ("Acreage Change", "sum")}),
        ("Only rows where the acreage went down.", {"rows": where("Acreage Change", "lt", 0)}),
        ("Histogram of Net % Change in County.", {"kinds": ["histogram"]}),
        ("What is the minimum Previous Acres?", {"action": "explain", "states_value": ("Previous Acres", "min")}),
        ("Show New Acres and Previous Acres as two charts in one image.", {"panels": 2}),
        ("Which risk value has the most rows?", {"action": NO_ERROR}),
        ("Summarise this file in one image for a customer.", {"action": NO_ERROR}),
    ]),
    (LOGS_2026 / "CA_Change_Log.xlsx", "Sheet1", 3, [
        ("Bar chart of average Net % Change in County by Risk Value.", {"kinds": ["bar"], "x": "Risk Value"}),
        ("Only counties with more than 1000 acres of change.", {"rows": where("Acreage Change", "gt", 1000)}),
        ("What is the maximum New Acres?", {"action": "explain", "states_value": ("New Acres", "max")}),
        ("Scatter of Acreage Change against Previous Acres.", {"kinds": ["scatter"]}),
        ("Colour the points by Risk Value.", {"action": NO_ERROR}),
        ("Remove all filters.", {"action": NO_ERROR}),
        ("Show the total New Acres for risk value 3.", {"action": NO_ERROR}),
        ("Box plot of Acreage Change by Risk Value.", {"kinds": ["box"]}),
        ("What percentage of rows have a negative change?", {"action": NO_ERROR}),
        ("Sum the New County % by Risk Value.", {"action": ["clarify", "image", "explain"]}),
    ]),
    (LOGS_2026 / "TX_Change_Log.xlsx", "Sheet1", 3, [
        ("Bar chart of total Previous Acres by Risk Value.", {"kinds": ["bar"], "x": "Risk Value"}),
        ("Now show New Acres instead.", {"action": NO_ERROR}),
        ("Which county has the largest New Acres?", {"action": NO_ERROR}),
        ("Only risk value 4 and 5.", {"action": NO_ERROR}),
        ("What is the median Acreage Change?", {"action": "explain", "states_value": ("Acreage Change", "median")}),
        ("Histogram of New County %.", {"kinds": ["histogram"]}),
        ("Scatter New County % against Previous County %.", {"kinds": ["scatter"]}),
        ("Show me the top 5 counties by acreage change.", {"action": NO_ERROR}),
        ("Make the bars teal.", {"action": NO_ERROR}),
        ("Title it Texas brushfire change.", {"action": NO_ERROR}),
    ]),
    (LOGS_2026 / "WA_Change_Log.xlsx", "Sheet1", 3, [
        ("Bar chart of Acreage Change by Risk Value.", {"kinds": ["bar"]}),
        ("Only positive changes.", {"rows": where("Acreage Change", "gt", 0)}),
        ("What is the total New Acres?", {"action": "explain", "states_value": ("New Acres", "sum")}),
        ("Show distribution of Previous Acres.", {"kinds": ["histogram", "box"]}),
        ("Which risk value grew the most?", {"action": NO_ERROR}),
        ("Compare acreage before and after by risk value.", {"action": NO_ERROR}),
        ("Show percent change by county.", {"action": NO_ERROR}),
        ("Clear all filters.", {"action": NO_ERROR}),
    ]),
    (LOGS_2025 / "CO_Change_Log.xlsx", "Sheet1", 3, [
        ("Bar chart of total Acreage Change by Risk Value.", {"kinds": ["bar"]}),
        ("What is the average Acreage Change?", {"action": "explain", "states_value": ("Acreage Change", "mean")}),
        ("Only risk value 1.", {"rows": risk(1)}),
        ("Scatter New Acres against Previous Acres coloured by Risk Value.", {"kinds": ["scatter"]}),
        ("How many rows are there?", {"action": "explain"}),
        ("Histogram of Acreage Change with 20 bins.", {"kinds": ["histogram"]}),
        ("Show the counties with the largest decrease.", {"action": NO_ERROR}),
        ("Give me a presentation image of this data.", {"action": NO_ERROR}),
    ]),
    (LOGS_2026 / "FL_Change_Log.xlsx", "Sheet1", 3, [
        ("Bar chart of New Acres by Risk Value.", {"kinds": ["bar"]}),
        ("What is the largest Net % Change in County?", {"action": "explain", "states_value": ("Net % Change in County", "max")}),
        ("Only counties where the percentage went up.", {"rows": where("Net % Change in County", "gt", 0)}),
        ("Box plot of New County % by Risk Value.", {"kinds": ["box"]}),
        ("What date range does this file cover?", {"action": NO_ERROR}),
        ("Show acreage change as a share of previous acres.", {"action": NO_ERROR}),
        ("Which risk value has the highest average acreage?", {"action": NO_ERROR}),
        ("Show both New Acres and Acreage Change in one image.", {"panels": 2}),
    ]),
    (LOGS_2026 / "NV_Change_Log.xlsx", "Sheet1", 3, [
        ("Compare this file with the 2025 version.", {"action": NO_ERROR}),
        ("Show the difference between this year and last year by county.", {"action": NO_ERROR}),
        ("Load the AZ file as well and compare.", {"action": NO_ERROR}),
        ("Predict next year's acreage for risk value 3.", {"action": NO_ERROR}),
        ("What changed since the previous change log?", {"action": NO_ERROR}),
        ("Merge all state files into one chart.", {"action": NO_ERROR}),
        ("Which state had the biggest change?", {"action": NO_ERROR}),
        ("Rank counties by percentage growth.", {"action": NO_ERROR}),
    ]),
    (LOGS_2026 / "OR_Change_Log.xlsx", "Sheet1", 3, [
        ("New acres minus previous acres by county.", {"action": NO_ERROR}),
        ("Ratio of New Acres to Previous Acres by risk value.", {"action": NO_ERROR}),
        ("Percent growth in acreage by risk value.", {"action": NO_ERROR}),
        ("Average acres per county.", {"action": NO_ERROR}),
        ("Show total acres across all risk values.", {"action": NO_ERROR}),
        ("Which counties gained more than they lost?", {"action": NO_ERROR}),
        ("Show me an image with the three most useful charts.", {"action": NO_ERROR}),
        ("Explain what Acreage Change means.", {"action": NO_ERROR}),
    ]),
    (LOGS_2026 / "ID_Change_Log.xlsx", "Sheet1", 3, [
        ("How many distinct County FIPS Codes are there?", {"action": "explain"}),
        ("What is the highest Previous County %?", {"action": "explain", "states_value": ("Previous County %", "max")}),
        ("What is the smallest Acreage Change?", {"action": "explain", "states_value": ("Acreage Change", "min")}),
        ("Bar chart of count of rows by Risk Value.", {"kinds": ["bar"], "rows": None}),
        ("Only rows where New Acres is above the average.", {"rows": lambda f: f[f["New Acres"] > f["New Acres"].mean()]}),
        ("Histogram of New Acres.", {"kinds": ["histogram"], "x": "New Acres"}),
        ("Show risk value distribution as a pie chart.", {"kinds": ["pie", "donut"]}),
        ("What is the total Previous Acres?", {"action": "explain", "states_value": ("Previous Acres", "sum")}),
    ]),
]


def numbers(values):
    if isinstance(values, dict) and "bdata" in values:
        return np.frombuffer(base64.b64decode(values["bdata"]), dtype=values["dtype"]).tolist()
    return list(values)


def check(step, expect, frame, outcome, artifact):
    """Return a list of failures; every expected number comes from pandas."""
    failures = []
    action, message = outcome["action"], outcome["message"]
    if "action" in expect:
        allowed = expect["action"] if isinstance(expect["action"], list) else [expect["action"]]
        if action not in allowed:
            failures.append(f"action {action} not in {allowed}: {message[:120]}")
    if expect.get("offer") and expect["offer"] not in message.casefold():
        failures.append(f"no {expect['offer']} alternative offered: {message[:120]}")
    if expect.get("states_value"):
        column, how = expect["states_value"]
        real = float(getattr(frame[column].dropna(), how)())
        stated = [float(value.replace(",", "")) for value in re.findall(r"-?\d[\d,]*\.?\d*", message)]
        if not any(abs(value - real) <= max(0.01, abs(real) * 0.001) for value in stated):
            failures.append(f"message does not state the real {how} of {column} ({real:,.2f}): {message[:120]}")
    if expect.get("no_invented_dates"):
        column = expect["no_invented_dates"]
        low, high = str(frame[column].min().date()), str(frame[column].max().date())
        years = {str(year) for year in frame[column].dt.year.unique()}
        if low not in message or high not in message:
            failures.append(f"answer omits the real {column} range {low}..{high}: {message[:120]}")
        for year in re.findall(r"\b(20\d{2})\b", message):
            if year not in years:
                failures.append(f"invented year {year} (real years {sorted(years)})")
    if artifact is None:
        if ("kinds" in expect or "panels" in expect or expect.get("rows") is not None) and "action" not in expect:
            failures.append(f"no image created ({action}): {message[:120]}")
        return failures
    specs = chart_specs(artifact.plan)
    kinds = [spec["chart"] for spec in specs]
    if "kinds" in expect and not set(kinds) & set(expect["kinds"]):
        failures.append(f"kinds {kinds} not in {expect['kinds']}")
    if "panels" in expect and len(specs) != expect["panels"]:
        failures.append(f"{len(specs)} panels, expected {expect['panels']}")
    if "panels_min" in expect and len(specs) < expect["panels_min"]:
        failures.append(f"{len(specs)} panels, expected at least {expect['panels_min']}")
    for field in ("x", "y"):
        if field in expect and expect[field] not in [spec.get(field) for spec in specs]:
            failures.append(f"{field} {[spec.get(field) for spec in specs]} != {expect[field]}")
    if expect.get("rows") is not None:
        expected_rows = len(expect["rows"](frame))
        if artifact.matched_rows != expected_rows:
            failures.append(f"{artifact.matched_rows} rows matched, pandas says {expected_rows}")
    elif "rows" in expect and artifact.matched_rows != len(frame):
        failures.append(f"{artifact.matched_rows} rows matched, expected all {len(frame)}")
    if expect.get("no_rate_sum"):
        for spec in specs:
            if spec.get("y") == expect["no_rate_sum"] and spec.get("agg") == "sum":
                failures.append(f"summed rate-like {expect['no_rate_sum']}")
    if expect.get("colors"):
        used = {str(color).lower() for spec in specs for color in spec.get("colors", [])}
        if used and not used & set(expect["colors"]):
            failures.append(f"colors {sorted(used)} not in {expect['colors']}")
    if expect.get("top_n") and any(spec.get("top_n") != expect["top_n"] for spec in specs):
        failures.append(f"top_n {[spec.get('top_n') for spec in specs]} != {expect['top_n']}")
    return failures


def run(scenarios, limit, pause, model):
    """Each step goes through app.ask, the same path the chat UI uses."""
    results = []
    budget = {"calls": 0}
    transport = gemini.request_json

    def counted(*args, **kwargs):
        budget["calls"] += 1
        time.sleep(pause)
        return transport(*args, **kwargs)

    gemini.request_json = counted
    try:
        for scenario in scenarios:
            name, sheet, header, steps = scenario if len(scenario) == 4 else (*scenario[:2], 0, scenario[2])
            label = name if isinstance(name, str) else f"{Path(name).parent.parent.name}/{Path(name).name}"
            frame, profiles = load(name, sheet, header)
            policy = PrivacyPolicy(tuple(default_allowed_columns(profiles)))
            root = {"datasets": {}}
            identity = f"{label}:{sheet}:{header}"
            data = {"id": identity, "frame": frame, "profiles": profiles}
            state = app.conversation(root, identity)
            for question, expect in steps:
                if budget["calls"] >= limit:
                    return results
                before = state["selected"]
                calls_before = budget["calls"]
                app.ask(state, data, question, True, policy, model)
                reply = state["messages"][-1] if state["messages"] else {"text": ""}
                artifact = app.current(state)
                artifact = artifact if artifact and artifact.id != before else None
                if artifact is not None:
                    action = "image"
                elif reply.get("error"):
                    action = "error"
                elif state.get("pending"):
                    action = "clarify"
                else:
                    action = "explain"
                outcome = {"action": action, "message": reply.get("text", "")}
                record = {"file": f"{label}/{sheet}", "question": question, "action": action,
                          "message": outcome["message"][:200], "model_call": budget["calls"] > calls_before}
                if artifact is not None:
                    record["charts"] = [{k: spec.get(k) for k in ("chart", "x", "y", "color", "agg", "top_n")}
                                        for spec in chart_specs(artifact.plan)]
                    record["filters"] = artifact.plan.get("filters", [])
                    record["matched_rows"] = artifact.matched_rows
                    record["source_rows"] = artifact.source_rows
                record["failures"] = check(question, expect, frame, outcome, artifact)
                results.append(record)
                status = "PASS" if not record["failures"] else "FAIL"
                print(f"[{status}] {record['file']}: {question}", flush=True)
                for failure in record["failures"]:
                    print(f"         - {failure}", flush=True)
    finally:
        gemini.request_json = transport
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=50, help="maximum live model calls")
    parser.add_argument("--pause", type=float, default=3.0, help="seconds between calls")
    parser.add_argument("--model", default=None)
    parser.add_argument("--suite", choices=["sample", "brushfire"], default="sample")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    scenarios = SCENARIOS if args.suite == "sample" else BRUSHFIRE_SCENARIOS
    results = run(scenarios, args.limit, args.pause, args.model)
    failed = [r for r in results if r["failures"]]
    summary = {"suite": args.suite, "steps": len(results), "passed": len(results) - len(failed),
               "failed": len(failed), "results": results}
    path = ROOT / (args.out or f".validation/query_eval_{args.suite}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\n{summary['passed']}/{summary['steps']} steps passed. Details: {path}")


if __name__ == "__main__":
    main()
