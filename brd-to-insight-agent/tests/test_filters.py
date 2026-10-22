import base64
import unittest

import numpy as np
import pandas as pd

from brd_insight_agent.filters import (
    boolean_filter, category_filter, date_filter, describe, filter_offers, number_filter,
)
from brd_insight_agent.intent import SpecError, apply_filters, parse_filters
from brd_insight_agent.privacy import PrivacyPolicy, outbound_payload
from brd_insight_agent.profiler import profile_frame
from brd_insight_agent.workflow import build_artifact


def frame():
    return pd.DataFrame({
        "OrderID": [f"A{index:03d}" for index in range(12)],
        "SaleDate": pd.to_datetime(
            ["2024-02-10", "2024-05-02", "2024-08-19", "2024-11-30",
             "2025-01-15", "2025-02-20", "2025-03-31", "2025-04-01",
             "2025-07-04", "2025-10-09", "2025-12-31", "2025-03-01"]),
        "Region": ["West", "East", "West", "East", "West", "East", "West", "East", "West", "East", "West", "East"],
        "Returned": [True, False, False, True, False, False, True, False, False, True, False, False],
        "NetRevenue": [10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 110., 120.],
    })


class OfferTests(unittest.TestCase):
    def setUp(self):
        self.frame = frame()
        self.profiles = profile_frame(self.frame)
        self.offers = {offer["column"]: offer for offer in filter_offers(self.frame, self.profiles)}

    def test_offers_match_column_kinds_and_exclude_identifiers(self):
        self.assertEqual({name: offer["kind"] for name, offer in self.offers.items()},
                         {"SaleDate": "date", "Region": "category", "Returned": "boolean", "NetRevenue": "number"})
        self.assertNotIn("OrderID", self.offers)

    def test_quarter_choices_are_real_calendar_ranges_present_in_the_data(self):
        labels = [period["label"] for period in self.offers["SaleDate"]["periods"]]
        self.assertIn("2025 Q1 (2025-01-01 to 2025-03-31)", labels)
        self.assertIn("2024 (full year)", labels)
        self.assertNotIn("2023", str(labels))
        first = next(p for p in self.offers["SaleDate"]["periods"] if p["label"].startswith("2025 Q1"))
        parsed = parse_filters([date_filter("SaleDate", first["start"], first["end"])], self.frame, self.profiles)
        matched = apply_filters(self.frame, parsed)
        self.assertEqual(len(matched), 4)
        self.assertEqual(matched["SaleDate"].max(), pd.Timestamp("2025-03-31"))

    def test_category_boolean_and_number_filters_match_expected_rows(self):
        cases = [
            (category_filter("Region", ["West"]), 6),
            (boolean_filter("Returned", True), 4),
            (number_filter("NetRevenue", 100, 120), 3),
        ]
        for spec, expected in cases:
            with self.subTest(column=spec["column"]):
                parsed = parse_filters([spec], self.frame, self.profiles)
                self.assertEqual(len(apply_filters(self.frame, parsed)), expected)

    def test_scope_summary_counts_rows_locally(self):
        scope = [date_filter("SaleDate", "2025-01-01", "2025-03-31"), category_filter("Region", ["West"])]
        summary, matched = describe(self.frame, self.profiles, scope)
        self.assertEqual(matched, 2)
        self.assertIn("SaleDate", summary)
        self.assertIn("Region", summary)
        self.assertEqual(describe(self.frame, self.profiles, []), ("All rows in the selected sheet", 12))

    def test_text_values_match_regardless_of_case(self):
        for value in ("west", "WEST", "West"):
            with self.subTest(value=value):
                parsed = parse_filters([category_filter("Region", [value])], self.frame, self.profiles)
                self.assertEqual(len(apply_filters(self.frame, parsed)), 6)
        excluded = parse_filters([{"column": "Region", "op": "neq", "value": ["west"]}], self.frame, self.profiles)
        self.assertEqual(len(apply_filters(self.frame, excluded)), 6)

    def test_invalid_filter_is_rejected_not_ignored(self):
        for bad in ({"column": "Missing", "op": "eq", "value": ["1"]},
                    {"column": "Region", "op": "gte", "value": ["1"]},
                    {"column": "SaleDate", "op": "between", "value": ["2025-01-01"]}):
            with self.subTest(filter=bad), self.assertRaises(SpecError):
                parse_filters([bad], self.frame, self.profiles)

    def test_columns_without_usable_values_offer_nothing(self):
        empty = pd.DataFrame({"Blank": [None, None], "Flat": [5., 5.], "Note": ["a", "b"]})
        offers = filter_offers(empty, profile_frame(empty))
        self.assertEqual([offer["column"] for offer in offers], ["Note"])


class ScopeBuildTests(unittest.TestCase):
    def setUp(self):
        self.frame = frame()
        self.profiles = profile_frame(self.frame)
        self.chart = {"chart": "bar", "x": "Region", "y": "NetRevenue", "agg": "sum", "title": "Revenue"}

    def plotted_total(self, artifact):
        trace = artifact.figure.data[0]
        values = trace.y if trace.orientation != "h" else trace.x
        if isinstance(values, dict):
            values = np.frombuffer(base64.b64decode(values["bdata"]), dtype=values["dtype"])
        return float(np.asarray(values, dtype=float).sum())

    def test_scope_narrows_the_image_and_is_disclosed(self):
        scope = [date_filter("SaleDate", "2025-01-01", "2025-03-31")]
        artifact = build_artifact(self.frame, "one", {"kind": "chart", "chart": self.chart,
                                                      "filters": [], "scope_filters": scope})
        self.assertEqual((artifact.matched_rows, artifact.source_rows), (4, 12))
        self.assertIn("SaleDate", artifact.scope)
        self.assertEqual(self.plotted_total(artifact), 300.)

    def test_scope_and_chat_filters_both_apply(self):
        plan = {"kind": "chart", "chart": self.chart,
                "filters": [category_filter("Region", ["West"])],
                "scope_filters": [date_filter("SaleDate", "2025-01-01", "2025-03-31")]}
        artifact = build_artifact(self.frame, "one", plan)
        self.assertEqual(artifact.matched_rows, 2)
        self.assertEqual(self.plotted_total(artifact), 120.)

    def test_empty_scope_result_is_an_explicit_error(self):
        plan = {"kind": "chart", "chart": self.chart, "filters": [],
                "scope_filters": [date_filter("SaleDate", "2024-01-01", "2024-01-02")]}
        with self.assertRaises(SpecError):
            build_artifact(self.frame, "one", plan)

    def test_scope_changes_produce_distinct_versions(self):
        base = {"kind": "chart", "chart": self.chart, "filters": []}
        first = build_artifact(self.frame, "one", {**base, "scope_filters": []})
        second = build_artifact(self.frame, "one", {**base, "scope_filters": [category_filter("Region", ["West"])]})
        self.assertNotEqual(first.id, second.id)
        self.assertEqual(first.matched_rows, 12)
        self.assertEqual(second.matched_rows, 6)


class ScopePrivacyTests(unittest.TestCase):
    def setUp(self):
        self.frame = frame()
        self.profiles = profile_frame(self.frame)
        self.policy = PrivacyPolicy(("SaleDate", "NetRevenue"))

    def payload(self, previous=None, scope=None):
        return outbound_payload(self.frame, self.profiles, "chart it", previous, self.policy,
                                "2025-06-01", scope=scope)

    def test_scope_on_unapproved_column_is_withheld_but_counted(self):
        scope = [date_filter("SaleDate", "2025-01-01", "2025-03-31"), category_filter("Region", ["West"])]
        payload = self.payload(scope=scope)
        self.assertEqual(payload["active_scope"], [scope[0]])
        self.assertEqual(payload["withheld_scope_filters"], 1)
        self.assertNotIn("West", str(payload))
        self.assertNotIn("Region", str(payload))

    def test_previous_plan_filters_on_unapproved_columns_are_redacted(self):
        previous = {"kind": "chart", "chart": {"chart": "bar", "x": "SaleDate", "agg": "count"},
                    "filters": [category_filter("Region", ["East"])],
                    "scope_filters": [date_filter("SaleDate", "2025-01-01", "2025-03-31")]}
        payload = self.payload(previous=previous)
        self.assertEqual(payload["previous_plan"]["filters"], [])
        self.assertEqual(payload["previous_plan"]["scope_filters"], previous["scope_filters"])
        self.assertEqual(payload["previous_plan"]["withheld_filters"], 1)
        self.assertNotIn("East", str(payload))
        self.assertEqual(previous["filters"][0]["value"], ["East"])

    def test_approved_scope_is_sent_unchanged(self):
        scope = [number_filter("NetRevenue", 10, 50)]
        payload = self.payload(scope=scope)
        self.assertEqual(payload["active_scope"], scope)
        self.assertEqual(payload["withheld_scope_filters"], 0)


if __name__ == "__main__":
    unittest.main()
