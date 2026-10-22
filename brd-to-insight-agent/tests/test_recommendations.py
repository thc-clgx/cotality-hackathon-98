import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from brd_insight_agent.profiler import profile_frame
from brd_insight_agent.recommendations import introduction, recommend_visuals, select_recommendation, type_message
from brd_insight_agent.workflow import build_artifact


class RecommendationTests(unittest.TestCase):
    def recommend(self, frame, identity="sample"):
        return recommend_visuals(frame, profile_frame(frame), identity)

    def kinds(self, bundle):
        return [group["chart"] for group in bundle["types"]]

    def options(self, bundle, chart_type):
        return next(group["options"] for group in bundle["types"] if group["chart"] == chart_type)

    def choose(self, bundle, chart_type):
        bundle["selected_type"] = chart_type
        return [type_message(bundle, chart_type)]

    def test_mixed_data_types_are_executable_and_do_not_sum_rates(self):
        frame = pd.DataFrame({"RecordID": [1, 2, 3, 4], "Region": ["A", "A", "B", "B"],
                              "ConversionRate": [.1, .2, .3, .4], "Amount": [10., 30., 20., 40.],
                              "RecordedOn": pd.date_range("2025-01-01", periods=4)})
        bundle = self.recommend(frame)
        self.assertEqual(set(self.kinds(bundle)), {"bar", "histogram", "line", "scatter", "pie", "box"})
        self.assertIn("Grouped bar chart is unavailable", introduction(bundle)["text"])
        for group in bundle["types"]:
            for option in group["options"]:
                chart = option["response"]["chart"]
                self.assertEqual(chart["chart"], group["chart"])
                self.assertNotIn("RecordID", [chart.get("x"), chart.get("y"), chart.get("color")])
                self.assertNotEqual(chart["agg"], "sum")
                artifact = build_artifact(frame, "sample", {"kind": "chart", "chart": chart, "filters": []})
                self.assertEqual(artifact.source_rows, 4)

    def test_category_only_offers_counts_and_explains_unavailable_types(self):
        bundle = self.recommend(pd.DataFrame({"Department": ["Support", "Sales", "Sales"]}))
        self.assertEqual(set(self.kinds(bundle)), {"bar", "pie"})
        self.assertEqual(self.options(bundle, "bar")[0]["response"]["chart"]["agg"], "count")
        text = introduction(bundle)["text"]
        self.assertIn("Histogram is unavailable: this sheet has no numeric column", text)
        self.assertIn("Line chart is unavailable: this sheet has no date column", text)

    def test_numeric_only_does_not_invent_groups(self):
        bundle = self.recommend(pd.DataFrame({"Measurement": [1., 2., 3.]}))
        self.assertEqual(self.kinds(bundle), ["histogram"])

    def test_identifier_text_or_empty_has_no_speculative_chart(self):
        for frame in (pd.DataFrame(), pd.DataFrame({"PersonID": [1, 2]}), pd.DataFrame({"Note": ["only one"]}),
                      pd.DataFrame({"Amount": [np.inf, np.nan]})):
            with self.subTest(columns=list(frame.columns)):
                bundle = self.recommend(frame)
                self.assertEqual(bundle["types"], [])
                self.assertIn("Correct column format", introduction(bundle)["text"])

    def test_complete_sheet_not_preview_used(self):
        frame = pd.DataFrame({"Group": ["A"] * 200 + ["B"], "Value": [1.] * 200 + [3.]})
        bundle = self.recommend(frame)
        self.assertEqual(self.options(bundle, "bar")[0]["response"]["chart"]["x"], "Group")

    def test_disjoint_measures_never_offer_scatter(self):
        frame = pd.DataFrame({"First": [1., 2., None, None], "Second": [None, None, 3., 4.]})
        bundle = self.recommend(frame)
        self.assertNotIn("scatter", self.kinds(bundle))
        self.assertIn("Scatter plot is unavailable", introduction(bundle)["text"])

    def test_render_failure_not_advertised(self):
        frame = pd.DataFrame({"Amount": [1., 2.]})
        with patch("brd_insight_agent.recommendations.build_artifact", side_effect=ValueError("Not drawable")):
            self.assertEqual(self.recommend(frame)["types"], [])

    def test_ordinal_selection_needs_a_chosen_type_and_copies_the_plan(self):
        bundle = self.recommend(pd.DataFrame({"Group": ["A", "B"], "Amount": [1., 2.]}))
        self.assertEqual(select_recommendation("recommendation 1", bundle, [])["action"], "clarify")
        messages = self.choose(bundle, "bar")
        response = select_recommendation("the second one", bundle, messages)
        self.assertEqual(response, self.options(bundle, "bar")[1]["response"])
        response["chart"]["x"] = "changed"
        self.assertNotEqual(self.options(bundle, "bar")[1]["response"]["chart"]["x"], "changed")
        self.assertIsNone(select_recommendation("the second one but only last year", bundle, messages))
        self.assertIsNone(select_recommendation("the second one", bundle, messages, {"question": "Which date?"}))
        self.assertIsNone(select_recommendation("the second one", bundle, [{"role": "assistant", "text": "Other choices"}]))

    def test_option_model_context_omits_computed_counts(self):
        frame = pd.DataFrame({"Amount": [1., 2., None]})
        bundle = self.recommend(frame)
        message = self.choose(bundle, "histogram")[0]
        self.assertIn("1 rows", message["text"])
        self.assertNotIn("1 rows", message["context_text"])

    def test_sampling_is_disclosed(self):
        frame = pd.DataFrame({"First": range(20), "Second": range(20)})
        with patch("brd_insight_agent.charts.MAX_SCATTER_POINTS", 5):
            bundle = self.recommend(frame)
        option = self.options(bundle, "scatter")[0]
        self.assertTrue(any("samples 5 of 20" in note for note in option["notes"]))

    def test_date_only_counts_without_inventing_measure(self):
        bundle = self.recommend(pd.DataFrame({"Submitted": pd.date_range("2025-01-01", periods=3)}))
        self.assertEqual(self.kinds(bundle), ["line"])
        chart = self.options(bundle, "line")[0]["response"]["chart"]
        self.assertEqual((chart["x"], chart["agg"]), ("Submitted", "count"))

    def test_type_options_are_limited_to_that_type(self):
        frame = pd.DataFrame({"Region": ["A", "B", "A", "B"], "Tier": ["X", "X", "Y", "Y"],
                              "Amount": [1., 2., 3., 4.]})
        bundle = self.recommend(frame)
        self.assertIn("grouped_bar", self.kinds(bundle))
        for chart_type in self.kinds(bundle):
            message = type_message(bundle, chart_type)
            self.assertIn(message["recommendation_options"], [chart_type])
            self.assertTrue(all(option["response"]["chart"]["chart"] == chart_type
                                for option in self.options(bundle, chart_type)))
        self.assertIsNone(type_message(bundle, "correlation"))

    def test_specific_selection_after_other_reply_requires_explicit_reference(self):
        bundle = self.recommend(pd.DataFrame({"Value": [1., 2.]}))
        messages = self.choose(bundle, "histogram") + [{"role": "assistant", "text": "Different choices now"}]
        self.assertIsNone(select_recommendation("the first one", bundle, messages))
        self.assertIsNotNone(select_recommendation("recommendation 1", bundle, messages))
        self.assertEqual(select_recommendation("recommendation 99", bundle, messages)["action"], "clarify")


if __name__ == "__main__":
    unittest.main()