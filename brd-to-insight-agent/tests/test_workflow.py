import unittest

import pandas as pd

from brd_insight_agent.intent import SpecError
from brd_insight_agent.workflow import build_artifact, chart_specs, dataset_identity, local_response, rendered_message, revise_plan


class ChartWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"City": ["A", "A", "B", "B"], "Type": ["Condo", "House"] * 2,
                                "Price": [100., 200., 300., 400.], "Size": [10., 20., 30., 40.]})
        self.identity = dataset_identity(b"synthetic", "Data", 0)
        self.first = revise_plan(None, {"action": "create", "message": "Price by city", "chart": {
            "chart": "bar", "x": "City", "y": "Price", "agg": "mean", "title": "Price by city"}})

    def test_followup_preserves_filters_and_original(self):
        second = revise_plan(self.first, {"action": "update", "filters": [{"column": "Type", "op": "eq", "value": ["Condo"]}]})
        third = revise_plan(second, local_response("make it horizontal", self.df, second))
        result = build_artifact(self.df, self.identity, third)
        self.assertEqual(result.matched_rows, 2)
        self.assertEqual(result.figure.data[0].orientation, "h")
        self.assertEqual(third["filters"], second["filters"])
        self.assertEqual(self.first["filters"], [])

    def test_new_question_drops_old_filters(self):
        prior = {**self.first, "filters": [{"column": "Type", "op": "eq", "value": ["Condo"]}]}
        new = revise_plan(prior, {"action": "create", "chart": {"chart": "scatter", "x": "Size", "y": "Price", "agg": "mean", "title": "Price vs size"}})
        self.assertEqual(build_artifact(self.df, self.identity, new).matched_rows, 4)
        self.assertEqual(new["filters"], [])

    def test_schema_forces_an_explicit_choice_for_core_fields(self):
        from brd_insight_agent.workflow import PLANNER_SCHEMA
        for node in (PLANNER_SCHEMA["properties"]["chart"], PLANNER_SCHEMA["properties"]["charts"]["items"]):
            self.assertEqual(set(node["required"]), {"chart", "x", "y", "color", "agg"})
            self.assertIn("mean", node["properties"]["agg"]["enum"])

    def test_create_null_agg_uses_default_and_null_color_means_no_split(self):
        plan = revise_plan(None, {"action": "create", "chart": {"chart": "bar", "x": "City", "y": "Price", "color": None, "agg": None}})
        self.assertNotIn("agg", plan["chart"])
        panel = build_artifact(self.df, self.identity, plan).figure.layout.meta["panels"][0]
        self.assertIsNone(panel["color"])

    def test_update_nulls_keep_current_values(self):
        plan = revise_plan(self.first, {"action": "update", "chart": {
            "chart": None, "x": None, "y": None, "color": None, "agg": None, "theme": "dark"}})
        self.assertEqual({k: plan["chart"][k] for k in ("chart", "x", "y", "agg", "theme")},
                         {"chart": "bar", "x": "City", "y": "Price", "agg": "mean", "theme": "dark"})

    def test_update_none_text_removes_a_split(self):
        split = revise_plan(self.first, {"action": "update", "chart": {"color": "Type"}})
        plan = revise_plan(split, {"action": "update", "chart": {
            "chart": None, "x": None, "y": None, "color": "none", "agg": None}})
        self.assertIsNone(build_artifact(self.df, self.identity, plan).figure.layout.meta["panels"][0]["color"])

    def test_blank_title_is_ignored_and_long_title_is_trimmed(self):
        blank = revise_plan(self.first, {"action": "update", "title": "   "})
        self.assertNotIn("title", blank)
        long = revise_plan(self.first, {"action": "update", "title": "x" * 300})
        self.assertEqual(len(long["title"]), 180)
        with self.assertRaises(SpecError):
            revise_plan(self.first, {"action": "update", "title": 7})

    def test_equal_size_upload_identity(self):
        self.assertNotEqual(dataset_identity(b"abc", "S", 0), dataset_identity(b"abd", "S", 0))

    def test_requested_scatter_color_survives_high_cardinality(self):
        frame = pd.DataFrame({"x": range(15), "y": range(15), "Group": [f"G{index}" for index in range(15)]})
        plan = revise_plan(None, {"action": "create", "chart": {"chart": "scatter", "x": "x", "y": "y", "color": "Group", "agg": "mean"}})
        fig = build_artifact(frame, self.identity, plan).figure
        self.assertEqual(len(fig.data), 15)
        self.assertFalse(fig.layout.meta["color_omitted"])

    def test_custom_color_and_category_assignments(self):
        plan = revise_plan(self.first, {"action": "update", "chart": {"colors": ["#123456"]}})
        self.assertEqual(build_artifact(self.df, self.identity, plan).figure.data[0].marker.color, "#123456")
        plan = revise_plan(self.first, {"action": "update", "chart": {"chart": "scatter", "x": "Size", "color": "City", "category_colors": [{"value": "A", "color": "magenta"}, {"value": "B", "color": "#008080"}]}})
        fig = build_artifact(self.df, self.identity, plan).figure
        self.assertEqual({trace.name: trace.marker.color for trace in fig.data}, {"A": "magenta", "B": "#008080"})
        plan["chart"]["category_colors"][0]["color"] = "not-a-color"
        with self.assertRaises(SpecError):
            build_artifact(self.df, self.identity, plan)

    def test_pie_applies_category_colors(self):
        plan = revise_plan(None, {"action": "create", "chart": {"chart": "pie", "x": "City", "agg": "count",
            "category_colors": [{"value": "A", "color": "orange"}, {"value": "B", "color": "#123456"}]}})
        trace = build_artifact(self.df, self.identity, plan).figure.data[0]
        self.assertEqual(dict(zip(trace.labels, trace.marker.colors)), {"A": "orange", "B": "#123456"})

    def test_filter_preserves_explicit_colors_for_remaining_categories(self):
        plan = revise_plan(self.first, {"action": "update", "chart": {
            "category_colors": [{"value": "A", "color": "red"}, {"value": "B", "color": "teal"}]}})
        filtered = revise_plan(plan, {"action": "update", "filters": [{"column": "City", "op": "eq", "value": ["B"]}]})
        result = build_artifact(self.df, self.identity, filtered)
        self.assertEqual(list(result.figure.data[0].marker.color), ["teal"])
        self.assertEqual(len(result.plan["chart"]["category_colors"]), 2)

    def test_custom_color_limits_are_enforced_locally(self):
        plan = revise_plan(self.first, {"action": "update", "chart": {"colors": ["red"] * 21}})
        with self.assertRaises(SpecError):
            build_artifact(self.df, self.identity, plan)

    def test_clarification_never_creates_chart(self):
        with self.assertRaises(SpecError):
            revise_plan(self.first, {"action": "clarify", "message": "Which measure?"})

    def test_type_change_removes_incompatible_orientation(self):
        prior = revise_plan(self.first, {"action": "update", "chart": {"orientation": "h"}})
        new = revise_plan(prior, {"action": "update", "chart": {"chart": "scatter", "x": "Size"}})
        self.assertNotIn("orientation", new["chart"])
        build_artifact(self.df, self.identity, new)

    def test_local_parser_does_not_swallow_filters(self):
        self.assertIsNone(local_response("average Price by City for condos only", self.df, self.first))
        self.assertIsNone(local_response("make it horizontal and only condos", self.df, self.first))

    def test_unsupported_fields(self):
        for response in ({"action": "update", "report": {}}, {"action": "update", "chart": {"wrong": 1}}):
            with self.assertRaises(SpecError):
                revise_plan(self.first, response)

    def multiple(self):
        return revise_plan(None, {"action": "create", "title": "Price and size", "charts": [
            {"chart": "histogram", "x": "Price", "agg": "count", "title": "Price distribution"},
            {"chart": "scatter", "x": "Size", "y": "Price", "agg": "mean", "title": "Price versus size"},
        ]})

    def test_both_charts_in_one_figure(self):
        artifact = build_artifact(self.df, self.identity, self.multiple())
        self.assertEqual(artifact.figure.layout.meta["panel_count"], 2)
        self.assertEqual([p["kind"] for p in artifact.figure.layout.meta["panels"]], ["histogram", "scatter"])
        self.assertNotEqual(artifact.figure.data[0].xaxis, artifact.figure.data[1].xaxis)
        self.assertIn("2 chart(s)", rendered_message(artifact))

    def test_shared_filter_and_panel_specific_refinement(self):
        original = self.multiple()
        plan = revise_plan(original, {"action": "update", "filters": [{"column": "Type", "op": "eq", "value": ["Condo"]}]})
        plan = revise_plan(plan, {"action": "update", "panel": 2, "chart": {"color": "City"}})
        result = build_artifact(self.df, self.identity, plan)
        self.assertEqual(result.matched_rows, 2)
        self.assertEqual(result.figure.layout.meta["panels"][0]["input_rows"], 2)
        self.assertEqual(result.figure.layout.meta["panels"][1]["color"], "City")
        self.assertNotIn("color", chart_specs(original)[1])

    def test_global_style_and_panel_removal(self):
        plan = revise_plan(self.multiple(), {"action": "update", "chart": {"theme": "dark", "palette": "blue"}})
        self.assertTrue(all(c["theme"] == "dark" for c in chart_specs(plan)))
        plan = revise_plan(plan, {"action": "update", "charts": [chart_specs(plan)[0]]})
        self.assertEqual(build_artifact(self.df, self.identity, plan).figure.layout.meta["panel_count"], 1)

    def test_limits_and_ambiguous_patch(self):
        for response in ({"action": "update", "chart": {"x": "Price"}},
                         {"action": "update", "panel": 3, "chart": {"title": "No"}},
                         {"action": "update", "charts": []},
                         {"action": "update", "charts": chart_specs(self.multiple()) * 3},
                         {"action": "update", "chart": {}, "charts": chart_specs(self.multiple())}):
            with self.subTest(response=response), self.assertRaises(SpecError):
                revise_plan(self.multiple(), response)

    def test_pie_and_heatmap_keep_domains_and_coloraxis(self):
        plan = revise_plan(None, {"action": "create", "charts": [
            {"chart": "pie", "x": "City", "agg": "count", "title": "City share"},
            {"chart": "correlation", "columns": ["Size", "Price"], "agg": "mean", "title": "Correlation"},
        ]})
        fig = build_artifact(self.df, self.identity, plan).figure
        self.assertLess(fig.data[0].domain.x[1], fig.layout.xaxis.domain[0])
        self.assertEqual(fig.data[1].coloraxis, "coloraxis2")

    def test_invalid_second_chart_prevents_entire_image(self):
        plan = self.multiple()
        plan["charts"][1]["y"] = "Absent"
        with self.assertRaises(SpecError):
            build_artifact(self.df, self.identity, plan)

    def test_composition_preserves_grouped_and_stacked_bars(self):
        plan = revise_plan(None, {"action": "create", "charts": [
            {"chart": kind, "x": "City", "y": "Price", "color": "Type", "agg": "sum", "title": kind}
            for kind in ("grouped_bar", "stacked_bar")
        ]})
        fig = build_artifact(self.df, self.identity, plan).figure
        self.assertEqual(fig.layout.barmode, "relative")
        self.assertNotEqual(fig.data[0].offsetgroup, fig.data[1].offsetgroup)
        self.assertEqual(fig.data[2].offsetgroup, fig.data[3].offsetgroup)


if __name__ == "__main__":
    unittest.main()