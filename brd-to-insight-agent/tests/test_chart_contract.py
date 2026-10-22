"""Local contract checks for validated requests and plotted values."""

from dataclasses import replace
from datetime import date
import hashlib
import json
import unittest

import numpy as np
import pandas as pd
import plotly.express as px
from pandas.testing import assert_frame_equal

from brd_insight_agent.charts import MAX_SCATTER_POINTS, _aggregate, build_figure
from brd_insight_agent.intent import RESPONSE_SCHEMA, Filter, ParsedRequest, SpecError, apply_filters, validate
from brd_insight_agent.profiler import profile_frame
from brd_insight_agent.spec import ChartSpec


class ChartContractTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "Category": ["A", "A", "B", "B", "C", None],
            "Revenue": [10., 20., 30., 40., 50., 60.],
            "Cost": [2., 4., 6., 8., 10., 12.],
            "Units": [1., 2., 3., 4., 5., 6.],
            "RowID": [101, 102, 103, 104, 105, 106],
            "ConversionRate": [.1, .2, .3, .4, .5, .6],
            "Flag": pd.Series([True, False, None, True, False, True], dtype="boolean"),
            "Text": ["a.b", "axb", None, "nan", "A.B", "plain"],
        })

    def parse(self, df=None, **changes):
        frame = self.df if df is None else df
        raw = {"chart": "bar", "x": "Category", "y": "Revenue", "agg": "sum",
               "filters": [], "title": "Contract", "interpretation": "Exact values", "confidence": "high"}
        raw.update(changes)
        return validate(raw, frame, profile_frame(frame))

    def filtered(self, column, op, values, df=None):
        frame = self.df if df is None else df
        parsed = self.parse(df=frame, filters=[{"column": column, "op": op, "value": values}])
        return parsed, apply_filters(frame, parsed.filters)

    def test_schema_and_defaults(self):
        props = RESPONSE_SCHEMA["properties"]
        self.assertNotIn("x", RESPONSE_SCHEMA["required"])
        self.assertEqual(props["orientation"]["enum"], ["auto", "h", "v"])
        self.assertEqual(props["palette"]["enum"], ["default", "blue", "viridis", "red_blue"])
        self.assertEqual(props["bins"], {"type": "integer", "minimum": 5, "maximum": 100})
        self.assertEqual(props["top_n"], {"type": "integer", "minimum": 1, "maximum": 40})
        self.assertEqual(props["filters"]["items"]["properties"]["op"]["enum"][-2:],
                         ["above_mean", "below_mean"])
        parsed = self.parse()
        self.assertIsInstance(parsed, ParsedRequest)
        self.assertEqual(parsed.spec.extra, {"grain": None, "orientation": "auto", "palette": "default", "bins": 30})
        self.assertEqual(parsed.summary(), "No filter applied - using every row.")

    def test_extras_are_strictly_bounded(self):
        for name, values in {"bins": [4, 101, 5.0, True, "10", None],
                             "top_n": [0, 41, 1.0, False, "3", None],
                             "orientation": ["horizontal", "", None],
                             "palette": ["rainbow", None, []],
                             "grain": ["hour", "", None]}.items():
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(SpecError):
                    self.parse(**{name: value})
        for bins, top_n in ((5, 1), (100, 40)):
            self.assertEqual(self.parse(bins=bins, top_n=top_n).spec.extra,
                             {"grain": None, "orientation": "auto", "palette": "default",
                              "bins": bins, "top_n": top_n})
        for grain in ("D", "W", "M", "Q", "Y"):
            self.assertEqual(self.parse(grain=grain).spec.extra["grain"], grain)

    def test_key_is_exact_canonical_json_hash(self):
        spec = ChartSpec("bar", "Title", "Category", "Revenue",
                         extra={"bins": 30, "orientation": "v", "nested": {"b": 2, "a": 1}})
        payload = {"kind": "bar", "x": "Category", "y": "Revenue", "color": None,
                   "agg": "sum", "extra": spec.extra}
        expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                            ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
        self.assertEqual(spec.key, expected)
        reordered = replace(spec, extra={"nested": {"a": 1, "b": 2}, "orientation": "v", "bins": 30})
        self.assertEqual(spec.key, reordered.key)
        self.assertEqual(spec.key, replace(spec, score=99, rationale="Different", title="Renamed").key)
        variants = [replace(spec, extra={**spec.extra, name: value}) for name, value in
                    (("bins", 31), ("orientation", "h"), ("palette", "blue"),
                     ("grain", "M"), ("top_n", 4), ("columns", ["Revenue", "Cost"]))]
        self.assertEqual(len({spec.key, *(s.key for s in variants)}), 7)
        self.assertNotEqual(ChartSpec("bar", "", "a|b", "c").key,
                            ChartSpec("bar", "", "a", "b|c").key)

    def test_correlation_explicit_subset_and_deduplication(self):
        parsed = self.parse(chart="correlation", x=None, y=None, columns=[" cost ", "Revenue", "COST"])
        self.assertEqual(parsed.spec.extra["columns"], ["Cost", "Revenue"])
        self.assertEqual(parsed.warnings, [])
        fig = build_figure(self.df, parsed.spec)
        self.assertEqual(list(fig.data[0].x), ["Cost", "Revenue"])
        self.assertEqual(list(fig.data[0].y), ["Cost", "Revenue"])
        self.assertEqual(fig.data[0].z.tolist(), [[1., 1.], [1., 1.]])
        self.assertEqual(fig.layout.meta["pairwise_counts"], [[6, 6], [6, 6]])

    def test_correlation_defaults_only_when_list_absent(self):
        raw = {"chart": "correlation", "agg": "sum", "filters": [], "title": "Correlations"}
        parsed = validate(raw, self.df, profile_frame(self.df))
        self.assertIsNone(parsed.spec.x)
        self.assertEqual(parsed.spec.extra["columns"], ["Revenue", "Cost", "Units", "ConversionRate"])
        self.assertEqual(validate(raw, self.df, {}).spec.extra["columns"], parsed.spec.extra["columns"])
        pair = self.parse(chart="correlation", x="Units", y="Revenue")
        self.assertEqual(pair.spec.extra["columns"], ["Units", "Revenue"])
        with self.assertRaisesRegex(SpecError, "two unique"):
            self.parse(chart="correlation", x="Revenue", y=None)

    def test_correlation_rejects_invalid_lists_without_broadening(self):
        for columns in ([], None, "Revenue", ["Revenue"], ["Revenue", "Revenue"],
                        ["Revenue", "Text"], ["Revenue", "RowID"], ["Revenue", "Flag"],
                        ["Revenue", "missing"], ["Revenue", ""], ["Revenue", None], [1, 2]):
            with self.subTest(columns=columns), self.assertRaises(SpecError):
                self.parse(chart="correlation", x="Revenue", y="Cost", columns=columns)
        for extra in ({}, {"columns": []}, {"columns": ["Revenue", "missing"]},
                      {"columns": ["Revenue", "Text"]}, {"columns": ["Revenue", "RowID"]}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                build_figure(self.df, ChartSpec("correlation", "Invalid", extra=extra))

    def test_correlation_keeps_constant_columns_and_full_labels(self):
        long_a, long_b = "A very long measure prefix one", "A very long measure prefix two"
        df = pd.DataFrame({long_a: [1., 2., 3.], long_b: [2., 4., 6.], "Constant": [1., 1., 1.]})
        spec = ChartSpec("correlation", "Subset", extra={"columns": [long_a, long_b, "Constant"]})
        fig = build_figure(df, spec)
        self.assertEqual(list(fig.data[0].x), [long_a, long_b, "Constant"])
        self.assertEqual(fig.data[0].z[:2, :2].tolist(), [[1., 1.], [1., 1.]])
        self.assertTrue(pd.isna(fig.data[0].z[2]).all())
        self.assertEqual(fig.layout.meta["undefined_columns"], ["Constant"])
        self.assertEqual(list(fig.layout.xaxis.tickvals), [long_a, long_b, "Constant"])

    def test_rate_sums_rejected_only_for_aggregating_kinds(self):
        for kind in ("bar", "grouped_bar", "stacked_bar", "line", "area", "pie", "donut"):
            with self.subTest(kind=kind), self.assertRaisesRegex(SpecError, "rate-like"):
                self.parse(chart=kind, y="ConversionRate", agg="sum")
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "rate-like"):
                build_figure(self.df, ChartSpec(kind, "Rate", "Category", "ConversionRate"))
        for kind, x, y in (("scatter", "Revenue", "ConversionRate"),
                           ("box", "Category", "ConversionRate"), ("histogram", "ConversionRate", None)):
            parsed = self.parse(chart=kind, x=x, y=y, agg="sum")
            self.assertEqual(parsed.spec.agg, "sum")
            self.assertIsNone(build_figure(self.df, parsed.spec).layout.meta["aggregation"])
        self.assertEqual(self.parse(y="ConversionRate", agg="mean").spec.agg, "mean")
        self.assertEqual(self.parse(y="ConversionRate", agg="count").spec.agg, "count")
        df = self.df.rename(columns={"Revenue": "CorporateRevenue"})
        self.assertEqual(self.parse(df=df, y="CorporateRevenue").spec.agg, "sum")

    def test_required_axes_and_missing_columns_are_checked(self):
        invalid = ({"chart": "scatter", "x": "Revenue", "y": None},
                   {"chart": "scatter", "x": "Category"}, {"chart": "histogram", "x": "Category"},
                   {"chart": "box", "x": "Category", "y": None}, {"y": "Text"},
                   {"y": "Flag"}, {"x": None}, {"y": None}, {"color": "missing"})
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(SpecError):
                self.parse(**changes)
        for kind in ("bar", "line", "histogram", "scatter", "box", "pie"):
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "not in this sheet"):
                build_figure(self.df, ChartSpec(kind, "Missing", "missing", "Revenue"))
        for kind, x, y in (("histogram", "Category", None), ("scatter", "Category", "Revenue"),
                           ("box", "Category", "Text")):
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "numeric"):
                build_figure(self.df, ChartSpec(kind, "Types", x, y))

    def test_duplicate_dataframe_and_grouping_keys_are_rejected(self):
        df = pd.DataFrame([[1, 2]], columns=["Revenue", "Revenue"])
        with self.assertRaisesRegex(SpecError, "unique"):
            validate({"chart": "histogram", "x": "Revenue"}, df, {})
        with self.assertRaisesRegex(ValueError, "unique"):
            build_figure(df, ChartSpec("histogram", "Duplicate", "Revenue"))
        with self.assertRaisesRegex(SpecError, "distinct grouping"):
            self.parse(color="Category", agg="count")
        with self.assertRaisesRegex(ValueError, "distinct grouping"):
            _aggregate(self.df, ChartSpec("bar", "Duplicate", "Category", color="Category", agg="count"), None)

    def test_filter_arity_and_value_shapes(self):
        for op in ("eq", "neq", "lt", "lte", "gt", "gte"):
            for values in ([], ["10", "20"], "10"):
                with self.subTest(op=op, values=values), self.assertRaises(SpecError):
                    self.filtered("Revenue", op, values)
        for op, values in (("between", ["10"]), ("between", ["10", "20", "30"]),
                           ("between", ["20", "10"]), ("in", []), ("not_in", []),
                           ("above_mean", ["0"]), ("below_mean", ["0"]), ("is_null", ["0"]),
                           ("not_null", ["0"]), ("eq", [None]), ("eq", [["10"]])):
            with self.subTest(op=op, values=values), self.assertRaises(SpecError):
                self.filtered("Revenue", op, values)
        for values in ([], ["a", "b"]):
            with self.assertRaises(SpecError):
                self.filtered("Text", "contains", values)
        for column in ("Text", "Flag"):
            with self.assertRaises(SpecError):
                self.filtered(column, "above_mean", [])

    def test_boolean_coercion_and_null_exclusion(self):
        for value in ("false", "no", "0", False):
            parsed, out = self.filtered("Flag", "eq", [value])
            self.assertIs(parsed.filters[0].value[0], False)
            self.assertEqual(out.index.tolist(), [1, 4])
        for value in ("true", "yes", "1", True):
            parsed, out = self.filtered("Flag", "eq", [value])
            self.assertIs(parsed.filters[0].value[0], True)
            self.assertEqual(out.index.tolist(), [0, 3, 5])
        for value in ("maybe", "2", "", None):
            with self.subTest(value=value), self.assertRaises(SpecError):
                self.filtered("Flag", "eq", [value])
        self.assertEqual(self.filtered("Flag", "neq", ["true"])[1].index.tolist(), [1, 4])
        self.assertEqual(self.filtered("Flag", "not_in", ["false"])[1].index.tolist(), [0, 3, 5])

    def test_numeric_coercion_is_finite_and_never_arithmetic(self):
        for value in ("NaN", "inf", "-inf", "1e999", "10 + 5", "mean(Revenue)"):
            with self.subTest(value=value), self.assertRaises(SpecError):
                self.filtered("Revenue", "gt", [value])
        parsed, out = self.filtered("Revenue", "in", ["$10.00", "50"])
        self.assertEqual(parsed.filters[0].value, [10., 50.])
        self.assertEqual(out.index.tolist(), [0, 4])
        self.assertEqual(self.filtered("Revenue", "between", ["20", "40"])[1].index.tolist(), [1, 2, 3])

    def test_scalar_boundaries_and_direct_filter_validation(self):
        for op, expected in (("gt", [3, 4, 5]), ("gte", [2, 3, 4, 5]),
                             ("lt", [0, 1]), ("lte", [0, 1, 2]), ("eq", [2]),
                             ("neq", [0, 1, 3, 4, 5])):
            self.assertEqual(self.filtered("Revenue", op, ["30"])[1].index.tolist(), expected)
        for f in (Filter("missing", "eq", ["1"]), Filter("Revenue", "gt", ["nan"]),
                  Filter("Flag", "eq", ["maybe"]), Filter("Revenue", "eq", ["1", "2"])):
            with self.subTest(filter=f), self.assertRaises(SpecError):
                apply_filters(self.df, [f])
        nulls = self.df.copy()
        nulls.loc[2, "Revenue"] = None
        for op in ("neq", "not_in"):
            self.assertEqual(self.filtered("Revenue", op, ["30"], nulls)[1].index.tolist(), [0, 1, 3, 4, 5])

    def test_filter_warnings_respect_operator_direction(self):
        for op, values in (("gt", ["0"]), ("gte", ["0"]), ("lt", ["100"]),
                           ("lte", ["100"]), ("neq", ["999"]), ("not_in", ["999"])):
            parsed, out = self.filtered("Revenue", op, values)
            self.assertEqual(parsed.warnings, [])
            self.assertEqual(out.index.tolist(), [0, 1, 2, 3, 4, 5])
        parsed, out = self.filtered("Revenue", "between", ["5", "15"])
        self.assertEqual(parsed.warnings, [])
        self.assertEqual(out.index.tolist(), [0])
        parsed, out = self.filtered("Revenue", "gt", ["100"])
        self.assertEqual(out.index.tolist(), [])
        self.assertEqual(parsed.warnings, ["Filter on 'Revenue' matches no rows in the sheet (observed range: 10 to 60)."])

    def test_literal_contains_and_negative_ops_preserve_nulls(self):
        self.assertEqual(self.filtered("Text", "contains", ["."])[1].index.tolist(), [0, 4])
        self.assertEqual(self.filtered("Text", "contains", ["nan"])[1].index.tolist(), [3])
        self.assertEqual(self.filtered("Text", "contains", ["["])[1].index.tolist(), [])
        for op in ("neq", "not_in"):
            self.assertEqual(self.filtered("Text", op, ["nan"])[1].index.tolist(), [0, 1, 4, 5])
        self.assertEqual(self.filtered("Text", "is_null", [])[1].index.tolist(), [2])
        self.assertEqual(self.filtered("Text", "not_null", [])[1].index.tolist(), [0, 1, 3, 4, 5])

    def test_mean_filters_use_original_provided_population_and_and_semantics(self):
        df = pd.DataFrame({"Category": ["A", "A", "B", "B"], "Revenue": [0., 40., 60., 100.]})
        for op, expected in (("below_mean", [0, 1]), ("above_mean", [])):
            filters = [Filter("Category", "in", ["A"]), Filter("Revenue", op, [])]
            self.assertEqual(apply_filters(df, filters).index.tolist(), expected)
            self.assertEqual(apply_filters(df, filters[::-1]).index.tolist(), expected)
        self.assertEqual(apply_filters(df.iloc[:3], [Filter("Category", "eq", ["A"]),
                                                    Filter("Revenue", "above_mean", [])]).index.tolist(), [1])
        parsed = self.parse(df=df, filters=[{"column": "Revenue", "op": "below_mean", "value": []}])
        self.assertEqual(parsed.summary(), "Filtered to Revenue below the sheet-wide mean (original input rows).")
        assert_frame_equal(df, pd.DataFrame({"Category": ["A", "A", "B", "B"], "Revenue": [0., 40., 60., 100.]}))
        nullable = pd.DataFrame({"Revenue": pd.Series([1., 3., 7., None], dtype="Float64")})
        self.assertEqual(apply_filters(nullable, [Filter("Revenue", "above_mean", [])]).index.tolist(), [2])

    def test_datetime_upper_bounds_include_full_day_but_not_next_day(self):
        df = pd.DataFrame({"Category": ["A"] * 5, "Revenue": [1.] * 5,
                           "When": [pd.Timestamp("2026-09-01"), pd.Timestamp("2026-09-01 12:00"),
                                    pd.Timestamp("2026-09-01 23:59:59.999999999"),
                                    pd.Timestamp("2026-09-02"), pd.NaT]})
        for value in ("2026-09-01", "09/01/2026", "September 1, 2026", date(2026, 9, 1)):
            parsed, out = self.filtered("When", "lte", [value], df)
            self.assertEqual(parsed.filters[0].value, [pd.Timestamp("2026-09-01 23:59:59.999999999")])
            self.assertEqual(out.index.tolist(), [0, 1, 2])
        parsed, out = self.filtered("When", "between", ["2026-09-01", "2026-09-01"], df)
        self.assertEqual(out.index.tolist(), [0, 1, 2])
        self.assertEqual(parsed.warnings, [])
        self.assertEqual(self.filtered("When", "lte", ["2026-09-01 00:00:00"], df)[1].index.tolist(), [0])
        self.assertEqual(self.filtered("When", "lt", ["2026-09-02"], df)[1].index.tolist(), [0, 1, 2])
        self.assertEqual(self.filtered("When", "gt", ["2026-08-01"], df)[0].warnings, [])
        self.assertEqual(apply_filters(df, [Filter("When", "lte", ["2026-09-01"])]).index.tolist(), [0, 1, 2])
        with self.assertRaisesRegex(SpecError, "ordered"):
            self.filtered("When", "between", ["2026-09-02", "2026-09-01"], df)
        with self.assertRaises(SpecError):
            self.filtered("When", "above_mean", [], df)
        utc = df.copy()
        utc["When"] = utc["When"].dt.tz_localize("UTC")
        self.assertEqual(self.filtered("When", "lte", ["2026-09-01"], utc)[1].index.tolist(), [0, 1, 2])

    def test_count_names_and_measure_grouping_collisions(self):
        df = pd.DataFrame({"count": ["A", "A", "B"], "count_2": ["x", "y", "x"], "Revenue": [5., 6., 7.]})
        spec = ChartSpec("grouped_bar", "Counts", "count", color="count_2", agg="count")
        data = _aggregate(df, spec, None)
        self.assertEqual(data.columns.tolist(), ["count", "count_2", "count_3"])
        self.assertEqual(data.to_dict("records"), [{"count": "A", "count_2": "x", "count_3": 1},
                                                  {"count": "A", "count_2": "y", "count_3": 1},
                                                  {"count": "B", "count_2": "x", "count_3": 1}])
        fig = build_figure(df, spec)
        self.assertEqual({t.name: list(t.y) for t in fig.data}, {"x": [1, 1], "y": [1]})
        self.assertEqual(fig.layout.meta["count_semantics"], "rows")
        repeated = pd.DataFrame({"Value": [2, 2, 3]})
        data = _aggregate(repeated, ChartSpec("bar", "Same measure", "Value", "Value"), None)
        self.assertEqual(data.to_dict("records"), [{"Value": 2, "Value_2": 4}, {"Value": 3, "Value_2": 3}])

    def test_sum_keeps_null_instead_of_zero_and_counts_rows(self):
        df = pd.DataFrame({"Category": ["A", "A", "B", "B"], "Revenue": [None, None, 5., None]})
        spec = ChartSpec("bar", "Missing", "Category", "Revenue")
        data = _aggregate(df, spec, None)
        self.assertEqual(data["Category"].tolist(), ["B", "A"])
        self.assertEqual(data["Revenue"].iloc[0], 5.)
        self.assertTrue(pd.isna(data["Revenue"].iloc[1]))
        self.assertEqual(data.attrs["chart_facts"]["eligible_rows"], 1)
        self.assertEqual(data.attrs["chart_facts"]["omitted_categories"], ["A"])
        counts = _aggregate(df, replace(spec, agg="count"), None)
        self.assertEqual(counts.to_dict("records"), [{"Category": "A", "count": 2}, {"Category": "B", "count": 2}])
        with self.assertRaisesRegex(ValueError, "non-null"):
            build_figure(df.iloc[:2], spec)

    def test_orientation_overrides_and_top_n_facts(self):
        df = pd.DataFrame({"Category": ["A", "A", "B"], "Revenue": [10., 20., 15.]})
        spec = ChartSpec("bar", "Bars", "Category", "Revenue", extra={"orientation": "h", "palette": "blue"})
        fig = build_figure(df, spec)
        self.assertEqual(fig.data[0].orientation, "h")
        self.assertEqual(list(fig.data[0].x), [15., 30.])
        self.assertEqual(list(fig.data[0].y), ["B", "A"])
        self.assertEqual(fig.data[0].marker.color, px.colors.sequential.Blues[2])
        vertical = build_figure(df, replace(spec, extra={"orientation": "v", "top_n": 1}))
        self.assertEqual(vertical.data[0].orientation, "v")
        self.assertEqual(list(vertical.data[0].x), ["A"])
        self.assertEqual(list(vertical.data[0].y), [30.])
        meta = vertical.layout.meta
        self.assertEqual({k: meta[k] for k in ("input_rows", "eligible_rows", "drawn_rows", "plotted_points")},
                         {"input_rows": 3, "eligible_rows": 3, "drawn_rows": 2, "plotted_points": 1})
        self.assertEqual(meta["omitted_categories"], ["B"])
        self.assertEqual(meta["aggregation"], "sum")
        self.assertEqual(meta["group_by"], ["Category"])
        self.assertFalse(meta["sampled"])
        long = df.assign(Category=["This is a long category"] * 3)
        self.assertEqual(build_figure(long, replace(spec, extra={"orientation": "v"})).data[0].orientation, "v")
        with self.assertRaisesRegex(ValueError, "Orientation overrides"):
            build_figure(df, replace(spec, kind="line"))

    def test_histogram_exact_bins_orientation_axes_metadata_and_palette(self):
        counts, edges = np.histogram(self.df["Revenue"].to_numpy(), bins=17)
        centers = edges[:-1] + np.diff(edges) / 2
        palettes = {"default": px.colors.qualitative.Plotly, "blue": px.colors.sequential.Blues[2:],
                    "viridis": px.colors.sequential.Viridis, "red_blue": px.colors.diverging.RdBu[::-1]}
        for orientation in ("auto", "h", "v"):
            for palette, colors in palettes.items():
                with self.subTest(orientation=orientation, palette=palette):
                    histogram = self.parse(chart="histogram", x="Revenue", y=None, bins=17,
                                           orientation=orientation, palette=palette)
                    fig = build_figure(self.df, histogram.spec)
                    trace = fig.data[0]
                    horizontal = orientation == "h"
                    self.assertEqual(trace.type, "bar")
                    self.assertEqual(trace.orientation, "h" if horizontal else "v")
                    np.testing.assert_array_equal(trace.x if horizontal else trace.y, counts)
                    np.testing.assert_array_equal(trace.y if horizontal else trace.x, centers)
                    np.testing.assert_array_equal(trace.width, np.diff(edges))
                    self.assertEqual(trace.marker.color, colors[0])
                    count_axis = fig.layout.xaxis if horizontal else fig.layout.yaxis
                    measure_axis = fig.layout.yaxis if horizontal else fig.layout.xaxis
                    self.assertEqual(count_axis.title.text, "count")
                    self.assertEqual(count_axis.tickformat, ",d")
                    self.assertEqual(measure_axis.title.text, "Revenue")
                    self.assertEqual(measure_axis.tickformat, "~s")
                    for axis in (count_axis, measure_axis):
                        self.assertEqual(axis.type, "linear")
                        self.assertIsNone(axis.tickvals)
                        self.assertIsNone(axis.ticktext)
                        self.assertIsNone(axis.tickangle)
                    meta = fig.layout.meta
                    self.assertEqual(meta["kind"], "histogram")
                    self.assertIsNone(meta["aggregation"])
                    self.assertEqual(meta["palette"], palette)
                    self.assertEqual(meta["orientation"], trace.orientation)
                    self.assertEqual(meta["bins"], 17)
                    self.assertEqual(meta["binning"], "numpy.histogram")
                    self.assertEqual(meta["bin_edges"], edges.tolist())
                    self.assertEqual(meta["bin_counts"], counts.tolist())
                    self.assertEqual(meta["count_semantics"], "rows")
                    self.assertEqual(meta["group_by"], [])
                    self.assertEqual({k: meta[k] for k in ("input_rows", "eligible_rows", "drawn_rows",
                                                          "plotted_points", "sampled")},
                                     {"input_rows": 6, "eligible_rows": 6, "drawn_rows": 6,
                                      "plotted_points": 17, "sampled": False})

    def test_histogram_colors_share_edges_and_include_final_endpoint(self):
        df = pd.DataFrame({"Revenue": [0., 1., 2., 3., 4., 5., 5.],
                           "Category": ["Low", "Low", "Low", "High", "High", "High", "High"]})
        counts, edges = np.histogram(df["Revenue"].to_numpy(), bins=5)
        self.assertEqual(counts.tolist(), [1, 1, 1, 1, 3])
        intervals = [[float(left), float(right), "]" if i == 4 else ")"]
                     for i, (left, right) in enumerate(zip(edges[:-1], edges[1:]))]
        for orientation in ("h", "v"):
            with self.subTest(orientation=orientation):
                spec = ChartSpec("histogram", "Groups", "Revenue", color="Category",
                                 extra={"bins": 5, "orientation": orientation, "palette": "viridis"})
                fig = build_figure(df, spec)
                self.assertEqual([trace.name for trace in fig.data], ["Low", "High"])
                total = np.zeros(5, dtype=int)
                for i, trace in enumerate(fig.data):
                    group = df.loc[df["Category"].eq(trace.name), "Revenue"].to_numpy()
                    expected, _ = np.histogram(group, bins=edges)
                    actual = trace.x if orientation == "h" else trace.y
                    positions = trace.y if orientation == "h" else trace.x
                    np.testing.assert_array_equal(actual, expected)
                    np.testing.assert_array_equal(positions, edges[:-1] + np.diff(edges) / 2)
                    np.testing.assert_array_equal(trace.width, np.diff(edges))
                    self.assertEqual([list(row) for row in trace.customdata], intervals)
                    self.assertIn("[%{customdata[0]}, %{customdata[1]}%{customdata[2]}", trace.hovertemplate)
                    self.assertIn("count=%{" + ("x" if orientation == "h" else "y") + ":d}",
                                  trace.hovertemplate)
                    self.assertEqual(trace.marker.color, px.colors.sequential.Viridis[i])
                    total += np.asarray(actual)
                np.testing.assert_array_equal(total, counts)
                self.assertEqual(fig.layout.barmode, "stack")
                self.assertEqual(fig.layout.meta["bin_counts"], counts.tolist())
                self.assertEqual(fig.layout.meta["bin_edges"], edges.tolist())
                self.assertEqual(fig.layout.meta["bin_interval"], "[left, right), except final bin [left, right]")
                self.assertEqual(fig.layout.meta["group_by"], ["Category"])
                self.assertEqual(fig.layout.meta["drawn_rows"], int(total.sum()))
                self.assertEqual(fig.layout.meta["plotted_points"], 10)
                self.assertEqual(fig.layout.meta["omitted_categories"], [])

    def test_histogram_eligible_rows_are_finite_and_have_requested_color(self):
        for dtype in ("float64", "Float64"):
            with self.subTest(dtype=dtype):
                df = pd.DataFrame({
                    "Revenue": pd.Series([0., 1., 2., 3., 4., 5., np.nan, np.inf, -np.inf, -999., 999.],
                                         dtype=dtype),
                    "Category": pd.Categorical(["A", "A", "B", "B", "B", "A", "C", "D", "E", None, None],
                                               categories=["A", "B", "C", "D", "E"] +
                                               [f"Unused {i}" for i in range(30)]),
                })
                original = df.copy(deep=True)
                for color, eligible in (("Category", np.arange(6, dtype=float)),
                                        (None, np.array([0., 1., 2., 3., 4., 5., -999., 999.]))):
                    spec = ChartSpec("histogram", "Finite", "Revenue", color=color, extra={"bins": 5})
                    fig = build_figure(df, spec)
                    counts, edges = np.histogram(eligible, bins=5)
                    self.assertEqual(fig.layout.meta["input_rows"], len(df))
                    self.assertEqual(fig.layout.meta["eligible_rows"], len(eligible))
                    self.assertEqual(fig.layout.meta["drawn_rows"], len(eligible))
                    self.assertEqual(fig.layout.meta["bin_edges"], edges.tolist())
                    self.assertEqual(fig.layout.meta["bin_counts"], counts.tolist())
                    self.assertEqual(sum(sum(trace.y) for trace in fig.data), len(eligible))
                    self.assertEqual(len(fig.data), 2 if color else 1)
                assert_frame_equal(df, original)

    def test_histogram_constant_single_and_nullable_values_match_numpy(self):
        series = (pd.Series([2., 2., None], dtype="Float64"),
                  pd.Series([2, None], dtype="Int64"),
                  pd.Series([0.1, 0.2, 0.3, None], dtype="Float32"))
        for values in series:
            with self.subTest(dtype=values.dtype):
                valid = values.dropna().to_numpy(dtype=values.dtype.numpy_dtype)
                counts, edges = np.histogram(valid, bins=5)
                spec = ChartSpec("histogram", "Constant or nullable", "Revenue", extra={"bins": 5})
                fig = build_figure(pd.DataFrame({"Revenue": values}), spec)
                self.assertEqual(fig.layout.meta["bin_edges"], edges.tolist())
                self.assertEqual(fig.layout.meta["bin_counts"], counts.tolist())
                np.testing.assert_array_equal(fig.data[0].y, counts)
                self.assertEqual(fig.layout.meta["drawn_rows"], len(valid))

    def test_histogram_serializes_only_bin_aggregates_not_raw_rows(self):
        df = pd.DataFrame({"Revenue": np.tile([0.123456789, 1.23456789, 2.3456789], 1_000),
                           "Private": "source-row-payload-must-not-be-serialized"})
        spec = ChartSpec("histogram", "Aggregated", "Revenue", extra={"bins": 5})
        fig = build_figure(df, spec)
        counts, edges = np.histogram(df["Revenue"].to_numpy(), bins=5)
        payload = fig.to_json()
        serialized = json.loads(payload)
        self.assertEqual(len(serialized["data"]), 1)
        trace = serialized["data"][0]
        self.assertEqual(trace["type"], "bar")
        self.assertEqual(trace["x"], (edges[:-1] + np.diff(edges) / 2).tolist())
        self.assertEqual(trace["y"], counts.tolist())
        self.assertEqual(trace["width"], np.diff(edges).tolist())
        self.assertEqual(len(trace["customdata"]), 5)
        self.assertTrue(all(len(row) == 3 for row in trace["customdata"]))
        self.assertNotIn("nbinsx", trace)
        self.assertNotIn("nbinsy", trace)
        self.assertNotIn("source-row-payload-must-not-be-serialized", payload)
        self.assertEqual(serialized["layout"]["meta"]["kind"], "histogram")
        self.assertEqual(serialized["layout"]["meta"]["drawn_rows"], 3_000)
        self.assertEqual(serialized["layout"]["meta"]["plotted_points"], 5)

    def test_histogram_rejects_empty_eligible_data_and_excessive_colors(self):
        spec = ChartSpec("histogram", "Limits", "Revenue", color="Category", extra={"bins": 5})
        for values, colors in (([], []), ([np.nan, np.inf, -np.inf], ["A", "B", "C"]), ([1.], [None])):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "no finite values"):
                build_figure(pd.DataFrame({"Revenue": pd.Series(values, dtype="Float64"),
                                           "Category": colors}), spec)
        df = pd.DataFrame({"Revenue": np.arange(21, dtype=float), "Category": range(21)})
        with self.assertRaisesRegex(ValueError, "Histogram color column 'Category'.*21.*at most 20"):
            build_figure(df, spec)
        fig = build_figure(df.iloc[:20], spec)
        self.assertEqual(len(fig.data), 20)
        self.assertEqual(sum(sum(trace.y) for trace in fig.data), 20)
        self.assertEqual(fig.layout.meta["drawn_rows"], 20)
        self.assertEqual(fig.layout.meta["omitted_categories"], [])

    def test_box_orientation_and_no_numeric_truncation(self):
        box = self.parse(chart="box", orientation="h")
        fig = build_figure(self.df, box.spec)
        self.assertEqual(list(fig.data[0].x), [10., 20., 30., 40., 50.])
        self.assertEqual(list(fig.data[0].y), ["A", "A", "B", "B", "C"])
        many = pd.DataFrame({"Revenue": range(30)})
        fig = build_figure(many, ChartSpec("box", "Raw", "Revenue", extra={"orientation": "v"}))
        self.assertEqual(list(fig.data[0].y), list(range(30)))
        self.assertEqual(fig.layout.meta["drawn_rows"], 30)

    def test_palette_changes_correlation_scale(self):
        spec = ChartSpec("correlation", "Colors", extra={"columns": ["Revenue", "Cost"], "palette": "viridis"})
        fig = build_figure(self.df, spec)
        self.assertEqual(fig.layout.coloraxis.colorscale[0][1], px.colors.sequential.Viridis[0])
        self.assertEqual(fig.layout.meta["palette"], "viridis")

    def test_line_and_area_sort_numeric_x_not_aggregate_value(self):
        df = pd.DataFrame({"X": [3, 1, 2], "Revenue": [2., 100., 9.]})
        for kind in ("line", "area"):
            fig = build_figure(df, ChartSpec(kind, "Ordered", "X", "Revenue"))
            self.assertEqual(list(fig.data[0].x), [1, 2, 3])
            self.assertEqual(list(fig.data[0].y), [100., 9., 2.])

    def test_pie_preserves_labels_and_merges_only_additive_values(self):
        categories = ["Other", "A", "B", "C", "D", "E", "F", "G", "H"]
        df = pd.DataFrame({"Category": categories, "Revenue": list(range(9, 0, -1))})
        spec = ChartSpec("pie", "Slices", "Category", "Revenue")
        fig = build_figure(df, spec)
        self.assertEqual(list(fig.data[0].labels), ["Other", "A", "B", "C", "D", "E", "F", "Other (2)"])
        self.assertEqual(list(fig.data[0].values), [9, 8, 7, 6, 5, 4, 3, 3])
        self.assertEqual(fig.layout.meta["merged_categories"], ["G", "H"])
        self.assertEqual(fig.layout.meta["omitted_categories"], [])
        self.assertEqual(fig.layout.meta["drawn_rows"], 9)
        for agg in ("mean", "median", "min", "max"):
            fig = build_figure(df, replace(spec, agg=agg))
            self.assertEqual(list(fig.data[0].labels), categories)
            self.assertEqual(list(fig.data[0].values), list(range(9, 0, -1)))
            self.assertEqual(fig.layout.meta["merged_categories"], [])
        names = ["A shared long category prefix one", "A shared long category prefix two"]
        long = pd.DataFrame({"Category": names, "Revenue": [2., 1.]})
        fig = build_figure(long, spec)
        self.assertEqual(list(fig.data[0].labels), names)
        self.assertEqual(list(fig.data[0].text), [name[:23] + "…" for name in names])

    def test_pie_count_remainder_and_negative_rejection(self):
        df = pd.DataFrame({"Category": list("ABCDEFGHI"), "Revenue": [-1.] * 9})
        spec = ChartSpec("donut", "Count", "Category", "Revenue", agg="count")
        fig = build_figure(df, spec)
        self.assertEqual(list(fig.data[0].labels), list("ABCDEFG") + ["Other"])
        self.assertEqual(list(fig.data[0].values), [1, 1, 1, 1, 1, 1, 1, 2])
        self.assertEqual(fig.layout.meta["merged_categories"], ["H", "I"])
        for agg in ("sum", "mean"):
            with self.assertRaisesRegex(ValueError, "negative"):
                build_figure(df, replace(spec, agg=agg))
        with self.assertRaisesRegex(ValueError, "positive"):
            build_figure(df.assign(Revenue=0.), replace(spec, agg="sum"))

    def test_scatter_sampling_metadata_and_exact_deterministic_sample(self):
        df = pd.DataFrame({"Revenue": range(MAX_SCATTER_POINTS + 3), "Cost": range(MAX_SCATTER_POINTS + 3)}, dtype=float)
        df.loc[len(df)] = [float("nan"), float("nan")]
        spec = ChartSpec("scatter", "Sample", "Revenue", "Cost")
        fig = build_figure(df, spec)
        expected = df.dropna().sample(MAX_SCATTER_POINTS, random_state=0)
        self.assertEqual(list(fig.data[0].x), expected["Revenue"].tolist())
        self.assertEqual(list(fig.data[0].y), expected["Cost"].tolist())
        meta = fig.layout.meta
        self.assertEqual({k: meta[k] for k in ("input_rows", "eligible_rows", "drawn_rows", "sampled", "sample_seed")},
                         {"input_rows": 5004, "eligible_rows": 5003, "drawn_rows": 5000,
                          "sampled": True, "sample_seed": 0})


if __name__ == "__main__":
    unittest.main()