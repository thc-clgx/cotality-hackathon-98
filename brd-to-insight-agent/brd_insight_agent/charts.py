"""Turn a ChartSpec into a Plotly figure."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .profiler import is_rate_like, profile_frame
from .spec import ChartSpec

CHART_KINDS = [
    "bar",
    "grouped_bar",
    "stacked_bar",
    "line",
    "area",
    "scatter",
    "pie",
    "donut",
    "histogram",
    "box",
    "correlation",
]

AGGS = ["sum", "mean", "median", "count", "min", "max"]
TIME_GRAINS = {"Auto": None, "Day": "D", "Week": "W", "Month": "M", "Quarter": "Q", "Year": "Y"}
AGGREGATING_KINDS = {"bar", "grouped_bar", "stacked_bar", "line", "area", "pie", "donut"}
ORIENTATIONS = ["auto", "h", "v"]
PALETTES = {"default": None, "blue": "Blues", "viridis": "Viridis", "red_blue": "RdBu_r"}

MAX_SCATTER_POINTS = 5_000
MAX_PIE_SLICES = 8
MAX_HISTOGRAM_COLORS = 20
MAX_CATEGORICAL_TRACES = 100
LABEL_TRUNCATE = 24
HORIZONTAL_BAR_LABEL_LEN = 16
TITLE_WRAP = 62


def build_figure(
    df: pd.DataFrame,
    spec: ChartSpec,
    top_n: int = 15,
    template: str = "plotly_white",
) -> go.Figure:
    _validate_spec(df, spec)
    kind = spec.kind
    horizontal = False

    if kind == "correlation":
        fig = _correlation(df, spec)
    elif kind == "histogram":
        fig = _histogram(df, spec)
        horizontal = fig.data[0].orientation == "h"
    elif kind == "box":
        fig = _box(df, spec)
        horizontal = fig.data[0].orientation == "h"
    elif kind == "scatter":
        fig = _scatter(df, spec)
    elif kind in ("pie", "donut"):
        fig = _pie(df, spec, kind)
    elif kind in ("line", "area"):
        fig = _timeseries(df, spec, kind)
    else:
        top_n = spec.extra.get("top_n", top_n)
        _bounded_int(top_n, "top_n", 1, 40)
        fig, horizontal = _bar(df, spec, kind, top_n)

    facts = dict(fig.layout.meta or {})
    mapping = {entry["value"]: entry["color"] for entry in spec.extra.get("category_colors", [])}
    if mapping and kind in {"pie", "donut"}:
        for trace in fig.data:
            palette = _colors(spec)["color_discrete_sequence"]
            trace.marker.colors = [mapping.get(str(label), palette[index % len(palette)]) for index, label in enumerate(trace.labels)]
    elif mapping and kind in {"bar", "grouped_bar", "stacked_bar"} and spec.color is None:
        for trace in fig.data:
            labels = trace.y if horizontal else trace.x
            palette = _colors(spec)["color_discrete_sequence"]
            trace.marker.color = [mapping.get(str(label), palette[index % len(palette)]) for index, label in enumerate(labels)]
    facts.update(kind=kind, aggregation=spec.agg if kind in AGGREGATING_KINDS else None,
                 palette=spec.extra.get("palette", "default"),
                 orientation=("h" if horizontal else "v")
                 if kind in {"bar", "grouped_bar", "stacked_bar", "histogram", "box"} else None)
    fig.update_layout(meta=facts)
    _apply_layout(fig, spec, template, horizontal=horizontal)
    return fig


def _numeric(series: pd.Series) -> bool:
    return (pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
            and not pd.api.types.is_complex_dtype(series))


def _bounded_int(value, name: str, lower: int, upper: int) -> None:
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"'{name}' must be an integer between {lower} and {upper}.")


def _correlation_columns(df: pd.DataFrame, spec: ChartSpec, profiles=None) -> list[str]:
    requested = spec.extra.get("columns")
    if not isinstance(requested, list) or any(not isinstance(c, str) for c in requested):
        raise ValueError("Correlation needs an explicit list of numeric non-ID columns.")
    cols = list(dict.fromkeys(requested))
    if len(cols) < 2:
        raise ValueError("Correlation needs at least two unique numeric non-ID columns.")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Column(s) not in this sheet: {', '.join(missing)}")
    actual_profiles = profile_frame(df[cols])
    bad = [c for c in cols if not _numeric(df[c]) or actual_profiles[c].is_identifier
           or (profiles is not None and c in profiles and profiles[c].is_identifier)]
    if bad:
        raise ValueError(f"Correlation requires numeric non-ID columns; invalid: {', '.join(bad)}")
    return cols


def _validate_spec(df: pd.DataFrame, spec: ChartSpec, profiles=None) -> None:
    if not df.columns.is_unique:
        raise ValueError("Column names must be unique.")
    if spec.kind not in CHART_KINDS:
        raise ValueError(f"'{spec.kind}' is not a supported chart type.")
    if spec.agg not in AGGS:
        raise ValueError(f"'{spec.agg}' is not a supported aggregation.")
    if not isinstance(spec.extra, dict):
        raise ValueError("Chart extras must be an object.")
    orientation = spec.extra.get("orientation", "auto")
    if orientation not in ORIENTATIONS:
        raise ValueError("Orientation must be auto, h, or v.")
    oriented = {"bar", "grouped_bar", "stacked_bar", "histogram", "box"}
    if orientation != "auto" and spec.kind not in oriented:
        raise ValueError(f"Orientation overrides are not supported for '{spec.kind}'.")
    palette = spec.extra.get("palette", "default")
    if not isinstance(palette, str) or palette not in PALETTES:
        raise ValueError("Palette must be default, blue, viridis, or red_blue.")
    colors = spec.extra.get("colors", [])
    assignments = spec.extra.get("category_colors", [])
    if not isinstance(colors, list) or len(colors) > 20:
        raise ValueError("Provide at most 20 custom colors.")
    if not isinstance(assignments, list) or len(assignments) > 100:
        raise ValueError("Provide at most 100 category color assignments.")
    for color in colors:
        _validate_color(color)
    seen = set()
    for assignment in assignments:
        if not isinstance(assignment, dict) or set(assignment) != {"value", "color"} or not isinstance(assignment["value"], str):
            raise ValueError("Category colors need a category value and a CSS color.")
        if assignment["value"] in seen:
            raise ValueError("Each category may have only one color assignment.")
        seen.add(assignment["value"])
        _validate_color(assignment["color"])
    if assignments:
        column = spec.x if spec.kind in {"pie", "donut"} or (spec.kind in {"bar", "grouped_bar", "stacked_bar"} and spec.color is None) else spec.color
        if column not in df.columns:
            raise ValueError("Category colors require a category/color column.")
        if spec.kind not in {"pie", "donut"} and _numeric(df[column]) and spec.kind in {"bar", "grouped_bar", "stacked_bar", "scatter"}:
            raise ValueError("Numeric color scales use custom colors, not category assignments.")
        actual_values = set(df[column].dropna().astype(str))
        if seen - actual_values:
            raise ValueError("A requested category color names a value not present in this data.")
    _bounded_int(spec.extra.get("bins", 30), "bins", 5, 100)
    if "top_n" in spec.extra:
        _bounded_int(spec.extra["top_n"], "top_n", 1, 40)
    if spec.extra.get("grain") not in (None, "auto", "D", "W", "M", "Q", "Y"):
        raise ValueError("Unsupported time grain.")
    axes = [c for c in (spec.x, spec.y, spec.color) if c is not None]
    if any(not isinstance(c, str) for c in axes):
        raise ValueError("Axis columns must be names.")
    missing = [c for c in axes if c not in df.columns]
    if missing:
        raise ValueError(f"Column(s) not in this sheet: {', '.join(missing)}")
    if spec.color and not _numeric(df[spec.color]) and df[spec.color].nunique() > MAX_CATEGORICAL_TRACES:
        raise ValueError(f"Requested color split exceeds {MAX_CATEGORICAL_TRACES} categories. Filter the data or choose another split; the split was not removed.")
    if spec.kind == "correlation":
        _correlation_columns(df, spec, profiles)
        return
    if spec.kind != "box" or spec.y is None:
        _require(spec.x, "Chart needs an X column.")
    numeric_axes = []
    if spec.kind == "scatter":
        _require(spec.y, "Scatter needs both an X and a Y column.")
        numeric_axes = [spec.x, spec.y]
    elif spec.kind == "histogram":
        numeric_axes = [spec.x]
    elif spec.kind == "box":
        numeric_axes = [spec.y or spec.x]
    elif spec.kind in AGGREGATING_KINDS:
        if spec.color is not None and spec.color == spec.x:
            raise ValueError("X and color must use distinct grouping columns.")
        if spec.agg != "count":
            _require(spec.y, f"Aggregation '{spec.agg}' needs a numeric measure.")
            numeric_axes = [spec.y]
        if spec.agg == "sum" and spec.y and is_rate_like(spec.y):
            raise ValueError(f"Cannot sum rate-like measure '{spec.y}'; choose mean or another aggregation.")
    for col in numeric_axes:
        if col is None or not _numeric(df[col]):
            raise ValueError(f"'{col}' must be a numeric column for {spec.kind}.")
        if profile_frame(df[[col]])[col].is_identifier:
            raise ValueError(f"'{col}' is an identifier, not a measure.")
    if spec.kind in ("pie", "donut") and spec.color is not None and spec.color != spec.x:
        raise ValueError("Pie slices are colored by their category. A second split cannot be silently discarded; use a grouped bar chart.")
    if spec.kind in ("pie", "donut") and spec.agg != "count" and df[spec.y].lt(0).any():
        raise ValueError("Pie and donut charts cannot contain negative values.")


def _validate_color(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 80:
        raise ValueError("Colors must be CSS color names, hex, rgb, or rgba strings.")
    try:
        go.scatter.Marker(color=value)
    except ValueError as exc:
        raise ValueError(f"Unsupported CSS color: {value!r}.") from exc


def _colors(spec: ChartSpec, continuous: bool = False) -> dict:
    name = spec.extra.get("palette", "default")
    sequences = {"default": px.colors.qualitative.Plotly, "blue": px.colors.sequential.Blues[2:],
                 "viridis": px.colors.sequential.Viridis,
                 "red_blue": px.colors.diverging.RdBu[::-1]}
    custom = spec.extra.get("colors", [])
    result = {"color_discrete_sequence": custom or sequences[name]}
    mapping = {entry["value"]: entry["color"] for entry in spec.extra.get("category_colors", [])}
    if mapping:
        result["color_discrete_map"] = mapping
    if continuous:
        result["color_continuous_scale"] = (px.colors.make_colorscale(custom if len(custom) > 1 else custom * 2)
                                            if custom else PALETTES[name] or "Viridis")
    return result


def _facts(fig: go.Figure, df: pd.DataFrame, eligible: pd.DataFrame,
           drawn: pd.DataFrame, **extra) -> go.Figure:
    """Drawn rows count represented source rows, not aggregate groups."""
    meta = {"input_rows": len(df), "eligible_rows": len(eligible), "drawn_rows": len(drawn),
            "plotted_points": len(drawn), "sampled": False,
            "omitted_categories": [], "merged_categories": []}
    meta.update(extra)
    fig.update_layout(meta=meta)
    return fig


def _histogram(df: pd.DataFrame, spec: ChartSpec) -> go.Figure:
    data = df.dropna(subset=[c for c in (spec.x, spec.color) if c is not None])
    # Preserve native precision, including pandas nullable numeric dtypes.
    values = data[spec.x].to_numpy(dtype=getattr(data[spec.x].dtype, "numpy_dtype", None))
    finite = np.isfinite(values)
    data, values = data.loc[finite], values[finite]
    if data.empty:
        raise ValueError(f"'{spec.x}' has no finite values to plot with non-missing grouping values.")
    if spec.color is not None:
        color_count = data[spec.color].nunique()
        if color_count > MAX_HISTOGRAM_COLORS:
            raise ValueError(
                f"Histogram color column '{spec.color}' has {color_count} eligible categories; "
                f"at most {MAX_HISTOGRAM_COLORS} are supported. "
                "Filter the data or choose another color column."
            )

    bins = spec.extra.get("bins", 30)
    counts, edges = np.histogram(values, bins=bins)
    widths = np.diff(edges)
    centers = (edges[:-1] + widths / 2).tolist()
    intervals = [[left, right, "]" if i == bins - 1 else ")"]
                 for i, (left, right) in enumerate(zip(edges[:-1].tolist(), edges[1:].tolist()))]
    groups = [(None, values)]
    if spec.color is not None:
        groups = [(label, values[indices]) for label, indices in
                  data.groupby(spec.color, observed=True, sort=False).indices.items()]
    palette = _colors(spec).get("color_discrete_sequence", px.colors.qualitative.Plotly)
    horizontal = spec.extra.get("orientation") == "h"
    count_axis = "x" if horizontal else "y"
    fig = go.Figure()
    for i, (label, group_values) in enumerate(groups):
        group_counts = counts if spec.color is None else np.histogram(group_values, bins=edges)[0]
        fig.add_trace(go.Bar(
            x=group_counts.tolist() if horizontal else centers,
            y=centers if horizontal else group_counts.tolist(),
            width=widths.tolist(), orientation="h" if horizontal else "v",
            name=str(label) if spec.color is not None else "",
            marker_color={entry["value"]: entry["color"] for entry in spec.extra.get("category_colors", [])}.get(str(label), palette[i % len(palette)]), customdata=intervals,
            hovertemplate=(f"{spec.x}: [%{{customdata[0]}}, %{{customdata[1]}}%{{customdata[2]}}"
                           f"<br>count=%{{{count_axis}:d}}<extra>%{{fullData.name}}</extra>"),
        ))
    fig.update_layout(barmode="stack", bargap=0,
                      xaxis_title="count" if horizontal else spec.x,
                      yaxis_title=spec.x if horizontal else "count")
    return _facts(fig, df, data, data, bins=bins, binning="numpy.histogram",
                  bin_edges=edges.tolist(), bin_counts=counts.tolist(),
                  bin_interval="[left, right), except final bin [left, right]",
                  count_semantics="rows", group_by=[spec.color] if spec.color is not None else [],
                  plotted_points=bins * len(groups),
                  missing_policy="finite X; non-missing color when grouped")


def _box(df: pd.DataFrame, spec: ChartSpec) -> go.Figure:
    cols = list(dict.fromkeys(c for c in (spec.x, spec.y, spec.color) if c))
    _require(cols, "Box plot needs at least one column.")
    data = df.dropna(subset=cols)
    if data.empty:
        raise ValueError("No rows left after dropping blanks.")
    eligible = data
    omitted = []
    top_n = spec.extra.get("top_n", 20)
    if spec.x and spec.y and data[spec.x].nunique() > top_n:
        keep = data[spec.x].value_counts().nlargest(top_n).index
        omitted = [str(v) for v in data[spec.x].unique() if v not in keep]
        data = data[data[spec.x].isin(keep)]
    orientation = spec.extra.get("orientation", "auto")
    if orientation == "auto":
        orientation = "v" if spec.y else "h"
    measure = spec.y or spec.x
    category = spec.x if spec.y else None
    x, y = (measure, category) if orientation == "h" else (category, measure)
    fig = px.box(data, x=x, y=y, color=spec.color, orientation=orientation, **_colors(spec))
    return _facts(fig, df, eligible, data, omitted_categories=omitted, top_n=top_n)


def _scatter(df: pd.DataFrame, spec: ChartSpec) -> go.Figure:
    color = spec.color
    data = df.dropna(subset=[c for c in (spec.x, spec.y, color) if c])
    if data.empty:
        raise ValueError("No rows with both values present.")
    eligible = data
    if len(data) > MAX_SCATTER_POINTS:
        data = data.sample(MAX_SCATTER_POINTS, random_state=0)
    fig = px.scatter(data, x=spec.x, y=spec.y, color=color, opacity=0.6, **_colors(spec, continuous=True))
    return _facts(fig, df, eligible, data, sampled=len(data) < len(eligible),
                  sample_limit=MAX_SCATTER_POINTS, sample_seed=0,
                  color_omitted=False)


def _pie(df: pd.DataFrame, spec: ChartSpec, kind: str) -> go.Figure:
    slice_spec = ChartSpec(**{**spec.__dict__, "color": None})
    data = _aggregate(df, slice_spec, top_n=None)
    value = data.columns[-1]
    meta = dict(data.attrs["chart_facts"])

    data = data.dropna(subset=[value]).sort_values(value, ascending=False, kind="stable")
    if data[value].lt(0).any():
        raise ValueError("Pie and donut charts cannot contain negative values.")
    if not data[value].gt(0).any():
        raise ValueError("Pie and donut charts need at least one positive value.")
    if len(data) > MAX_PIE_SLICES and spec.agg in ("sum", "count"):
        head = data.head(MAX_PIE_SLICES - 1).copy()
        tail = data.iloc[MAX_PIE_SLICES - 1:]
        rest = tail[value].sum(min_count=1)
        other, suffix = "Other", 2
        labels = set(data[spec.x].astype(str))
        while other in labels:
            other, suffix = f"Other ({suffix})", suffix + 1
        meta.update(merged_categories=tail[spec.x].astype(str).tolist(), merged_label=other)
        data = pd.concat([head, pd.DataFrame({spec.x: [other], value: [rest]})], ignore_index=True)

    fig = px.pie(data, names=spec.x, values=value,
                 color=spec.x if spec.extra.get("category_colors") else None,
                 hole=0.45 if kind == "donut" else 0.0, **_colors(spec))
    fig.update_traces(
        text=[_short(label) for label in data[spec.x]],
        textposition="inside",
        texttemplate="%{percent}",
        insidetextorientation="horizontal",
        hovertemplate="%{label}<br>%{value:,.4~g} (%{percent})<extra></extra>",
    )
    meta.update(plotted_points=len(data))
    fig.update_layout(uniformtext_minsize=11, uniformtext_mode="hide", meta=meta)
    return fig


def _timeseries(df: pd.DataFrame, spec: ChartSpec, kind: str) -> go.Figure:
    _require(spec.x, "Line chart needs an X column.")
    grain = spec.extra.get("grain")
    if grain in (None, "auto"):
        grain = _auto_grain(df[spec.x])
    data = _aggregate(df, spec, top_n=None, time_grain=grain)
    value = data.columns[-1]
    _require(data[value].notna().any(), "No non-null measure values to plot.")
    if _numeric(df[spec.x]):
        data = data.sort_values(spec.x, kind="stable")
    plot = px.line if kind == "line" else px.area
    fig = plot(data, x=spec.x, y=value, color=spec.color, **_colors(spec))
    fig.update_traces(mode="lines+markers" if len(data) <= 40 else "lines")
    fig.update_layout(meta=data.attrs["chart_facts"])
    return fig


def _bar(df: pd.DataFrame, spec: ChartSpec, kind: str, top_n: int) -> tuple[go.Figure, bool]:
    data = _aggregate(df, spec, top_n=top_n)
    value = data.columns[-1]
    _require(data[value].notna().any(), "No non-null measure values to plot.")
    labels = data[spec.x].astype(str)
    horizontal = bool(labels.str.len().max() > HORIZONTAL_BAR_LABEL_LEN or len(data) > 18)
    orientation = spec.extra.get("orientation", "auto")
    if orientation != "auto":
        horizontal = orientation == "h"
    barmode = "relative" if kind == "stacked_bar" else "group"

    if horizontal:
        data = data.sort_values(value, kind="stable")
        fig = px.bar(data, x=value, y=spec.x, color=spec.color, barmode=barmode,
                     orientation="h", **_colors(spec, continuous=True))
    else:
        fig = px.bar(data, x=spec.x, y=value, color=spec.color, barmode=barmode,
                     orientation="v", **_colors(spec, continuous=True))

    if len(data) <= 12 and not spec.color:
        fig.update_traces(
            text=[_compact(v) for v in data[value]],
            textposition="outside",
            cliponaxis=False,
        )
    fig.update_layout(meta=data.attrs["chart_facts"])
    return fig, horizontal


def _correlation(df: pd.DataFrame, spec: ChartSpec) -> go.Figure:
    cols = _correlation_columns(df, spec)
    corr = df[cols].corr().round(2)
    if not any(pd.notna(corr.iloc[i, j]) for i in range(len(cols)) for j in range(i)):
        raise ValueError("Selected columns have no defined pairwise correlation.")

    fig = px.imshow(
        corr.values,
        x=cols,
        y=cols,
        text_auto=".2f" if len(cols) <= 8 else False,
        aspect="auto",
        color_continuous_scale=(_colors(spec, continuous=True)["color_continuous_scale"]
                    if spec.extra.get("colors") else PALETTES[spec.extra.get("palette", "default")] or "RdBu_r"),
        zmin=-1,
        zmax=1,
    )
    fig.update_xaxes(tickangle=-45, tickmode="array", tickvals=cols, ticktext=[_short(c, 18) for c in cols])
    fig.update_yaxes(tickmode="array", tickvals=cols, ticktext=[_short(c, 18) for c in cols])
    present = df[cols].notna().astype(int)
    eligible = df[present.sum(axis=1).ge(2)]
    return _facts(fig, df, eligible, eligible, columns=cols,
                  plotted_points=int(corr.notna().to_numpy().sum()),
                  missing_policy="pairwise; eligible rows have at least two selected values",
                  pairwise_counts=present.T.dot(present).values.tolist(),
                  undefined_columns=corr.columns[corr.isna().all()].tolist())


def _aggregate(
    df: pd.DataFrame,
    spec: ChartSpec,
    top_n: int | None,
    time_grain: str | None = None,
) -> pd.DataFrame:
    _validate_spec(df, spec)
    if top_n is not None:
        _bounded_int(top_n, "top_n", 1, 40)
    if time_grain not in (None, "D", "W", "M", "Q", "Y"):
        raise ValueError("Unsupported time grain.")
    keys = [c for c in (spec.x, spec.color) if c]
    _require(keys, "Chart needs at least an X column.")
    missing = [c for c in keys + ([spec.y] if spec.y else []) if c not in df.columns]
    if missing:
        raise ValueError(f"Column(s) not in this sheet: {', '.join(missing)}")

    data = df.dropna(subset=keys).copy()
    if data.empty:
        raise ValueError("No rows left after dropping blanks.")

    x_is_time = pd.api.types.is_datetime64_any_dtype(data[spec.x]) if spec.x else False
    if time_grain and x_is_time:
        data[spec.x] = data[spec.x].dt.to_period(time_grain).dt.to_timestamp()

    base = "count" if spec.agg == "count" else spec.y
    value, suffix = base, 2
    while value in keys:
        value, suffix = f"{base}_{suffix}", suffix + 1
    grouped = data.groupby(keys, dropna=False, observed=True, sort=False)
    if spec.agg == "count":
        out = grouped.size().rename(value).reset_index()
        eligible = data
    else:
        measures = grouped[spec.y]
        result = measures.sum(min_count=1) if spec.agg == "sum" else measures.agg(spec.agg)
        out = result.rename(value).reset_index()
        eligible = data.dropna(subset=[spec.y])

    if top_n and not x_is_time:
        # Rank groups without adding nonadditive statistics.
        keep = out[value].abs().groupby(out[spec.x], observed=True).max().nlargest(top_n).index
        out = out[out[spec.x].isin(keep)]

    out = out.sort_values(spec.x if x_is_time else value, ascending=x_is_time, kind="stable")
    valid = out.dropna(subset=[value])
    drawn = eligible.merge(valid[keys].drop_duplicates(), on=keys, how="inner")
    kept = valid[spec.x].unique()
    out.attrs["chart_facts"] = {
        "input_rows": len(df), "eligible_rows": len(eligible), "drawn_rows": len(drawn),
        "plotted_points": len(valid), "sampled": False, "aggregation": spec.agg,
        "count_semantics": "rows" if spec.agg == "count" else None,
        "group_by": keys, "value_column": value, "grain": time_grain,
        "top_n": top_n if not x_is_time else None,
        "omitted_categories": [str(v) for v in data[spec.x].unique() if v not in kept],
        "merged_categories": [],
    }
    return out


def _apply_layout(fig: go.Figure, spec: ChartSpec, template: str, horizontal: bool) -> None:
    title = _wrap(spec.title)
    is_pie = spec.kind in ("pie", "donut")

    fig.update_layout(
        template=template,
        title=dict(text=title, x=0.01, xanchor="left", y=0.97, yanchor="top", font=dict(size=17)),
        # Keep the legend clear of the title.
        showlegend=_legend_entry_count(fig) > 1 or is_pie,
        legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02,
                    title_text="", font=dict(size=11)),
        margin=dict(l=10, r=10, t=40 + 20 * title.count("<br>"), b=10),
        height=470,
        hovermode="closest" if spec.kind in ("scatter", "histogram", "box") else "x unified",
        uirevision=spec.key,
    )

    if is_pie or spec.kind == "correlation":
        fig.update_layout(margin=dict(l=10, r=10, t=66, b=10))
        return

    fig.update_xaxes(automargin=True, title_standoff=12)
    fig.update_yaxes(automargin=True, title_standoff=12)

    if spec.kind == "histogram":
        # Histogram bin positions are numeric, not categorical bar labels.
        measure_axis = fig.update_yaxes if horizontal else fig.update_xaxes
        count_axis = fig.update_xaxes if horizontal else fig.update_yaxes
        measure_axis(type="linear", tickformat="~s", separatethousands=True)
        count_axis(type="linear", tickformat=",d", separatethousands=True, rangemode="tozero")
        return

    if _axis_is_categorical(fig, horizontal):
        _truncate_ticks(fig, horizontal)
        if not horizontal:
            fig.update_xaxes(tickangle=-35)

    value_axis = fig.update_xaxes if horizontal else fig.update_yaxes
    value_axis(tickformat="~s", separatethousands=True)


def _axis_is_categorical(fig: go.Figure, horizontal: bool) -> bool:
    for trace in fig.data:
        values = getattr(trace, "y" if horizontal else "x", None)
        if values is not None and len(values) > 0 and isinstance(values[0], str):
            return True
    return False


def _truncate_ticks(fig: go.Figure, horizontal: bool) -> None:
    seen: list[str] = []
    for trace in fig.data:
        values = getattr(trace, "y" if horizontal else "x", None)
        if values is None:
            continue
        for v in values:
            if isinstance(v, str) and v not in seen:
                seen.append(v)
    if not seen or max(len(s) for s in seen) <= LABEL_TRUNCATE:
        return
    update = fig.update_yaxes if horizontal else fig.update_xaxes
    update(tickmode="array", tickvals=seen, ticktext=[_short(s) for s in seen])


def _legend_entry_count(fig: go.Figure) -> int:
    return len({t.name for t in fig.data if getattr(t, "name", None)})


def _auto_grain(series: pd.Series) -> str | None:
    """Daily points across several years render as unreadable noise, so widen the bucket."""
    if not pd.api.types.is_datetime64_any_dtype(series):
        return None
    valid = series.dropna()
    if valid.empty:
        return None
    span_days = (valid.max() - valid.min()).days
    if span_days > 1_460:
        return "Q"
    if span_days > 420:
        return "M"
    if span_days > 120:
        return "W"
    return "D"


def _short(text: object, limit: int = LABEL_TRUNCATE) -> str:
    s = str(text)
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _wrap(title: str, limit: int = TITLE_WRAP) -> str:
    lines, current = [], ""
    for word in str(title).split():
        if current and len(current) + len(word) + 1 > limit:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    lines.append(current)
    return "<br>".join(lines[:2])


def _compact(value: float) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    for threshold, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= threshold:
            return f"{v / threshold:.1f}{suffix}"
    return f"{v:,.0f}" if abs(v) >= 10 else f"{v:,.2f}"


def _require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)
