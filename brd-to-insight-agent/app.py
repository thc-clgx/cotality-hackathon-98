"""Select a workbook, ask for a chart, refine it, and download the result."""

from __future__ import annotations

import datetime as dt
import copy
import hashlib
import io
import json
import zipfile
import re
import os
import time

import pandas as pd
import streamlit as st

from pathlib import Path

from brd_insight_agent import gemini
from brd_insight_agent import brd as brd_agent
from brd_insight_agent import warehouse
from brd_insight_agent.answers import data_answer
from brd_insight_agent.export import ExportError, figures_to_images
from brd_insight_agent.filters import (
    MAX_SCOPE_FILTERS, boolean_filter, category_filter, date_filter, describe, filter_offers, number_filter,
)
from brd_insight_agent.intent import SpecError
from brd_insight_agent.loader import clean_frame, correct_column, detect_header_row, load_sheet, workbook_sheet_names
from brd_insight_agent.privacy import PrivacyError, PrivacyPolicy, default_allowed_columns, outbound_payload, prepare_privacy_index
from brd_insight_agent.profiler import profile_frame
from brd_insight_agent.recommendations import group_for, introduction, recommend_visuals, select_recommendation, type_message
from brd_insight_agent.workflow import (
    PLANNER_SCHEMA, PLANNER_SYSTEM, build_artifact, dataset_identity,
    chart_specs, fingerprint, image_title, local_response, rendered_message, revise_plan,
)

STATE_KEY = "excel_chart_chat_v3"
MAX_FILE_BYTES = 100 * 1024 * 1024
MAX_TOTAL_UPLOAD_BYTES = int(os.environ.get("CHART_MAX_UPLOAD_MB", "250")) * 1024 * 1024
MAX_FRAME_BYTES = int(os.environ.get("CHART_MAX_FRAME_MB", "256")) * 1024 * 1024
MAX_RESULT_BYTES = int(os.environ.get("CHART_MAX_RESULTS_MB", "128")) * 1024 * 1024
MAX_FILES, MAX_DATASETS, MAX_VERSIONS, MAX_MESSAGES = 10, 20, 30, 100
MAX_MESSAGE_CHARS = 8000


def _reserve(state, size):
    budget = state.setdefault("budget", {"used": 0})
    if budget["used"] + size > MAX_RESULT_BYTES:
        raise ValueError("Saved images/downloads reached the session budget. Download what you need, then clear saved history.")
    budget["used"] += size
    state["stored_bytes"] = state.get("stored_bytes", 0) + size


def _clear_saved(root):
    root["datasets"].clear()
    root.setdefault("budget", {"used": 0})["used"] = 0


def workspace():
    if STATE_KEY not in st.session_state:
        st.session_state.pop("excel_chart_workspace_v2", None)
        st.session_state[STATE_KEY] = {"datasets": {}, "sheet_names": {}, "loaded": None}
    root = st.session_state[STATE_KEY]
    if "budget" not in root:
        root["budget"] = {"used": 0}
        for state in root["datasets"].values():
            size = sum(len(version.figure_json.encode("utf-8")) + len(version.plan_json.encode("utf-8")) for version in state["versions"])
            size += sum(len(content) for _, content in state["exports"].values())
            state.update(stored_bytes=size, budget=root["budget"])
            root["budget"]["used"] += size
    return root


def conversation(root, identity):
    if identity not in root["datasets"]:
        if len(root["datasets"]) >= MAX_DATASETS:
            raise ValueError("Too many saved sheets. Clear saved history before opening another dataset.")
        root["datasets"][identity] = {"messages": [], "versions": [], "selected": None,
                                      "pending": None, "exports": {}, "epoch": 0, "last_payload": None,
                                      "context_start": 0, "stored_bytes": 0, "scope": [],
                                      "budget": root.setdefault("budget", {"used": 0})}
    return root["datasets"][identity]


def current(state):
    return next((v for v in state["versions"] if v.id == state["selected"]), None)


def ensure_recommendations(state, data):
    if state.get("recommendations", {}).get("dataset_id") == data["id"]:
        return
    bundle = recommend_visuals(data["frame"], data["profiles"], data["id"])
    message = introduction(bundle)
    _reserve(state, len(json.dumps(bundle, ensure_ascii=False).encode("utf-8"))
             + len(json.dumps(message, ensure_ascii=False).encode("utf-8")))
    state["recommendations"] = bundle
    state["messages"].insert(0, message)
    if state.get("context_start", 0):
        state["context_start"] += 1


def choose_type(state, data, chart_type):
    """List the suggestions this sheet supports for one chart type; no model call is needed."""
    bundle = state.get("recommendations", {})
    group = group_for(bundle, chart_type)
    if group is None or bundle.get("dataset_id") != data["id"]:
        return
    if len(state["messages"]) > MAX_MESSAGES - 2:
        state["notice"] = "History limit reached. Clear saved history to continue; no API call was made."
        return
    bundle["selected_type"] = chart_type
    message = type_message(bundle, chart_type)
    _reserve(state, len(json.dumps(message, ensure_ascii=False).encode("utf-8")))
    state["messages"].append({"role": "user", "text": f"{group['label']} options"})
    state["messages"].append(message)


def _context_message(message, bundle, policy):
    """Recommendation text sent as context is rebuilt under the current column approvals."""
    if message.get("recommendation_options"):
        rebuilt = type_message(bundle, message["recommendation_options"], set(policy.allowed_columns))
        return message if rebuilt is None else {**message, "context_text": rebuilt["context_text"]}
    if message.get("recommendation_types"):
        return {**message, "context_text": introduction(bundle)["context_text"]}
    return message


def reset_context(state, selected=None):
    state.update(selected=selected, pending=None, epoch=state["epoch"] + 1,
                 context_start=len(state["messages"]))


def filter_panel(state, data):
    """Build the data scope locally from real column values; no model call is involved."""
    if "filter_offers" not in data:
        data["filter_offers"] = filter_offers(data["frame"], data["profiles"])
    offers = data["filter_offers"]
    scope = []
    with st.expander("Filter data", expanded=bool(state.get("scope"))):
        if not offers:
            st.caption("No column in this sheet offers a filter. Ask in chat for anything more specific.")
            state["scope"] = []
            return
        st.caption("Filters apply to every new image and combine with filters you ask for in chat.")
        for offer in offers:
            column, kind = offer["column"], offer["kind"]
            key = f"scope_{data['id']}_{column}"
            if kind == "date":
                labels = ["All dates"] + [period["label"] for period in offer["periods"]] + ["Custom range"]
                choice = st.selectbox(f"{column} period", labels, key=key)
                if choice == "Custom range":
                    start, end = st.date_input(
                        f"{column} range", value=(dt.date.fromisoformat(offer["min"]), dt.date.fromisoformat(offer["max"])),
                        min_value=dt.date.fromisoformat(offer["min"]), max_value=dt.date.fromisoformat(offer["max"]),
                        key=f"{key}_range", format="YYYY-MM-DD")
                    scope.append(date_filter(column, start, end))
                elif choice != "All dates":
                    period = offer["periods"][labels.index(choice) - 1]
                    scope.append(date_filter(column, period["start"], period["end"]))
            elif kind == "category":
                chosen = st.multiselect(f"{column}", offer["values"], key=key,
                                        help="No selection means every value is included.")
                if chosen:
                    scope.append(category_filter(column, chosen))
            elif kind == "boolean":
                choice = st.selectbox(f"{column}", ["All", "True", "False"], key=key)
                if choice != "All":
                    scope.append(boolean_filter(column, choice == "True"))
            else:
                low, high = st.slider(f"{column}", offer["min"], offer["max"],
                                      (offer["min"], offer["max"]), key=key)
                if (low, high) != (offer["min"], offer["max"]):
                    scope.append(number_filter(column, low, high))
        if len(scope) > MAX_SCOPE_FILTERS:
            st.error(f"Use at most {MAX_SCOPE_FILTERS} panel filters. The scope was not applied.")
            state["scope"] = []
            return
        try:
            summary, matched = describe(data["frame"], data["profiles"], scope)
        except (SpecError, ValueError) as exc:
            st.error(f"{exc} The scope was not applied.")
            state["scope"] = []
            return
        state["scope"] = scope
        st.caption(f"{summary}. {matched:,} of {len(data['frame']):,} rows match.")
        if not matched:
            st.warning("No rows match this scope, so a new image cannot be built until you widen it.")


SOURCE_COLUMN = "Source"
MAX_COMPARE = 6
EXCEL_SOURCE = "Excel upload"
BRD_RUN_MODE = "Run a BRD"
LINK_MODE = "Paste a table link"
WAREHOUSE_SOURCE = "BRD to BigQuery"


def _sheet_frame(content, filename, sheet, header):
    return clean_frame(load_sheet(content, filename, sheet, header))


def _combined(parts):
    """Stack sheets on their shared columns and label each row with its file."""
    shared = [column for column in parts[0][1].columns
              if all(column in frame.columns for _, frame in parts[1:])]
    if not shared:
        raise ValueError("These sheets share no common columns, so they cannot be compared.")
    labelled = []
    for label, frame in parts:
        piece = frame[shared].copy()
        piece.insert(0, SOURCE_COLUMN, label)
        labelled.append(piece)
    return pd.concat(labelled, ignore_index=True), shared


def selected_data(root):
    with st.sidebar:
        if st.button("Clear saved chats and images", key="clear_saved"):
            _clear_saved(root)
            st.rerun()
        source = st.radio("Data source", [EXCEL_SOURCE, WAREHOUSE_SOURCE], key="data_source",
                          help="Excel keeps the original workflow. The BRD option runs the "
                               "BRD-to-BigQuery agent and charts the tables it creates.")
    if source == WAREHOUSE_SOURCE:
        return warehouse_data(root)
    return excel_data(root)


def excel_data(root):
    with st.sidebar:
        uploads = st.file_uploader("Excel files", type=["xlsx", "xlsm", "xls", "csv"],
                                   accept_multiple_files=True, max_upload_size=100, key="uploads")
    if not uploads:
        root.update(datasets={}, sheet_names={}, loaded=None, corrections={})
        root.setdefault("budget", {"used": 0})["used"] = 0
        return None
    if len(uploads) > MAX_FILES or sum(upload.size for upload in uploads) > MAX_TOTAL_UPLOAD_BYTES:
        st.error(f"Select at most {MAX_FILES} files totaling {MAX_TOTAL_UPLOAD_BYTES // (1024 * 1024)} MB. No files were partially loaded.")
        return None
    files = {}
    for upload in uploads:
        if upload.size > MAX_FILE_BYTES:
            st.error(f"{upload.name}: file exceeds the 100 MB limit.")
            continue
        content = upload.getvalue()
        token = fingerprint([upload.name, hashlib.sha256(content).hexdigest()])
        files[token] = (upload.name, content)
    if not files:
        return None
    # Browser uploads carry no folder, so identical names need a distinguishing suffix.
    counts = {}
    labels = {}
    for token, (name, content) in files.items():
        counts[name] = counts.get(name, 0) + 1
    seen = {}
    for token, (name, content) in files.items():
        if counts[name] > 1:
            seen[name] = seen.get(name, 0) + 1
            labels[token] = f"{name} (#{seen[name]}, {len(content) // 1024} KB)"
        else:
            labels[token] = name
    root["sheet_names"] = {key: value for key, value in root["sheet_names"].items() if key in files}
    with st.sidebar:
        file_id = st.selectbox("Workbook", list(files), format_func=lambda key: labels[key])
        filename, content = files[file_id]
        try:
            if file_id not in root["sheet_names"]:
                root["sheet_names"][file_id] = workbook_sheet_names(content, filename)
            sheet = st.selectbox("Sheet", root["sheet_names"][file_id], key=f"sheet_{file_id}")
            detected = detect_header_row(content, filename, sheet)
            with st.expander("Header setting"):
                st.caption(f"Detected header on row {detected} (0-based). Change it if the preview looks wrong.")
                header = int(st.number_input("Header row (0-based)", min_value=0, max_value=1000,
                                             value=detected, key=f"header_{file_id}_{sheet}_{detected}"))
            others = [key for key in files if key != file_id]
            compare = st.multiselect("Compare with", others, format_func=lambda key: labels[key],
                                     max_selections=MAX_COMPARE - 1, key=f"compare_{file_id}_{sheet}",
                                     help="Stacks the other files on their shared columns and labels each row by file.")
        except Exception as exc:
            st.error(f"Cannot open workbook: {exc}")
            return None
    identity = dataset_identity(content, f"{file_id}:{sheet}", header, {"compare": sorted(compare)})
    if not root["loaded"] or root["loaded"].get("base_id", root["loaded"]["id"]) != identity:
        try:
            with st.spinner("Reading the selected sheet..."):
                raw = load_sheet(content, filename, sheet, header)
                frame = clean_frame(raw)
                sources = 1
                if compare:
                    parts = [(filename, frame)]
                    for key in compare:
                        other_name, other_content = files[key]
                        other_sheet = workbook_sheet_names(other_content, other_name)[0]
                        other_header = detect_header_row(other_content, other_name, other_sheet)
                        parts.append((other_name, _sheet_frame(other_content, other_name, other_sheet, other_header)))
                    frame, shared = _combined(parts)
                    sources = len(parts)
                corrected_id = identity
                for correction in root.setdefault("corrections", {}).get(identity, []):
                    frame = correct_column(frame, **correction)
                    corrected_id = fingerprint([corrected_id, correction])
                if frame.memory_usage(index=True, deep=True).sum() > MAX_FRAME_BYTES:
                    raise ValueError("Expanded sheet exceeds the memory budget. Use a smaller sheet; rows were not truncated.")
                root["loaded"] = {"id": corrected_id, "base_id": identity, "frame": frame, "profiles": profile_frame(frame),
                                  "raw_rows": len(raw), "filename": filename, "sheet": sheet, "sources": sources}
        except Exception as exc:
            hint = " Close the file in Excel and try again." if isinstance(exc, PermissionError) else ""
            st.error(f"Cannot read the selected sheet: {exc}{hint}")
            return None
    data = root["loaded"]
    if data["frame"].empty:
        st.warning("No data at this header. Select another sheet or header row.")
        return None
    try:
        state = conversation(root, data["id"])
    except ValueError as exc:
        st.error(str(exc))
        return None
    state["file_id"] = file_id
    removed = [key for key, value in root["datasets"].items() if value.get("file_id") not in files and key != data["id"]]
    for key in removed:
        root["budget"]["used"] -= root["datasets"].pop(key).get("stored_bytes", 0)
    return data, state


def brd_panel(root):
    """Upload a BRD with its source files, run the agent framework, and keep the tables."""
    st.sidebar.caption(f"Agent mode: {brd_agent.configured_mode()}. {warehouse.backend_detail()}")
    uploads = st.sidebar.file_uploader(
        "BRD and source files", type=[s.lstrip(".") for s in brd_agent.UPLOAD_SUFFIXES],
        accept_multiple_files=True, key="brd_upload",
        help="Exactly one BRD document (DOCX, PDF, TXT, MD) plus the CSV/Excel sources it registers.")
    dry_run = st.sidebar.checkbox(
        "Dry run first", value=True, key="brd_dry_run",
        help="Validates and plans without creating tables or loading data. "
             "Clear it to let the framework write to BigQuery.")
    force = st.sidebar.checkbox(
        "Force re-run (ignore existing tables)", value=False, key="brd_force",
        help="Runs the framework even when the tables already exist. A real run costs "
             "model calls and writes to BigQuery unless 'Dry run first' is set.")
    documents = [u for u in (uploads or []) if Path(u.name).suffix.lower() in brd_agent.BRD_SUFFIXES]
    sources = [u for u in (uploads or []) if Path(u.name).suffix.lower() in brd_agent.SOURCE_SUFFIXES]
    run = st.sidebar.button("Run", key="brd_run", disabled=not (documents and sources), width="stretch",
                            help="Passes the uploaded input folder to the BRD-to-BigQuery framework.")
    previous_run = brd_agent.last_successful_run()
    replay = st.sidebar.button(
        "Table build", key="brd_replay", disabled=previous_run is None, width="stretch",
        help="Shows the full recorded build of these managed tables, step by step, from the "
             "framework's own run report. The tables already exist and are read live.")
    if previous_run is None:
        st.sidebar.caption("No completed framework run is on disk to show.")
    if replay and previous_run:
        transcript = list(previous_run.get("transcript") or [])
        # The recorded report predates the 02 KPI layer, so the live state of every managed
        # dataset is read from BigQuery now and appended as its own clearly labelled section.
        datasets = warehouse.sibling_datasets(previous_run["project"], previous_run["dataset"])
        live = warehouse.qualified_tables(previous_run["project"], datasets)
        live_statuses = []
        for reference in live:
            ref_project, ref_dataset, ref_table = _split_reference(
                reference, previous_run["project"], previous_run["dataset"])
            try:
                live_statuses.append((ref_dataset, ref_table,
                                      warehouse.table_status(ref_table, ref_project, ref_dataset)))
            except Exception:
                continue
        if live_statuses:
            transcript += ["", "── live verification against BigQuery (read just now) ──"]
            for dataset_name in datasets:
                rows = [s for s in live_statuses if s[0] == dataset_name]
                if not rows:
                    continue
                transcript.append(f"   dataset {dataset_name} · {len(rows)} tables")
                for _, table_name, info in rows:
                    count = info.get("rows")
                    shown = f"{int(count):>8,} rows" if isinstance(count, (int, float)) else "rows unavailable"
                    modified = info.get("modified") or ""
                    transcript.append(f"      {table_name:46} {shown}  {modified}")
            total = sum(int(s[2]["rows"]) for s in live_statuses
                        if isinstance(s[2].get("rows"), (int, float)))
            transcript += [
                "",
                f"   {len(live_statuses)} managed tables across {len(datasets)} datasets · "
                f"{total:,} rows live in BigQuery",
            ]
        log_lines: list[str] = []
        with st.status("Building managed BigQuery tables from the BRD contract...",
                       expanded=True) as status:
            st.caption(f"Recorded run {previous_run['run_id']} · {previous_run['finished_at']} · "
                       f"{previous_run['project']}")
            panel = st.empty()
            pace = max(0.25, min(1.2, 60.0 / max(len(transcript), 1)))
            for line in transcript:
                log_lines.append(line)
                panel.code("\n".join(log_lines[-26:]), language="log")
                time.sleep(0.05 if not line.strip() else pace)
            status.update(
                label=(f"Table build complete — {len(live_statuses) or len(previous_run['row_counts'])} "
                       f"managed tables across {len(datasets)} datasets"),
                state="complete", expanded=False)
        curated = warehouse.curated_only(live or previous_run["tables"])
        root["brd"] = {
            "brd_name": previous_run["brd_file"] or "recorded run",
            "tables": curated, "all_tables": previous_run["tables"],
            "qualified": [], "already_existed": curated,
            "project": previous_run["project"], "dataset": previous_run["dataset"],
            "status": f"{previous_run['status']} (recorded run {previous_run['run_id'][:10]})",
            "simulated": False, "issues": [],
            "summary": (f"{len(previous_run['row_counts'])} managed tables, "
                        f"{sum(int(c) for c in previous_run['row_counts'].values()):,} rows, "
                        f"built {previous_run['finished_at']}."),
            "statuses": [],
            "skip_reason": "These tables were built by the replayed run and still exist.",
            "log": log_lines, "replayed": True, "detail": "", "ok": bool(curated),
        }
        statuses = []
        for name in curated:
            table_project, table_dataset, table_name = _split_reference(
                name, previous_run["project"], previous_run["dataset"])
            statuses.append(warehouse.table_status(table_name, table_project, table_dataset))
        root["brd"]["statuses"] = statuses
        root["warehouse_refresh"] = root.get("warehouse_refresh", 0) + 1
    if uploads and not (documents and sources):
        st.sidebar.warning("Upload both a BRD document (DOCX, PDF, TXT, MD) and at least one "
                           "CSV or Excel source file.")
    if run and uploads:
        try:
            payload = [(u.name, u.getvalue()) for u in uploads]
            folder, brd_path = brd_agent.save_inputs(payload)
            planned = brd_agent.planned_tables(brd_path)
            qualified = brd_agent.planned_tables(brd_path, qualified=True)
            curated = warehouse.curated_only(planned)
            project = os.environ.get("BRD_BQ_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
            dataset = os.environ.get("BRD_BQ_DATASET")
            # 00/01/02 live in separate datasets, so each table is checked in the dataset the
            # BRD names for it. Checking them all against one dataset made 02 look missing.
            curated_refs = warehouse.curated_only(qualified) if qualified else curated
            statuses = []
            if planned:
                for reference in curated_refs:
                    ref_project, ref_dataset, ref_table = _split_reference(reference, project, dataset)
                    statuses.append(warehouse.table_status(ref_table, ref_project, ref_dataset))
            existing = [s["table"] for s in statuses if s["exists"]]
            # A table name existing does not mean it was built from the files just uploaded.
            fingerprint_now = brd_agent.source_fingerprint(payload)
            seen = root.setdefault("brd_source_fingerprints", {})
            previous = seen.get(fingerprint([sorted(curated), project, dataset]))
            changed = previous is not None and previous != fingerprint_now
            skip = bool(existing) and len(existing) == len(curated_refs) and not force and not changed
            if skip:
                reason = ("The source files match the last run recorded in this session."
                          if previous == fingerprint_now else
                          "This session has no record of these tables being built from different sources.")
                root["brd"] = {"brd_name": brd_path.name, "tables": curated,
                               "all_tables": planned, "qualified": qualified, "already_existed": existing,
                               "project": project, "dataset": dataset,
                               "status": "ALREADY_EXISTS", "simulated": False, "issues": [],
                               "summary": f"{len(existing)} table(s) already exist; the framework was not run.",
                               "statuses": statuses, "skip_reason": reason,
                               "detail": "", "ok": True}
                # Establish the baseline on the first skip, otherwise a later change to the
                # sources has nothing to be compared against and would be skipped as well.
                seen.setdefault(fingerprint([sorted(curated), project, dataset]), fingerprint_now)
            else:
                log_lines: list[str] = []
                with st.status("Running the BRD-to-BigQuery agent...", expanded=True) as status:
                    st.caption("Live output from the BRD-to-BigQuery framework. "
                               "Account identifiers are masked for screen sharing.")
                    panel = st.empty()

                    def show(line: str) -> None:
                        log_lines.append(line)
                        panel.code("\n".join(log_lines[-20:]), language="log")

                    result = brd_agent.run_agent(brd_path, dry_run=dry_run, source_dir=folder,
                                                 on_line=show)
                    status.update(label=f"Framework finished: {result.status or 'completed'}",
                                  state="complete" if result.ok else "error", expanded=False)
                root["brd"] = {"brd_name": brd_path.name, "brd_path": str(brd_path),
                               "tables": warehouse.curated_only(result.curated or result.tables),
                               "all_tables": result.tables, "qualified": qualified, "already_existed": [],
                               "project": result.project, "dataset": result.dataset,
                               "simulated": result.simulated, "mode": result.mode,
                               "status": result.status, "issues": result.issues,
                               "run_id": result.run_id, "report_dir": result.report_dir,
                               "statuses": statuses, "forced": bool(force), "sources_changed": changed,
                               "log": log_lines,
                               "summary": result.summary, "detail": result.detail, "ok": result.ok}
                if result.ok and curated:
                    seen[fingerprint([sorted(curated), project, dataset])] = fingerprint_now
            root["warehouse_refresh"] = root.get("warehouse_refresh", 0) + 1
        except brd_agent.BrdError as exc:
            root["brd"] = None
            st.sidebar.error(str(exc))
        except Exception as exc:
            root["brd"] = None
            st.sidebar.error(f"The BRD run failed: {type(exc).__name__}: {exc}")
    return root.get("brd")


def warehouse_data(root):
    with st.sidebar:
        mode = st.radio("Source", [BRD_RUN_MODE, LINK_MODE], key="wh_mode",
                        help="Run a BRD to build tables, or point at tables that already exist.")
    if mode == LINK_MODE:
        return linked_data(root)
    return brd_data(root)


def linked_data(root):
    """Fetch tables that already exist from a pasted BigQuery link or identifier."""
    with st.sidebar:
        text = st.text_input("BigQuery table or dataset link", key="wh_link",
                             placeholder="console link, or project.dataset.table")
    if not text.strip():
        st.info("Paste a BigQuery console link, or a project.dataset or project.dataset.table "
                "identifier, to chart tables that already exist.")
        return None
    try:
        ref = warehouse.parse_reference(text)
        tables = [ref["table"]] if ref.get("table") else warehouse.list_tables(ref["project"], ref["dataset"])
    except warehouse.WarehouseError as exc:
        st.error(str(exc))
        return None
    if not tables:
        st.error(f"No tables found in {ref['project']}.{ref['dataset']}.")
        return None
    return warehouse_table(root, warehouse.curated_only(tables), ref["project"], ref["dataset"],
                           review=tables)


def brd_data(root):
    run = brd_panel(root)
    if not run:
        st.info("Upload a BRD document and its source files in the sidebar, then press Run. "
                "The tables the framework creates become the data for the dashboard and chat.")
        return None
    if run.get("status"):
        st.caption(f"Run status: {run['status']}. {run.get('summary', '')}")
    if run.get("already_existed") and not run.get("replayed"):
        st.success(f"These tables already exist and were not rebuilt: "
                   f"{', '.join(run['already_existed'])}. They are charted directly below.")
        if run.get("skip_reason"):
            st.caption(run["skip_reason"] + " Tick 'Force re-run (ignore existing tables)' in the "
                       "sidebar to rebuild them from the uploaded sources.")
    if run.get("sources_changed"):
        st.warning("The uploaded source files differ from the last run in this session, so the "
                   "existing tables were rebuilt rather than reused.")
    elif run.get("forced"):
        st.info("Force re-run was set, so the framework ran even though the tables already existed.")
    for status in run.get("statuses") or []:
        st.caption(warehouse.describe_status(status))
    if run.get("log"):
        with st.expander(f"Framework run log ({len(run['log'])} lines)"):
            st.caption("Captured from the BRD-to-BigQuery framework. "
                       "Account identifiers are masked for screen sharing.")
            st.code("\n".join(run["log"][-400:]), language="log")
    if run.get("simulated"):
        if warehouse.backend() == "bigquery":
            st.warning("Simulated run: no framework was called and no table was created. "
                       "The tables below already exist in the configured BigQuery dataset.")
        else:
            st.warning("Simulated run: no framework was called and no BigQuery table was created. "
                       "The tables below are local files standing in for warehouse tables.")
    for issue in (run.get("issues") or [])[:5]:
        if isinstance(issue, dict):
            st.warning(f"{issue.get('check', 'Issue')}: {issue.get('result', '')} "
                       f"Action: {issue.get('action', '')}")
    tables = run.get("tables") or []
    # 00/01/02 live in separate datasets, so include every managed layer that exists.
    if warehouse.backend() == "bigquery":
        datasets = warehouse.sibling_datasets(run.get("project"), run.get("dataset"))
        across = warehouse.qualified_tables(run.get("project"), datasets)
        chartable = warehouse.curated_only(across)
        if chartable:
            tables, review_all = chartable, across
        else:
            review_all = run.get("qualified") or run.get("all_tables") or tables
    else:
        review_all = run.get("qualified") or run.get("all_tables") or tables
    if not tables:
        st.error(run.get("detail") or "The run reported no tables.")
        return None
    return warehouse_table(root, tables, run.get("project"), run.get("dataset"), review=review_all)


def warehouse_table(root, tables, project, dataset, review=None):
    """Shared loader: pick a curated table and read it into the standard data dict."""
    with st.sidebar:
        table = st.selectbox("Table", tables, key="brd_table",
                             format_func=lambda name: str(name).split(".")[-1],
                             help="Curated 01 facts and 02 KPI/summary tables are listed.")
        # A fully qualified choice carries its own dataset, so 02 can be read while the
        # configured dataset still points at 01.
        table_project, table_dataset, table = _split_reference(table, project, dataset)
        limit = int(st.number_input("Row limit", min_value=1000, max_value=warehouse.MAX_ROW_LIMIT,
                                    value=min(warehouse.DEFAULT_ROW_LIMIT, warehouse.MAX_ROW_LIMIT),
                                    step=1000, key="brd_limit",
                                    help="Rows are read into memory. Any truncation is disclosed."))
        if st.button("Re-read table", key="brd_reread", width="stretch",
                     help="Reads the table again from the warehouse. Use after the data has "
                          "changed; the chart cache is keyed by table name, not by contents."):
            root["warehouse_refresh"] = root.get("warehouse_refresh", 0) + 1
            root["loaded"] = None
            st.rerun()
    identity = dataset_identity(f"{table_project}.{table_dataset}.{table}".encode("utf-8"),
                                f"warehouse:{table_dataset}:{table}", limit,
                                {"project": table_project, "refresh": root.get("warehouse_refresh", 0)})
    if not root["loaded"] or root["loaded"].get("base_id", root["loaded"]["id"]) != identity:
        try:
            with st.spinner(f"Reading table {table}..."):
                frame, info = warehouse.load_table(table, table_project, table_dataset, limit)
                if frame.empty:
                    st.warning(f"Table '{table}' has no rows to chart.")
                    return None
                if frame.memory_usage(index=True, deep=True).sum() > MAX_FRAME_BYTES:
                    raise ValueError("The table exceeds the memory budget. Lower the row limit.")
                root["loaded"] = {"id": identity, "base_id": identity, "frame": frame,
                                  "profiles": profile_frame(frame), "raw_rows": info["read_rows"],
                                  "filename": info["source"], "sheet": table, "sources": 1,
                                  "warehouse": info}
        except warehouse.WarehouseError as exc:
            st.error(str(exc))
            return None
        except Exception as exc:
            st.error(f"Cannot read table '{table}': {type(exc).__name__}: {exc}")
            return None
    data = root["loaded"]
    try:
        state = conversation(root, data["id"])
    except ValueError as exc:
        st.error(str(exc))
        return None
    state["file_id"] = f"warehouse:{table_dataset}:{table}"
    data["review_tables"] = list(review or tables)
    data["review_project"], data["review_dataset"] = project, dataset
    return data, state


def column_corrections(root, data):
    with st.expander("Correct column format"):
        st.caption("Corrections are local and start a new dataset revision. No nonblank values may be discarded.")
        with st.form(f"types_{data['id']}"):
            column = st.selectbox("Column", list(data["frame"].columns))
            kind = st.selectbox("Interpret as", ["numeric", "datetime", "boolean", "text"])
            left, right = st.columns(2)
            decimal = left.selectbox("Decimal separator", [".", ","])
            thousands = right.selectbox("Thousands separator", [",", ".", " ", ""])
            currency = st.text_input("Currency symbol to remove", max_chars=8)
            percent = st.checkbox("Divide numeric values by 100 (percentage points to fractions)")
            date_format = st.text_input("Exact date format", value="%d/%m/%Y", max_chars=80)
            apply = st.form_submit_button("Apply correction")
        if apply:
            try:
                if len(root["datasets"]) >= MAX_DATASETS:
                    raise ValueError("Clear saved history before making another dataset revision.")
                if len(root.get("corrections", {}).get(data.get("base_id", data["id"]), [])) >= 30:
                    raise ValueError("Correction limit reached. Re-upload the original file to start a fresh correction history.")
                frame = correct_column(data["frame"], column, kind, decimal, thousands, currency, percent, date_format)
                if frame.memory_usage(index=True, deep=True).sum() > MAX_FRAME_BYTES:
                    raise ValueError("Corrected frame exceeds the memory budget.")
                correction = dict(column=column, kind=kind, decimal=decimal, thousands=thousands,
                                  currency=currency, percent=percent, date_format=date_format)
                identity = fingerprint([data["id"], correction])
                root.setdefault("corrections", {}).setdefault(data.get("base_id", data["id"]), []).append(correction)
                data.update(id=identity, frame=frame, profiles=profile_frame(frame))
                data["correction_notice"] = f"{column} is now {kind}. All {len(frame):,} rows retained. Start a new conversation for this corrected revision."
                data.pop("privacy_index", None)
                data.pop("privacy_index_key", None)
                data.pop("filter_offers", None)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))


def ai_settings(data):
    identity = data["id"]
    try:
        configured = gemini.available()
        model = gemini.get_model()
    except gemini.GeminiUnavailable as exc:
        configured, model = False, ""
        st.sidebar.error(str(exc))
    with st.sidebar:
        st.divider()
        st.subheader("Gemini")
        if configured:
            st.caption(f"Configured model: {model}")
        else:
            st.error("No Gemini key found. Set GEMINI_API_KEY in the project's .env file.")
        approved = st.checkbox("Allow selected column names and my messages to Gemini",
                               key=f"approval_{identity}")
        with st.expander("Metadata sharing"):
            allowed = st.multiselect("Columns", list(data["frame"].columns),
                                     default=default_allowed_columns(data["profiles"]), key=f"allowed_{identity}")
            st.caption("No worksheet rows or numeric summaries are sent. Messages and prior filters may contain "
                       "values you supplied. Use synthetic or company-approved data; automated checks are not a privacy guarantee.")
    return configured and approved, PrivacyPolicy(tuple(allowed)), model


def _offer_instead(exc):
    """Turn a validation refusal into a concrete supported alternative instead of a dead end."""
    message = str(exc)
    missing = re.search(r"Column '([^']+)' is not in this sheet", message)
    if missing:
        return (f"This sheet has no column called {missing[1]}. "
                f"Which column should I use? The data preview lists them all.")
    rate = re.search(r"Cannot sum rate-like measure '([^']+)'", message)
    if rate:
        return (f"Adding up {rate[1]} would be misleading because it is a rate or score. "
                f"Shall I show the average {rate[1]} instead?")
    flag = re.search(r"'([^']+)' must be a numeric column for (\w+)", message)
    if flag:
        return (f"{flag[1]} holds labels rather than numbers, so it cannot be measured directly. "
                f"Shall I count records split by {flag[1]}, or choose a numeric measure?")
    if "Every panel needs a chart type" in message:
        # The reply named no chart. It may have carried a filter, a title, or nothing at
        # all, so offer the supported types rather than assuming what it did contain.
        return ("I understood the request but not which chart to draw. Shall I use a bar, "
                "line, area, pie, donut, histogram, scatter, box or correlation chart? "
                "Naming the columns too, such as \"bar chart of NetSales by Category\", "
                "gets you there in one step.")
    orientation = re.search(r"Orientation overrides are not supported for '([^']+)'", message)
    if orientation:
        return (f"A {orientation[1]} chart has no horizontal or vertical form. "
                f"Shall I switch it to a bar chart, or leave it as it is?")
    if "No rows matched" in message:
        return ("No rows matched that combination, so the previous image is unchanged. "
                "Shall I widen it, or list the values available for that column?")
    return None


def _replies_to_question(question):
    """A short or affirmative message answers the pending question; anything fuller is a new request."""
    text = question.strip().rstrip(".!? ")
    if re.match(r"^(yes|no|yeah|yep|nope|sure|ok|okay|please|do it|go ahead|neither|both)\b", text, re.IGNORECASE):
        return True
    return len(text.split()) <= 4


def accept_response(state, data, response, question):
    previous = current(state)
    if not isinstance(response, dict) or response.get("action") not in {"create", "update", "clarify", "explain", "unsupported"}:
        raise SpecError("The model returned an unsupported response. Try rephrasing the request.")
    message = response.get("message", "")
    if not isinstance(message, str) or not message.strip():
        raise SpecError("The model response is missing its interpretation.")
    if response["action"] in {"clarify", "explain", "unsupported"}:
        if response["action"] == "clarify":
            original = state["pending"]["request"] if state["pending"] else question
            state["pending"] = {"request": original, "question": message}
        else:
            state["pending"] = None
        entry = {"role": "assistant", "text": message}
        if isinstance(response.get("context_text"), str):
            entry["context_text"] = response["context_text"]
        state["messages"].append(entry)
        return
    plan = revise_plan(previous.plan if previous else None, response)
    plan["scope_filters"] = copy.deepcopy(state.get("scope", []))
    if re.fullmatch(r"(?:yes[, ]+)?(?:please )?(?:(?:do|show|create|include) )?both[.! ]*", question.strip(), re.IGNORECASE):
        if len(chart_specs(plan)) != 2:
            raise SpecError("Both charts were requested, but the returned plan did not contain two. The previous image is unchanged.")
    artifact = build_artifact(data["frame"], data["id"], plan,
                              previous.id if previous and response["action"] == "update" else None)
    if len(state["versions"]) >= MAX_VERSIONS:
        raise ValueError("Saved-image limit reached. Clear saved history to continue; the current image is unchanged.")
    _reserve(state, len(artifact.figure_json.encode("utf-8")) + len(artifact.plan_json.encode("utf-8")))
    state["versions"].append(artifact)
    state.update(selected=artifact.id, pending=None, epoch=state["epoch"] + 1)
    state["messages"].append({"role": "assistant", "text": rendered_message(artifact, response["action"] == "update"),
                              "context_text": rendered_message(artifact, response["action"] == "update", include_counts=False),
                              "artifact_id": artifact.id})


def ask(state, data, question, enabled, policy, model=None):
    if len(question) > MAX_MESSAGE_CHARS or len(state["messages"]) > MAX_MESSAGES - 2:
        state["notice"] = "Message/history limit reached. Shorten the message or clear saved history; no API call was made."
        return
    if len(state["versions"]) >= MAX_VERSIONS or state.get("budget", {}).get("used", 0) >= MAX_RESULT_BYTES:
        state["notice"] = "Saved-result budget reached. Download results and clear saved history before another request. No API call was made."
        return
    new_pattern = r"^(?:new (?:image|chart|question|analysis)\b|start (?:a )?new\b)"
    explicit_new = bool(re.match(new_pattern, question.strip(), re.IGNORECASE))
    pending = {} if explicit_new else (state["pending"] or {})
    if pending and not _replies_to_question(question):
        pending = {}
        state["pending"] = None
    new_request = explicit_new or bool(re.match(new_pattern, pending.get("request", ""), re.IGNORECASE))
    history = state["messages"][state.get("context_start", 0):][-6:]
    bundle = state.get("recommendations")
    recommendation_response = None
    if bundle and bundle["dataset_id"] == data["id"]:
        recommendation_response = select_recommendation(question, bundle, history, pending)
        if recommendation_response and recommendation_response["action"] == "create":
            new_request = True
            pending = {}
        history = [_context_message(message, bundle, policy) for message in history]
    if new_request:
        history = []
    state["messages"].append({"role": "user", "text": question})
    previous = None if new_request else current(state)
    try:
        response = recommendation_response
        if response is None:
            response = data_answer(question, data["frame"], data["profiles"],
                                   previous.scope if previous else None)
        if response is None:
            response = local_response(question, data["frame"], previous.plan if previous else None)
        if response is None:
            if not enabled:
                raise PrivacyError("Approve metadata sharing in the sidebar and check your Gemini key, then send your question again. Nothing was sent.")
            index_key = (data["id"], tuple(sorted(policy.allowed_columns)))
            if data.get("privacy_index_key") != index_key:
                data["privacy_index"] = prepare_privacy_index(data["frame"], data["profiles"], policy)
                data["privacy_index_key"] = index_key
            payload = outbound_payload(data["frame"], data["profiles"], question,
                                       previous.plan if previous else None, policy, dt.date.today().isoformat(),
                                       pending.get("question"), pending.get("request"), history=history,
                                       index=data["privacy_index"], scope=state.get("scope"))
            state["last_payload"] = payload
            response = gemini.request_json(payload, PLANNER_SCHEMA, PLANNER_SYSTEM, model=model)
        if new_request and response.get("action") == "update":
            raise SpecError("A new image was requested, but the model returned an edit. The previous image is unchanged.")
        if explicit_new and response.get("action") == "clarify":
            state["pending"] = None
        accept_response(state, data, response, question)
        if new_request and response.get("action") == "create":
            state["context_start"] = len(state["messages"]) - 2
    except (PrivacyError, SpecError, ValueError, gemini.GeminiUnavailable, gemini.RateLimited) as exc:
        offer = _offer_instead(exc) if isinstance(exc, (SpecError, ValueError)) else None
        if offer:
            state["pending"] = {"request": question, "question": offer}
            state["messages"].append({"role": "assistant", "text": offer})
        else:
            state["messages"].append({"role": "assistant", "text": str(exc), "error": True})
    except Exception:
        state["messages"].append({"role": "assistant", "text": "The request failed. The previous chart is unchanged; no automatic retry was made.", "error": True})


def _split_reference(identifier, project, dataset):
    """Return (project, dataset, table) for a short or fully qualified identifier."""
    parts = str(identifier).split(".")
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    return project, dataset, parts[-1]


def validation_panel(state, data, tables, project, dataset):
    """Show every table the run produced, with a console link and real rows, then gate charts.

    Charts are not built until a reviewer confirms the previews match BigQuery itself.
    """
    approved = state.get("validated") == data["id"]
    heading = st.columns([4, 1])
    heading[0].subheader("Validate the pipeline output")
    if heading[1].button("Approve and build dashboard" if not approved else "Re-validate",
                         key=f"approve_{data['id']}", width="stretch", type="primary"):
        if approved:
            state.pop("validated", None)
            state.pop("dashboard", None)
        else:
            state["validated"] = data["id"]
        st.rerun()
    if approved:
        st.success("Approved. The dashboard below was built from these tables.")
        return True
    st.caption("Open each table in BigQuery and check the rows and counts against the preview "
               "below. Charts are built only after you approve.")
    for identifier in tables:
        table_project, table_dataset, table_name = _split_reference(identifier, project, dataset)
        with st.expander(f"{table_name}", expanded=table_name.startswith(warehouse.CURATED_PREFIXES)):
            link = warehouse.console_link(table_project, table_dataset, table_name)
            try:
                frame, info = warehouse.preview_table(table_name, table_project, table_dataset, rows=10)
            except warehouse.WarehouseError as exc:
                st.warning(str(exc))
                if link:
                    st.markdown(f"[Open {table_name} in BigQuery]({link})")
                continue
            st.caption(f"{info['source']} | {info['total_rows']:,} rows | {info['columns']} columns. "
                       f"Showing the first {info['read_rows']} rows read from "
                       f"{'the simulated warehouse' if info.get('simulated') else 'BigQuery'}.")
            if link:
                st.markdown(f"[Open {table_name} in BigQuery]({link})")
            st.dataframe(frame, width="stretch", hide_index=True)
            if info.get("dropped_columns"):
                st.caption(f"{len(info['dropped_columns'])} repeated (ARRAY) column(s) cannot be "
                           f"charted and are not shown: {', '.join(info['dropped_columns'])}.")
    return False


def build_dashboard(state, data):
    """Build every chart this table supports, locally, with no AI call.

    Each recommendation already carries a validated chart plan, so the whole dashboard is
    computed by pandas and drawn by Plotly. Charts that fail validation are skipped and
    counted rather than silently dropped.
    """
    bundle = state.get("recommendations") or {}
    built, failed = [], 0
    for group in bundle.get("types", []):
        for option in group.get("options", []):
            plan = {"kind": "chart", "chart": copy.deepcopy(option["response"]["chart"]), "filters": []}
            try:
                artifact = build_artifact(data["frame"], data["id"], plan)
            except (ValueError, SpecError):
                failed += 1
                continue
            built.append({"label": option.get("label") or image_title(plan),
                          "reason": option.get("reason", ""), "artifact": artifact,
                          "chart": group["label"]})
    state["dashboard"] = {"dataset_id": data["id"], "charts": built, "failed": failed}
    return state["dashboard"]


def dashboard_panel(state, data):
    """Show the generated dashboard; every figure comes from the table's own rows."""
    board = state.get("dashboard")
    if not board or board.get("dataset_id") != data["id"]:
        with st.spinner("Building every chart this table supports..."):
            board = build_dashboard(state, data)
    charts = board["charts"]
    if not charts:
        st.info("This table did not produce any validated chart. Ask for one in the chat.")
        return
    header = st.columns([3, 1, 1])
    header[0].subheader(f"Dashboard: {len(charts)} charts")
    fmt = header[1].selectbox("Format", ["PNG", "SVG", "JPEG"], key=f"dash_fmt_{data['id']}").lower()
    if header[2].button("Rebuild", key=f"rebuild_{data['id']}", width="stretch"):
        state.pop("dashboard", None)
        st.rerun()
    note = f"All {len(charts)} charts were computed locally from {len(data['frame']):,} rows; no AI call was used."
    if board["failed"]:
        note += f" {board['failed']} suggestion(s) could not be validated and were skipped."
    st.caption(note)
    bundle_key = (data["id"], fmt, "bundle")
    if st.button(f"Prepare all {len(charts)} charts as {fmt.upper()}", key=f"dash_all_{data['id']}_{fmt}",
                 disabled=bundle_key in state["exports"]):
        try:
            with st.spinner(f"Rendering {len(charts)} images..."):
                figures = [(item["label"], item["artifact"].figure) for item in charts]
                rendered = figures_to_images(figures, fmt, 1280, 720, scale=1, timeout=30 * len(charts))
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                for filename, content in rendered:
                    archive.writestr(filename, content)
            payload = buffer.getvalue()
            _reserve(state, len(payload))
            state["exports"][bundle_key] = (f"dashboard_{fmt}.zip", payload)
        except (ExportError, ValueError) as exc:
            state["exports"].pop(bundle_key, None)
            st.error(str(exc))
    if bundle_key in state["exports"]:
        filename, payload = state["exports"][bundle_key]
        st.download_button(f"Download all {len(charts)} charts (ZIP)", payload, filename, "application/zip",
                           key=f"dash_zip_{fingerprint(bundle_key)}", on_click="ignore")
    columns = st.columns(2)
    for index, item in enumerate(charts):
        with columns[index % 2]:
            st.plotly_chart(item["artifact"].figure, width="stretch", theme=None,
                            key=f"dash_{data['id']}_{index}",
                            config={"displaylogo": False, "modeBarButtonsToRemove": ["toImage"]})
            st.caption(f"{item['chart']} | {item['artifact'].matched_rows:,} of "
                       f"{item['artifact'].source_rows:,} rows. {item['reason']}")
            key = (data["id"], fmt, index)
            if st.button(f"Prepare {fmt.upper()}", key=f"dash_prep_{data['id']}_{index}_{fmt}",
                         disabled=key in state["exports"]):
                try:
                    with st.spinner("Rendering image..."):
                        output = figures_to_images([(item["label"], item["artifact"].figure)],
                                                   fmt, 1280, 720, scale=1, timeout=30)
                    _reserve(state, len(output[0][1]))
                    state["exports"][key] = output[0]
                except (ExportError, ValueError) as exc:
                    state["exports"].pop(key, None)
                    st.error(str(exc))
            if key in state["exports"]:
                filename, content = state["exports"][key]
                st.download_button(f"Download {fmt.upper()}", content, filename,
                                   {"png": "image/png", "svg": "image/svg+xml", "jpeg": "image/jpeg"}[fmt],
                                   key=f"dash_dl_{fingerprint(key)}", on_click="ignore")


def downloads(state, artifact):
    with st.expander("Download image", expanded=True):
        size = dict(artifact.figure.layout.meta or {})
        minimum_height = size.get("export_height", 720)
        controls = st.columns(3)
        fmt = controls[0].selectbox("Format", ["PNG", "SVG", "JPEG"], key="export_format").lower()
        width = int(controls[1].number_input("Width (px)", 640, 2000, 1280, step=80, key="export_width"))
        height = int(controls[2].number_input("Height (px)", minimum_height, 2000, minimum_height,
                                             step=40, key=f"export_height_{minimum_height}"))
        key = (artifact.id, fmt, width, height)
        if st.button("Prepare image", key=f"prepare_{artifact.id}", disabled=key in state["exports"]):
            try:
                with st.spinner("Rendering image..."):
                    output = figures_to_images([(image_title(artifact.plan), artifact.figure)],
                                               fmt, width, height, scale=1, timeout=30)
                _reserve(state, len(output[0][1]))
                state["exports"][key] = output[0]
            except (ExportError, ValueError) as exc:
                state["exports"].pop(key, None)
                st.error(str(exc))
        if key in state["exports"]:
            filename, content = state["exports"][key]
            st.download_button(f"Download {fmt.upper()}", content, filename,
                               {"png": "image/png", "svg": "image/svg+xml", "jpeg": "image/jpeg"}[fmt],
                               key=f"download_{fingerprint(key)}", on_click="ignore")
        if fmt == "svg":
            st.caption("SVG retains vector labels; heatmaps and WebGL traces may include raster content.")


def main():
    st.set_page_config(page_title="BRD to Insight Agent", page_icon=":bar_chart:", layout="wide")
    st.title("BRD to Insight Agent")
    root = workspace()
    selection = selected_data(root)
    if selection is None:
        return
    data, state = selection
    if data.get("correction_notice"):
        st.success(data.pop("correction_notice"))
    column_corrections(root, data)
    filter_panel(state, data)
    try:
        if "recommendations" not in state:
            with st.spinner("Checking chart options for this sheet..."):
                ensure_recommendations(state, data)
    except ValueError as exc:
        st.warning(f"Could not prepare recommendations: {exc}")
    enabled, policy, model = ai_settings(data)
    sources = data.get("sources", 1)
    combined = f" | {sources} files combined, labelled by {SOURCE_COLUMN}" if sources > 1 else ""
    st.caption(f"{data['filename']} / {data['sheet']} | {len(data['frame']):,} rows{combined}")
    if data.get("warehouse"):
        st.caption(warehouse.disclosure(data["warehouse"]))
    with st.expander("Data preview"):
        st.dataframe(data["frame"].head(100), width="stretch")
        st.dataframe([{"Column": name, "Detected type": str(data["frame"][name].dtype), "Role": profile.role}
                      for name, profile in data["profiles"].items()], hide_index=True, width="stretch")
        st.caption(f"Preview: first 100 rows. Analysis uses all {len(data['frame']):,} nonblank rows.")
        if len(data["frame"]) < data["raw_rows"]:
            st.caption(f"Removed {data['raw_rows'] - len(data['frame']):,} entirely blank rows.")
    toolbar = st.columns([3, 1])
    if state["versions"]:
        choices = [None] + [v.id for v in state["versions"]]
        labels = {None: "New chart"}
        labels.update({v.id: f"{idx + 1}. {image_title(v.plan)}" for idx, v in enumerate(state["versions"])})
        selected = toolbar[0].selectbox("Image history", choices,
                                         index=choices.index(state["selected"]), format_func=lambda key: labels[key],
                                         key=f"history_{data['id']}_{state['epoch']}")
        if selected != state["selected"]:
            reset_context(state, selected)
            st.rerun()
    if toolbar[1].button("New image", key="new_chart"):
        reset_context(state)
        st.rerun()
    if data.get("warehouse"):
        approved = validation_panel(state, data, data.get("review_tables") or [],
                                    data.get("review_project"), data.get("review_dataset"))
        st.divider()
        if approved:
            dashboard_panel(state, data)
            st.divider()
    chart_area, chat_area = st.columns([3, 2], gap="large")
    artifact = current(state)
    with chart_area:
        if artifact:
            fig = artifact.figure
            st.plotly_chart(fig, width="stretch", theme=None, key=f"plot_{artifact.id}",
                            config={"displaylogo": False, "modeBarButtonsToRemove": ["toImage"]})
            st.caption(f"{artifact.matched_rows:,} of {artifact.source_rows:,} rows matched. {artifact.scope}")
            meta = dict(fig.layout.meta or {})
            if meta.get("sampled"):
                st.caption(f"Displaying {meta['drawn_rows']:,} of {meta['eligible_rows']:,} eligible points.")
            if meta.get("omitted_categories"):
                st.caption(f"Display limit omitted {len(meta['omitted_categories'])} categories.")
            if meta.get("panel_count", 1) > 1:
                st.caption(f"{meta['panel_count']} charts in this image; the same filters apply to each.")
            downloads(state, artifact)
    with chat_area:
        st.subheader("Chat")
        if state.get("notice"):
            st.warning(state.pop("notice"))
        with st.container(height=480):
            for index, message in enumerate(state["messages"]):
                with st.chat_message(message["role"]):
                    if message.get("error"):
                        st.error(message["text"])
                    else:
                        st.text(message["text"])
                    if message.get("recommendation_types") == data["id"]:
                        for group in state.get("recommendations", {}).get("types", []):
                            if st.button(group["label"], key=f"rectype_{data['id']}_{index}_{group['chart']}",
                                         help=f"{len(group['options'])} suggestion(s) built from this sheet", width="stretch"):
                                choose_type(state, data, group["chart"])
                                st.rerun()
                    if message.get("recommendation_options"):
                        options = group_for(state.get("recommendations", {}), message["recommendation_options"])
                        for number, option in enumerate((options or {}).get("options", []), 1):
                            if st.button(f"{number}. {option['label']}", key=f"recommend_{data['id']}_{index}_{option['id']}",
                                         help=option["reason"], width="stretch"):
                                ask(state, data, f"recommendation {number}", enabled, policy, model)
                                st.rerun()
        question = st.chat_input("Ask for a chart or modify the current one", key=f"chat_{data['id']}")
        if question:
            with st.spinner("Preparing your chart..."):
                ask(state, data, question, enabled, policy, model)
            st.rerun()
        if state["last_payload"]:
            with st.expander("Last request metadata"):
                st.json(state["last_payload"])


if __name__ == "__main__":
    main()