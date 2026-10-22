"""Read tables produced by the BRD agent into the same pandas frames the app already uses.

Two backends:

``bigquery``   real tables, used when ``google-cloud-bigquery`` is installed and a project is set
``simulated``  local workbooks standing in for tables, used when BigQuery is not reachable

The simulated backend never invents values: it serves real rows from local files and is
always labelled as simulated in the UI. Row limits are applied and disclosed, never hidden.
"""

from __future__ import annotations

import datetime as dt
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from brd_insight_agent.loader import clean_frame, detect_header_row, load_sheet, workbook_sheet_names

DEFAULT_ROW_LIMIT = int(os.environ.get("BRD_BQ_ROW_LIMIT", "200000"))
MAX_ROW_LIMIT = 1_000_000
SIMULATED_DIR = Path(os.environ.get("BRD_SIMULATED_DIR", "testdata"))
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_$\-]+$")
CURATED_PREFIX = "01_"
# 02 holds the KPI/summary layer and 01 the curated facts; both are chartable.
CURATED_PREFIXES = ("02_", "01_")

# BigQuery console links carry the project and dataset as !1s<project>!2s<dataset>[!3s<table>].
_CONSOLE_LINK = re.compile(r"!1s(?P<project>[A-Za-z0-9\-:.]+)!2s(?P<dataset>[A-Za-z0-9_]+)(?:!3s(?P<table>[A-Za-z0-9_$\-]+))?")
# Plain identifiers such as project.dataset or project.dataset.table.
_PLAIN_REF = re.compile(r"^\s*`?(?P<project>[A-Za-z0-9\-:.]+)\.(?P<dataset>[A-Za-z0-9_]+)(?:\.(?P<table>[A-Za-z0-9_$\-]+))?`?\s*$")


def parse_reference(text: str) -> dict:
    """Read a BigQuery console link or a project.dataset[.table] identifier.

    Returns the project, dataset and optional table. Raises WarehouseError when the text
    carries no usable reference, so the caller can tell the user exactly what to paste.
    """
    value = (text or "").strip()
    if not value:
        raise WarehouseError("Paste a BigQuery table or dataset link.")
    match = _CONSOLE_LINK.search(value) or _PLAIN_REF.match(value)
    if not match:
        raise WarehouseError(
            "That does not look like a BigQuery reference. Paste a console link, or "
            "project.dataset or project.dataset.table."
        )
    found = match.groupdict()
    return {"project": found["project"], "dataset": found["dataset"], "table": found.get("table")}



class WarehouseError(Exception):
    """A table could not be listed or read."""


def bigquery_ready() -> bool:
    try:
        import google.cloud.bigquery  # noqa: F401
    except Exception:
        return False
    return bool(os.environ.get("BRD_BQ_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT"))


def backend() -> str:
    return "bigquery" if bigquery_ready() else "simulated"


def backend_detail() -> str:
    if backend() == "bigquery":
        project = os.environ.get("BRD_BQ_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
        return f"BigQuery, project {project}."
    try:
        import google.cloud.bigquery  # noqa: F401
        reason = "no project is set (BRD_BQ_PROJECT)"
    except Exception:
        reason = "google-cloud-bigquery is not installed"
    return (f"Simulated warehouse: {reason}. Local files in '{SIMULATED_DIR}' stand in for tables. "
            "Rows are real file contents, not BigQuery data.")


def _check_name(value: str, label: str) -> str:
    if not value or not _SAFE_NAME.match(str(value)):
        raise WarehouseError(f"Unsafe {label} name: {value!r}")
    return str(value)


def simulated_tables() -> list[str]:
    folder = SIMULATED_DIR
    if not folder.exists():
        return []
    names = []
    for path in sorted(folder.glob("*.xls*")) + sorted(folder.glob("*.csv")):
        if path.name.startswith("~$"):
            continue
        names.append(path.stem)
    return names


def list_tables(project: str | None = None, dataset: str | None = None) -> list[str]:
    if backend() == "simulated":
        return simulated_tables()
    from google.cloud import bigquery

    project = project or os.environ.get("BRD_BQ_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    dataset = dataset or os.environ.get("BRD_BQ_DATASET")
    if not dataset:
        raise WarehouseError("No BigQuery dataset is set. Set BRD_BQ_DATASET or pick one after the run.")
    try:
        client = bigquery.Client(project=project)
        reference = f"{project}.{dataset}" if project else dataset
        return sorted(table.table_id for table in client.list_tables(reference))
    except Exception as exc:
        raise WarehouseError(f"Could not list tables in {project}.{dataset}: {type(exc).__name__}: {exc}") from exc


def _drop_repeated(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Remove BigQuery ARRAY columns, which no chart type can use.

    Repeated fields arrive as numpy arrays and break profiling, which only ever sees
    scalar spreadsheet values. The removal is reported so the UI can disclose it.
    """
    dropped = []
    for name in list(frame.columns):
        series = frame[name]
        if series.dtype != object:
            continue
        sample = series.dropna().head(20)
        if any(isinstance(value, (list, tuple, set, np.ndarray)) for value in sample):
            dropped.append(name)
    return (frame.drop(columns=dropped) if dropped else frame), dropped


def table_exists(table: str, project: str | None = None, dataset: str | None = None) -> bool:
    """True when the table is already present, so a BRD run need not rebuild it."""
    return table_status(table, project, dataset)["exists"]


def table_status(table: str, project: str | None = None, dataset: str | None = None) -> dict:
    """Existence plus the freshness evidence a reviewer needs to judge a skip.

    A name existing is not evidence the table holds the current data, so the row count
    and last-modified time are reported alongside it. Values are read from the
    warehouse's own metadata; nothing here is estimated.
    """
    blank = {"table": table, "exists": False, "rows": None, "modified": None,
             "simulated": backend() == "simulated"}
    if backend() == "simulated":
        matches = [path for path in SIMULATED_DIR.glob(f"{table}.*") if not path.name.startswith("~$")]
        if not matches:
            return blank
        stamp = dt.datetime.fromtimestamp(matches[0].stat().st_mtime, dt.timezone.utc)
        return {**blank, "exists": True, "modified": stamp.strftime("%Y-%m-%d %H:%M UTC")}
    from google.cloud import bigquery

    project = project or os.environ.get("BRD_BQ_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    dataset = dataset or os.environ.get("BRD_BQ_DATASET")
    try:
        meta = bigquery.Client(project=project).get_table(
            ".".join(part for part in (project, dataset, table) if part))
    except Exception:
        return blank
    modified = getattr(meta, "modified", None)
    return {**blank, "exists": True, "rows": int(meta.num_rows or 0),
            "modified": modified.strftime("%Y-%m-%d %H:%M UTC") if modified else None}


def describe_status(status: dict) -> str:
    """One line of evidence about a table, so a skip is never an unexplained claim."""
    if not status.get("exists"):
        return f"{status['table']} — not found"
    parts = [f"{status['rows']:,} rows"] if status.get("rows") is not None else []
    if status.get("modified"):
        parts.append(f"last modified {status['modified']}")
    where = "local stand-in file" if status.get("simulated") else "BigQuery"
    return f"{status['table']} — {', '.join(parts) or 'exists'} ({where})"


def existing_tables(tables, project: str | None = None, dataset: str | None = None) -> list[str]:
    """Return the subset of the given table names that already exist."""
    return [name for name in tables if table_exists(name, project, dataset)]


def curated_only(tables) -> list[str]:
    """Chartable layers: the 02 KPI/summary tables and the 01 curated facts."""
    curated = [name for name in tables
               if name.split(".")[-1].startswith(CURATED_PREFIXES)]
    return curated or list(tables)


def sibling_datasets(project: str | None, dataset: str | None) -> list[str]:
    """Datasets of the same family that hold the other managed layers.

    The framework writes 00/01/02 to separate datasets whose names differ only by that
    layer number, so the companions are derived from the configured one rather than
    guessed. Only datasets that actually exist are returned.
    """
    if not dataset:
        return []
    match = re.search(r"(0[012])", dataset)
    if not match:
        return [dataset]
    found = []
    for layer in ("00", "01", "02"):
        candidate = dataset[:match.start()] + layer + dataset[match.end():]
        if candidate == dataset:
            found.append(candidate)
            continue
        try:
            if list_tables(project, candidate):
                found.append(candidate)
        except WarehouseError:
            continue
    return found or [dataset]


def qualified_tables(project: str | None, datasets) -> list[str]:
    """Fully qualified table identifiers across the given datasets, newest layer first."""
    names: list[str] = []
    for dataset in datasets:
        try:
            for table in list_tables(project, dataset):
                names.append(f"{project}.{dataset}.{table}" if project else f"{dataset}.{table}")
        except WarehouseError:
            continue
    return names


def console_link(project: str | None, dataset: str | None, table: str | None = None) -> str | None:
    """Google Cloud console URL for a dataset or table, so a reviewer can open the source."""
    if not project or not dataset:
        return None
    base = "https://console.cloud.google.com/bigquery"
    if table:
        ws = f"!1m5!1m4!4m3!1s{project}!2s{dataset}!3s{table}"
    else:
        ws = f"!1m4!1m3!3m2!1s{project}!2s{dataset}"
    return f"{base}?ws={ws}&project={project}"


def preview_table(table: str, project: str | None = None, dataset: str | None = None,
                  rows: int = 10) -> tuple[pd.DataFrame, dict]:
    """Read a few real rows so a reviewer can compare them against BigQuery itself."""
    rows = max(1, min(int(rows), 200))
    if backend() == "simulated":
        frame, info = _load_simulated(table, rows)
        return frame, info
    from google.cloud import bigquery

    project = project or os.environ.get("BRD_BQ_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    dataset = dataset or os.environ.get("BRD_BQ_DATASET")
    _check_name(table, "table")
    _check_name(dataset or "", "dataset")
    reference = ".".join(part for part in (project, dataset, table) if part)
    try:
        client = bigquery.Client(project=project)
        meta = client.get_table(reference)
        frame = client.list_rows(meta, max_results=rows).to_dataframe()
    except Exception as exc:
        raise WarehouseError(f"Could not preview {reference}: {type(exc).__name__}: {exc}") from exc
    frame, dropped = _drop_repeated(frame)
    return frame, {"table": table, "source": reference, "total_rows": int(meta.num_rows or 0),
                   "read_rows": len(frame), "columns": len(meta.schema),
                   "dropped_columns": dropped, "simulated": False,
                   "link": console_link(project, dataset, table)}


def load_table(table: str, project: str | None = None, dataset: str | None = None,
               limit: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Return a cleaned frame plus a disclosure describing exactly what was read."""
    rows = max(1, min(int(limit or DEFAULT_ROW_LIMIT), MAX_ROW_LIMIT))
    if backend() == "simulated":
        return _load_simulated(table, rows)
    return _load_bigquery(table, project, dataset, rows)


def _load_simulated(table: str, rows: int) -> tuple[pd.DataFrame, dict]:
    matches = [path for path in SIMULATED_DIR.glob(f"{table}.*") if not path.name.startswith("~$")]
    if not matches:
        raise WarehouseError(f"No local file stands in for table '{table}'.")
    path = matches[0]
    content = path.read_bytes()
    if path.suffix.lower() == ".csv":
        sheet = "csv"
    else:
        sheet = workbook_sheet_names(content, path.name)[0]
    header = detect_header_row(content, path.name, sheet)
    raw = load_sheet(content, path.name, sheet, header)
    frame = clean_frame(raw)
    total = len(frame)
    truncated = total > rows
    if truncated:
        frame = frame.head(rows).copy()
    return frame, {
        "backend": "simulated",
        "dropped_columns": [],
        "simulated": True,
        "table": table,
        "source": str(path),
        "total_rows": total,
        "read_rows": len(frame),
        "columns": len(frame.columns),
        "truncated": truncated,
        "limit": rows,
    }


def _load_bigquery(table: str, project: str | None, dataset: str | None, rows: int) -> tuple[pd.DataFrame, dict]:
    from google.cloud import bigquery

    project = project or os.environ.get("BRD_BQ_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    dataset = dataset or os.environ.get("BRD_BQ_DATASET")
    _check_name(table, "table")
    _check_name(dataset or "", "dataset")
    if project:
        _check_name(project.replace(":", "-"), "project")
    reference = ".".join(part for part in (project, dataset, table) if part)
    try:
        client = bigquery.Client(project=project)
        meta = client.get_table(reference)
        total = int(meta.num_rows or 0)
        query = f"SELECT * FROM `{reference}` LIMIT {rows}"
        frame = client.query(query).to_dataframe()
    except Exception as exc:
        raise WarehouseError(f"Could not read {reference}: {type(exc).__name__}: {exc}") from exc
    frame, dropped = _drop_repeated(frame)
    frame = clean_frame(frame)
    return frame, {
        "backend": "bigquery",
        "dropped_columns": dropped,
        "simulated": False,
        "table": table,
        "source": reference,
        "total_rows": total or len(frame),
        "read_rows": len(frame),
        "columns": len(frame.columns),
        "truncated": bool(total and total > len(frame)),
        "limit": rows,
    }


def disclosure(info: dict) -> str:
    """One line the UI shows so the scope of the read is never hidden."""
    where = "simulated warehouse" if info.get("simulated") else "BigQuery"
    if info.get("truncated"):
        text = (f"Read {info['read_rows']:,} of {info['total_rows']:,} rows from {where} table "
                f"'{info['table']}' (limit {info['limit']:,}). Charts cover only the rows read.")
    else:
        text = f"Read all {info['read_rows']:,} rows from {where} table '{info['table']}'."
    dropped = info.get("dropped_columns") or []
    if dropped:
        text += (f" {len(dropped)} repeated (ARRAY) column(s) cannot be charted and were "
                 f"excluded: {', '.join(dropped)}.")
    return text
