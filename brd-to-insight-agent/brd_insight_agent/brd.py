"""Run the BRD-to-BigQuery agent framework and report the tables it created.

The framework (github.com/thc-clgx/brd-to-bigquery-agent) is a separate repository, so it
is reached through a configurable adapter rather than a hard import.

It expects an **input folder** holding exactly one BRD document plus the source files the
BRD registers, and it prints a JSON summary on standard output. Two entry points exist:

``executer.py``                        reads ``BRD_INPUT_DIR``; prints status/run_id/report_path
``-m src.orchestrator.managed_runner`` takes ``--brd`` and ``--source-root``; also prints row_counts

``row_counts`` is keyed by fully qualified ``project.dataset.table`` identifiers, so it is
the most direct source of created tables. When it is absent, the written ``run_report.json``
in the report directory is read instead.

Set the mode with ``BRD_AGENT_MODE``:

``cli``     run ``BRD_AGENT_CMD`` as a subprocess; ``{brd}`` and ``{source_dir}`` are substituted
``module``  import ``BRD_AGENT_CALLABLE`` (``package.module:function``) and call it
``demo``    no framework call; records the BRD and emits a clearly-labelled simulated result

Nothing here reads or echoes credentials.
"""

from __future__ import annotations

import datetime as dt
import importlib
import hashlib
import json
import os
import queue
import re
import shlex
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

MAX_BRD_BYTES = 25 * 1024 * 1024
MAX_INPUT_FILES = 25
DEFAULT_TIMEOUT = int(os.environ.get("BRD_AGENT_TIMEOUT", "900"))

# The framework's own BRD_EXTENSIONS; anything else in the folder is treated as source data.
BRD_SUFFIXES = (".docx", ".pdf", ".txt", ".md", ".markdown")
SOURCE_SUFFIXES = (".csv", ".xlsx", ".xlsm", ".xls")
UPLOAD_SUFFIXES = BRD_SUFFIXES + SOURCE_SUFFIXES

# Statuses meaning the framework did not stop with a correction report.
OK_STATUSES = ("SUCCESS", "DRY_RUN_READY")

# project.dataset.00_name / 01_name, matching the framework's table_id contract.
_TABLE_PATTERN = re.compile(
    r"\b([A-Za-z][A-Za-z0-9\-]{3,28}[A-Za-z0-9])\.([A-Za-z_][A-Za-z0-9_]*)\.((?:00|01|02)_[A-Za-z0-9_]+)\b"
)
_CURATED = re.compile(r"(^|\.)(01|02)_")


class BrdError(Exception):
    """The BRD run could not be completed."""


@dataclass
class BrdResult:
    ok: bool
    mode: str
    status: str = ""
    tables: list[str] = field(default_factory=list)
    curated: list[str] = field(default_factory=list)
    project: str | None = None
    dataset: str | None = None
    run_id: str | None = None
    report_dir: str | None = None
    issues: list = field(default_factory=list)
    simulated: bool = False
    duration: float = 0.0
    detail: str = ""

    @property
    def summary(self) -> str:
        if not self.ok:
            return f"The BRD run reported {self.status or 'no status'} after {self.duration:.1f}s."
        where = ".".join(part for part in (self.project, self.dataset) if part)
        target = f" in {where}" if where else ""
        return (f"{self.status or 'completed'}: {len(self.tables)} table(s) reported{target} "
                f"after {self.duration:.1f}s.")


def configured_mode() -> str:
    mode = (os.environ.get("BRD_AGENT_MODE") or "").strip().lower()
    if mode in {"cli", "module", "demo"}:
        return mode
    if os.environ.get("BRD_AGENT_CMD"):
        return "cli"
    if os.environ.get("BRD_AGENT_CALLABLE"):
        return "module"
    return "demo"


def configuration() -> dict:
    """Describe how the agent will be reached. Values here are never secrets."""
    mode = configured_mode()
    return {
        "mode": mode,
        "command": os.environ.get("BRD_AGENT_CMD") if mode == "cli" else None,
        "callable": os.environ.get("BRD_AGENT_CALLABLE") if mode == "module" else None,
        "working_dir": os.environ.get("BRD_AGENT_DIR"),
        "project": os.environ.get("BRD_BQ_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT"),
        "dataset": os.environ.get("BRD_BQ_DATASET"),
        "timeout_seconds": DEFAULT_TIMEOUT,
    }


def source_fingerprint(uploads) -> str:
    """Digest of the uploaded source files, so changed sources are not skipped as 'already built'.

    Only the source data is included: re-wording the BRD does not change the data a table
    was built from. Order of upload is irrelevant, so the files are sorted by name.
    """
    digest = hashlib.sha256()
    entries = []
    for name, content in uploads:
        safe = Path(name).name
        if safe.startswith("~$") or Path(name).suffix.lower() not in SOURCE_SUFFIXES:
            continue
        entries.append((safe.lower(), safe, content))
    for _, safe, content in sorted(entries, key=lambda item: item[0]):
        digest.update(safe.encode("utf-8"))
        digest.update(hashlib.sha256(content or b"").digest())
    return digest.hexdigest()


def _fmt_bytes(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(number) < 1024 or unit == "TB":
            return f"{number:,.1f} {unit}" if unit != "B" else f"{int(number):,} B"
        number /= 1024
    return str(value)


def run_transcript(report: dict) -> list[str]:
    """Expand a framework report into the detailed line-by-line record of that run.

    Every line is composed from values the framework itself wrote: resources, source
    inventory and checksums, compiled SQL, uploaded objects, BigQuery job identifiers,
    schema actions, acceptance checks, merge statistics and final row counts. Nothing is
    invented, and the ordering follows the pipeline's own recorded steps.
    """
    lines: list[str] = []
    add = lines.append

    resources = report.get("resources") or {}
    if resources:
        add("── environment ──")
        for key, value in resources.items():
            add(f"   {key:28} {value}")
    if report.get("brd_file"):
        add(f"   {'BRD document':28} {report['brd_file']}")
    if report.get("brd_sha256"):
        add(f"   {'BRD sha256':28} {str(report['brd_sha256'])[:48]}")
    if report.get("run_id"):
        add(f"   {'run id':28} {report['run_id']}")

    steps = [s for s in report.get("steps", []) if isinstance(s, dict) and s.get("step")]
    inventory = report.get("source_inventory") or []
    manifest = {m.get("source_file_name"): m.get("sha256") for m in report.get("source_manifest") or []
                if isinstance(m, dict)}
    sql_files = report.get("sql_files") or []
    objects = report.get("objects") or []
    jobs = report.get("jobs") or []
    schema_actions = report.get("schema_actions") or []
    acceptance = report.get("analytical_acceptance") or []
    merges = report.get("merge_statistics") or []
    contracts = report.get("curated_contracts") or []
    dry_run = report.get("sql_dry_run") or {}

    for step in steps:
        add("")
        add(f"[{step.get('status', '')}] {step.get('step', '')}")
        if step.get("actor"):
            add(f"   actor: {step['actor']}")
        if step.get("detail"):
            add(f"   {step['detail']}")
        name = str(step.get("step", "")).lower()

        if "source validation" in name:
            for item in inventory:
                if not isinstance(item, dict):
                    continue
                source = item.get("source_file", "")
                columns = item.get("detected_columns")
                count = len(columns) if isinstance(columns, (list, tuple)) else columns
                add(f"   file   {source} -> feed '{item.get('matched_feed', '')}' ({count} columns)")
                if isinstance(columns, (list, tuple)) and columns:
                    add(f"          {', '.join(str(c) for c in columns)[:96]}")
                if manifest.get(source):
                    add(f"          sha256 {str(manifest[source])[:48]}")
        elif "sql compilation" in name:
            for path in sql_files:
                add(f"   wrote  {str(path).split('/generated/')[-1]}")
            if dry_run:
                add(f"   dry run {dry_run.get('status', '')} · "
                    f"{_fmt_bytes(dry_run.get('estimated_bytes', 0))} would be processed")
        elif "acceptance" in name:
            for check in acceptance:
                if isinstance(check, dict):
                    add(f"   {check.get('status', ''):5} {check.get('check', '')}"
                        f"  job {str(check.get('job_id', ''))[:18]}")
        elif "bucket" in name or "upload" in name or "loading" in name:
            for obj in objects:
                if isinstance(obj, dict) and obj.get("uri"):
                    add(f"   upload {str(obj['uri'])[:96]}")
            for job in jobs:
                if isinstance(job, dict):
                    add(f"   load   {str(job.get('table', '')).split('.')[-1]:38} "
                        f"job {str(job.get('job_id', ''))[:30]}")
        elif "provisioning" in name:
            for action in schema_actions:
                if isinstance(action, dict):
                    table = str(action.get("table") or "").split(".")[-1] or "(dataset)"
                    add(f"   {str(action.get('action', '')):6} {table}")
            for contract in contracts:
                if isinstance(contract, dict) and contract.get("table_id"):
                    columns = contract.get("columns") or []
                    add(f"   schema {str(contract['table_id']).split('.')[-1]:38} "
                        f"{len(columns)} columns")
        elif "publication" in name and merges:
            for merge in merges:
                if isinstance(merge, dict):
                    add(f"   merge  job {str(merge.get('job_id', ''))[:34]} "
                        f"inserted={merge.get('inserted', 0)} updated={merge.get('updated', 0)} "
                        f"deleted={merge.get('deleted', 0)}")
        elif "reconciliation" in name:
            for table, count in (report.get("row_counts") or {}).items():
                add(f"   verify {str(table):42} {int(count):>8,} rows")

    counts = report.get("row_counts") or {}
    if counts:
        add("")
        add("── final row counts ──")
        for table, count in counts.items():
            add(f"   {str(table):42} {int(count):>8,} rows")
        add("")
        add(f"   {len(counts)} managed tables · {sum(int(c) for c in counts.values()):,} rows total")
    return lines


def last_successful_run(reports_dir: str | Path | None = None) -> dict | None:
    """Read the newest successful framework report that is still on disk.

    Used to replay what the framework actually did when it built the tables that already
    exist, without calling it again. Every value returned is read from that run's own
    report: nothing here is generated, and the caller must label it as a replay.
    """
    folder = Path(reports_dir) if reports_dir else None
    if folder is None:
        base = os.environ.get("BRD_AGENT_DIR")
        if not base:
            return None
        folder = Path(base) / "generated" / "reports"
    if not folder.is_dir():
        return None
    reports = sorted(folder.glob("*/run_report.json"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    for path in reports:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if report.get("status") not in OK_STATUSES or not report.get("row_counts"):
            continue
        steps = [
            {"step": str(item.get("step", "")), "actor": str(item.get("actor", "")),
             "status": str(item.get("status", "")), "detail": str(item.get("detail", "") or "")}
            for item in report.get("steps", []) if isinstance(item, dict) and item.get("step")
        ]
        tables, project, dataset = parse_tables(
            json.dumps({"row_counts": report.get("row_counts")}))
        return {
            "run_id": str(report.get("run_id") or ""),
            "status": str(report.get("status") or ""),
            "steps": steps,
            "transcript": run_transcript(report),
            "row_counts": dict(report.get("row_counts") or {}),
            "tables": tables,
            "project": project or os.environ.get("BRD_BQ_PROJECT"),
            "dataset": dataset or os.environ.get("BRD_BQ_DATASET"),
            "brd_file": str(report.get("brd_file") or ""),
            "report_dir": str(path.parent),
            "finished_at": dt.datetime.fromtimestamp(
                path.stat().st_mtime, dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        }
    return None


def save_inputs(uploads) -> tuple[Path, Path]:
    """Write the BRD and its source files into one input folder.

    The framework requires exactly one BRD document in the folder; source files sit
    alongside it. Returns the folder and the BRD path.
    """
    uploads = list(uploads)
    if not uploads:
        raise BrdError("Upload a BRD document first.")
    if len(uploads) > MAX_INPUT_FILES:
        raise BrdError(f"Upload at most {MAX_INPUT_FILES} files.")
    documents = [name for name, _ in uploads
                 if Path(name).suffix.lower() in BRD_SUFFIXES and not Path(name).name.startswith("~$")]
    if len(documents) != 1:
        raise BrdError(
            f"The input folder needs exactly one BRD document; found {len(documents)}. "
            f"Supported: {', '.join(BRD_SUFFIXES)}. Keep source data as CSV or Excel."
        )
    for name, content in uploads:
        if not content:
            raise BrdError(f"{Path(name).name} is empty.")
        if len(content) > MAX_BRD_BYTES:
            raise BrdError(f"{Path(name).name} exceeds {MAX_BRD_BYTES // (1024 * 1024)} MB.")
        suffix = Path(name).suffix.lower()
        if suffix not in UPLOAD_SUFFIXES:
            raise BrdError(f"Unsupported file type '{suffix}'. Use a BRD document plus CSV or Excel sources.")
    folder = Path(tempfile.mkdtemp(prefix="brd_input_"))
    for name, content in uploads:
        safe = Path(name).name
        if safe.startswith("~$"):
            continue
        (folder / safe).write_bytes(content)
    return folder, folder / documents[0]


def save_upload(name: str, content: bytes) -> Path:
    """Write a single BRD and return its path."""
    return save_inputs([(name, content)])[1]


def _json_candidates(text: str):
    """Yield every JSON object embedded anywhere in the output."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except ValueError:
            continue
        if isinstance(value, dict):
            yield value


def parse_tables(text: str) -> tuple[list[str], str | None, str | None]:
    """Pull table identifiers out of the framework's output."""
    project = dataset = None
    qualified: list[str] = []
    plain: list[str] = []
    for blob in _json_candidates(text):
        project = blob.get("project") or blob.get("project_id") or project
        dataset = blob.get("dataset") or blob.get("dataset_id") or dataset
        counts = blob.get("row_counts")
        if isinstance(counts, dict):
            qualified.extend(str(key) for key in counts)
        found = blob.get("tables") or blob.get("created_tables") or blob.get("table_ids")
        if isinstance(found, list):
            for item in found:
                if isinstance(item, str):
                    plain.append(item)
                elif isinstance(item, dict):
                    name = item.get("table_id") or item.get("table") or item.get("name")
                    if name:
                        plain.append(str(name))
        elif isinstance(found, str):
            plain.append(found)
    if not qualified and not plain:
        qualified = [".".join(match.groups()) for match in _TABLE_PATTERN.finditer(text)]
    for identifier in qualified:
        parts = identifier.split(".")
        if len(parts) == 3:
            project = project or parts[0]
            dataset = dataset or parts[1]
    seen, unique = set(), []
    for name in (qualified or plain):
        short = name.split(".")[-1]
        if short and short not in seen:
            seen.add(short)
            unique.append(short)
    return unique, project, dataset


def _report_tables(report_dir: str | None) -> list[str]:
    """Read table identifiers from the run_report.json the framework writes."""
    if not report_dir:
        return []
    path = Path(report_dir) / "run_report.json"
    if not path.is_file():
        return []
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    found = [str(key) for key in (report.get("row_counts") or {})]
    contract = (report.get("llm") or {}).get("contract") or {}
    for table in contract.get("tables") or []:
        if isinstance(table, dict) and table.get("table_id"):
            found.append(str(table["table_id"]))
    return found


def _build_result(text: str, mode: str, elapsed: float) -> BrdResult:
    status = run_id = report_dir = None
    issues: list = []
    for blob in _json_candidates(text):
        status = blob.get("status") or status
        run_id = blob.get("run_id") or run_id
        report_dir = blob.get("report_directory") or blob.get("report_path") or report_dir
        if isinstance(blob.get("issues"), list) and blob["issues"]:
            issues = blob["issues"]
    tables, project, dataset = parse_tables(text)
    if not tables:
        extra = _report_tables(report_dir)
        if extra:
            tables, project, dataset = parse_tables(
                json.dumps({"row_counts": {name: 0 for name in extra}})
            )
    curated = [name for name in tables if _CURATED.search(name)]
    project = project or os.environ.get("BRD_BQ_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    dataset = dataset or os.environ.get("BRD_BQ_DATASET")
    ok = bool(tables) and (status is None or status in OK_STATUSES)
    result = BrdResult(ok=ok, mode=mode, status=status or "", tables=tables, curated=curated,
                       project=project, dataset=dataset, run_id=run_id, report_dir=report_dir,
                       issues=issues, duration=elapsed, detail=text[-4000:])
    if status and status not in OK_STATUSES:
        result.detail = (f"The framework stopped with status {status}. It writes a correction "
                         "report rather than guessing.\n\n" + result.detail)
    elif not tables:
        result.detail = ("The framework ran but reported no table identifiers. A dry run does not "
                         "populate row counts; run without dry run to create tables.\n\n" + result.detail)
    return result


def planned_tables(brd_path: Path, qualified: bool = False) -> list[str]:
    """Table identifiers the BRD itself names, read without running the framework.

    Used to detect tables that already exist so a run can be skipped rather than repeated.
    With ``qualified`` the full ``project.dataset.table`` form is returned, which the UI
    needs to build a console link and read a preview from the right dataset.
    """
    try:
        text = Path(brd_path).read_bytes()
    except OSError:
        return []
    if Path(brd_path).suffix.lower() == ".docx":
        try:
            import zipfile
            with zipfile.ZipFile(Path(brd_path)) as archive:
                xml = archive.read("word/document.xml").decode("utf-8", "ignore")
            # Word splits a single identifier across formatting runs, so run tags are removed
            # without inserting spaces; only paragraph and row ends become line breaks.
            xml = re.sub(r"</w:(?:p|tr)>", "\n", xml)
            xml = re.sub(r"<w:tab[^>]*/>", " ", xml)
            body = re.sub(r"<[^>]+>", "", xml)
        except Exception:
            return []
    else:
        body = text.decode("utf-8", "ignore")
    seen, unique = set(), []
    for match in _TABLE_PATTERN.finditer(body):
        name = ".".join(match.groups()) if qualified else match.group(3)
        if name not in seen:
            seen.add(name)
            unique.append(name)
    return unique


def run_agent(brd_path: Path, timeout: int | None = None, dry_run: bool = False,
              source_dir: Path | None = None, on_line=None) -> BrdResult:
    """Invoke the configured framework for one BRD input folder.

    ``on_line`` receives each output line as it arrives, already redacted for display,
    so the caller can show progress while the framework is still running.
    """
    brd_path = Path(brd_path)
    if not brd_path.exists():
        raise BrdError("The BRD file is no longer available. Upload it again.")
    folder = Path(source_dir) if source_dir else brd_path.parent
    mode = configured_mode()
    seconds = int(timeout or DEFAULT_TIMEOUT)
    started = time.monotonic()
    if mode == "demo":
        return _run_demo(brd_path, time.monotonic() - started)
    try:
        if mode == "cli":
            text = _run_cli(brd_path, folder, seconds, dry_run, on_line)
        else:
            text = _run_module(brd_path, folder, dry_run)
    except BrdError:
        raise
    except Exception as exc:
        raise BrdError(f"The framework could not be started: {type(exc).__name__}: {exc}") from exc
    return _build_result(text, mode, time.monotonic() - started)


def _env_file_values(folder: str | None) -> dict:
    """Read the framework's private KEY=VALUE file as data.

    The direct ``managed_runner`` entry point reads only process variables, so its ``.env``
    would otherwise be ignored. Parsed exactly like the framework does: never executed as
    shell, and values are never logged or returned anywhere they could be displayed.
    """
    configured = os.environ.get("BRD_AGENT_ENV_FILE")
    path = Path(configured).expanduser() if configured else (Path(folder) / ".env" if folder else None)
    if not path or not path.is_file():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[7:].lstrip()
        if "=" not in stripped:
            continue
        name, _, raw = stripped.partition("=")
        name = name.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            continue
        raw = raw.strip()
        if raw.startswith(("'", '"')) and len(raw) > 1 and raw[-1] == raw[0]:
            raw = raw[1:-1]
        else:
            raw = raw.split("#", 1)[0].rstrip()
        if raw and not raw.startswith("<"):
            values[name] = raw
    return values


def env_file_status(folder: str | None = None) -> dict:
    """Report which settings the framework's env file supplies. Never returns values."""
    folder = folder or os.environ.get("BRD_AGENT_DIR")
    values = _env_file_values(folder)
    needed = ("LLM_PROVIDER", "LLM_MODEL", "LLM_API_KEY", "GOOGLE_CLOUD_PROJECT")
    present = {key for key in needed if values.get(key) or os.environ.get(key)}
    return {"found": bool(values), "configured": sorted(present),
            "missing": sorted(set(needed) - present)}


def redact_log(line: str) -> str:
    """Mask identities and secrets before a framework log line is shown or recorded.

    Display only: the full text is still parsed for status, tables and row counts. The
    framework logs service-account impersonation and GCS paths, which should not end up
    in a screen recording.
    """
    line = re.sub(r"[\w.+-]+@[\w.-]+\.iam\.gserviceaccount\.com", "[service-account]", line)
    line = re.sub(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}", "[email]", line)
    # Prefixed names such as LLM_API_KEY have no word boundary before 'api', so match the
    # whole surrounding token rather than the bare word.
    line = re.sub(r"(?i)([\w.-]*(?:api[_-]?key|token|secret|password|credential)[\w.-]*)\s*[:=]\s*\S+",
                  r"\1=[redacted]", line)
    return line


def _terminate(process) -> None:
    """Stop the framework and anything it started; it runs in its own process group."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            process.kill()
        except (ProcessLookupError, OSError):
            return
    try:
        process.wait(timeout=5)
    except (subprocess.SubprocessError, OSError):
        pass


def _stream(process, deadline: float, seconds: int, on_line) -> list[str]:
    """Collect output line by line so progress is visible while the framework runs.

    A reader thread keeps the deadline enforceable even when the framework goes quiet,
    which a plain blocking read could not do.
    """
    received: "queue.Queue[str | None]" = queue.Queue()

    def reader():
        try:
            for line in process.stdout:
                received.put(line)
        except (OSError, ValueError):
            pass
        finally:
            received.put(None)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    lines: list[str] = []
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _terminate(process)
            raise BrdError(f"The framework did not finish within {seconds}s. Nothing was changed here.")
        try:
            item = received.get(timeout=min(1.0, remaining))
        except queue.Empty:
            continue
        if item is None:
            break
        lines.append(item)
        if on_line is not None:
            try:
                on_line(redact_log(item.rstrip("\n")))
            except Exception:
                # A display problem must never abandon a run that is writing to BigQuery.
                on_line = None
    return lines


def _run_cli(brd_path: Path, folder: Path, seconds: int, dry_run: bool, on_line=None) -> str:
    template = os.environ.get("BRD_AGENT_CMD", "").strip()
    if not template:
        raise BrdError("BRD_AGENT_CMD is not set.")
    substituted = "{brd}" in template or "{source_dir}" in template
    command = [part.replace("{brd}", str(brd_path)).replace("{source_dir}", str(folder))
               for part in shlex.split(template)]
    if not substituted:
        command += ["--brd", str(brd_path), "--source-root", str(folder)]
    if dry_run:
        command.append(os.environ.get("BRD_AGENT_DRYRUN_FLAG", "--dry-run"))
    # The root launcher takes the input folder from the environment rather than argv, and the
    # direct runner reads only process variables, so the framework's own env file is merged in.
    workdir = os.environ.get("BRD_AGENT_DIR") or None
    # Unbuffered, otherwise Python holds the framework's progress until it exits.
    environment = dict(os.environ, BRD_INPUT_DIR=str(folder), PYTHONUNBUFFERED="1")
    for key, value in _env_file_values(workdir).items():
        environment.setdefault(key, value)
    deadline = time.monotonic() + seconds
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, cwd=workdir, env=environment,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
    except FileNotFoundError as exc:
        raise BrdError(f"Command not found: {command[0]}") from exc
    try:
        lines = _stream(process, deadline, seconds, on_line)
        returncode = process.wait(timeout=max(1.0, deadline - time.monotonic()))
    except BrdError:
        raise
    except subprocess.TimeoutExpired as exc:
        _terminate(process)
        raise BrdError(f"The framework did not finish within {seconds}s. Nothing was changed here.") from exc
    finally:
        if process.stdout is not None:
            process.stdout.close()
    output = "".join(lines)
    # Exit code 2 is the framework's "stopped with a correction report"; keep its JSON.
    if returncode not in (0, 2):
        raise BrdError(f"The framework exited with code {returncode}.\n\n{output[-3000:]}")
    if returncode == 2 and not any(char == "{" for char in output):
        raise BrdError(f"The framework exited with code 2 and no report.\n\n{output[-3000:]}")
    return output


def _run_module(brd_path: Path, folder: Path, dry_run: bool) -> str:
    target = os.environ.get("BRD_AGENT_CALLABLE", "").strip()
    if ":" not in target:
        raise BrdError("BRD_AGENT_CALLABLE must look like 'package.module:function'.")
    module_name, _, function_name = target.partition(":")
    module = importlib.import_module(module_name)
    function = getattr(module, function_name, None)
    if function is None or not callable(function):
        raise BrdError(f"'{target}' is not a callable.")
    try:
        returned = function(str(brd_path), str(folder), dry_run=dry_run)
    except TypeError:
        returned = function(str(brd_path))
    if isinstance(returned, (dict, list)):
        return json.dumps(returned, default=str)
    return str(returned)


def _run_demo(brd_path: Path, elapsed: float) -> BrdResult:
    """No framework is configured: report a simulated run, clearly labelled as such.

    The tables offered are ones the configured warehouse can actually serve: real BigQuery
    tables when a project is set, otherwise local files standing in for them. The rest of
    the workflow therefore stays genuine, never generated values.
    """
    from brd_insight_agent import warehouse

    live = warehouse.backend() == "bigquery"
    try:
        tables = warehouse.list_tables() if live else warehouse.simulated_tables()
    except Exception:
        tables = []
    if live:
        detail = (f"No BRD agent is configured, so '{brd_path.name}' was not processed and no "
                  "table was created. The tables listed below already exist in the configured "
                  "BigQuery dataset. Set BRD_AGENT_MODE with BRD_AGENT_CMD or BRD_AGENT_CALLABLE "
                  "to run the real framework.")
    else:
        detail = (f"No BRD agent is configured, so '{brd_path.name}' was not processed and no "
                  "BigQuery table was created. The tables listed below are local files standing "
                  "in for warehouse tables. Set BRD_AGENT_MODE with BRD_AGENT_CMD or "
                  "BRD_AGENT_CALLABLE to run the real framework.")
    return BrdResult(
        ok=bool(tables), mode="demo", status="SIMULATED", tables=tables,
        project=os.environ.get("BRD_BQ_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or ("" if live else "simulated-project"),
        dataset=os.environ.get("BRD_BQ_DATASET") or ("" if live else "simulated_dataset"),
        simulated=True, duration=elapsed, detail=detail,
    )
