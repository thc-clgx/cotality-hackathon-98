"""Render and package charts without leaving a stalled browser in the app."""

from __future__ import annotations

import io
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from numbers import Integral, Real
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image

DEFAULT_SCALE = 2
DEFAULT_SIZE = (1280, 720)
DEFAULT_TIMEOUT = 30
MAX_FIGURES = 50
MAX_DIMENSION = 8192
MAX_IMAGE_PIXELS = 16_777_216
MAX_TOTAL_PIXELS = 128_000_000
MAX_PNG_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024
MAX_REQUEST_BYTES = 64 * 1024 * 1024
CLEANUP_TIMEOUT = 5
_IS_WINDOWS = sys.platform == "win32"
_CREATE_NO_WINDOW = 0x08000000
_CREATE_NEW_PROCESS_GROUP = 0x00000200


class ExportError(RuntimeError):
    """Raised when an export is invalid, unavailable, incomplete, or stalled."""


def _positive_number(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ExportError(f"{name} must be a finite positive number.")
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise ExportError(f"{name} must be a finite positive number.") from exc
    if not math.isfinite(number) or number <= 0:
        raise ExportError(f"{name} must be a finite positive number.")
    return number


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ExportError(f"{name} must be a positive integer.")
    return int(value)


def _validate_dimensions(width: int, height: int) -> None:
    width = _positive_integer(width, "Width")
    height = _positive_integer(height, "Height")
    if max(width, height) > MAX_DIMENSION or width * height > MAX_IMAGE_PIXELS:
        raise ExportError(
            f"Each image is limited to {MAX_DIMENSION}px per side and "
            f"{MAX_IMAGE_PIXELS:,} pixels; reduce size or scale."
        )


def _validate_figures(figures: list[tuple[str, go.Figure]]) -> None:
    if not isinstance(figures, (list, tuple)) or not figures:
        raise ExportError("Nothing to render; provide a nonempty list of titled figures.")
    if len(figures) > MAX_FIGURES:
        raise ExportError(f"Export at most {MAX_FIGURES} charts at a time.")
    for item in figures:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ExportError("Each chart must be a (title, go.Figure) pair.")
        title, figure = item
        if not isinstance(title, str) or not isinstance(figure, go.Figure):
            raise ExportError("Each chart needs a string title and a go.Figure.")


def _renderer_environment() -> dict[str, str]:
    excluded = {"GEMINI_API_KEY", "GOOGLE_API_KEY", "PYTHONPATH", "PYTHONHOME"}
    return {key: value for key, value in os.environ.items() if key.upper() not in excluded}


def _terminate_process_tree(process: subprocess.Popen) -> str:
    deadline = time.monotonic() + CLEANUP_TIMEOUT
    issues = []
    try:
        if _IS_WINDOWS:
            taskkill = os.path.join(
                os.environ.get("SystemRoot", r"C:\Windows"), "System32", "taskkill.exe"
            )
            result = subprocess.run(
                [taskkill, "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=_renderer_environment(),
                creationflags=_CREATE_NO_WINDOW,
                timeout=CLEANUP_TIMEOUT,
                check=False,
            )
            if result.returncode:
                issues.append("could not confirm browser-tree termination")
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except (OSError, subprocess.SubprocessError):
        issues.append("could not confirm browser-tree termination")
    try:
        if process.poll() is None:
            process.kill()
    except ProcessLookupError:
        pass
    except OSError:
        issues.append("could not terminate the renderer")
    try:
        process.wait(timeout=max(0.0, deadline - time.monotonic()))
    except (OSError, subprocess.SubprocessError):
        issues.append("renderer did not confirm exit within the cleanup deadline")
    return (" Cleanup warning: " + "; ".join(issues) + ".") if issues else ""


def _engine_error(detail: str) -> str:
    lowered = detail.lower()
    if any(word in lowered for word in ("chrome", "chromium", "browser", "kaleido")):
        return (
            "Image rendering failed. Kaleido needs an installed, runnable Chrome/Chromium "
            "browser and compatible Plotly/Kaleido versions. Check the browser installation "
            f"and try again. Details: {detail}"
        )
    return f"Image rendering failed: {detail}"


def _figure_json(figure: go.Figure) -> str:
    try:
        return pio.to_json(figure, validate=True, pretty=False, remove_uids=True)
    except Exception as exc:
        raise ExportError(f"Invalid chart data: {exc}") from exc


def figures_to_images(
    figures: list[tuple[str, go.Figure]],
    image_format: str = "png",
    width: int = DEFAULT_SIZE[0],
    height: int = DEFAULT_SIZE[1],
    scale: int = DEFAULT_SCALE,
    timeout: int = DEFAULT_TIMEOUT,
) -> list[tuple[str, bytes]]:
    if image_format not in {"png", "svg", "jpeg"}:
        raise ExportError("Choose PNG, SVG, or JPEG.")
    _validate_figures(figures)
    width = _positive_integer(width, "Width")
    height = _positive_integer(height, "Height")
    _validate_dimensions(width, height)
    multiplier = _positive_number(scale, "Scale")
    seconds = _positive_number(timeout, "Timeout")
    if multiplier > 4:
        raise ExportError("Scale must be at most 4.")
    expected_size = (int(width * multiplier), int(height * multiplier))
    _validate_dimensions(*expected_size)
    if expected_size[0] * expected_size[1] * len(figures) > MAX_TOTAL_PIXELS:
        raise ExportError(f"The batch exceeds {MAX_TOTAL_PIXELS:,} total pixels.")

    payload = json.dumps(
        {
            "figures": [json.loads(_figure_json(figure)) for _, figure in figures],
            "width": width,
            "height": height,
            "scale": multiplier,
            "format": image_format,
        },
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(payload) > MAX_REQUEST_BYTES:
        raise ExportError("Chart data is too large; export fewer charts or fewer data points.")

    try:
        with tempfile.TemporaryDirectory(prefix="excel-chart-export-") as temporary:
            directory = Path(temporary).resolve()
            request = directory / "request.json"
            request.write_bytes(payload)
            log_path = directory / "worker.log"
            worker = Path(__file__).with_name("export_worker.py").resolve()
            options = (
                {"creationflags": _CREATE_NO_WINDOW | _CREATE_NEW_PROCESS_GROUP}
                if _IS_WINDOWS
                else {"start_new_session": True}
            )
            with log_path.open("wb") as log:
                process = subprocess.Popen(
                    [sys.executable, "-I", str(worker), str(request)],
                    cwd=str(directory),
                    env=_renderer_environment(),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=log,
                    close_fds=True,
                    **options,
                )
                try:
                    returncode = process.wait(timeout=seconds)
                except subprocess.TimeoutExpired as exc:
                    cleanup = _terminate_process_tree(process)
                    raise ExportError(
                        f"Image rendering exceeded {seconds:g}s and was stopped. "
                        f"Try fewer charts or a smaller image.{cleanup}"
                    ) from exc
                except BaseException:
                    _terminate_process_tree(process)
                    raise
                cleanup = _terminate_process_tree(process) if returncode else ""

            if returncode:
                with log_path.open("rb") as log:
                    log.seek(max(0, log_path.stat().st_size - 8192))
                    detail = log.read(8192).decode("utf-8", errors="replace").strip()
                raise ExportError(_engine_error(detail or f"worker exited with {returncode}") + cleanup)

            rendered = []
            total_bytes = 0
            for index, (title, _) in enumerate(figures, start=1):
                path = directory / f"{index:02d}.{image_format}"
                if not path.is_file() or not 0 < path.stat().st_size <= MAX_PNG_BYTES:
                    raise ExportError(f"Renderer output {index} is missing, empty, or too large.")
                total_bytes += path.stat().st_size
                if total_bytes > MAX_TOTAL_BYTES:
                    raise ExportError("Rendered images exceed the total byte limit.")
                data = path.read_bytes()
                if image_format == "svg":
                    if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
                        raise ExportError("Unexpected SVG document declarations.")
                    document = ET.fromstring(data)
                    if document.tag != "{http://www.w3.org/2000/svg}svg":
                        raise ExportError("Renderer did not produce an SVG image.")
                elif image_format == "png":
                    if _png_dimensions(data) != expected_size:
                        raise ExportError(f"Renderer output {index} has unexpected PNG dimensions.")
                else:
                    with Image.open(io.BytesIO(data)) as image:
                        if image.format != "JPEG" or image.size != expected_size:
                            raise ExportError("Renderer produced an invalid JPEG image.")
                        image.verify()
                rendered.append((f"{index:02d}_{_slug(title)}.{image_format}", data))
            return rendered
    except ExportError:
        raise
    except Exception as exc:
        raise ExportError(f"Image export failed: {exc}") from exc


def _png_dimensions(data: bytes) -> tuple[int, int]:
    if not isinstance(data, bytes) or not 0 < len(data) <= MAX_PNG_BYTES:
        raise ExportError("Each image must contain nonempty PNG bytes within the size limit.")
    try:
        with io.BytesIO(data) as stream, Image.open(stream) as image:
            if image.format != "PNG":
                raise ExportError("Only PNG images can be exported.")
            dimensions = image.size
            _validate_dimensions(*dimensions)
            image.verify()
        with io.BytesIO(data) as stream, Image.open(stream) as image:
            image.load()
        return dimensions
    except ExportError:
        raise
    except Exception as exc:
        raise ExportError(f"Invalid or incomplete PNG image: {exc}") from exc


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")[:60] or "chart"

