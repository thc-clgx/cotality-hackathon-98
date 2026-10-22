"""Export regression tests with no renderer processes, Chrome, or API access."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
import zipfile
from contextlib import ExitStack
from html import escape
from pathlib import Path
from unittest import mock

import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image


SOURCE = Path(__file__).resolve().parents[1] / "brd_insight_agent"


def _load_file(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SOURCE / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


export = _load_file("_export_under_test", "export.py")
worker = _load_file("_export_worker_under_test", "export_worker.py")


def _png(width: int = 16, height: int = 8, format: str = "PNG") -> bytes:
    with io.BytesIO() as stream, Image.new("RGB", (width, height), "white") as image:
        image.save(stream, format=format)
        return stream.getvalue()


def _figure() -> go.Figure:
    return go.Figure(go.Bar(x=["a", "b"], y=[1, 2]), layout={"template": None})


class RendererTests(unittest.TestCase):
    def setUp(self) -> None:
        self.figures = [("Sales / North", _figure()), ("Profit", _figure())]
        self.process = mock.Mock(pid=424242)
        self.process.wait.return_value = 0
        self.process.poll.return_value = 0
        self.directory = None
        self.log = None
        self.outputs = None
        self.diagnostic = b""
        patches = ExitStack()
        self.addCleanup(patches.close)
        self.popen = patches.enter_context(mock.patch.object(export.subprocess, "Popen", side_effect=self._launch))
        self.taskkill = patches.enter_context(mock.patch.object(export.subprocess, "run"))
        self.taskkill.return_value.returncode = 0
        self.killpg = patches.enter_context(mock.patch.object(export.os, "killpg", create=True))
        patches.enter_context(mock.patch.object(export.signal, "SIGKILL", 9, create=True))

    def _launch(self, command, **kwargs):
        request_path = Path(command[-1])
        self.directory = request_path.parent
        self.log = kwargs["stderr"]
        request = json.loads(request_path.read_text(encoding="utf-8"))
        outputs = self.outputs
        if outputs is None:
            data = _png(int(request["width"] * request["scale"]), int(request["height"] * request["scale"]))
            outputs = {index: data for index in range(1, len(request["figures"]) + 1)}
        for index, data in outputs.items():
            (self.directory / f"{index:02d}.png").write_bytes(data)
        self.log.write(self.diagnostic)
        return self.process

    def _render(self, **kwargs):
        options = {"width": 16, "height": 8, "scale": 1}
        options.update(kwargs)
        return export.figures_to_images(self.figures, "png", **options)

    def _assert_clean(self) -> None:
        self.assertIsNotNone(self.directory)
        self.assertFalse(self.directory.exists())
        self.assertTrue(self.log.closed)

    def test_completion_preserves_order_names_bytes_and_default_timeout(self) -> None:
        result = self._render()
        self.assertEqual(result, [("01_sales_north.png", _png()), ("02_profit.png", _png())])
        self.process.wait.assert_called_once_with(timeout=30)
        self.popen.assert_called_once()
        self.taskkill.assert_not_called()
        self.killpg.assert_not_called()
        self._assert_clean()

    def test_current_interpreter_explicit_worker_path_and_sanitized_environment(self) -> None:
        environment = {
            "GEMINI_API_KEY": "test-placeholder",
            "google_api_key": "test-placeholder",
            "PYTHONPATH": "untrusted-path",
            "PYTHONHOME": "untrusted-home",
            "PATH": "retained-path",
            "SystemRoot": "C:\\Windows",
        }
        with mock.patch.dict(os.environ, environment, clear=True):
            self._render()
            self.assertIn("GEMINI_API_KEY", os.environ)
        command = self.popen.call_args.args[0]
        options = self.popen.call_args.kwargs
        self.assertEqual(command[:2], [sys.executable, "-I"])
        self.assertEqual(Path(command[2]), (SOURCE / "export_worker.py").resolve())
        self.assertTrue(Path(command[3]).is_absolute())
        self.assertNotIn("-m", command)
        forbidden = {"GEMINI_API_KEY", "GOOGLE_API_KEY", "PYTHONPATH", "PYTHONHOME"}
        self.assertFalse(forbidden.intersection(key.upper() for key in options["env"]))
        self.assertEqual(options["env"]["PATH"], "retained-path")
        self.assertEqual(Path(options["cwd"]), self.directory)
        self.assertTrue(options["close_fds"])
        self.assertEqual(options["stdin"], subprocess.DEVNULL)
        self.assertEqual(options["stdout"], subprocess.DEVNULL)
        self._assert_clean()

    def test_fractional_scale_and_custom_timeout(self) -> None:
        result = self._render(scale=1.5, timeout=2.5)
        self.assertEqual(export._png_dimensions(result[0][1]), (24, 12))
        self.process.wait.assert_called_once_with(timeout=2.5)

    def test_missing_partial_outputs_fail_the_entire_batch(self) -> None:
        for outputs in ({}, {1: _png()}):
            with self.subTest(outputs=list(outputs)):
                self.outputs = outputs
                with self.assertRaisesRegex(export.ExportError, "missing, empty"):
                    self._render()
                self._assert_clean()

    def test_empty_corrupt_truncated_non_png_and_wrong_dimensions_fail(self) -> None:
        for bad in (b"", b"not an image", _png()[:-16], _png(format="JPEG"), _png(4, 4)):
            with self.subTest(length=len(bad)):
                self.outputs = {1: _png(), 2: bad}
                with self.assertRaises(export.ExportError):
                    self._render()
                self._assert_clean()

    def test_worker_failure_does_not_return_existing_valid_images(self) -> None:
        self.process.wait.return_value = 1
        self.diagnostic = b"ChromeNotFoundError: Chrome is not installed"
        with mock.patch.object(export, "_IS_WINDOWS", True):
            with self.assertRaisesRegex(export.ExportError, "Check the browser installation"):
                self._render()
        self.taskkill.assert_called_once()
        self._assert_clean()

    def test_empty_worker_error_still_explains_exit_status(self) -> None:
        self.process.wait.return_value = 7
        with mock.patch.object(export, "_IS_WINDOWS", False):
            with self.assertRaisesRegex(export.ExportError, "worker exited with 7"):
                self._render()
        self._assert_clean()

    def test_windows_timeout_kills_only_worker_tree_and_reaps_bounded(self) -> None:
        self.process.wait.side_effect = [subprocess.TimeoutExpired("worker", 30), 0]
        self.process.poll.return_value = None
        with mock.patch.object(export, "_IS_WINDOWS", True):
            with self.assertRaisesRegex(export.ExportError, "exceeded 30s"):
                self._render()
        command = self.taskkill.call_args.args[0]
        self.assertTrue(command[0].endswith("taskkill.exe"))
        self.assertEqual(command[1:], ["/PID", "424242", "/T", "/F"])
        self.assertNotIn("/IM", command)
        self.assertLessEqual(self.taskkill.call_args.kwargs["timeout"], export.CLEANUP_TIMEOUT)
        self.assertEqual(self.popen.call_args.kwargs["creationflags"], 0x08000200)
        self.assertNotIn("start_new_session", self.popen.call_args.kwargs)
        self.process.kill.assert_called_once()
        self.assertEqual(len(self.process.wait.call_args_list), 2)
        self.assertLessEqual(self.process.wait.call_args.kwargs["timeout"], export.CLEANUP_TIMEOUT)
        self.killpg.assert_not_called()
        self._assert_clean()

    def test_posix_timeout_kills_new_process_group_and_reaps(self) -> None:
        self.process.wait.side_effect = [subprocess.TimeoutExpired("worker", 0.1), 0]
        with mock.patch.object(export, "_IS_WINDOWS", False):
            with self.assertRaisesRegex(export.ExportError, "exceeded 0.1s"):
                self._render(timeout=0.1)
        self.assertTrue(self.popen.call_args.kwargs["start_new_session"])
        self.assertNotIn("creationflags", self.popen.call_args.kwargs)
        self.killpg.assert_called_once_with(424242, 9)
        self.taskkill.assert_not_called()
        self._assert_clean()

    def test_cleanup_failure_is_bounded_and_reported_without_a_real_wait(self) -> None:
        self.process.wait.side_effect = [
            subprocess.TimeoutExpired("worker", 30),
            subprocess.TimeoutExpired("worker", 0),
        ]
        self.process.poll.return_value = None
        self.taskkill.side_effect = subprocess.TimeoutExpired("taskkill", 5)
        with mock.patch.object(export, "_IS_WINDOWS", True), mock.patch.object(
            export.time, "monotonic", side_effect=[100, 106]
        ):
            with self.assertRaisesRegex(export.ExportError, "Cleanup warning"):
                self._render()
        self.process.kill.assert_called_once()
        self.process.wait.assert_called_with(timeout=0)
        self._assert_clean()

    def test_unexpected_wait_failure_also_cleans_up(self) -> None:
        self.process.wait.side_effect = [OSError("wait failed"), 0]
        with mock.patch.object(export, "_IS_WINDOWS", False):
            with self.assertRaisesRegex(export.ExportError, "wait failed"):
                self._render()
        self.killpg.assert_called_once()
        self._assert_clean()

    def test_cleanup_reaps_worker_even_if_it_exits_during_kill(self) -> None:
        self.process.wait.side_effect = [subprocess.TimeoutExpired("worker", 30), 0]
        self.process.poll.return_value = None
        self.process.kill.side_effect = ProcessLookupError("already exited")
        with mock.patch.object(export, "_IS_WINDOWS", False):
            with self.assertRaisesRegex(export.ExportError, "exceeded 30s"):
                self._render()
        self.assertEqual(self.process.wait.call_count, 2)
        self._assert_clean()

    def test_launch_failure_removes_request_and_log(self) -> None:
        original_launch = self._launch

        def fail(command, **kwargs):
            original_launch(command, **kwargs)
            raise OSError("cannot start interpreter")

        self.popen.side_effect = fail
        with self.assertRaisesRegex(export.ExportError, "cannot start interpreter"):
            self._render()
        self._assert_clean()

    def test_invalid_numeric_options_never_launch_a_worker(self) -> None:
        invalid = [
            {"width": value} for value in (0, -1, True, 1.5, float("nan"), float("inf"), "16")
        ] + [
            {"height": value} for value in (0, False, float("nan"), float("inf"))
        ] + [
            {"scale": value} for value in (0, -1, True, 5, float("nan"), float("inf"), "2", 0.001)
        ] + [
            {"timeout": value} for value in (0, -1, True, float("nan"), float("inf"), "30")
        ]
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(export.ExportError):
                self._render(**options)
        self.popen.assert_not_called()

    def test_invalid_figures_and_resource_limits_never_launch(self) -> None:
        for figures in ([], None, "chart", [("bad", {})], [(None, _figure())], [("short",)]):
            with self.subTest(figures=figures), self.assertRaises(export.ExportError):
                export.figures_to_images(figures)
        with self.assertRaises(export.ExportError):
            export.figures_to_images([self.figures[0]] * (export.MAX_FIGURES + 1))
        for options in ({"width": 8193}, {"width": 8192, "height": 8192}, {"width": 4096, "scale": 4}):
            with self.subTest(options=options), self.assertRaises(export.ExportError):
                self._render(**options)
        with mock.patch.object(export, "MAX_TOTAL_PIXELS", 200), self.assertRaises(export.ExportError):
            self._render()
        with mock.patch.object(export, "MAX_REQUEST_BYTES", 1), self.assertRaises(export.ExportError):
            self._render()
        self.popen.assert_not_called()

    def test_rendered_byte_limits_fail_instead_of_returning_partial_data(self) -> None:
        for limit, value in (("MAX_PNG_BYTES", 1), ("MAX_TOTAL_BYTES", len(_png()))):
            with self.subTest(limit=limit), mock.patch.object(export, limit, value):
                with self.assertRaises(export.ExportError):
                    self._render()
                self._assert_clean()


class WorkerTests(unittest.TestCase):
    def test_single_batch_call_without_application_package_import(self) -> None:
        request = {"figures": [{"data": []}, {"data": []}], "width": 16, "height": 8, "scale": 2}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "request.json"
            path.write_text(json.dumps(request), encoding="utf-8")
            blocked = {"brd_insight_agent": None, "brd_insight_agent.gemini": None, "dotenv": None}
            with mock.patch.dict(sys.modules, blocked), mock.patch.object(pio, "write_images") as write:
                isolated = _load_file("_worker_isolation_test", "export_worker.py")
                self.assertEqual(isolated.main([str(path)]), 0)
            write.assert_called_once_with(
                request["figures"], [path.parent / "01.png", path.parent / "02.png"],
                format="png", width=16, height=8, scale=2,
            )

    def test_worker_reports_render_failure_as_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "request.json"
            path.write_text(json.dumps({"figures": [{}], "width": 16, "height": 8, "scale": 1}), encoding="utf-8")
            with mock.patch.object(pio, "write_images", side_effect=RuntimeError("Chrome not found")):
                with mock.patch.object(sys, "stderr", new_callable=io.StringIO) as stderr:
                    self.assertEqual(worker.main([str(path)]), 1)
                    self.assertIn("Chrome not found", stderr.getvalue())

    def test_worker_handles_usage_invalid_requests_and_old_plotly(self) -> None:
        with mock.patch.object(sys, "stderr", new_callable=io.StringIO):
            self.assertEqual(worker.main([]), 2)
            self.assertEqual(worker.main(["relative.json"]), 1)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory).resolve() / "request.json"
                for text in ("invalid JSON", '{}', '{"figures":[]}'):
                    path.write_text(text, encoding="utf-8")
                    with mock.patch.object(pio, "write_images") as write:
                        self.assertEqual(worker.main([str(path)]), 1)
                        write.assert_not_called()
                path.write_text('{"figures":[{}]}', encoding="utf-8")
                with mock.patch.object(pio, "write_images", None):
                    self.assertEqual(worker.main([str(path)]), 1)


if __name__ == "__main__":
    unittest.main()