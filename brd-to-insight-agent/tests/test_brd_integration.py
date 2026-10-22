"""Tests for the BRD agent adapter and the warehouse reader."""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brd_insight_agent import brd, warehouse  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class ParseTables(unittest.TestCase):
    def test_reads_json_object(self):
        text = json.dumps({"project": "p1", "dataset": "d1", "tables": ["orders", "customers"]})
        tables, project, dataset = brd.parse_tables(text)
        self.assertEqual(tables, ["orders", "customers"])
        self.assertEqual((project, dataset), ("p1", "d1"))

    def test_reads_table_dicts(self):
        text = json.dumps({"dataset_id": "d2", "created_tables": [{"table_id": "sales"}, {"name": "returns"}]})
        tables, _, dataset = brd.parse_tables(text)
        self.assertEqual(tables, ["sales", "returns"])
        self.assertEqual(dataset, "d2")

    def test_falls_back_to_qualified_names(self):
        text = "Created my-proj.my_set.01_fact_sales\nCreated my-proj.my_set.00_raw_date\n"
        tables, project, dataset = brd.parse_tables(text)
        self.assertEqual(tables, ["01_fact_sales", "00_raw_date"])
        self.assertEqual((project, dataset), ("my-proj", "my_set"))

    def test_deduplicates_and_strips_prefixes(self):
        tables, _, _ = brd.parse_tables(
            "proj-id.ds.01_t1\nproj-id.ds.01_t1\nproj-id.ds.01_t2")
        self.assertEqual(tables, ["01_t1", "01_t2"])

    def test_no_tables_found(self):
        tables, _, _ = brd.parse_tables("nothing useful here")
        self.assertEqual(tables, [])


class SaveUpload(unittest.TestCase):
    def test_rejects_empty(self):
        with self.assertRaises(brd.BrdError):
            brd.save_upload("a.txt", b"")

    def test_rejects_unknown_type(self):
        with self.assertRaises(brd.BrdError):
            brd.save_upload("a.exe", b"x")

    def test_rejects_spreadsheet_as_brd(self):
        with self.assertRaises(brd.BrdError):
            brd.save_upload("sources.xlsx", b"x")

    def test_rejects_oversized(self):
        with self.assertRaises(brd.BrdError):
            brd.save_upload("a.txt", b"x" * (brd.MAX_BRD_BYTES + 1))

    def test_writes_file(self):
        path = brd.save_upload("spec.md", b"hello")
        self.addCleanup(lambda: path.exists() and path.unlink())
        self.assertEqual(path.read_bytes(), b"hello")


class ConfiguredMode(unittest.TestCase):
    def test_demo_without_configuration(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(brd.configured_mode(), "demo")

    def test_cli_inferred_from_command(self):
        with mock.patch.dict(os.environ, {"BRD_AGENT_CMD": "python run.py"}, clear=True):
            self.assertEqual(brd.configured_mode(), "cli")

    def test_module_inferred_from_callable(self):
        with mock.patch.dict(os.environ, {"BRD_AGENT_CALLABLE": "pkg.mod:fn"}, clear=True):
            self.assertEqual(brd.configured_mode(), "module")

    def test_explicit_mode_wins(self):
        with mock.patch.dict(os.environ, {"BRD_AGENT_MODE": "demo", "BRD_AGENT_CMD": "x"}, clear=True):
            self.assertEqual(brd.configured_mode(), "demo")

    def test_configuration_reports_no_secret_values(self):
        with mock.patch.dict(os.environ, {"BRD_AGENT_CMD": "python run.py", "LITELLM_API_KEY": "secret"}, clear=True):
            self.assertNotIn("secret", json.dumps(brd.configuration()))


class RunAgent(unittest.TestCase):
    def test_missing_file(self):
        with self.assertRaises(brd.BrdError):
            brd.run_agent(Path("/nonexistent/brd.txt"))

    def test_cli_mode_parses_stdout(self):
        path = brd.save_upload("spec.txt", b"brd")
        self.addCleanup(lambda: path.exists() and path.unlink())
        script = 'import json; print(json.dumps({"project":"p","dataset":"d","tables":["t1"]}))'
        with mock.patch.dict(os.environ, {"BRD_AGENT_MODE": "cli",
                                          "BRD_AGENT_CMD": f"{sys.executable} -c {json.dumps(script)}"}, clear=True):
            result = brd.run_agent(path)
        self.assertTrue(result.ok)
        self.assertEqual(result.tables, ["t1"])
        self.assertEqual((result.project, result.dataset), ("p", "d"))
        self.assertFalse(result.simulated)

    def test_cli_failure_is_reported(self):
        path = brd.save_upload("spec.txt", b"brd")
        self.addCleanup(lambda: path.exists() and path.unlink())
        with mock.patch.dict(os.environ, {"BRD_AGENT_MODE": "cli",
                                          "BRD_AGENT_CMD": f"{sys.executable} -c \"raise SystemExit(3)\""}, clear=True):
            with self.assertRaises(brd.BrdError):
                brd.run_agent(path)

    def test_cli_missing_command(self):
        path = brd.save_upload("spec.txt", b"brd")
        self.addCleanup(lambda: path.exists() and path.unlink())
        with mock.patch.dict(os.environ, {"BRD_AGENT_MODE": "cli", "BRD_AGENT_CMD": ""}, clear=True):
            with self.assertRaises(brd.BrdError):
                brd.run_agent(path)

    def test_module_mode_calls_callable(self):
        path = brd.save_upload("spec.txt", b"brd")
        self.addCleanup(lambda: path.exists() and path.unlink())
        module = mock.Mock()
        module.build = mock.Mock(return_value={"dataset": "ds", "tables": ["alpha"]})
        with mock.patch.dict(os.environ, {"BRD_AGENT_MODE": "module",
                                          "BRD_AGENT_CALLABLE": "fake_mod:build"}, clear=True):
            with mock.patch.object(brd.importlib, "import_module", return_value=module):
                result = brd.run_agent(path)
        module.build.assert_called_once_with(str(path), str(path.parent), dry_run=False)
        self.assertEqual(result.tables, ["alpha"])

    def test_module_mode_rejects_bad_target(self):
        path = brd.save_upload("spec.txt", b"brd")
        self.addCleanup(lambda: path.exists() and path.unlink())
        with mock.patch.dict(os.environ, {"BRD_AGENT_MODE": "module",
                                          "BRD_AGENT_CALLABLE": "nocolon"}, clear=True):
            with self.assertRaises(brd.BrdError):
                brd.run_agent(path)

    def test_demo_mode_is_flagged_simulated(self):
        path = brd.save_upload("spec.txt", b"brd")
        self.addCleanup(lambda: path.exists() and path.unlink())
        with mock.patch.dict(os.environ, {"BRD_AGENT_MODE": "demo"}, clear=True):
            result = brd.run_agent(path)
        self.assertTrue(result.simulated)
        self.assertIn("no BigQuery table was created", result.detail)


class RealFrameworkContract(unittest.TestCase):
    """The output shapes the BRD-to-BigQuery framework actually prints."""

    MANAGED_RUNNER_STDOUT = json.dumps({
        "status": "SUCCESS",
        "run_id": "2026-10-20-abc123",
        "report_directory": "/tmp/generated/reports/generic_2026-10-20-abc123",
        "row_counts": {
            "cotality-sbx.brd_demo.00_raw_rainfall": 1200,
            "cotality-sbx.brd_demo.01_fact_daily_station_rainfall": 950,
            "cotality-sbx.brd_demo.01_dim_station_watershed": 48,
        },
        "issues": None,
    }, indent=2)

    LAUNCHER_STDOUT = json.dumps({
        "status": "BLOCKED",
        "run_id": None,
        "report_path": "/tmp/generated/reports/executer_20261020T101500.md",
        "issues": [{"check": "input configuration", "result": "BRD_INPUT_DIR is missing.",
                    "action": "Set BRD_INPUT_DIR."}],
        "error": "BRD_INPUT_DIR is missing.",
    }, indent=2)

    def test_row_counts_are_read_as_tables(self):
        tables, project, dataset = brd.parse_tables(self.MANAGED_RUNNER_STDOUT)
        self.assertEqual(tables, ["00_raw_rainfall", "01_fact_daily_station_rainfall",
                                  "01_dim_station_watershed"])
        self.assertEqual((project, dataset), ("cotality-sbx", "brd_demo"))

    def test_success_result_separates_curated_tables(self):
        result = brd._build_result(self.MANAGED_RUNNER_STDOUT, "cli", 1.0)
        self.assertTrue(result.ok)
        self.assertEqual(result.status, "SUCCESS")
        self.assertEqual(result.curated, ["01_fact_daily_station_rainfall", "01_dim_station_watershed"])
        self.assertEqual(result.run_id, "2026-10-20-abc123")
        self.assertIn("generic_2026-10-20-abc123", result.report_dir)

    def test_blocked_status_is_not_ok_and_keeps_issues(self):
        result = brd._build_result(self.LAUNCHER_STDOUT, "cli", 1.0)
        self.assertFalse(result.ok)
        self.assertEqual(result.status, "BLOCKED")
        self.assertEqual(result.issues[0]["check"], "input configuration")
        self.assertIn("correction report", result.detail)

    def test_dry_run_without_row_counts_is_reported(self):
        text = json.dumps({"status": "DRY_RUN_READY", "run_id": "r1", "row_counts": {}})
        result = brd._build_result(text, "cli", 1.0)
        self.assertFalse(result.ok)
        self.assertIn("dry run does not", result.detail)

    def test_tables_read_from_run_report_json(self):
        directory = Path(brd.tempfile.mkdtemp(prefix="brd_report_"))
        (directory / "run_report.json").write_text(json.dumps({
            "row_counts": {"proj-id.ds.01_fact_sales": 10},
            "llm": {"contract": {"tables": [{"table_id": "proj-id.ds.00_raw_sales"}]}},
        }))
        text = json.dumps({"status": "SUCCESS", "report_directory": str(directory)})
        result = brd._build_result(text, "cli", 1.0)
        self.assertEqual(result.tables, ["01_fact_sales", "00_raw_sales"])
        self.assertEqual(result.curated, ["01_fact_sales"])

    def test_qualified_names_in_plain_text(self):
        tables, project, dataset = brd.parse_tables(
            "Created cotality-sbx.brd_demo.01_fact_sales and cotality-sbx.brd_demo.00_raw_sales")
        self.assertEqual(tables, ["01_fact_sales", "00_raw_sales"])
        self.assertEqual((project, dataset), ("cotality-sbx", "brd_demo"))


class InputFolder(unittest.TestCase):
    def test_folder_holds_brd_and_sources(self):
        folder, brd_path = brd.save_inputs([
            ("requirements.docx", b"brd"), ("daily.csv", b"a,b\n1,2\n"), ("map.csv", b"x\n1\n"),
        ])
        self.addCleanup(lambda: [p.unlink() for p in folder.iterdir()] and None)
        self.assertEqual(brd_path.name, "requirements.docx")
        self.assertEqual(sorted(p.name for p in folder.iterdir()),
                         ["daily.csv", "map.csv", "requirements.docx"])

    def test_requires_exactly_one_brd(self):
        with self.assertRaises(brd.BrdError) as caught:
            brd.save_inputs([("a.docx", b"x"), ("b.md", b"y")])
        self.assertIn("exactly one BRD", str(caught.exception))

    def test_requires_at_least_one_brd(self):
        with self.assertRaises(brd.BrdError):
            brd.save_inputs([("only_source.csv", b"a\n1\n")])

    def test_ignores_office_lock_files(self):
        folder, brd_path = brd.save_inputs([("~$req.docx", b"lock"), ("req.docx", b"brd")])
        self.addCleanup(lambda: [p.unlink() for p in folder.iterdir()] and None)
        self.assertEqual([p.name for p in folder.iterdir()], ["req.docx"])
        self.assertEqual(brd_path.name, "req.docx")

    def test_too_many_files(self):
        with self.assertRaises(brd.BrdError):
            brd.save_inputs([("a.docx", b"x")] + [(f"s{i}.csv", b"1") for i in range(brd.MAX_INPUT_FILES)])


class CliInvocation(unittest.TestCase):
    def _echo_argv(self):
        return (f"{sys.executable} -c "
                + json.dumps("import sys,json,os;"
                             "print(json.dumps({'argv':sys.argv[1:],"
                             "'input_dir':os.environ.get('BRD_INPUT_DIR'),"
                             "'status':'SUCCESS','row_counts':{'p-roj.ds.01_t':1}}))"))

    def test_appends_brd_and_source_root_when_no_placeholders(self):
        folder, brd_path = brd.save_inputs([("req.docx", b"x"), ("s.csv", b"a\n1\n")])
        with mock.patch.dict(os.environ, {"BRD_AGENT_MODE": "cli",
                                          "BRD_AGENT_CMD": self._echo_argv()}, clear=True):
            result = brd.run_agent(brd_path, source_dir=folder)
        payload = json.loads(result.detail[result.detail.index("{"):])
        self.assertIn("--brd", payload["argv"])
        self.assertIn("--source-root", payload["argv"])
        self.assertEqual(payload["input_dir"], str(folder))
        self.assertEqual(result.tables, ["01_t"])

    def test_dry_run_flag_is_added(self):
        folder, brd_path = brd.save_inputs([("req.docx", b"x")])
        with mock.patch.dict(os.environ, {"BRD_AGENT_MODE": "cli",
                                          "BRD_AGENT_CMD": self._echo_argv()}, clear=True):
            result = brd.run_agent(brd_path, dry_run=True, source_dir=folder)
        payload = json.loads(result.detail[result.detail.index("{"):])
        self.assertIn("--dry-run", payload["argv"])

    def test_placeholders_are_substituted(self):
        folder, brd_path = brd.save_inputs([("req.docx", b"x")])
        template = self._echo_argv() + " --brd {brd} --source-root {source_dir}"
        with mock.patch.dict(os.environ, {"BRD_AGENT_MODE": "cli",
                                          "BRD_AGENT_CMD": template}, clear=True):
            result = brd.run_agent(brd_path, source_dir=folder)
        payload = json.loads(result.detail[result.detail.index("{"):])
        self.assertEqual(payload["argv"].count("--brd"), 1)
        self.assertIn(str(brd_path), payload["argv"])

    def test_exit_code_two_keeps_the_correction_report(self):
        folder, brd_path = brd.save_inputs([("req.docx", b"x")])
        script = ("import json,sys;"
                  "print(json.dumps({'status':'BLOCKED','issues':[{'check':'c','result':'r','action':'a'}]}));"
                  "sys.exit(2)")
        with mock.patch.dict(os.environ, {"BRD_AGENT_MODE": "cli",
                                          "BRD_AGENT_CMD": f"{sys.executable} -c {json.dumps(script)}"},
                             clear=True):
            result = brd.run_agent(brd_path, source_dir=folder)
        self.assertFalse(result.ok)
        self.assertEqual(result.status, "BLOCKED")

    def test_other_exit_codes_raise(self):
        folder, brd_path = brd.save_inputs([("req.docx", b"x")])
        with mock.patch.dict(os.environ, {"BRD_AGENT_MODE": "cli",
                                          "BRD_AGENT_CMD": f"{sys.executable} -c \"raise SystemExit(3)\""},
                             clear=True):
            with self.assertRaises(brd.BrdError):
                brd.run_agent(brd_path, source_dir=folder)


class WarehouseBackend(unittest.TestCase):
    def test_simulated_without_bigquery_project(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(warehouse.backend(), "simulated")
            self.assertIn("Simulated warehouse", warehouse.backend_detail())

    def test_simulated_tables_are_local_files(self):
        with mock.patch.object(warehouse, "SIMULATED_DIR", ROOT / "testdata"):
            names = warehouse.simulated_tables()
        self.assertIn("retail_sales", names)
        self.assertTrue(all(not n.endswith(".xlsx") for n in names))

    def test_load_simulated_returns_real_rows(self):
        with mock.patch.object(warehouse, "SIMULATED_DIR", ROOT / "testdata"):
            frame, info = warehouse.load_table("retail_sales")
        self.assertEqual(len(frame), 5000)
        self.assertEqual(info["total_rows"], 5000)
        self.assertFalse(info["truncated"])
        self.assertIn("NetSales", frame.columns)

    def test_row_limit_truncates_and_discloses(self):
        with mock.patch.object(warehouse, "SIMULATED_DIR", ROOT / "testdata"):
            frame, info = warehouse.load_table("retail_sales", limit=1000)
        self.assertEqual(len(frame), 1000)
        self.assertTrue(info["truncated"])
        message = warehouse.disclosure(info)
        self.assertIn("1,000 of 5,000", message)
        self.assertIn("only the rows read", message)

    def test_disclosure_for_full_read(self):
        message = warehouse.disclosure({"simulated": False, "table": "t", "read_rows": 10,
                                        "total_rows": 10, "truncated": False, "limit": 100})
        self.assertIn("Read all 10 rows", message)

    def test_unknown_table(self):
        with mock.patch.object(warehouse, "SIMULATED_DIR", ROOT / "testdata"):
            with self.assertRaises(warehouse.WarehouseError):
                warehouse.load_table("no_such_table")

    def test_rejects_unsafe_identifiers(self):
        with self.assertRaises(warehouse.WarehouseError):
            warehouse._check_name("bad;DROP", "table")
        with self.assertRaises(warehouse.WarehouseError):
            warehouse._check_name("", "dataset")

    def test_limit_is_capped(self):
        with mock.patch.object(warehouse, "SIMULATED_DIR", ROOT / "testdata"):
            _, info = warehouse.load_table("retail_sales", limit=99_999_999)
        self.assertEqual(info["limit"], warehouse.MAX_ROW_LIMIT)



class EnvFile(unittest.TestCase):
    def _write(self, text):
        folder = Path(brd.tempfile.mkdtemp(prefix="brd_env_"))
        (folder / ".env").write_text(text, encoding="utf-8")
        return folder

    def test_reads_quoted_values_without_executing(self):
        folder = self._write('LLM_PROVIDER="litellm"\nexport LLM_MODEL=flash-lite\n'
                             'BAD=$(rm -rf /)\n# comment\n')
        values = brd._env_file_values(str(folder))
        self.assertEqual(values["LLM_PROVIDER"], "litellm")
        self.assertEqual(values["LLM_MODEL"], "flash-lite")
        self.assertEqual(values["BAD"], "$(rm -rf /)")

    def test_placeholders_are_ignored(self):
        folder = self._write('LLM_API_KEY="<PASTE-YOUR-KEY-HERE>"\nLLM_PROVIDER="litellm"\n')
        values = brd._env_file_values(str(folder))
        self.assertNotIn("LLM_API_KEY", values)
        self.assertIn("LLM_PROVIDER", values)

    def test_status_reports_missing_without_values(self):
        folder = self._write('LLM_PROVIDER="litellm"\nLLM_API_KEY="real-secret-value"\n')
        with mock.patch.dict(os.environ, {}, clear=True):
            status = brd.env_file_status(str(folder))
        self.assertIn("LLM_API_KEY", status["configured"])
        self.assertIn("GOOGLE_CLOUD_PROJECT", status["missing"])
        self.assertNotIn("real-secret-value", json.dumps(status))

    def test_missing_file_is_not_an_error(self):
        self.assertEqual(brd._env_file_values("/nonexistent/path"), {})
        self.assertEqual(brd._env_file_values(None), {})

    def test_env_file_reaches_the_subprocess(self):
        folder = self._write('DEMO_FROM_ENV_FILE="present"\n')
        inputs, brd_path = brd.save_inputs([("req.docx", b"x")])
        script = ("import os,json;"
                  "print(json.dumps({'seen':os.environ.get('DEMO_FROM_ENV_FILE'),"
                  "'status':'SUCCESS','row_counts':{'p-roj.ds.01_t':1}}))")
        with mock.patch.dict(os.environ, {
                "BRD_AGENT_MODE": "cli", "BRD_AGENT_DIR": str(folder),
                "BRD_AGENT_CMD": f"{sys.executable} -c {json.dumps(script)}"}, clear=True):
            result = brd.run_agent(brd_path, source_dir=inputs)
        self.assertIn('"seen": "present"', result.detail)


class PlannedTables(unittest.TestCase):
    """Table identifiers must survive Word splitting a name across formatting runs."""

    def _docx(self, runs):
        import zipfile
        folder = Path(brd.tempfile.mkdtemp(prefix="brd_docx_"))
        path = folder / "spec.docx"
        body = "".join(f"<w:r><w:t>{piece}</w:t></w:r>" for piece in runs)
        xml = ('<?xml version="1.0"?><w:document '
               'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
               f"<w:body><w:p>{body}</w:p></w:body></w:document>")
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("word/document.xml", xml)
        return path

    def test_identifier_split_across_runs_is_rejoined(self):
        path = self._docx(["proj-id.ds.01_fact", "_sales", " and more text"])
        self.assertEqual(brd.planned_tables(path), ["01_fact_sales"])

    def test_plain_text_brd(self):
        folder = Path(brd.tempfile.mkdtemp(prefix="brd_txt_"))
        path = folder / "spec.md"
        path.write_text("Create proj-id.ds.00_raw_x and proj-id.ds.01_fact_x today.")
        self.assertEqual(brd.planned_tables(path), ["00_raw_x", "01_fact_x"])

    def test_missing_file_returns_empty(self):
        self.assertEqual(brd.planned_tables(Path("/nonexistent/a.docx")), [])


if __name__ == "__main__":
    unittest.main()
