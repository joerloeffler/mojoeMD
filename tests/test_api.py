from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mojoemd import PipelineExecutionError
from mojoemd.api import _load_engine, equilibrate_system, prepare_system, run_pipeline, run_production
from mojoemd.cli import build_parser, main


class FakeEngine:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.executed = None

    def parse_arguments(self, argv):
        config = Path(argv[1])
        data = json.loads(config.read_text())
        return SimpleNamespace(
            output_dir=data["output_dir"],
            prepare=data.get("prepare", False),
            equilibrate=data.get("equilibrate", False),
            produce=data.get("produce", False),
        )

    def execute_pipeline(self, args):
        if self.fail:
            raise SystemExit(1)
        self.executed = args


class PipelineApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = self.root / "run.json"
        self.output = self.root / "output"
        self.config.write_text(
            json.dumps({"output_dir": str(self.output), "prepare": True})
        )

    def test_prepare_convenience_function_forces_only_prepare(self):
        engine = FakeEngine()
        with patch("mojoemd.api._select_engine", return_value=engine):
            result = prepare_system(self.config)

        self.assertEqual(result.stages, ("prepare",))
        self.assertTrue(engine.executed.prepare)
        self.assertFalse(engine.executed.equilibrate)
        self.assertFalse(engine.executed.produce)

    def test_configured_stages_are_preserved(self):
        engine = FakeEngine()
        with patch("mojoemd.api._select_engine", return_value=engine):
            result = run_pipeline(self.config)

        self.assertEqual(result.stages, ("prepare",))

    def test_legacy_system_exit_becomes_library_exception(self):
        with patch("mojoemd.api._select_engine", return_value=FakeEngine(fail=True)):
            with self.assertRaises(PipelineExecutionError):
                prepare_system(self.config)

    def test_production_restart_has_a_first_class_cli_flag(self):
        args = build_parser().parse_args(
            ["produce", "--config", str(self.config), "--restart"]
        )
        self.assertTrue(args.restart)

    def test_cli_defaults_to_modular_and_accepts_both_explicit_backends(self):
        parser = build_parser()
        self.assertEqual(parser.parse_args(["prepare", "--pdb", "test.pdb"]).backend, "modular")
        for backend in ("modular", "legacy"):
            self.assertEqual(parser.parse_args([
                "prepare", "--pdb", "test.pdb", "--backend", backend,
            ]).backend, backend)

    def test_api_defaults_to_modular(self):
        for function in (run_pipeline, prepare_system, equilibrate_system, run_production):
            with self.subTest(function=function.__name__):
                with patch("mojoemd.api._select_engine", return_value=FakeEngine()) as select:
                    function(self.config)
                select.assert_called_once_with("modular")

    def test_explicit_legacy_backend_remains_available(self):
        engine = FakeEngine()
        with patch("mojoemd.api._load_engine", return_value=engine) as load:
            result = run_pipeline(self.config, backend="legacy")
        load.assert_called_once_with()
        self.assertEqual(result.stages, ("prepare",))

    def test_direct_prepare_derives_output_directory(self):
        pdb = self.root / "TEST.pdb"
        pdb.touch()
        result = SimpleNamespace(stages=("prepare",), output_dir=self.root / "TEST_mojoemd")
        with patch("mojoemd.cli.run_pipeline", return_value=result) as run:
            self.assertEqual(main(["prepare", "--pdb", str(pdb)]), 0)

        self.assertIsNone(run.call_args.args[0])
        self.assertEqual(run.call_args.kwargs["stages"], ("prepare",))
        self.assertIn(
            str((self.root / "TEST_mojoemd").resolve()),
            run.call_args.kwargs["cli_overrides"],
        )

    def test_real_engine_loader_does_not_resolve_to_package(self):
        engine = _load_engine()
        self.assertTrue(callable(engine.parse_arguments))
        self.assertTrue(callable(engine.execute_pipeline))
        self.assertTrue(str(engine.__file__).endswith("mojoeMD.py"))


if __name__ == "__main__":
    unittest.main()
