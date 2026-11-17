"""Diagnostics must inspect the same engine selected by the default CLI."""
import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from mojoemd.doctor import run_doctor


class DoctorTests(unittest.TestCase):
    def test_checks_default_backend_and_reports_import_failures(self):
        engine = SimpleNamespace(parse_arguments=lambda: None,
                                 execute_pipeline=lambda: None, __file__="modular/pipeline.py")
        def dependency(name):
            if name == "espaloma":
                raise ImportError("broken DGL")
            return SimpleNamespace()
        output = io.StringIO()
        with patch("mojoemd.api._select_engine", return_value=engine) as select, \
             patch("mojoemd.api._load_engine", side_effect=AssertionError("legacy loaded")), \
             patch("mojoemd.doctor.importlib.import_module", side_effect=dependency), \
             redirect_stdout(output):
            self.assertFalse(run_doctor())
        select.assert_called_once_with("modular")
        self.assertIn("Pipeline engine (modular): modular/pipeline.py", output.getvalue())
        self.assertIn("Espaloma: BROKEN (ImportError: broken DGL)", output.getvalue())

    def test_sparse_library_failure_is_not_hidden_by_successful_imports(self):
        engine = SimpleNamespace(parse_arguments=lambda: None,
                                 execute_pipeline=lambda: None, __file__="modular/pipeline.py")
        def broken_sparse():
            raise FileNotFoundError("missing sparse library")
        graph = SimpleNamespace(adj=broken_sparse)
        module = SimpleNamespace(graph=lambda *args, **kwargs: graph)
        output = io.StringIO()
        with patch("mojoemd.api._select_engine", return_value=engine), \
             patch("mojoemd.doctor.importlib.import_module", return_value=module), \
             redirect_stdout(output):
            self.assertFalse(run_doctor())
        self.assertIn("[!!] DGL/PyTorch sparse backend: missing sparse library", output.getvalue())
