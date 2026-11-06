"""Legacy parity and public entry-point regressions for the extracted engine."""
import ast
import copy
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from mojoemd.api import _load_engine, _select_engine, run_pipeline
from mojoemd.cli import main

ROOT = Path(__file__).resolve().parents[1]


class ModularParityTests(unittest.TestCase):
    def test_all_legacy_functions_and_constants_are_preserved(self):
        legacy = ast.parse((ROOT / 'mojoeMD.py').read_text())
        modules = [ROOT / 'mojoemd' / f'{name}.py' for name in
                   ('config', 'runtime', 'pipeline', 'prep', 'equil', 'production')]
        modules += list((ROOT / 'mojoemd/chemistry').glob('*.py'))
        extracted = {}
        for path in modules:
            for node in ast.parse(path.read_text()).body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    extracted[node.name] = node
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            extracted[target.id] = node
        for node in legacy.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                name = node.name
            elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
            else:
                continue
            with self.subTest(name=name):
                self.assertIn(name, extracted)
                if name == 'run_production':
                    # Intentional bug fix: OpenMM accepts strings/file objects,
                    # but the legacy code supplies a pathlib.Path on restart.
                    node = copy.deepcopy(node)
                    for call in ast.walk(node):
                        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                                and call.func.attr == 'loadCheckpoint'):
                            call.args[0] = ast.Call(func=ast.Name(id='str', ctx=ast.Load()),
                                                    args=[call.args[0]], keywords=[])
                self.assertEqual(ast.dump(node), ast.dump(extracted[name]))

    def test_configuration_matches_legacy_for_every_protein_force_field(self):
        legacy = _load_engine()
        modular = _select_engine('modular')
        for forcefield in legacy.PROTEIN_FORCE_FIELDS:
            with self.subTest(forcefield=forcefield):
                argv = ['--protein_ff', forcefield, '--prepare', '--receptor', 'test.pdb']
                self.assertEqual(vars(legacy.parse_arguments(argv)), vars(modular.parse_arguments(argv)))

    def test_modular_backend_does_not_load_legacy_engine(self):
        with patch('mojoemd.api._load_engine', side_effect=AssertionError('Legacy loaded')):
            engine = _select_engine('modular')
            self.assertEqual(engine.run_prepare.__module__, 'mojoemd.prep')

    def test_unknown_backend_is_rejected(self):
        with self.assertRaises(ValueError):
            _select_engine('unknown')

    def test_api_dispatches_only_the_requested_modular_stage(self):
        engine = _select_engine('modular')
        with tempfile.TemporaryDirectory() as directory, \
             patch('mojoemd.api._load_engine', side_effect=AssertionError('Legacy loaded')), \
             patch.object(engine, 'run_prepare') as prepare, \
             patch.object(engine, 'run_equilibrate', side_effect=AssertionError('Unexpected equilibration')), \
             patch.object(engine, 'run_production', side_effect=AssertionError('Unexpected production')):
            result = run_pipeline(stages=('prepare',), cli_overrides=(
                '--receptor', str(ROOT / 'tests/fixtures/preparation/protein.pdb'),
                '--output_dir', directory,
            ))
            prepare.assert_called_once()
            self.assertEqual(result.stages, ('prepare',))

    def test_cli_validation_errors_are_argparse_errors(self):
        with self.assertRaises(SystemExit) as error:
            main(['prepare'])
        self.assertEqual(error.exception.code, 2)

    def test_analysis_functions_are_preserved(self):
        for old, new in [('analyse_MD.py', 'analyse'), ('contact_analysis.py', 'contacts')]:
            legacy = ast.parse((ROOT / 'analysis' / old).read_text())
            modular = ast.parse((ROOT / 'mojoemd' / f'{new}.py').read_text())
            old_funcs = {n.name: n for n in legacy.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
            new_funcs = {n.name: n for n in modular.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
            for name, node in old_funcs.items():
                if name == 'main':
                    continue  # CLI now accepts an explicit argv.
                with self.subTest(module=new, name=name):
                    self.assertEqual(ast.dump(node), ast.dump(new_funcs[name]))


if __name__ == '__main__':
    unittest.main()
