"""Backend selection, source-system preservation and checkpoint compatibility."""
import io
import json
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import openmm as mm
from openmm import unit

from mojoemd import acceleration as accel
from mojoemd.api import PipelineExecutionError, run_pipeline
from mojoemd.cli import build_parser, main
from mojoemd.config import parse_arguments


def pme_system():
    system = mm.System()
    system.addParticle(12)
    system.addParticle(12)
    system.setDefaultPeriodicBoxVectors(mm.Vec3(3, 0, 0), mm.Vec3(0, 3, 0), mm.Vec3(0, 0, 3))
    force = mm.NonbondedForce()
    force.setNonbondedMethod(force.PME)
    force.setCutoffDistance(1 * unit.nanometer)
    force.addParticle(0.1, 0.3, 0.1)
    force.addParticle(-0.1, 0.3, 0.1)
    force.setForceGroup(2)
    system.addForce(force)
    return system


class AccelerationTests(unittest.TestCase):
    def test_config_defaults_and_explicit_cli_override(self):
        self.assertEqual(parse_arguments([]).acceleration, 'openmm')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps({'acceleration': 'cupotential'}))
            self.assertEqual(parse_arguments(['-c', str(path)]).acceleration, 'cupotential')
            self.assertEqual(parse_arguments(['-c', str(path), '--acceleration', 'openmm']).acceleration, 'openmm')
            path.write_text(json.dumps({'acceleration': 'typo'}))
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_arguments(['-c', str(path)])

    def test_cli_routes_acceleration_and_doctor(self):
        result = SimpleNamespace(stages=('produce',), output_dir=Path('/tmp/output'))
        with patch('mojoemd.cli.run_pipeline', return_value=result) as run, redirect_stdout(io.StringIO()):
            main(['produce', '-o', '/tmp/output', '--acceleration', 'cupotential'])
        self.assertIn('--acceleration', run.call_args.kwargs['cli_overrides'])
        self.assertIn('cupotential', run.call_args.kwargs['cli_overrides'])
        with patch('mojoemd.cli.run_doctor', return_value=False) as doctor, redirect_stdout(io.StringIO()):
            self.assertEqual(main(['doctor', '--acceleration', 'cupotential']), 1)
        doctor.assert_called_once_with(acceleration='cupotential')
        self.assertEqual(build_parser().parse_args(['doctor']).acceleration, 'openmm')

    def test_legacy_rejects_acceleration_from_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps({'acceleration': 'cupotential', 'produce': True, 'output_dir': directory}))
            with self.assertRaisesRegex(PipelineExecutionError, 'modular backend'):
                run_pipeline(path, backend='legacy')

    def test_preparation_does_not_load_optional_plugin(self):
        from mojoemd import pipeline
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(pipeline, 'run_prepare'), \
             patch.object(accel, 'load_cupotential', side_effect=AssertionError('plugin loaded')):
            run_pipeline(stages=('prepare',), cli_overrides=(
                '--acceleration', 'cupotential', '--receptor',
                str(Path(__file__).parent / 'fixtures/preparation/protein.pdb'),
                '--output_dir', directory))

    def test_unavailable_plugin_fails_before_any_pipeline_stage(self):
        from mojoemd import pipeline
        with patch.object(accel, 'load_cupotential', side_effect=accel.AccelerationError('no CUDA')), \
             patch.object(pipeline, 'execute_pipeline') as execute:
            with self.assertRaisesRegex(PipelineExecutionError, 'no CUDA'):
                run_pipeline(stages=('produce',), cli_overrides=('--acceleration', 'cupotential'))
        execute.assert_not_called()

    def test_unsupported_host_is_explicit(self):
        with patch.object(accel.platform, 'system', return_value='Darwin'):
            with self.assertRaisesRegex(accel.AccelerationError, 'Linux x86-64'):
                accel.load_cupotential()

    def test_native_library_load_is_forced_after_soft_import(self):
        class BrokenPlugin:
            def __getattr__(self, name):
                raise RuntimeError('native library missing')
        with patch.object(accel.platform, 'system', return_value='Linux'), \
             patch.object(accel.platform, 'machine', return_value='x86_64'), \
             patch.object(accel, '_version', side_effect=lambda name: '8.4.0' if name == 'openmm' else '0.3.5'), \
             patch.object(mm.Platform, 'getPlatformByName'), \
             patch.object(accel.importlib, 'import_module', return_value=BrokenPlugin()):
            with self.assertRaisesRegex(accel.AccelerationError, 'native library missing'):
                accel.load_cupotential()

    def test_incompatible_openmm_abi_is_rejected_before_plugin_import(self):
        with patch.object(accel.platform, 'system', return_value='Linux'), \
             patch.object(accel.platform, 'machine', return_value='x86_64'), \
             patch.object(accel, '_version', return_value='8.6.1'), \
             patch.object(accel.importlib, 'import_module') as load:
            with self.assertRaisesRegex(accel.AccelerationError, 'OpenMM 8.4.x'):
                accel.load_cupotential()
        load.assert_not_called()

    def test_conversion_preserves_source_other_forces_groups_and_settings(self):
        source = pme_system()
        source.addForce(mm.CMMotionRemover(7))
        source.addForce(mm.MonteCarloBarostat(1.5 * unit.bar, 300 * unit.kelvin, 17))
        source.addForce(mm.CustomExternalForce('k*x^2'))
        source.getForce(3).addGlobalParameter('k', 0)
        xml = mm.XmlSerializer.serialize(source)
        captured = {}
        def nonbonded(force):
            captured['nonbonded'] = mm.XmlSerializer.serialize(force)
            return mm.CustomExternalForce('0')
        def barostat(*args):
            captured['barostat'] = args
            return mm.MonteCarloBarostat(*args)
        def integrator(*args, **kwargs):
            captured['integrator'] = kwargs
            return mm.LangevinMiddleIntegrator(*args)
        plugin = SimpleNamespace(NonbondedForce=nonbonded, MonteCarloBarostat=barostat,
                                 LangevinMiddleIntegrator=integrator)
        with patch.object(accel, 'load_cupotential', return_value=plugin), \
             patch('mojoemd.runtime.select_platform', return_value='CUDA'):
            converted, integ, platform = accel.cupotential_dynamics(source, 2)
        self.assertEqual(mm.XmlSerializer.serialize(source), xml)
        self.assertEqual(platform, 'CUDA')
        self.assertEqual(integ.getStepSize(), 2 * unit.femtoseconds)
        self.assertEqual(captured['integrator'], {'comFrequency': 7})
        self.assertEqual(captured['barostat'], (1.5 * unit.bar, 300 * unit.kelvin, 17))
        self.assertFalse(any(isinstance(f, mm.CMMotionRemover) for f in converted.getForces()))
        self.assertEqual(converted.getNumForces(), source.getNumForces() - 1)
        self.assertIn(2, [f.getForceGroup() for f in converted.getForces()])
        self.assertEqual(captured['nonbonded'], mm.XmlSerializer.serialize(source.getForce(0)))

    def test_incompatible_systems_reject_before_conversion(self):
        def triclinic(s):
            s.setDefaultPeriodicBoxVectors(mm.Vec3(3, 0, 0), mm.Vec3(.2, 3, 0), mm.Vec3(0, 0, 3))
        def offset(s):
            f = s.getForce(0)
            f.addGlobalParameter('lambda', 0)
            f.addParticleParameterOffset('lambda', 0, 1, 0, 0)
        mutations = [
            ('rectangular', triclinic),
            ('PME', lambda s: s.getForce(0).setNonbondedMethod(mm.NonbondedForce.LJPME)),
            ('CustomNonbondedForce', lambda s: s.addForce(mm.CustomNonbondedForce('0'))),
            ('offsets', offset),
            ('tolerance', lambda s: s.getForce(0).setEwaldErrorTolerance(1e-5)),
            ('automatic PME', lambda s: s.getForce(0).setPMEParameters(3, 32, 32, 32)),
            ('force group', lambda s: s.getForce(0).setReciprocalSpaceForceGroup(3)),
            ('twice', lambda s: s.getForce(0).setCutoffDistance(2 * unit.nanometer)),
        ]
        for message, mutate in mutations:
            with self.subTest(message=message):
                system = pme_system()
                mutate(system)
                with self.assertRaisesRegex(accel.AccelerationError, message):
                    accel.validate_system(system)

    def test_backend_record_prevents_cross_backend_and_version_restarts(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output_dir=directory, acceleration='cupotential', restart=False)
            record = accel.production_backend(args)
            accel.write_production_backend(args, record)
            args.restart = True
            self.assertEqual(accel.production_backend(args), record)
            args.acceleration = 'openmm'
            with self.assertRaisesRegex(accel.AccelerationError, 'differs'):
                accel.production_backend(args)
            args.acceleration = 'cupotential'
            with patch.object(accel, '_version', return_value='different'):
                with self.assertRaisesRegex(accel.AccelerationError, 'versions'):
                    accel.production_backend(args)

    def test_old_stock_checkpoints_remain_usable_but_not_with_cupotential(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'production.chk').touch()
            args = SimpleNamespace(output_dir=directory, acceleration='openmm', restart=True)
            self.assertEqual(accel.production_backend(args)['acceleration'], 'openmm')
            args.acceleration = 'cupotential'
            with self.assertRaisesRegex(accel.AccelerationError, 'Stock OpenMM checkpoints'):
                accel.production_backend(args)

    def test_corrupt_backend_record_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'production_backend.json')
            path.write_text('[]')
            args = SimpleNamespace(output_dir=directory, acceleration='openmm', restart=False)
            with self.assertRaisesRegex(accel.AccelerationError, 'schema'):
                accel.production_backend(args)
            self.assertEqual(path.read_text(), '[]')
