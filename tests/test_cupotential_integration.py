"""Opt-in real CUDA accuracy, serialization and checkpoint checks on the cluster."""
import os
import unittest

import numpy as np
import openmm as mm
from openmm import app, unit

from mojoemd.acceleration import cupotential_dynamics, load_cupotential


@unittest.skipUnless(os.environ.get('MOJOEMD_CUPOTENTIAL_INTEGRATION') == '1',
                     'Set MOJOEMD_CUPOTENTIAL_INTEGRATION=1 on a compatible CUDA node')
class CuPotentialIntegrationTests(unittest.TestCase):
    def test_water_force_accuracy_xml_and_npt_checkpoint_continuation(self):
        load_cupotential()
        forcefield = app.ForceField('amber14/tip3p.xml')
        model = app.Modeller(app.Topology(), [] * unit.nanometer)
        model.addSolvent(forcefield, model='tip3p', boxSize=mm.Vec3(2.4, 2.4, 2.4) * unit.nanometer)
        system = forcefield.createSystem(model.topology, nonbondedMethod=app.PME,
                                        nonbondedCutoff=1 * unit.nanometer,
                                        constraints=app.HBonds, rigidWater=True)
        # A short, frequent-barostat run exercises NPT checkpoint state.
        system.addForce(mm.MonteCarloBarostat(1 * unit.bar, 300 * unit.kelvin, 5))
        converted, integrator, cuda = cupotential_dynamics(system, 2)
        serialized = mm.XmlSerializer.serialize(converted)
        restored = mm.XmlSerializer.deserialize(serialized)
        self.assertEqual(mm.XmlSerializer.serialize(restored), serialized)
        stock_integrator = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond,
                                                     2 * unit.femtoseconds)
        stock = mm.Context(system, stock_integrator, cuda)
        accelerated = mm.Context(restored, integrator, cuda)
        stock.setPositions(model.positions)
        accelerated.setPositions(model.positions)
        reference = stock.getState(getEnergy=True, getForces=True)
        observed = accelerated.getState(getEnergy=True, getForces=True)
        f_ref = reference.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
        f_observed = observed.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
        self.assertTrue(np.isfinite(f_observed).all())
        relative_force_error = np.linalg.norm(f_observed - f_ref) / np.linalg.norm(f_ref)
        # A regression threshold, not a claim of universal scientific accuracy.
        self.assertLess(relative_force_error, 0.01)
        e_ref = reference.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        e_observed = observed.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        self.assertTrue(np.isfinite(e_observed))
        self.assertLess(abs(e_observed - e_ref) / max(abs(e_ref), 1), 0.01)
        del stock, stock_integrator
        accelerated.setVelocitiesToTemperature(300 * unit.kelvin, 412)
        integrator.step(25)
        checkpoint = accelerated.createCheckpoint()
        integrator.step(20)
        expected = accelerated.getState(getPositions=True, getVelocities=True, getEnergy=True)
        # Rebuild both the force conversion and Context, as production restart does.
        fresh_system, fresh_integrator, cuda = cupotential_dynamics(system, 2)
        fresh = mm.Context(fresh_system, fresh_integrator, cuda)
        fresh.loadCheckpoint(checkpoint)
        fresh_integrator.step(20)
        actual = fresh.getState(getPositions=True, getVelocities=True, getEnergy=True)
        self.assertEqual(actual.getStepCount(), expected.getStepCount())
        self.assertEqual(actual.getTime(), expected.getTime())
        np.testing.assert_allclose(actual.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
                                   expected.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
                                   atol=1e-6, rtol=1e-6)
        self.assertTrue(np.isfinite(actual.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)))
        np.testing.assert_allclose(actual.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer),
                                   expected.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer),
                                   atol=1e-6, rtol=1e-6)
        print(f'cuPotential water probe: relative force error={relative_force_error:.6g}, '
              f'relative energy error={abs(e_observed-e_ref)/max(abs(e_ref),1):.6g}')
        del fresh, fresh_integrator, accelerated, integrator
