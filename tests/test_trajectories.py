"""Real DCD regressions for restarted trajectory analysis."""
import json
from pathlib import Path
import tempfile
import unittest

import mdtraj as md
import numpy as np

from mojoemd.analyse import load_traj, main
from mojoemd.trajectories import resolve_trajectories


class TrajectorySeriesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        topology = md.Topology()
        chain = topology.add_chain()
        residue = topology.add_residue('ALA', chain)
        topology.add_atom('CA', md.element.carbon, residue)
        self.top = self.root / 'equilibrated.pdb'
        md.Trajectory(np.zeros((1, 1, 3)), topology).save_pdb(str(self.top))
        for name, values in [('production_010.dcd', [5, 6]),
                             ('production_002.dcd', [3, 4]),
                             ('production_000.dcd', [0, 1, 2])]:
            xyz = np.zeros((len(values), 1, 3), dtype=np.float32)
            xyz[:, 0, 0] = values
            md.Trajectory(xyz, topology).save_dcd(str(self.root / name))

    def test_numeric_order_and_all_supported_automatic_inputs(self):
        for value in [self.root, self.root / 'production_*.dcd',
                      self.root / 'production_000.dcd', self.root / 'production.dcd']:
            with self.subTest(value=str(value)):
                trajectory = load_traj(self.top, value)
                np.testing.assert_allclose(trajectory.xyz[:, 0, 0], np.arange(7), atol=1e-5)

    def test_stride_is_global_across_unequal_segments(self):
        trajectory = load_traj(self.top, self.root, stride=2)
        np.testing.assert_allclose(trajectory.xyz[:, 0, 0], [0, 2, 4, 6], atol=1e-5)

    def test_explicit_list_preserves_selection_and_order(self):
        trajectory = load_traj(self.top, [self.root / 'production_002.dcd', self.root / 'production_000.dcd'])
        np.testing.assert_allclose(trajectory.xyz[:, 0, 0], [3, 4, 0, 1, 2], atol=1e-5)
        self.assertEqual(load_traj(self.top, [self.root / 'production_000.dcd']).n_frames, 3)

    def test_stride_across_read_chunks_and_file_boundaries(self):
        topology = md.load(str(self.top)).topology
        xyz = np.zeros((105, 1, 3), dtype=np.float32)
        xyz[:, 0, 0] = np.arange(105)
        md.Trajectory(xyz, topology).save_dcd(str(self.root / 'production_000.dcd'))
        result = load_traj(self.top, self.root, stride=7)
        expected = np.concatenate([np.arange(105), [3, 4, 5, 6]])[::7]
        np.testing.assert_allclose(result.xyz[:, 0, 0], expected, atol=1e-5)

    def test_errors_do_not_silently_skip_files(self):
        for value in [[], self.root / 'absent_*.dcd', [self.root / 'absent.dcd']]:
            with self.subTest(value=value), self.assertRaises(FileNotFoundError):
                resolve_trajectories(value)
        with self.assertRaises(ValueError):
            resolve_trajectories([self.root / 'production_000.dcd'] * 2)
        with self.assertRaises(ValueError):
            load_traj(self.top, self.root, stride=0)

    def test_legacy_single_file_still_loads(self):
        directory = self.root / 'legacy'
        directory.mkdir()
        source = self.root / 'production_000.dcd'
        target = directory / 'production.dcd'
        target.write_bytes(source.read_bytes())
        self.assertEqual(load_traj(self.top, target).n_frames, 3)
        self.assertEqual(load_traj(self.top, directory).n_frames, 3)

    def test_analysis_writes_one_combined_processed_dcd(self):
        config = self.root / 'analyse.json'
        output = self.root / 'analysis'
        config.write_text(json.dumps({'topology_pdb': str(self.top),
            'output_dir': str(output), 'imaging': {'enable': False},
            'alignment': {'enable': False}, 'bfactor_rmsf': {'enable': False},
            'sasa': {'enable': False}}))
        main(['--config', str(config)])
        trajectory = md.load(str(output / 'processed.dcd'), top=str(output / 'processed_topology.pdb'))
        np.testing.assert_allclose(trajectory.xyz[:, 0, 0], np.arange(7), atol=1e-5)
        self.assertEqual(len((output / 'rmsd.csv').read_text().splitlines()), 8)


if __name__ == '__main__':
    unittest.main()
