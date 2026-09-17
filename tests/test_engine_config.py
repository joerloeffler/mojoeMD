from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import mojoeMD
from openmm import Vec3
from openmm.app import Topology
from openmm.unit import nanometer


class EngineConfigTests(unittest.TestCase):
    def test_placeholder_input_box_is_discarded_before_solvation(self):
        topology = Topology()
        topology.setPeriodicBoxVectors(
            (
                Vec3(0.1, 0.0, 0.0),
                Vec3(0.0, 0.1, 0.0),
                Vec3(0.0, 0.0, 0.1),
            ) * nanometer
        )

        self.assertTrue(mojoeMD.discard_input_periodic_box(topology))
        self.assertIsNone(topology.getPeriodicBoxVectors())
        self.assertFalse(mojoeMD.discard_input_periodic_box(topology))

    def test_paths_are_relative_to_config_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config" / "prepare.json"
            config.parent.mkdir()
            config.write_text(
                json.dumps(
                    {
                        "receptor": "inputs/protein.pdb",
                        "output_dir": "runs/test",
                        "prepare": True,
                    }
                )
            )

            args = mojoeMD.parse_arguments(["--config", str(config)])

            self.assertEqual(args.config, str(config))
            self.assertEqual(args.receptor, str((config.parent / "inputs/protein.pdb").resolve()))
            self.assertEqual(args.output_dir, str((config.parent / "runs/test").resolve()))

    def test_production_segments_never_reuse_an_existing_number(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "production_000.dcd").touch()
            (root / "production_002.csv").touch()
            (root / "production_final_004.chk").touch()

            self.assertEqual(mojoeMD.next_production_segment(root), 5)


if __name__ == "__main__":
    unittest.main()
