from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

import mojoeMD
from openmm import Vec3
from openmm.app import ForceField, Topology
from openmm.unit import nanometer


class EngineConfigTests(unittest.TestCase):
    def test_default_force_fields_are_preserved(self):
        args = mojoeMD.parse_arguments([])
        self.assertEqual(args.protein_ff, "amber14/protein.ff14SB.xml")
        self.assertEqual(args.water_ff, "amber14/tip3pfb.xml")
        self.assertEqual(args.lipid_ff, "amber14/lipid17.xml")

    def test_all_protein_options_load_with_default_water(self):
        for protein_ff in mojoeMD.PROTEIN_FORCE_FIELDS:
            with self.subTest(protein_ff=protein_ff):
                args = mojoeMD.parse_arguments(["--protein_ff", protein_ff])
                forcefield = ForceField(args.protein_ff, args.water_ff)
                self.assertIn("ALA", forcefield._templates)

    def test_modern_amber_options_load_with_glycam(self):
        for protein_ff in ("amber14/protein.ff14SB.xml", "amber19/protein.ff19SB.xml"):
            with self.subTest(protein_ff=protein_ff):
                args = mojoeMD.parse_arguments(["--protein_ff", protein_ff, "--glycoprotein"])
                forcefield = ForceField(args.protein_ff, args.water_ff, args.glycam_xml)
                self.assertIn("NLN", forcefield._templates)

    def test_charmm_uses_matching_water_and_bundled_lipids(self):
        for protein_ff in ("charmm36.xml", "charmm36_2024.xml"):
            args = mojoeMD.parse_arguments(["--protein_ff", protein_ff, "--membrane"])
            self.assertEqual(args.water_ff, protein_ff.removesuffix(".xml") + "/water.xml")
            self.assertEqual(args.lipid_ff, protein_ff)

    def test_incompatible_force_field_combinations_fail_early(self):
        combinations = [
            ["--protein_ff", "charmm36.xml", "--glycoprotein"],
            ["--protein_ff", "amber99sb.xml", "--glycoprotein"],
            ["--protein_ff", "amber14/protein.ff15ipq.xml", "--glycoprotein"],
            ["--protein_ff", "amber19/protein.ff19ipq.xml", "--glycoprotein"],
            ["--protein_ff", "charmm36.xml", "--ligand", "ligand.sdf"],
            ["--protein_ff", "charmm36.xml", "--ncaa"],
            ["--protein_ff", "charmm36.xml", "--water_ff", "amber14/tip3pfb.xml"],
            ["--protein_ff", "charmm36_2024.xml", "--water_ff", "charmm36/water.xml"],
            ["--water_ff", "charmm36/water.xml"],
            ["--protein_ff", "amber99sb.xml", "--water_ff", "amber14/tip3pfb.xml"],
        ]
        for argv in combinations:
            with self.subTest(argv=argv), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as error:
                    mojoeMD.parse_arguments(argv)
                self.assertEqual(error.exception.code, 2)

    def test_json_force_field_selection_and_invalid_choice(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "prepare.json"
            config.write_text(json.dumps({"protein_ff": "amber19/protein.ff19SB.xml"}))
            args = mojoeMD.parse_arguments(["--config", str(config)])
            self.assertEqual(args.protein_ff, "amber19/protein.ff19SB.xml")
            config.write_text(json.dumps({"protein_ff": "unknown.xml"}))
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                mojoeMD.parse_arguments(["--config", str(config)])

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
