from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import mojoeMD_DP


PDB = """\
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  CA  GLU A   2       3.800   0.000   0.000  1.00  0.00           C
ATOM      3  CA  LYS A   3       7.600   0.000   0.000  1.00  0.00           C
END
"""


class CalvadosAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "protein.pdb").write_text(PDB)

    def write_config(self, ionic=0.15):
        config = self.root / f"{ionic}.json"
        config.write_text(json.dumps({
            "input_pdb": "protein.pdb",
            "output_dir": f"run-{ionic}",
            "ionic_strength": ionic,
            "prod_ns": 500,
        }))
        return config

    def test_inspect_pdb_extracts_one_bead_per_residue(self):
        summary = mojoeMD_DP.inspect_pdb(self.root / "protein.pdb")
        self.assertEqual(summary.sequence, "AEK")
        self.assertEqual(summary.residues, 3)
        self.assertEqual(summary.chains, 1)

    def test_prepare_writes_job_without_running_calvados(self):
        job = mojoeMD_DP.prepare_job(self.write_config())
        output = Path(job.output_dir)
        self.assertEqual(job.production_steps, 50_000_000)
        self.assertTrue(job.prepared_only)
        self.assertTrue((output / "config.yaml").is_file())
        self.assertTrue((output / "components.yaml").is_file())
        self.assertTrue((output / "build.py").is_file())
        self.assertTrue((output / "equilibrate.py").is_file())
        self.assertTrue((output / "production.py").is_file())
        self.assertTrue((output / "run.py").is_file())
        self.assertFalse((output / "top.pdb").exists())
        self.assertIn("ionic: 0.15", (output / "config.yaml").read_text())
        self.assertIn("equilibration_steps: 100000", (output / "config.yaml").read_text())

    def test_production_runner_requires_equilibration_checkpoint(self):
        job = mojoeMD_DP.prepare_job(self.write_config())
        runner = (Path(job.output_dir) / "run.py").read_text()
        self.assertIn("Missing equilibration/checkpoint.cif", runner)
        self.assertIn('choices=("build", "equilibrate", "produce")', runner)
        self.assertIn("MOJOEMD_DP_PLATFORM", runner)

    def test_noncanonical_residue_is_rejected(self):
        bad = self.root / "bad.pdb"
        bad.write_text(PDB.replace("ALA", "2MR"))
        with self.assertRaisesRegex(ValueError, "Noncanonical residues"):
            mojoeMD_DP.inspect_pdb(bad)

    def test_staged_pdb_removes_caps_and_normalizes_histidine(self):
        source = self.root / "capped.pdb"
        source.write_text(
            "HETATM    1  CH3 ACE A   0       0.000   0.000   0.000  1.00  0.00           C\n"
            "HETATM    2  CA  HIE A   1       0.000   0.000   0.000  1.00  0.00           C\n"
            "ATOM      3  CA  ALA A   2       3.800   0.000   0.000  1.00  0.00           C\n"
            "END\n"
        )
        staged = self.root / "staged.pdb"
        mojoeMD_DP.write_calvados_pdb(source, staged)
        text = staged.read_text()
        self.assertNotIn("ACE", text)
        self.assertIn("HIS", text)
        self.assertNotIn("HIE", text)


if __name__ == "__main__":
    unittest.main()
