from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import mojoeMD


class EngineConfigTests(unittest.TestCase):
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
