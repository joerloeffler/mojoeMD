"""Opt-in real preparation tests; normal unit discovery remains fast."""

from __future__ import annotations

import os
from pathlib import Path
import unittest

import preparation_matrix


@unittest.skipUnless(os.environ.get("MOJOEMD_PREPARE_INTEGRATION") == "1",
                     "Set MOJOEMD_PREPARE_INTEGRATION=1 to run real preparation tests")
class PrepareIntegrationTests(unittest.TestCase):
    def test_selected_preparation_matrix(self):
        suite = os.environ.get("MOJOEMD_PREPARE_SUITE", "matrix")
        argv = ["--suite", suite, "--jobs", os.environ.get("MOJOEMD_PREPARE_JOBS", "1")]
        output = os.environ.get("MOJOEMD_PREPARE_OUTPUT")
        if output:
            argv.extend(("--output", str(Path(output))))
        self.assertEqual(preparation_matrix.main(argv), 0, "Preparation regressions failed; inspect summary.json")


if __name__ == "__main__":
    unittest.main()
