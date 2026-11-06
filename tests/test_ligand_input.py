"""Regression for malformed SDF bond orders caught by preparation tests."""

from pathlib import Path
import unittest

from mojoemd.api import _load_engine


engine = _load_engine()


@unittest.skipUnless(engine._has_openff, "Requires optional ligand dependencies")
class LigandInputTests(unittest.TestCase):
    def test_nonring_aromatic_bonds_are_rejected_before_toolkit_conversion(self):
        fixture = Path(__file__).parent / "fixtures/preparation/invalid_aromatic_ligand.sdf"
        with self.assertRaisesRegex(ValueError, "6 aromatic bond\\(s\\) outside rings"):
            engine.prepare_ligand(fixture)

    def test_valid_ring_aromatic_bonds_and_implicit_hydrogens_are_accepted(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem
        import tempfile

        # Ordinary benzene aromatic bonds must not be rejected by the input check.
        molecule = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
        self.assertEqual(AllChem.EmbedMolecule(molecule, randomSeed=20261001), 0)
        molecule = Chem.RemoveHs(molecule)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "benzene.sdf"
            with Chem.SDWriter(str(path)) as writer:
                writer.write(molecule)
            ligand = engine.prepare_ligand(path)
        self.assertEqual(ligand.n_atoms, 12)
        self.assertEqual(ligand.name, "BEN")


if __name__ == "__main__":
    unittest.main()
