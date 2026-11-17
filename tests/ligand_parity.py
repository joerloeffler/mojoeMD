"""Compare real ligand systems from the legacy and extracted parameterizers."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    from openmm import XmlSerializer
    from openmm.app import ForceField, NoCutoff
    from openmm.unit import nanometer
    from mojoemd.api import _load_engine
    from mojoemd.chemistry import ligands

    legacy = _load_engine()
    reports = []
    for method in ('smirnoff', 'espaloma'):
        systems = []
        positions = []
        for engine in (legacy, ligands):
            molecule = engine.prepare_ligand(ROOT / 'tests/fixtures/preparation/ligand.sdf', residue_name='LIG')
            forcefield = ForceField('amber14/protein.ff14SB.xml', 'amber14/tip3pfb.xml')
            topology, xyz = engine.parametrize_ligand(molecule, forcefield, method)
            systems.append(XmlSerializer.serialize(forcefield.createSystem(topology, nonbondedMethod=NoCutoff)))
            positions.append(np.asarray(xyz.value_in_unit(nanometer)))
        assert systems[0] == systems[1], f'{method}: ligand parameters changed'
        np.testing.assert_array_equal(positions[0], positions[1])
        reports.append({'method': method, 'system_xml_equal': True, 'coordinates_equal': True})
    print(json.dumps(reports), flush=True)


if __name__ == '__main__':
    main()
