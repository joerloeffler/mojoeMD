"""Real CPU prepare/equilibrate/produce/restart/analysis regression.

Run each backend in a separate process. No protocol steps are shortened.
The short 50 ps production segments test handoffs, not scientific convergence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('legacy', 'modular'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume-after-production', action='store_true',
                        help='Validate an existing completed initial run, then test restart and analysis')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=args.resume_after_production)
    for key in ('OMP_NUM_THREADS', 'OPENMM_CPU_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '2'
    os.environ['MPLCONFIGDIR'] = str(output / 'cache/matplotlib')
    os.environ['XDG_CACHE_HOME'] = str(output / 'cache')

    import numpy as np
    from openmm import Platform, XmlSerializer
    from openmm.unit import nanometer, kilojoules_per_mole
    import mdtraj as md
    from mojoemd.api import _result, _select_engine, run_pipeline
    from mojoemd import analyse

    config = output / 'config.json'
    if not args.resume_after_production:
        config.write_text(json.dumps({'receptor': str(ROOT / 'tests/fixtures/preparation/protein.pdb'),
            'output_dir': str(output / 'simulation'), 'prepare': True, 'equilibrate': True,
            'produce': True, 'prod_ns': 0.05, 'timestep_fs': 2.0,
            'checkpoint_interval_ps': 10.0}, indent=2) + '\n')
    else:
        assert (output / 'simulation/production_final_000.chk').exists(), 'Initial production incomplete'
        assert not (output / 'simulation/production_001.dcd').exists(), 'Restart already exists'
    engine = _select_engine(args.backend)
    platform_owner = engine if args.backend == 'legacy' else __import__('mojoemd.runtime', fromlist=['select_platform'])
    # Extracted stages import the platform selector directly.
    owners = [platform_owner] if args.backend == 'legacy' else [
        __import__('mojoemd.equil', fromlist=['select_platform']),
        __import__('mojoemd.production', fromlist=['select_platform'])]
    from contextlib import ExitStack
    with ExitStack() as stack:
        for owner in owners:
            stack.enter_context(patch.object(owner, 'select_platform', return_value=Platform.getPlatformByName('CPU')))
        result = (_result(output / 'simulation', ('prepare', 'equilibrate', 'produce'))
                  if args.resume_after_production else run_pipeline(config, backend=args.backend))
        first_state = XmlSerializer.deserialize((result.output_dir / 'equilibrated.xml').read_text())
        assert np.isfinite(np.asarray(first_state.getPositions(asNumpy=True).value_in_unit(nanometer))).all()
        restarted = run_pipeline(config, stages=('produce',), cli_overrides=('--restart',), backend=args.backend)
    assert result.production_dcd.name == 'production_000.dcd'
    assert restarted.production_dcd.name == 'production_001.dcd'
    first = md.load(str(result.production_dcd), top=str(result.equilibrated_pdb))
    second = md.load(str(restarted.production_dcd), top=str(result.equilibrated_pdb))
    assert len(first) >= 1 and len(second) >= 1
    assert np.isfinite(first.xyz).all() and np.isfinite(second.xyz).all()
    import csv
    times = []
    for segment in (0, 1):
        with (result.output_dir / f'production_{segment:03d}.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        assert rows, 'Missing production reports'
        assert all(np.isfinite(float(row['Potential Energy (kJ/mole)'])) for row in rows), 'Nonfinite production energy'
        times.append([float(row['Time (ps)']) for row in rows])
    assert times[1][0] > times[0][-1], 'Restart did not advance simulation time'
    joined = first.join(second)
    joined.save_dcd(str(output / 'joined.dcd'))
    analysis_config = output / 'analysis.json'
    analysis_config.write_text(json.dumps({'topology_pdb': str(result.equilibrated_pdb),
        'trajectory': str(output / 'joined.dcd'), 'output_dir': str(output / 'analysis'),
        'strip': {'remove_water': True, 'remove_ions': True}, 'sasa': {'enable': True}}))
    analyse.main(['--config', str(analysis_config)])
    for filename in ('rmsd.csv', 'rmsf_atoms.csv', 'rmsf_residues.csv', 'sasa_global.csv', 'processed.dcd'):
        assert (output / 'analysis' / filename).stat().st_size > 0, filename
    # Compare the original analysis on the same trajectory, isolating analysis parity
    # from independent stochastic dynamics.
    import importlib.util
    spec = importlib.util.spec_from_file_location('_legacy_analysis', ROOT / 'analysis/analyse_MD.py')
    legacy_analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy_analysis)
    legacy_cfg = json.loads(analysis_config.read_text())
    legacy_cfg['output_dir'] = str(output / 'legacy_analysis')
    legacy_config = output / 'legacy_analysis.json'
    legacy_config.write_text(json.dumps(legacy_cfg))
    with patch.object(sys, 'argv', ['analyse_MD.py', '--config', str(legacy_config)]):
        legacy_analysis.main()
    for filename in ('rmsd.csv', 'rmsf_atoms.csv', 'rmsf_residues.csv', 'sasa_global.csv'):
        assert (output / 'analysis' / filename).read_bytes() == (output / 'legacy_analysis' / filename).read_bytes(), filename
    record = {'backend': args.backend, 'status': 'passed', 'equilibration_steps': 130000,
        'production_ps_per_segment': 50, 'frames': [len(first), len(second)],
        'report_times_ps': times, 'analysis_parity': 'CSV byte equality on identical input',
        'legacy_sha256': hashlib.sha256((ROOT / 'mojoeMD.py').read_bytes()).hexdigest()}
    (output / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
