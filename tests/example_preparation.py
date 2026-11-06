"""Run every numbered example with explicitly mapped local inputs, preparation only.

Examples are templates: originals are preserved and every input/stage/output
substitution is recorded. Large inputs are local development assets, not fixtures
shipped with the repository. No equilibration or production is permitted.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['MOJOEMD_TEST_BACKEND'] = 'modular'
from preparation_matrix import Case, run_case


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolved_examples():
    glycan = ROOT / 'dev/refactor_retests/ncaa_glycan/AiB_test_AIB_minimized.pdb'
    glycan_xml = ROOT / 'dev/refactor_retests/ncaa_glycan/ffxml/AIB.xml'
    ncaa = ROOT / 'dev/refactor_retests/ncaa/5MU0.pdb'
    ncaa_xml = ROOT / 'dev/ncaa_retest/5MU0/ffxml/CIR.xml'
    protein = ROOT / 'dev/inputs/protein.pdb'
    membrane = ROOT / 'dev/membrane/protein.pdb'
    membrane_ligand = ROOT / 'dev/membrane/ligand.sdf'
    test_ligand = ROOT / 'tests/fixtures/preparation/ligand.sdf'
    result = []
    for example in sorted((ROOT / 'examples').glob('[0-9][0-9]_*.json')):
        original = json.loads(example.read_text())
        config = dict(original)
        number = int(example.name[:2])
        notes = []
        if number == 11:
            config.update(receptor=str(membrane), ligand=str(membrane_ligand))
            notes.append('Matching membrane receptor and ligand from dev/membrane; coordinates unchanged.')
        elif config.get('glycoprotein'):
            config['receptor'] = str(glycan)
            notes.append('Full local 33-sugar glycoprotein/AIB structure; coordinates unchanged.')
            if config.get('ligand'):
                config['ligand'] = str(test_ligand)
                notes.append('Synthetic ethanol software-test ligand; not a biological bound ligand.')
            if config.get('ncaa'):
                config.update(ncaa_resnames=['AIB'], ncaa_ffxml=[str(glycan_xml)])
                notes.append('NCA placeholders resolved to the actual AIB residue and matching parameters.')
            else:
                notes.append('NCAA mode remains disabled: normal preparation may replace the source AIB with a canonical residue.')
        elif config.get('ncaa'):
            config.update(receptor=str(ncaa), ncaa_resnames=['CIR', 'HYP'], ncaa_ffxml=[str(ncaa_xml)])
            notes.append('NCA placeholders resolved to the real CIR/HYP system and matching CIR parameters; HYP uses built-in support.')
            if config.get('ligand'):
                config['ligand'] = str(test_ligand)
                notes.append('Synthetic ethanol software-test ligand; not a biological bound ligand.')
        else:
            config['receptor'] = str(protein)
            notes.append('Local copper-containing protein; metal-aware protonation settings preserved.')
            if config.get('ligand'):
                config['ligand'] = str(test_ligand)
                notes.append('Synthetic ethanol for positive chemistry coverage; dev/inputs/ligand.sdf is separately checked as malformed.')
        config.update(prepare=True, equilibrate=False, produce=False)
        if number == 9:
            notes.append('Debug target 527 is retained from the example; this local structure reports that the target is absent.')
        paths = [Path(config['receptor'])] + ([Path(config['ligand'])] if config.get('ligand') else [])
        paths += [Path(p) for p in config.get('ncaa_ffxml', [])]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        changes = {key: {'original': original.get(key), 'effective': value}
                   for key, value in config.items() if original.get(key) != value}
        result.append((Case(example.stem, config), {
            'example': str(example.relative_to(ROOT)), 'example_sha256': digest(example),
            'original': original, 'input_changes': changes, 'notes': notes,
            'inputs_sha256': {str(p.relative_to(ROOT)): digest(p) for p in paths},
        }))
    if len(result) != 11:
        raise ValueError(f'Expected 11 numbered examples, found {len(result)}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'dev/modular_prep_check')
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--timeout', type=float, default=3600)
    args = parser.parse_args()
    if args.jobs < 1 or args.timeout <= 0:
        parser.error('jobs and timeout must be positive')
    cases = resolved_examples()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'backend': 'modular', 'stages': ['prepare'], 'python': sys.version,
        'executable': sys.executable, 'examples': [meta for _, meta in cases],
        'output_policy': 'Each example writes to <case>/output; original output_dir is ignored.',
        'sources_sha256': {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / 'mojoemd').rglob('*.py'))}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    os.environ['MPLCONFIGDIR'] = str(output / 'cache/matplotlib')
    os.environ['XDG_CACHE_HOME'] = str(output / 'cache')
    print(f'Running all {len(cases)} resolved examples, preparation only -> {output}', flush=True)
    reports = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(run_case, case, output, args.timeout): (case, meta) for case, meta in cases}
        for future in as_completed(futures):
            case, meta = futures[future]
            try:
                report = future.result()
            except Exception as error:
                report = {'name': case.name, 'status': 'failed', 'error': f'{type(error).__name__}: {error}'}
            (output / case.name / 'input_mapping.json').write_text(json.dumps(meta, indent=2) + '\n')
            reports.append(report)
            (output / 'summary.json').write_text(json.dumps(sorted(reports, key=lambda r: r['name']), indent=2) + '\n')
            print(f"{report['status'].upper()}: {case.name}" + (f" — {report['error']}" if report.get('error') else ''), flush=True)
    # The malformed local ligand must stay rejected; never infer/repair its chemistry.
    from mojoemd.chemistry.ligands import prepare_ligand
    malformed = ROOT / 'dev/inputs/ligand.sdf'
    try:
        prepare_ligand(malformed)
    except ValueError as error:
        if 'aromatic' not in str(error).lower():
            raise
        negative = {'status': 'expected_rejection', 'source': str(malformed.relative_to(ROOT)),
                    'sha256': digest(malformed), 'error': str(error)}
    else:
        raise AssertionError('Malformed original ligand unexpectedly accepted')
    (output / 'malformed_ligand.json').write_text(json.dumps(negative, indent=2) + '\n')
    failed = sum(r['status'] != 'passed' for r in reports)
    print(f'{len(reports)-failed}/{len(reports)} examples passed; {failed} failed.', flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
