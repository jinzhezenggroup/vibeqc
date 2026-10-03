"""Verify every complete endpoint pair before summarizing a frozen AO experiment."""
import lzma
import hashlib
import json
import sys
from pathlib import Path
from statistics import median
import numpy as np

def require(condition, message):
    """Keep every evidence acceptance gate active even under ``python -O``."""
    if not condition:
        raise ValueError(message)


root = Path(__file__).resolve().parent
atoms = int(sys.argv[1])
manifest = json.loads((root/'manifest.json').read_text())
expected = manifest['source']
summary = {'atoms': atoms, 'variants': {}, 'input_identity': expected}
for mode in manifest['comparison_modes'][str(atoms)]:
    path = root / f'matched{atoms}-{mode}.json.xz'
    stored = path.read_bytes()
    entry = manifest['files'][path.name]
    require(len(stored) == entry['bytes'] and hashlib.sha256(stored).hexdigest() == entry['sha256'], 'Stored report size or SHA-256 mismatch')
    payload = lzma.decompress(stored)
    require(len(payload) == entry['decoded_bytes'] and hashlib.sha256(payload).hexdigest() == entry['decoded_sha256'], 'Decoded report size or SHA-256 mismatch')
    d = json.loads(payload)
    require(d['status'] == 'measured' and d['stage'] == 'complete' and d['accepted'], 'Report is not a completed accepted measurement')
    require(d['native_build']['library_sha256'] == expected['library_sha256'], 'Native library identity mismatch')
    require(d['native_build']['probe']['source_identity'] == expected['source_identity'], 'Native source identity mismatch')
    ns = [d['native_cold'], d['native_priming'], *d['native_samples']]
    rs = [d['reference_cold'], d['reference_priming'], *d['reference_samples']]
    require(len(ns) == len(rs) == 5, 'Expected exactly five endpoint pairs')
    errors = {}
    for key, gate in (('energies_hartree', 1e-8), ('forces_hartree_per_bohr', 1e-7)):
        values = []
        for n, r in zip(ns, rs, strict=True):
            a, b = np.asarray(n[key]), np.asarray(r[key])
            require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), 'Energy/force arrays must have matching shapes and finite values')
            values.append(float(np.max(np.abs(a-b))))
        require(max(values) <= gate, f'Energy/force acceptance gate exceeded: {key}: {values}')
        errors[key] = values
    for s in ns + rs:
        require(all(x['converged'] for x in s['convergence']), 'Unconverged endpoint')
    require(all(x['iterations'] == 1 for s in ns[1:]+rs[1:] for x in s['convergence']), 'Warm and priming endpoints must use exactly one SCF iteration')
    require(all(s['reference_xc_backend'] and all(x['on_gpu'] for x in s['reference_xc_backend']) for s in rs), 'Reference XC must execute on GPU')
    if mode == 'sparse':
        for s in ns:
            work = s['force_work']['active_ao_maps']
            require(work['discovery_density_contractions'] == 0, 'Active AO discovery performed density contractions')
            require(work['point_ao_square_sum'] <= work['dense_point_ao_square_sum'], 'Force active AO work exceeds dense work')
        require(all(s['force_work']['active_ao_maps']['discoveries'] == 0 for s in ns[1:]), 'Warm/priming force maps were rediscovered')
    # The disabled force caller does not expose cache counters. Preserve absent
    # evidence rather than filling it with zero or an estimated dense count.
    work = ns[-1]['force_work'].get('active_ao_maps')
    for s in ns:
        scf = s['native_scf_ao_work']
        require(scf['requested'] == scf['selected'] == (1 if mode == 'sparse' else 0), 'Actual SCF AO selection differs from the requested mode')
        require(scf['xc_evaluations'] > 0, 'No SCF XC evaluations recorded')
        require(scf['point_ao_square_sum'] <= scf['dense_point_ao_square_sum'], 'SCF active AO work exceeds dense work')
        for key in ('tiles','empty_tiles','active_sum','max_active','host_peak_bytes','discovery_seconds'):
            require(scf[key] == ns[0]['native_scf_ao_work'][key], 'SCF AO map identity or discovery work changed between endpoints')
    for engine in ('native','reference'):
        total = d[f'{engine}_complete_cold']
        require(total['seconds'] == total['prepare_seconds'] + total['first_execute_seconds'], 'Complete cold total does not include preparation and first execution')
        require(total['prepare_seconds'] == d[f'{engine}_prepare_seconds'], 'Complete cold preparation differs from reported preparation')
        require(total['first_execute_seconds'] == d[f'{engine}_cold']['seconds'], 'Complete cold execution differs from the cold endpoint')
    scf = ns[-1]['native_scf_ao_work']
    nv, rv = [s['seconds'] for s in d['native_samples']], [s['seconds'] for s in d['reference_samples']]
    summary['variants'][mode] = {
        'raw_sha256': hashlib.sha256(payload).hexdigest(), 'all_pair_errors': errors,
        'native_warm_seconds': nv, 'reference_warm_seconds': rv,
        'native_median_seconds': median(nv), 'reference_median_seconds': median(rv),
        'native_over_reference': median(nv)/median(rv),
        'execute_only_cold_seconds': {e: d[f'{e}_cold']['seconds'] for e in ('native','reference')},
        'reported_prepare_seconds': {e: d[f'{e}_prepare_seconds'] for e in ('native','reference')},
        'force_warm_ao_work': work,
        'force_gm2_fraction': None if work is None else work['point_ao_square_sum']/work['dense_point_ao_square_sum'],
        'all_reference_xc_on_gpu': True,
        'complete_cold': {e:d[f'{e}_complete_cold'] for e in ('native','reference')},
        'native_scf_cold_work': ns[0]['native_scf_ao_work'],
        'native_scf_warm_work': scf,
        'scf_gm2_fraction': scf['point_ao_square_sum']/scf['dense_point_ao_square_sum'],
        'cold_iterations': {e:[x['iterations'] for x in d[f'{e}_cold']['convergence']] for e in ('native','reference')},
    }
# Read-only verification preserves the retained scientific records.
for mode,v in summary['variants'].items():
    print(mode, 'native',v['native_median_seconds'],'reference',v['reference_median_seconds'], 'ratio',v['native_over_reference'],'GM2',v['force_gm2_fraction'])
print('Every E/F pair, source/library identity, actual AO selection, reference XC backend and complete-cold total passed.')
