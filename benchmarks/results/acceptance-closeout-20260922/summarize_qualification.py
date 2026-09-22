"""Extract bounded acceptance facts; failed and incomplete cells remain explicit."""
import json
import re
import statistics
from pathlib import Path

out = Path(__file__).resolve().parent
summary = {
    'production_source': '0d89ab6fb6219d9af6641d5cfafe0b43f8387206',
    'library_sha256': '7bb9c7ccd6381dbdaefb3cff752b8ede66445f3578b36ee79253874fa131c364',
    'test_repairs_pr': 1046,
    'native_baseline': {'passed': 81, 'failed': 5, 'total': 86, 'job': '11195'},
    'python_full_replay': {'passed': 247, 'failed': 4, 'job': '11204',
        'scope': 'Explicit MP2 energy request only; later focused repairs are separate receipts'},
    'host_checks': [], 'focused_checks': [], 'df_endpoints': [],
}
for name in ('issue487-tests', 'issue487-compat-tests', 'issue487-compat-expanded-tests',
             'issue971-tests', 'issue949-tests', 'scf-structure-tests'):
    p = out / f'{name}.log'
    summary['host_checks'].append({'receipt': p.name,
        'source': '4a56f4d5b92c2a011ba6525c1d2f68da88bbe1aa' if 'compat' in name else summary['production_source'],
        'pytest_summary': next((s.strip() for s in reversed(p.read_text().splitlines()) if re.search(r'\d+ passed', s)), None)})
for name in ('test_mp2_contract_fixed', 'mp2-extended-fixed-cpu', 'mp2-extended-fixed-gpu',
             'public-dft-fixed', 'hf-contracts-fixed', 'hf-contracts-fixed-v2',
             'test_density_fitting_fixed', 'test_density_fitting_fixed_v2', 'test_dft_api_fixed_v2'):
    p = out / f'{name}.log'
    if p.exists():
        summary['focused_checks'].append({'receipt': p.name,
            'result': p.read_text().strip().splitlines()[-1] if p.stat().st_size else 'incomplete'})
for p in sorted(out.glob('df-*-*.json')):
    if p.name.endswith(('runs.json', 'disposition.json')): continue
    x = json.loads(p.read_text())
    for policy in x.get('policies', []):
        samples = [s for s in x['samples'] if s['policy'] == policy]
        diagnostics = [s for s in x.get('diagnostics', []) if s['policy'] == policy]
        row = {'receipt': p.name, 'policy': policy, 'samples': len(samples),
               'seconds': [s['seconds'] for s in samples],
               'median_seconds': statistics.median(s['seconds'] for s in samples) if samples else None,
               'iterations': [s['iterations'] for s in samples],
               'maximum_energy_error': max((s['maximum_energy_error'] for s in samples), default=None),
               'maximum_force_error': max((s['maximum_force_error'] for s in samples), default=None),
               'gates': x['gates']}
        row['force_response_counters'] = [
            {k: v for k,v in op['counter_sums'].items() if k.startswith('response_') or k == 'tensor_host_to_device_bytes'}
            for d in diagnostics for op in d['components']['groups'] if op['operation'] == 'force_response']
        summary['df_endpoints'].append(row)
x=json.loads((out/'r2scan3c-qualification.json').read_text())
summary['r2scan3c'] = [{k:v for k,v in row.items() if k in ('case','aos','passed','error','maximum_energy_error','maximum_force_error','changed_fresh_force_error','finite_differences')} for row in x['cases']]
summary['r2scan3c_failure_isolation'] = x['ragged_failure_isolation']
(out/'qualification-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:summary[k] for k in ('focused_checks','df_endpoints')},indent=2))
