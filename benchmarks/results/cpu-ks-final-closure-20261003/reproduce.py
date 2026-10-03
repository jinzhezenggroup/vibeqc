"""Bounded, opt-in scientific reproducer; default verification runs no chemistry."""
import argparse, hashlib, json, os, sys
from pathlib import Path
from types import SimpleNamespace
from common import (THREADS, GATES, ADDITIONAL_GATES, atomic_json, identity, read, require,
                    git, resource_admission, sha, canonical)

ROOT=Path(__file__).resolve().parent
# Keep the shared storage reader in the publication checkout. Historical
# scientific checkouts predate this reader; their Python numerical source still
# comes exclusively from --source/python via the child environment.
PUBLICATION_REPOSITORY=ROOT.parents[2]
if (PUBLICATION_REPOSITORY/"tools/generativeqc_validation/record.py").is_file():
    sys.path.insert(0,str(PUBLICATION_REPOSITORY))



def build_record(source,library):
    library=library.resolve();build=library.parent
    required=[library,build/'compile_commands.json',build/'CMakeCache.txt']
    missing=[str(path) for path in required if not path.is_file()]
    require(not missing,'Missing build prerequisites; configure with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON: '+', '.join(missing))
    return dict(source=str(source),revision=git(source,'rev-parse','HEAD'),tree=git(source,'rev-parse','HEAD^{tree}'),library=identity(library),compile_commands=identity(build/'compile_commands.json'),cmake_cache=identity(build/'CMakeCache.txt'))


def declared_oracle_densities(oracle):
    rows=[]
    for sample in oracle['samples']:
        row=identity(sample['density_file'])
        require(row['sha256']==sample['density_sha256'],'Oracle density differs from its declared receipt hash')
        rows.append(row)
    require(rows,'Oracle has no density samples')
    return rows



def validate_independent_intake(native,oracle,manifest):
    require(native.get('input_sha256')==manifest['input']['sha256'],'Native input differs from reproduction input')
    require(native['identity']['tree']==manifest['source']['tree'],'Native source tree differs from reproduction source')
    require(oracle.get('schema')=='pbe96-fresh-reproduction-oracle-v1','Portable independent mode requires explicitly fresh reproduction oracle; original historical receipt is checksum-only evidence here')
    require(oracle.get('input_sha256')==manifest['input']['sha256'],'Fresh oracle input differs from reproduction input')
    require(oracle.get('source',{}).get('tree')==manifest['source']['tree'],'Fresh oracle source tree differs from reproduction source')
    require(isinstance(oracle.get('manifest_sha256'),str) and len(oracle['manifest_sha256'])==64,'Fresh oracle manifest identity missing')


def verify():
    from tools.generativeqc_validation.publication import validate_publication
    from tools.generativeqc_validation.record import load_publication_record
    manifest=read(ROOT/'publication.json')
    validate_publication(manifest,{row['path']:(ROOT/row['path']).read_bytes() for row in manifest['files']})
    samples=load_publication_record(ROOT,role='samples',name='samples.json')
    old=samples['original_native'];new=samples['corrected_native'];audit=samples['same_coordinate_v2']
    require(old['iterations']==17 and old['fock_builds']==19,'Old outcome changed')
    require(new['iterations']==21 and new['fock_builds']==25,'New outcome changed')
    require(old['history_prefix_rows']==17 and canonical(new['history'][:17])==new['closure_progress']['first17_history_sha256'],'Historical prefix changed')
    require([x['iteration'] for x in new['history']]==list(range(1,22)),'History gap')
    require([x['iteration'] for x in new['history'] if x['energy_change'] is None]==[1,18],'Stage baseline changed')
    require(new['snapshot_extra_fock_builds_observed']==0 and new['pre_post_diagnostic_equal'],'Export work changed')
    for key,value in audit['metrics'].items():require(value<=GATES[key],'Scientific gate failed: '+key)
    for key,value in audit['additional_metrics'].items():require(value<=ADDITIONAL_GATES[key],'Additional gate failed: '+key)
    require(audit['original_cross_node_grid_gate_passed'] is False,'Old grid failure concealed')
    require(audit['tail_domain']['points']==331776 and len(audit['grid_identity_checks'])==12,'Grid coverage changed')
    require(all(x['object_identity_preserved'] and x['build_attempts']==0 for x in audit['grid_identity_checks']),'Grid identity changed')
    require(samples['same_coordinate_v1']['physical_computations_valid'] is False,'Wrong-grid result admitted')
    print('Selective publication, scalar gates, histories and original failures verified; no science executed')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('verify','native','oracle','independent'),nargs='?',default='verify')
    p.add_argument('--source',type=Path);p.add_argument('--library',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--cpu',type=int,default=2)
    p.add_argument('--native',type=Path);p.add_argument('--native-supervisor',type=Path)
    p.add_argument('--oracle',type=Path)
    p.add_argument('--child',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--manifest',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--manifest-sha256',help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.mode=='verify':verify();return
    require(a.source is not None and a.output is not None,'Scientific modes need --source and --output')
    a.source=a.source.resolve();a.output=a.output.resolve()
    require(a.cpu in os.sched_getaffinity(0),'Requested CPU unavailable')
    require(all(os.environ.get(k)=='1' for k in THREADS),'Set all six thread caps to 1')
    require(sys.dont_write_bytecode,'Use python -B')
    require(not any(k in os.environ for k in ('GENERATIVEQC_CPU_XC_POTENTIAL','GENERATIVEQC_CPU_ORBITAL_EIGEN')),'Experimental selector set')
    os.environ['PYTHONPATH']=str(a.source/'python')+os.pathsep+str(a.source)
    if a.library is not None:os.environ['GENERATIVEQC_LIBRARY']=str(a.library.resolve())
    if a.child:
        # Only one explicitly requested mode is imported/executed per process.
        a.arm='candidate';a.phase='original'
        module=__import__({'native':'native_state','oracle':'oracle_state','independent':'independent_state'}[a.mode])
        module.main(a);return
    require(not a.output.exists(),'Output already exists; retain prior runs')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    admission=resource_admission(a.output.parent)
    require(not git(a.source,'status','--porcelain'),'Build a clean reconstructed source')
    arms={}
    if a.library is not None:
        arms['candidate']=build_record(a.source,a.library)
    require(a.mode!='native' or arms,'Native mode needs --library')
    manifest=dict(source=dict(path=str(a.source),revision=git(a.source,'rev-parse','HEAD'),tree=git(a.source,'rev-parse','HEAD^{tree}')),input=identity(ROOT/'input.json'),gates=GATES,additional_gates=ADDITIONAL_GATES,arms=arms,harness_files=[identity(x) for x in sorted(ROOT.glob('*.py'))],retained_inputs=[],admission=admission)
    historical=a.output.with_name(a.output.name+'.historical.json')
    from tools.generativeqc_validation.record import load_publication_record
    samples=load_publication_record(ROOT,role='samples',name='samples.json');old=samples['original_native']
    atomic_json(historical,dict(native_result=dict(iterations=17),ks_diagnostic=dict(history=samples['corrected_native']['history'][:old['history_prefix_rows']],fock_builds=19)),exclusive=True)
    manifest['historical_failed_native']=identity(historical)
    if a.mode=='independent':
        require(a.native is not None and a.native_supervisor is not None and a.oracle is not None,'Independent mode needs retained native directory, successful supervisor and oracle JSON')
        a.native=a.native.resolve();a.native_supervisor=a.native_supervisor.resolve();a.oracle=a.oracle.resolve()
        native=read(a.native/'result.json');oracle=read(a.oracle)
        require(oracle['status']=='completed','Unaccepted oracle receipt')
        validate_independent_intake(native,oracle,manifest)
        require(native['status']=='passed' and native['native_owners_closed'],'Native owner not closed')
        manifest['oracle']=identity(a.oracle)
        manifest['retained_inputs']=[identity(a.native/'result.json'),identity(a.native_supervisor),identity(a.oracle),*declared_oracle_densities(oracle),*native['arrays'].values()]
        manifest['retained_failed_independent']=samples['original_cross_node_audit']['raw_identity']
    path=a.output.with_name(a.output.name+'.manifest.json');atomic_json(path,manifest,exclusive=True)
    command=['taskset','-c',str(a.cpu),sys.executable,'-B',str(Path(__file__).resolve()),a.mode,'--child','--source',str(a.source),'--output',str(a.output),'--cpu',str(a.cpu),'--manifest',str(path),'--manifest-sha256',sha(path)]
    for flag in ('library','native','native_supervisor','oracle'):
        value=getattr(a,flag)
        if value is not None:command+=['--'+flag.replace('_','-'),str(value.resolve())]
    from safety import supervise
    receipt=supervise(command,dict(os.environ),a.output.with_name(a.output.name+'.log'))
    atomic_json(a.output.with_name(a.output.name+'.supervisor.json'),receipt,exclusive=True)
    require(receipt['status']=='completed' and receipt['returncode']==0,'Reproduction failed; all output preserved')


if __name__=='__main__':main()
