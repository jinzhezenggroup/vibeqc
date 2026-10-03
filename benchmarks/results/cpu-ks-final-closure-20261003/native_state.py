"""Portable actual-primary PBE96 snapshot; no external oracle imports."""
import argparse, dataclasses, gc, importlib.abc, os, resource, sys, traceback
from common import *

class BlockOracle(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'pyscf','gpu4pyscf','cupy'}:raise RuntimeError('External oracle forbidden in native state: '+fullname)

def main(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    r=dict(schema='pbe96-primary-native-state-v1',status='starting',arm=a.arm,phase=a.phase,performance_measurement=False,native_execute_calls=0,snapshot_attempts=0,external_oracle_calls=0,started_utc=now())
    save=lambda:atomic_json(out/'result.json',r)
    save()
    try:
        release=dict(cpu=a.cpu)
        require(not any(x in sys.modules for x in ('numpy','generativeqc','pyscf')),'Numerical imports preceded resource cap')
        resource.setrlimit(resource.RLIMIT_AS,(AS_LIMIT,AS_LIMIT))
        require(sorted(os.sched_getaffinity(0))==[release['cpu']],'Unexpected affinity')
        require(all(os.environ.get(k)=='1' for k in THREADS),'Thread cap mismatch')
        require(all(k not in os.environ for k in ('GENERATIVEQC_CPU_XC_POTENTIAL','GENERATIVEQC_CPU_ORBITAL_EIGEN')),'Experimental selector present')
        require(sys.dont_write_bytecode,'Bytecode writing forbidden')
        m=verify_manifest(a.manifest,a.manifest_sha256);case=read(m['input']['path']);arm=m['arms'][a.arm]
        sys.meta_path.insert(0,BlockOracle())
        import numpy as np
        import generativeqc
        from generativeqc import Atom,Calculator,_native
        from generativeqc_compiler.dft import NativeAO
        from generativeqc._dft_gradient import StationaryKsState
        from generativeqc.ks_diagnostics import read_ks_diagnostic
        package=Path(generativeqc.__file__).resolve();require(package==Path(arm['source'])/'python/generativeqc/__init__.py','Wrong Python package')
        library=Path(_native.load_library()._name).resolve();require(str(library)==arm['library']['path'] and sha(library)==arm['library']['sha256'],'Wrong loaded native library')
        xyz=case['atoms_bohr' if a.phase=='original' else 'moved_atoms_bohr']
        calc=Calculator(**SETTINGS,initial_guess=None)
        labels=[s+str(i) for i,(s,p) in enumerate(xyz)];exact={s:[] for s in labels}
        for shell in calc._shells_for_atoms(tuple(Atom.from_value(x) for x in xyz)):
            exact[labels[shell.atom_index]].append([shell.angular_momentum,*[[p.exponent,p.coefficient] for p in shell.primitives]])
        require(exact==case['exact_primitive_input'],'Current primitive input differs')
        require(json.loads(json.dumps(dataclasses.asdict(calc.ks_options.grid)))==case['grid_spec'],'Resolved grid differs')
        require(calc.ks_options.tile_points==256 and calc.ks_options.scf_domain=='semilocal-scaled-v1/pbe-spin-c2-1e-18','Tile/domain changed')
        r.update(manifest_sha256=a.manifest_sha256,input_sha256=m['input']['sha256'],identity=arm,atoms_bohr=xyz,settings=SETTINGS,grid_spec=case['grid_spec'],grid_points=331776,exact_primitive_input=exact,library=identity(library),package=str(package),rlimit_as=list(resource.getrlimit(resource.RLIMIT_AS)),thread_caps={k:os.environ[k] for k in THREADS},affinity=sorted(os.sched_getaffinity(0)),snapshot_semantics=dict(primary_matrices='Actual retained CPU physical D and F from this one public execute; no replacement solve',export_orbitals='New eigendecomposition of retained F during existing snapshot validation; not the lagged primary iteration frame',extra_export_eigensolves=1,extra_export_fock_builds=0,extra_export_scf_solves=0,source_proof='src/methods/dft_method.cpp read_final_state and src/dft/ks_final_state.cpp',outside_energy_only_timing=True))
        save()
        with calc.prepare_batch([xyz],charges=[0],multiplicities=[1],warm_start=False) as batch:
            r['status']='native_running';r['native_execute_calls']=1;save()
            returned=batch.execute(strict=False,properties=('energy',));require(len(returned.items)==1,'Wrong item count');item=returned.items[0]
            fields=('index','status','status_message','energy','converged','iterations','energy_change','density_rms','physical_residual_rms','restart_origin','warm_start_used','warm_start_fallback','executed_backend','precision','initial_guess','basis_metadata')
            r['native_result']={k:getattr(item,k,None) for k in fields};r['native_result']['succeeded']=item.succeeded;r['native_result']['forces_absent']=item.forces is None
            r['ks_diagnostic']=None if item.ks_diagnostic is None else item.ks_diagnostic.to_payload();r['resource_diagnostics']=batch.resource_diagnostics;save()
            require(item.succeeded and item.converged,'Native endpoint failed or did not converge')
            require(item.restart_origin=='cold' and not item.warm_start_used and not item.warm_start_fallback and item.initial_guess is None,'Native owner was seeded/replayed')
            require(item.forces is None and item.executed_backend=='cpu_reference','Unexpected native properties/backend')
            require(0<item.iterations<=150,'Iteration cap changed')
            for name,gate in (('energy_change',1e-12),('density_rms',1e-10),('physical_residual_rms',1e-10)):
                v=getattr(item,name);require(v is not None and math.isfinite(v) and abs(v)<=gate,'Native convergence '+name+' failed')
            require(item.precision and item.precision.get('requested_mode')=='fp64' and item.precision.get('effective_bits')==64,'Not FP64')
            for name in ('execution_retries','fallback_count'):require(item.precision.get(name,0)==0,'Native precision retry/fallback')
            require(r['ks_diagnostic'] is not None,'KS diagnostics absent')
            diagnostic=r['ks_diagnostic']
            require(not diagnostic['initial_density_used'] and diagnostic['grid_points']==331776 and diagnostic['tile_points']==256 and diagnostic['ao_order']==1,'Wrong primary XC work/input controls')
            require(diagnostic['scf_domain']=='semilocal-scaled-v1/pbe-spin-c2-1e-18' and diagnostic['fock_builds']>0,'Wrong primary domain/Fock count')
            r['closure_progress']=closure_progress(r['native_result'],diagnostic,read(m['historical_failed_native']['path']))
            save()
            # Ancillary NativeAO only normalizes/exports the same basis and collocates
            # 60 probes. It performs no molecular solve or integral tensor build.
            with NativeAO(xyz,basis='def2-svp',representation='spherical',charge=0,multiplicity=1) as ao:
                r['status']='snapshot_running';r['snapshot_attempts']=1;save()
                def consume(state):
                    source=state._source
                    require(source.backend=='cpu' and source.metadata[0]==2 and source.metadata[1:4]==(96,1,12),'Unexpected snapshot wire/backend/dimensions')
                    require(source.metadata[5]==331776 and source.metadata[6:8]==(1,1),'Unexpected point count/functional/domain')
                    require(source.energy()==item.energy,'Snapshot energy differs from returned primary result')
                    require(source.fock_provider_proof()==('exact',None,0.0),'Non-exact J/K provider')
                    require(source.coefficients==(1.,1.,0.) and source.hamiltonian=='all-electron','Unexpected composition/Hamiltonian')
                    require(json.loads(json.dumps(dataclasses.asdict(source.grid_spec)))==case['grid_spec'],'Snapshot grid differs')
                    probes=np.vstack([np.asarray(p)+np.asarray([[.137,.293,.419],[-.271,.183,.337],[.311,-.227,.149],[-.199,-.317,.251],[.233,.179,-.283]]) for s,p in xyz])
                    arrays={name:np.array(getattr(state,name),copy=True) for name in ('density','fock','coefficients','orbital_energies','occupations','weighted_density','overlap')}
                    arrays.update(packed_basis=np.array(ao.packed,copy=True),grid_points=np.array(state.grid.points,copy=True),grid_weights=np.array(state.grid.weights,copy=True),grid_owners=np.asarray(state.grid.owners,dtype=np.float64),atomic_weights=np.array(source.atomic_weights,copy=True),snapshot_wire_values=np.array(source.values,copy=True),ao_probe_points=probes,ao_probe_values=np.array(ao.evaluate(probes,order=0)[0],copy=True))
                    r['arrays']={}
                    for name,array in arrays.items():
                        require(np.isfinite(array).all(),'Nonfinite exported '+name);path=out/(name+'.npy');np.save(path,array,allow_pickle=False)
                        r['arrays'][name]=dict(**identity(path),shape=list(array.shape),dtype=str(array.dtype),raw_sha256=hashlib.sha256(array.tobytes(order='C')).hexdigest())
                    r.update(snapshot_metadata=list(source.metadata),snapshot_state_identity=state.identity.to_payload(),snapshot_energy=source.energy(),snapshot_residual=state.physical_residual,live_token_checks_before_and_after_capture=True,detached_files_are_not_live_derivative_leases=True,scientific_export_extra_fock_builds=0)
                    save()
                detached_capture(batch,ao,StationaryKsState.from_native,consume)
            r['snapshot_closed_before_batch']=True
            after=read_ks_diagnostic(batch._library,batch._batch,index=0,expected_domain=calc.ks_options.scf_domain).to_payload()
            require(after==r['ks_diagnostic'],'Snapshot export changed primary KS history/work')
            r['post_snapshot_ks_diagnostic']=after
            r['snapshot_extra_fock_builds_observed']=after['fock_builds']-r['ks_diagnostic']['fock_builds']
            save()
        del batch,calc,item,returned,ao;gc.collect()
        r.update(native_owners_closed=True,status='passed',finished_utc=now(),process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        verify_arm(arm);save()
    except BaseException:
        record_failure(r,status='failed')
        r['process_peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;save();raise
