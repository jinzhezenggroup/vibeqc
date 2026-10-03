"""Execution-gated PySCF fixed-D audit after native process has exited; no SCF."""
import argparse, os, resource, sys, traceback
from common import *
from grid_audit import independent_same_coordinate_weights
from explicit_grid import ExplicitGridGuard

def main(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    r=dict(schema='pbe96-independent-same-coordinate-audit-v2',qualification_scope='same-coordinate weights and unchanged fixed-D physical/oracle gates; original cross-node audit remains failed',original_cross_node_audit_status='failed',native_execute_calls=0,status='starting',arm=a.arm,phase=a.phase,performance_measurement=False,scf_kernel_calls=0,started_utc=now())
    save=lambda:atomic_json(out/'result.json',r)
    save()
    try:
        release=dict(cpu=a.cpu)
        require(not any(x in sys.modules for x in ('numpy','pyscf','generativeqc')),'Imports preceded AS cap')
        resource.setrlimit(resource.RLIMIT_AS,(AS_LIMIT,AS_LIMIT))
        require(sorted(os.sched_getaffinity(0))==[release['cpu']],'Unexpected affinity')
        require(all(os.environ.get(k)=='1' for k in THREADS),'Thread caps changed')
        m=verify_manifest(a.manifest,a.manifest_sha256);case=read(m['input']['path']);native=read(Path(a.native)/'result.json')
        supervisor=read(a.native_supervisor)
        require(supervisor['status']=='completed' and supervisor['returncode']==0,'Native process did not finish successfully')
        require(native['status']=='passed' and native['native_owners_closed'] is True and native['arm']==a.arm and native['phase']==a.phase,'Unaccepted primary state')
        for record in native['arrays'].values():check(record)
        import numpy as np
        import pyscf
        from pyscf import gto,dft,scf,lib
        from generativeqc import Atom
        from generativeqc_compiler.dft.grid import MolecularGrid,GridSpec
        # Imports above are pure Python contracts; never call Calculator/NativeAO/native library.
        lib.num_threads(1);require(pyscf.__version__=='2.14.0','PySCF differs from accepted oracle')
        arrays={name:np.load(info['path'],allow_pickle=False) for name,info in native['arrays'].items()}
        D=arrays['density'][0];F=arrays['fock'][0];C=arrays['coefficients'][0];eps=arrays['orbital_energies'][0];occ=arrays['occupations'][0];W=arrays['weighted_density'][0];S_native=arrays['overlap']
        require(D.shape==(96,96) and F.shape==(96,96),'Primary array shape')
        xyz=case['atoms_bohr' if a.phase=='original' else 'moved_atoms_bohr'];labels=[s+str(i) for i,(s,p) in enumerate(xyz)];exact=case['exact_primitive_input']
        mol=gto.M(atom=[(label,p) for label,(s,p) in zip(labels,xyz,strict=True)],basis=exact,unit='Bohr',cart=False,charge=0,spin=0,verbose=0,max_memory=1024)
        mol.incore_anyway=False
        require((mol.nao_nr(),mol.nao_cart(),mol.nelectron,mol.natm)==(96,100,40,12),'Independent dimensions differ')
        require([(mol.bas_atom(i),mol.bas_angular(i),mol.bas_nprim(i)) for i in range(mol.nbas)]==[(i,q[0],len(q)-1) for i,label in enumerate(labels) for q in exact[label]],'Independent shell order differs')
        maximum=lambda x:float(np.max(np.abs(x)))
        ao_ref=mol.eval_gto('GTOval_sph',arrays['ao_probe_points'])
        ao_error=maximum(ao_ref-arrays['ao_probe_values']);require(ao_error<=ADDITIONAL_GATES['ao_alignment'],'Independent AO ordering/normalization failed')
        gdict=dict(case['grid_spec']);gdict['element_radii']=tuple(tuple(x) for x in gdict['element_radii'])
        oracle_topology=MolecularGrid(tuple(Atom.from_value(x) for x in xyz),spec=GridSpec(**gdict))
        oracle_grid=oracle_topology.explicit(max_points=1_000_000)
        grid_checks={}
        for key,expected in (('grid_points',oracle_grid.points),('grid_weights',oracle_grid.weights),('grid_owners',oracle_grid.owners)):
            actual=arrays[key];expected=np.asarray(expected)
            ok=np.array_equal(actual,expected) if key=='grid_owners' else np.allclose(actual,expected,atol=ADDITIONAL_GATES['grid_atol'],rtol=ADDITIONAL_GATES['grid_rtol'])
            grid_checks[key]=dict(bitwise_equal=actual.shape==expected.shape and actual.tobytes()==expected.astype(actual.dtype).tobytes(),maximum_absolute_error=maximum(actual-expected),passed=bool(ok))
            # Retain the exact original cross-node comparison and its failure.
            # Points and owners remain required; weights get a separately named
            # same-coordinate construction, never a relaxed original gate.
            r['original_grid_checks']=grid_checks;save()
            if key!='grid_weights':require(ok,'Native/historical-generator grid mismatch: '+key)
        r.update(original_grid_checks=grid_checks,original_cross_node_grid_gate_passed=all(x['passed'] for x in grid_checks.values()),retained_failed_independent=m['retained_failed_independent'])
        save()
        raw_reference,weight_reference=independent_same_coordinate_weights(oracle_topology,arrays['grid_points'],arrays['grid_owners'])
        same_coordinate_checks={}
        for key,expected in (('atomic_weights',raw_reference),('grid_weights',weight_reference)):
            actual=arrays[key]
            ok=np.allclose(actual,expected,atol=ADDITIONAL_GATES['grid_atol'],rtol=ADDITIONAL_GATES['grid_rtol'])
            same_coordinate_checks[key]=dict(maximum_absolute_error=maximum(actual-expected),passed=bool(ok),expected_construction='Independent Python raw radial/angular quadrature; for molecular weights multiply Python Becke at unchanged native coordinates; native weights never used')
            r['same_coordinate_grid_checks']=same_coordinate_checks;save()
            require(ok,'Independent same-coordinate grid mismatch: '+key)
        r['same_coordinate_grid_checks']=same_coordinate_checks;save()
        mf=dft.RKS(mol);mf.xc='PBE';mf.max_memory=1024;mf.direct_scf=True;mf.direct_scf_tol=1e-13;mf.small_rho_cutoff=0
        native_guard=ExplicitGridGuard(mf.grids,arrays['grid_points'],arrays['grid_weights'])
        r['grid_identity_checks']=[native_guard.check('native_after_binding')];save()
        ni=dft.numint.NumInt();S=mf.get_ovlp();H=mf.get_hcore();Enuc=float(mol.energy_nuc())
        # Call independent full-density J/K explicitly, never incremental get_veff
        # or kernel. K is recorded but PBE's coefficient remains exactly zero.
        def audit_density(dm,grid,guard,label):
            r['grid_identity_checks'].append(guard.check(label+'_before_physics'));save()
            require(mf._eri is None,'Unexpected in-core ERI cache')
            J,K=scf.hf.SCF.get_jk(mf,mol=mol,dm=dm,hermi=1,with_j=True,with_k=True)
            require(mf._eri is None,'Independent J/K allocated retained dense ERIs')
            r['grid_identity_checks'].append(guard.check(label+'_before_numint'));save()
            nexc,Exc,V=ni.nr_rks(mol,grid,'PBE',dm,hermi=1,max_memory=256)
            r['grid_identity_checks'].append(guard.check(label+'_after_numint'));save()
            f=H+J+V
            energy=float(np.einsum('ij,ji->',dm,H)+.5*np.einsum('ij,ji->',dm,J)+Exc+Enuc)
            return dict(J=J,K=K,Vxc=V,F=f,Exc=float(Exc),grid_electron_count=float(nexc),energy=energy,comm=f@dm@S-S@dm@f)
        actual=audit_density(D,mf.grids,native_guard,'native')
        oracle=read(m['oracle']['path']);sample=next(x for x in oracle['samples'] if x['phase']==('cold' if a.phase=='original' else 'changed_geometry'))
        D0=np.load(sample['density_file'],allow_pickle=False)
        grid0=dft.gen_grid.Grids(mol)
        oracle_guard=ExplicitGridGuard(grid0,np.array(oracle_grid.points),np.array(oracle_grid.weights))
        r['grid_identity_checks'].append(oracle_guard.check('oracle_after_binding'));save()
        original=audit_density(D0,grid0,oracle_guard,'oracle')
        e,U=np.linalg.eigh(S);require(np.all(e>0),'Independent overlap not positive definite');root=(U*np.sqrt(e))@U.T
        metric=root@(D-D0)@root/2
        metrics=dict(energy_error=abs(native['native_result']['energy']-sample['energy']),density_error=maximum(D-D0),metric_projector_error=maximum(metric),electron_error=abs(float(np.trace(D@S))-40),idempotency=maximum(D@S@D-2*D),overlap_error=maximum(S-S_native),independent_fock_error=maximum(actual['F']-F),independent_commutator=maximum(actual['comm']),native_physical_residual_rms=float(native['native_result']['physical_residual_rms']),oracle_commutator=maximum(original['comm']))
        extra=dict(oracle_energy_reconstruction=abs(original['energy']-sample['energy']),independent_energy_consistency=abs(actual['energy']-native['native_result']['energy']),density_symmetry=maximum(D-D.T),coefficient_metric=maximum(C.T@S@C-np.eye(96)),density_from_coefficients=maximum((C*occ)@C.T-D),weighted_density=maximum((C*(eps*occ))@C.T-W),generalized_eigen_residual=maximum(F@C-(S@C)*eps),ao_alignment=ao_error)
        r.update(status='physics_evaluated_arrays_pending',metrics=metrics,additional_metrics=extra,tail_observation_complete=False);save()
        outputs=dict(independent_overlap=S,independent_hcore=H,independent_coulomb=actual['J'],independent_exchange_unused=actual['K'],independent_vxc=actual['Vxc'],independent_fock=actual['F'],independent_commutator=actual['comm'],oracle_density=D0,oracle_fock=original['F'],oracle_commutator=original['comm'],metric_projector_difference=metric,independent_ao_probes=ao_ref)
        outputs.update(historical_grid_points=np.asarray(oracle_grid.points),historical_grid_weights=np.asarray(oracle_grid.weights),historical_grid_owners=np.asarray(oracle_grid.owners,dtype=np.float64),independent_atomic_weights=raw_reference,independent_same_coordinate_weights=weight_reference)
        r['arrays']={}
        for name,array in outputs.items():
            require(np.isfinite(array).all(),'Nonfinite independent '+name);path=out/(name+'.npy');np.save(path,array,allow_pickle=False)
            r['arrays'][name]=dict(**identity(path),shape=list(array.shape),dtype=str(array.dtype),raw_sha256=hashlib.sha256(array.tobytes(order='C')).hexdigest())
        failed=[k for k,v in metrics.items() if not math.isfinite(v) or v>GATES[k]]+[k for k,v in extra.items() if not math.isfinite(v) or v>ADDITIONAL_GATES[k]]
        r.update(status='physics_evaluated_tail_pending' if not failed else 'scientific_gates_failed',failed_gates=failed,metrics=metrics,additional_metrics=extra,grid_checks=grid_checks,tail_observation_complete=False,native_receipt=identity(Path(a.native)/'result.json'),native_supervisor=identity(a.native_supervisor),manifest_sha256=a.manifest_sha256,independent_controls=dict(scf_kernel_calls=0,fixed_density_audits=2,full_density_JK_calls=2,numint_calls=2,direct_scf_tol=1e-13,max_memory_mb=1024,NumInt_max_memory_mb=256,exact_exchange_coefficient=0.,in_core_eri_cache_absent=mf._eri is None),energy_native=native['native_result']['energy'],energy_independent_at_native_D=actual['energy'],energy_retained_oracle=sample['energy'],energy_reconstructed_retained_oracle=original['energy'],Exc_at_native_D=actual['Exc'],grid_electron_count_at_native_D=actual['grid_electron_count'],native_owner_released_before_independent_process=True,physics_checkpoint_utc=now(),process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        save();require(not failed,'Predeclared scientific gates failed')
        r['grid_identity_checks'].extend((native_guard.check('native_before_tail'),oracle_guard.check('oracle_before_tail')));save()
        # Retain the unmodified NumInt/Libxc tail policy and count the original
        # density domain. These observations do not enlarge any gate or clip rho.
        tail=dict(points=0,zero_density_points=0,negative_density_points=0,below_1e_18=0,below_1e_12=0,minimum_density=None,maximum_density=None,absolute_weighted_electrons_below_1e_12=0.)
        r['tail_domain']=tail;save()
        for start in range(0,331776,256):
            pts=mf.grids.coords[start:start+256];weight=mf.grids.weights[start:start+256]
            expected_points=min(256,331776-start)
            r['tail_progress']=dict(start=start,expected_points=expected_points,point_shape=list(pts.shape),weight_shape=list(weight.shape))
            require(pts.shape==(expected_points,3) and weight.shape==(expected_points,) and expected_points>0,'Tail tile inventory changed')
            ao=ni.eval_ao(mol,pts,deriv=1);rho=ni.eval_rho(mol,ao,D,xctype='GGA',hermi=1)[0]
            r['tail_progress']['rho_shape']=list(rho.shape)
            require(rho.shape==(expected_points,),'Tail density tile inventory changed')
            tail['points']+=len(rho);tail['zero_density_points']+=int(np.count_nonzero(rho==0));tail['negative_density_points']+=int(np.count_nonzero(rho<0))
            for threshold,key in ((1e-18,'below_1e_18'),(1e-12,'below_1e_12')):tail[key]+=int(np.count_nonzero((rho>0)&(rho<threshold)))
            lo=float(np.min(rho));hi=float(np.max(rho));tail['minimum_density']=lo if tail['minimum_density'] is None else min(tail['minimum_density'],lo);tail['maximum_density']=hi if tail['maximum_density'] is None else max(tail['maximum_density'],hi)
            mask=(rho>=0)&(rho<1e-12);tail['absolute_weighted_electrons_below_1e_12']+=float(np.sum(np.abs(weight[mask]*rho[mask])))
        require(tail['points']==331776,'Tail did not cover the full frozen point inventory')
        r['grid_identity_checks'].extend((native_guard.check('native_after_tail'),oracle_guard.check('oracle_after_tail')));save()
        tail.update(native_domain='semilocal-scaled-v1/pbe-spin-c2-1e-18',spin_policy='RKS positive spin fractions both 1/2; C2 minority-spin extension inactive. Exact vacuum native zero; Libxc/AO screening remains its independently versioned default.',reference_policy='Unmodified PySCF 2.14.0 NumInt/PBE Libxc with full explicit grids and small_rho_cutoff=0; this disables grid pruning, not all Libxc/AO low-density screening.',density_clipping=False,libxc_version=str(getattr(dft.libxc,'__version__','unavailable')),numint_cutoff=float(ni.cutoff),grid_cutoff=float(mf.grids.cutoff),tail_domain_bitwise_equivalence_claim=False)
        r.update(status='passed',tail_domain=tail,tail_observation_complete=True,finished_utc=now(),process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        verify_manifest(a.manifest,a.manifest_sha256)
        save();require(not failed,'Predeclared scientific gates failed')
    except BaseException:
        record_failure(r,status='scientific_gates_failed' if r['status']=='scientific_gates_failed' else 'failed')
        r['process_peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;save();raise
