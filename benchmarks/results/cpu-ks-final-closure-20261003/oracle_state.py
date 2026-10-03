"""Fresh, separately requested oracle generation; independent audit never calls SCF."""
import os, resource, sys
from pathlib import Path
from common import AS_LIMIT, THREADS, atomic_json, identity, read, require, verify_manifest
from explicit_grid import ExplicitGridGuard


def main(a):
    require(not any(x in sys.modules for x in ('numpy','pyscf','generativeqc')), 'Premature scientific import')
    resource.setrlimit(resource.RLIMIT_AS,(AS_LIMIT,AS_LIMIT))
    require(sorted(os.sched_getaffinity(0))==[a.cpu], 'Affinity mismatch')
    require(all(os.environ.get(k)=='1' for k in THREADS), 'Thread cap mismatch')
    m=verify_manifest(a.manifest,a.manifest_sha256);case=read(m['input']['path'])
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import pyscf
    from pyscf import dft,gto,lib
    from generativeqc import Atom
    from generativeqc_compiler.dft.grid import MolecularGrid,GridSpec
    require(pyscf.__version__=='2.14.0','Use measured PySCF 2.14.0')
    lib.num_threads(1)
    xyz=case['atoms_bohr'];labels=[s+str(i) for i,(s,p) in enumerate(xyz)]
    mol=gto.M(atom=[(label,p) for label,(s,p) in zip(labels,xyz,strict=True)],basis=case['exact_primitive_input'],unit='Bohr',cart=False,charge=0,spin=0,verbose=0)
    gdict=dict(case['grid_spec']);gdict['element_radii']=tuple(tuple(x) for x in gdict['element_radii'])
    grid=MolecularGrid(tuple(Atom.from_value(x) for x in xyz),spec=GridSpec(**gdict)).explicit(max_points=1_000_000)
    mf=dft.RKS(mol);mf.xc='PBE';mf.small_rho_cutoff=0
    guard=ExplicitGridGuard(mf.grids,np.asarray(grid.points),np.asarray(grid.weights))
    mf.max_cycle=150;mf.conv_tol=1e-13;mf.conv_tol_grad=1e-10;mf.init_guess='1e'
    checks=[guard.check('before_scf')]
    mf.kernel()
    checks.append(guard.check('after_scf'))
    dm=mf.make_rdm1();fock=mf.get_fock(dm=dm);overlap=mf.get_ovlp()
    residual=fock@dm@overlap-overlap@dm@fock
    path=out/'density.npy';np.save(path,dm,allow_pickle=False)
    sample=dict(phase='cold',energy=float(mf.e_tot),converged=bool(mf.converged),iterations=int(mf.cycles),physical_residual_rms=float(np.sqrt(np.mean(residual**2))),density_file=str(path),density_sha256=identity(path)['sha256'])
    passed=sample['converged'] and sample['physical_residual_rms']<1e-9
    verify_manifest(a.manifest,a.manifest_sha256)
    atomic_json(out/'result.json',dict(schema='pbe96-fresh-reproduction-oracle-v1',manifest_sha256=a.manifest_sha256,input_sha256=m['input']['sha256'],source=m['source'],status='completed' if passed else 'failed',scope='fresh reproduction oracle, not original retained oracle',scf_kernel_calls=1,native_execute_calls=0,pyscf_version=pyscf.__version__,grid_identity_checks=checks,samples=[sample]))
    require(passed,'Fresh independent oracle convergence failed')
