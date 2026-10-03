"""Portable project reproducer helpers; original numerical gates unchanged."""
import datetime, hashlib, json, math, os, shutil, struct, subprocess, tempfile, traceback
from pathlib import Path
THREADS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')

AS_LIMIT = 8*1024**3

RSS_LIMIT = 13*1024**3//2

MIN_AVAILABLE = 7*1024**3

MIN_DISK = 2*1024**3

TIMEOUT = 900

SETTINGS = dict(method='pbe-rks', basis='def2-svp', device='cpu', basis_representation='spherical',max_iterations=150,energy_tolerance=1e-12,density_tolerance=1e-10)

GATES = dict(energy_error=1e-8,density_error=2e-6,metric_projector_error=2e-6,electron_error=1e-9,idempotency=1e-9,overlap_error=1e-10,independent_fock_error=1e-7,independent_commutator=1e-8,native_physical_residual_rms=1e-9,oracle_commutator=1e-9)

ADDITIONAL_GATES = dict(native_energy_change=1e-12,native_density_rms=1e-10,native_residual_rms=1e-10,oracle_energy_reconstruction=1e-8,independent_energy_consistency=1e-8,density_symmetry=1e-11,coefficient_metric=1e-9,density_from_coefficients=2e-6,weighted_density=1e-9,generalized_eigen_residual=1e-8,ao_alignment=1e-12,grid_atol=1e-11,grid_rtol=1e-10)

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

def require(value, message):
    if not value: raise RuntimeError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def identity(path):
    path=Path(path).resolve();return dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)

def check(info):
    require(sha(info['path'])==info['sha256'],'Identity mismatch: '+info['path'])

def read(path):return json.loads(Path(path).read_text())

def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def diagnostic_json_value(value):
    """Preserve every binary64 diagnostic, including nested NaN/Inf, without NaN JSON.

    Finite numbers remain numbers. Nonfinite tags are evidence, not replacement
    numeric inputs; raw in-memory values still reach every existing gate. Any
    attempted success receipt carrying a nonfinite value is failed closed.
    """
    paths=[]
    def visit(item,path):
        if isinstance(item,float) and not math.isfinite(item):
            bits=struct.unpack('>Q',struct.pack('>d',item))[0]
            fraction=bits & ((1<<52)-1)
            kind='nan' if fraction else 'infinity'
            paths.append(path)
            return {'__nonfinite_float64__':dict(classification=kind,
                sign='negative' if bits>>63 else 'positive',sign_bit=bits>>63,
                encoding='ieee754-binary64-big-endian',bits_hex=f'{bits:016x}',
                quiet_nan=bool(fraction & (1<<51)) if fraction else None)}
        if isinstance(item,dict):
            return {key:visit(val,path+'/'+str(key).replace('~','~0').replace('/','~1')) for key,val in item.items()}
        if isinstance(item,(list,tuple)):
            return [visit(val,path+'/'+str(i)) for i,val in enumerate(item)]
        return item
    result=visit(value,'')
    if paths and isinstance(result,dict):
        previous=result.get('status')
        guard=dict(nonfinite_encoded=True,json_pointer_paths=paths,
            finite_values_unchanged=True,numeric_gates_are_not_satisfied_by_tags=True)
        if previous in ('passed','completed'):
            guard['rejected_success_status']=previous
            result['status']='failed_nonfinite_payload'
        result['serialization_guard']=guard
    return result

def record_failure(data,*,status='failed',error=None):
    """Append failure evidence; never replace a previously captured exception."""
    error=traceback.format_exc() if error is None else error
    data.setdefault('failure_events',[]).append(dict(status_before_exception=data.get('status'),traceback=error,recorded_utc=now()))
    data.setdefault('exception',error)
    data.update(status=status,finished_utc=now())
    return data

def atomic_json(path,value,exclusive=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(diagnostic_json_value(value),f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        if exclusive:os.link(tmp,path);os.unlink(tmp)
        else:os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

def git(source,*args):return subprocess.check_output(['git','-C',str(source),*args],text=True).strip()

def verify_arm(arm):
    source=Path(arm['source']);require(git(source,'rev-parse','HEAD')==arm['revision'],'Revision changed')
    require(git(source,'rev-parse','HEAD^{tree}')==arm['tree'],'Tree changed')
    require(not git(source,'status','--porcelain'),'Source became dirty')
    check(arm['library']);check(arm['compile_commands']);check(arm['cmake_cache'])

def detached_capture(batch,basis,state_factory,consume):
    """Use existing snapshot live checks. No renewed token or derivative-authority claim."""
    state=None
    try:
        state=state_factory(batch,basis)
        state._source.check_current()
        result=consume(state)
        state._source.check_current()
        return result
    finally:
        if state is not None:state._source.close()

def memory_availability():
    fields={s.split(':')[0]:int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith(('MemAvailable:','MemTotal:','SwapTotal:'))}
    require('MemAvailable' in fields,'Memory telemetry missing')
    rel=next((x.split(':',2)[2] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::')),None)
    require(rel is not None,'cgroup v2 telemetry unavailable')
    root=Path('/sys/fs/cgroup');current=root/rel.lstrip('/');limits=[]
    while True:
        if (current/'memory.max').exists():
            maximum=(current/'memory.max').read_text().strip();used=int((current/'memory.current').read_text())
            if maximum!='max':limits.append(dict(path=str(current),limit_bytes=int(maximum),current_bytes=used,remaining_bytes=max(0,int(maximum)-used)))
        if current==root:break
        require(root in current.parents,'Invalid cgroup hierarchy');current=current.parent
    return dict(**fields,effective_available_bytes=min([fields['MemAvailable']]+[i['remaining_bytes'] for i in limits]),visible_cgroup_limits=limits,hidden_ancestor_limits_not_certified=True,checked_utc=now())

def resource_admission(directory):
    mem=memory_availability();disk=shutil.disk_usage(directory)
    require(mem['MemAvailable']>=MIN_AVAILABLE and mem['effective_available_bytes']>=MIN_AVAILABLE,'At least 7 GiB available is required; no launch')
    require(disk.free>=MIN_DISK,'At least 2 GiB free disk is required; no launch')
    return dict(memory=mem,disk_free_bytes=disk.free,minimum_available_bytes=MIN_AVAILABLE,minimum_disk_bytes=MIN_DISK)

def closure_progress(native,diagnostic,historical):
    """Exact historical prefix and bounded work, without claiming hidden audit observables."""
    old=historical['ks_diagnostic']['history'];history=diagnostic['history']
    require(len(old)==17 and historical['native_result']['iterations']==17 and
            historical['ks_diagnostic']['fock_builds']==19,'Historical failure changed')
    require(len(history)>=17 and canonical(history[:17])==canonical(old),'First17 native history rows changed; stop for review')
    iterations=native['iterations'];builds=diagnostic['fock_builds'];extra=builds-iterations
    require(isinstance(iterations,int) and 0<iterations<=150 and len(history)==iterations,'Iteration/history bound changed')
    require([row['iteration'] for row in history]==list(range(1,iterations+1)),'Noncontiguous primary history')
    require(isinstance(builds,int) and extra>=2 and extra%2==0,'Fock count is inconsistent with completed two-build audits')
    attempts=extra//2
    require(attempts<=iterations//2 and builds<=2*150,'Final audit budget inconsistent')
    return dict(first17_history_exact=True,first17_history_sha256=canonical(old),
                primary_iterations=iterations,fock_builds=builds,
                completed_finalization_attempts_inferred=attempts,
                accounting_equation='Fock builds = primary iterations + 2 * completed finalizations',
                accounting_basis='Existing native counters reconciled with source-reviewed and mock-counted controller',
                per_audit_maxima_available=False,per_audit_maxima=None,
                intermediate_audit_arrays_available=False,
                historical_first_rejected_maximum_not_remeasured=True)

def verify_manifest(path,expected):
    require(sha(path)==expected,'Reproducer manifest changed');m=read(path)
    source=m['source'];require(git(source['path'],'rev-parse','HEAD')==source['revision'] and git(source['path'],'rev-parse','HEAD^{tree}')==source['tree'] and not git(source['path'],'status','--porcelain'),'Source identity changed')
    require(m['gates']==GATES and m['additional_gates']==ADDITIONAL_GATES,'Gate changed')
    for row in [m['input'],*m['harness_files'],*m.get('retained_inputs',[])]:check(row)
    for arm in m.get('arms',{}).values():verify_arm(arm)
    return m
