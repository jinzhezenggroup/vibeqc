"""Small offline harness controls; no native/AO/J/K/NumInt/SCF calculations."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE=Path(__file__).resolve().parent


def module(name):
    spec=importlib.util.spec_from_file_location(name,HERE/(name+'.py'))
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


class Grid:
    """Model PySCF's resetting policy setter, without a scientific import."""
    def __init__(self):self.coords=None;self.weights=None;self.policy=None
    @property
    def radii_adjust(self):return self.policy
    @radii_adjust.setter
    def radii_adjust(self,value):self.policy=value;self.coords=None;self.weights=None


@pytest.mark.parametrize('change',['replacement','mutation','reset','build'])
def test_explicit_grid_rejects_every_drift(change):
    guard_module=module('explicit_grid');grid=Grid()
    points=np.arange(12,dtype=float).reshape(4,3);weights=np.ones(4)
    guard=guard_module.ExplicitGridGuard(grid,points,weights,expected_points=4)
    assert guard.check('before')['object_identity_preserved']
    if change=='replacement':grid.coords=points.copy()
    elif change=='mutation':grid.weights[0]+=1
    elif change=='reset':grid.radii_adjust=None
    else:
        with pytest.raises(RuntimeError,match='rebuild'):grid.build()
    with pytest.raises(RuntimeError):guard.check('after')


def test_configuration_precedes_array_binding():
    grid=Grid();points=np.arange(12,dtype=float).reshape(4,3);weights=np.ones(4)
    guard=module('explicit_grid').ExplicitGridGuard(grid,points,weights,expected_points=4)
    assert grid.radii_adjust is None
    assert grid.coords is points and grid.weights is weights
    assert guard.check('valid')['build_attempts']==0


def test_serialization_does_not_hide_unexpected_nonfinite(monkeypatch):
    monkeypatch.syspath_prepend(str(HERE));common=module('common')
    r=common.diagnostic_json_value(dict(status='passed',delta=float('inf'),finite=1.23456789012345))
    assert r['status']=='failed_nonfinite_payload'
    assert r['finite']==1.23456789012345
    assert r['delta']['__nonfinite_float64__']['bits_hex']=='7ff0000000000000'
    json.dumps(r,allow_nan=False)


def test_native_mode_refuses_missing_thread_cap_before_science(tmp_path):
    environment=dict(os.environ);environment['OPENBLAS_NUM_THREADS']='2'
    command=[sys.executable,'-B',str(HERE/'reproduce.py'),'native','--source',str(HERE),'--output',str(tmp_path/'must-not-exist'),'--cpu',str(min(os.sched_getaffinity(0))) ]
    run=subprocess.run(command,env=environment,text=True,capture_output=True,timeout=10)
    assert run.returncode!=0 and 'six thread caps' in run.stderr
    assert not (tmp_path/'must-not-exist').exists()


@pytest.mark.parametrize('mutating_hook',[False,True])
def test_reconstruction_checks_full_tree_without_changing_source(tmp_path,mutating_hook):
    source=tmp_path/'source';source.mkdir()
    def git(*args):return subprocess.check_output(['git','-C',str(source),*args],text=True).strip()
    git('init','-q');git('config','user.name','Test');git('config','user.email','test@example.invalid')
    (source/'a').write_text('before\n');(source/'unrelated').write_text('keep\n');git('add','.');git('commit','-qm','base')
    base=git('rev-parse','HEAD');(source/'a').write_text('after\n');git('add','a');tree=git('write-tree');git('commit','-qm','correction')
    import hashlib
    package=tmp_path/'package';package.mkdir();(package/'reconstruct_source.py').write_bytes((HERE/'reconstruct_source.py').read_bytes())
    provenance=dict(measured_source=dict(revision='0'*40,tree=tree,public_base=base,postimages={'a':dict(sha256=hashlib.sha256((source/'a').read_bytes()).hexdigest())}))
    (package/'provenance.json').write_text(json.dumps(provenance))
    destination=tmp_path/'reconstructed';head=git('rev-parse','HEAD')
    if mutating_hook:
        hook=source/'.git/hooks/pre-commit';hook.write_text('#!/bin/sh\necho hook-change > unrelated\ngit add unrelated\n');hook.chmod(0o755)
    run=subprocess.run([sys.executable,str(package/'reconstruct_source.py'),'--repository',str(source),'--destination',str(destination)],text=True,capture_output=True,timeout=20)
    if mutating_hook:
        assert run.returncode!=0 and 'changed the expected clean tree' in run.stderr
        assert (destination/'unrelated').read_text()=='hook-change\n'
    else:
        assert run.returncode==0,run.stderr
        assert json.loads(run.stdout.splitlines()[-1])['tree']==tree
        assert (destination/'unrelated').read_text()=='keep\n'
    assert git('rev-parse','HEAD')==head and not git('status','--porcelain')


def reproducer_module(monkeypatch):
    monkeypatch.syspath_prepend(str(HERE))
    return module('reproduce')


def test_missing_compile_database_fails_before_native_work(tmp_path,monkeypatch):
    repro=reproducer_module(monkeypatch)
    library=tmp_path/'lib.so';library.write_bytes(b'not executed')
    (tmp_path/'CMakeCache.txt').write_text('configuration')
    monkeypatch.setattr(repro,'git',lambda *args:pytest.fail('Read Git before checking prerequisites'))
    with pytest.raises(RuntimeError,match='CMAKE_EXPORT_COMPILE_COMMANDS=ON'):
        repro.build_record(tmp_path,library)


@pytest.mark.parametrize('mutated',[False,True])
def test_oracle_density_honors_receipt_hash(tmp_path,monkeypatch,mutated):
    repro=reproducer_module(monkeypatch)
    path=tmp_path/'density.npy';path.write_bytes(b'original density')
    expected=repro.identity(path)['sha256']
    if mutated:path.write_bytes(b'changed density')
    oracle=dict(samples=[dict(density_file=str(path),density_sha256=expected)])
    if mutated:
        with pytest.raises(RuntimeError,match='declared receipt hash'):repro.declared_oracle_densities(oracle)
    else:assert repro.declared_oracle_densities(oracle)[0]['sha256']==expected


def test_oracle_receipt_binding_and_final_recheck_are_present():
    import ast
    tree=ast.parse((HERE/'oracle_state.py').read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    calls=[n for n in ast.walk(function) if isinstance(n,ast.Call)]
    verify=[n for n in calls if isinstance(n.func,ast.Name) and n.func.id=='verify_manifest']
    kernel=next(n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='kernel')
    save=next(n for n in calls if isinstance(n.func,ast.Name) and n.func.id=='atomic_json')
    assert len(verify)==2 and verify[0].lineno<kernel.lineno<verify[1].lineno<save.lineno
    result=save.args[1]
    names={k.arg for k in result.keywords}
    assert {'manifest_sha256','input_sha256','source'}<=names


@pytest.mark.parametrize('changed',['none','native_input','native_tree','oracle_input','oracle_tree','missing_fresh_identity'])
def test_independent_intake_requires_explicit_matching_fresh_provenance(monkeypatch,changed):
    repro=reproducer_module(monkeypatch)
    manifest=dict(input=dict(sha256='input'),source=dict(tree='tree'))
    native=dict(input_sha256='input',identity=dict(tree='tree'))
    oracle=dict(schema='pbe96-fresh-reproduction-oracle-v1',input_sha256='input',source=dict(tree='tree',revision='different-commit-same-tree'),manifest_sha256='a'*64)
    if changed=='native_input':native['input_sha256']='wrong'
    elif changed=='native_tree':native['identity']['tree']='wrong'
    elif changed=='oracle_input':oracle['input_sha256']='wrong'
    elif changed=='oracle_tree':oracle['source']['tree']='wrong'
    elif changed=='missing_fresh_identity':del oracle['schema']
    if changed=='none':repro.validate_independent_intake(native,oracle,manifest)
    else:
        with pytest.raises(RuntimeError):repro.validate_independent_intake(native,oracle,manifest)
