"""Retain #240 build ownership after the complete empty Release build passes."""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT=Path('/home/jzzeng/codes/vibeqc-acceptance-closeout-20260922')
OUT=Path(__file__).resolve().parent
BUILD=ROOT/'build-acceptance'
library=BUILD/'libvibeqc.so'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def actions(log):
    """Count actual compiler/link commands, excluding diagnostics and progress text."""
    commands=[line for line in log.splitlines() if re.match(r'^\[\d+/\d+\]',line)]
    return {
        'command_count':len(commands),
        'cuda_compilations':sum('nvcc ' in line and ' -c ' in line for line in commands),
        'device_links':sum(' -dlink ' in line for line in commands),
        'objects':[m.group(1) for line in commands for m in re.finditer(r' -o (\S+\.o)(?: |$)',line)],
    }


original=digest(library)
report={'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'library_bytes':library.stat().st_size,'library_sha256':original,
        'cold_build_actions':actions((OUT/'build.log').read_text()),'increments':[]}
for owner in ('src/scf/cuda/rhf_graph.cpp','src/scf/cuda/rhf_bucket.cpp','src/scf/cuda_rhf.cpp'):
    path=ROOT/owner
    before=digest(path)
    os.utime(path,None)
    start=time.monotonic()
    result=subprocess.run(['cmake','--build',str(BUILD),'--target','vibeqc','--parallel','8','-v'],
                          cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=300)
    elapsed=time.monotonic()-start
    log=OUT/f'incremental-{path.stem}.log'
    log.write_text(result.stdout)
    assert digest(path)==before
    row={'owner':owner,'seconds':elapsed,'returncode':result.returncode,
         'library_sha256':digest(library),**actions(result.stdout)}
    report['increments'].append(row)
    (OUT/'build-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(row),flush=True)
    assert result.returncode==0 and row['cuda_compilations']==0 and row['device_links']==0
    assert row['library_sha256']==original, 'Content-neutral owner rebuild changed the library'
report['git_status']=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
assert not report['git_status']
report['largest_scf_units']=sorted(
    [{'path':str(p.relative_to(ROOT)),'lines':len(p.read_bytes().splitlines()),'bytes':p.stat().st_size}
     for p in (ROOT/'src/scf').rglob('*') if p.suffix in ('.cpp','.cu','.hpp','.cuh')],
    key=lambda x:x['lines'],reverse=True)[:15]
(OUT/'build-audit.json').write_text(json.dumps(report,indent=2)+'\n')
