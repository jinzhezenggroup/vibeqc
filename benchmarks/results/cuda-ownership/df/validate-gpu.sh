#!/usr/bin/env bash
# Run through a finite srun allocation. Every output directory must be new.
set -euo pipefail
: "${SLURM_JOB_ID:?Run this script through Slurm}"
: "${CUDA_VISIBLE_DEVICES:?Preserve Slurm device visibility}"
: "${ROOT:?Supply the exact clean candidate checkout}"
: "${BUILD:?Supply its matching optimized CUDA build}"
: "${VALIDATION_ROOT:?Supply the exact clean validation checkout recorded in validation.json}"
: "${BASELINE_ROOT:?Supply the exact clean baseline checkout}"
: "${BASELINE_BUILD:?Supply its matching optimized CUDA build}"
: "${OUTPUT_DIR:?Supply a new output directory}"
PYTHON=${PYTHON:-python3}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
mkdir "$OUTPUT_DIR"
cd "$ROOT"
export PYTHONPATH="$ROOT/python:$ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="$BUILD:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VIBEQC_LIBRARY="$BUILD/libvibeqc.so"
"$PYTHON" - "$OUTPUT_DIR" <<'PY'
import ctypes
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy
import pyscf

from vibeqc.autotune import source_identity

validation_root = Path(os.environ["VALIDATION_ROOT"])
for root in (Path.cwd(), validation_root):
    if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"]).strip():
        raise RuntimeError("validation requires clean endpoint and validation checkouts")
lib = ctypes.CDLL(os.environ["VIBEQC_LIBRARY"])
lib.vibeqc_get_source_identity.restype = ctypes.c_char_p
identity = lib.vibeqc_get_source_identity().decode()
if identity != source_identity(Path.cwd()) or identity != source_identity(validation_root):
    raise RuntimeError("candidate library does not match its source checkout")
record = {
    "revision": subprocess.check_output(["git", "-C", str(validation_root), "rev-parse", "HEAD"], text=True).strip(),
    "endpoint_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "validation_scope": "Six complete suites in separate processes; original exact checkpoint assertion uses its documented serial one-electron response. Production inputs and library match the endpoint source.",
    "native_source_identity": identity,
    "library_sha256": hashlib.sha256(Path(os.environ["VIBEQC_LIBRARY"]).read_bytes()).hexdigest(),
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    "python": platform.python_version(),
    "numpy": numpy.__version__,
    "pyscf": pyscf.__version__,
    "gpu": subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"
    ], text=True).strip(),
}
(Path(sys.argv[1]) / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")
PY
"$PYTHON" tools/benchmark_cuda_ownership.py compare --domain df --process-scope case \
  --baseline-root "$BASELINE_ROOT" --baseline-build "$BASELINE_BUILD" \
  --candidate-root "$ROOT" --candidate-build "$BUILD" --samples 5 \
  --output "$OUTPUT_DIR/endpoints" > "$OUTPUT_DIR/endpoints.log" 2>&1
ctest --test-dir "$BUILD" --output-on-failure > "$OUTPUT_DIR/native.log" 2>&1
export VIBEQC_DF_DERIVATIVE_CUDA_TEST=1 VIBEQC_ONE_ELECTRON_DERIVATIVE_CUDA_TEST=1
export VIBEQC_ONE_ELECTRON_CUDA_TEST=1 VIBEQC_RESOURCE_CUDA_TEST=1
export VIBEQC_CHECKPOINT_DEVICE=cuda VIBEQC_PROJECTION_CUDA_TEST=1
# Release each suite's CUDA context before checkpoint subprocesses start.
cd "$VALIDATION_ROOT"
export PYTHONPATH="$VALIDATION_ROOT/python:$VALIDATION_ROOT"
"$PYTHON" - "$OUTPUT_DIR" <<'PY'
import hashlib,json,subprocess,sys
from pathlib import Path
from xml.etree import ElementTree
out=Path(sys.argv[1]); combined=ElementTree.Element('testsuites'); records=[]; executed=[]
names=('df_derivatives_cuda','one_electron_derivatives_cuda','one_electron_values_cuda','hf_resources_cuda','checkpoint','basis_projection_cuda')
files=['tests/python/test_'+name+'.py' for name in names]
with (out/'gpu-collected.log').open('w') as stream:
 subprocess.run([sys.executable,'-m','pytest',*files,'--collect-only','-q'],stdout=stream,stderr=subprocess.STDOUT,check=True)
collected=[line.strip() for line in (out/'gpu-collected.log').read_text().splitlines() if line.startswith('tests/python/') and '::' in line]
for name in names:
 path=Path('tests/python/test_'+name+'.py'); xml=out/('gpu-'+name+'.xml'); log=out/('gpu-'+name+'.log')
 with log.open('w') as stream:
  subprocess.run([sys.executable,'-m','pytest',str(path),'-x','-q','--junitxml='+str(xml)],stdout=stream,stderr=subprocess.STDOUT,check=True)
 tree=ElementTree.parse(xml).getroot(); suites=list(tree.iter('testsuite'))
 assert suites and all(int(s.attrib['errors'])==int(s.attrib['failures'])==int(s.attrib['skipped'])==0 for s in suites)
 for suite in suites:
  combined.append(suite)
  executed += [case.attrib['classname'].replace('.','/')+'.py::'+case.attrib['name'] for case in suite.iter('testcase')]
 records.append({'source':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'xml':xml.name,'log':log.name,'tests':sum(int(s.attrib['tests']) for s in suites)})
 (out/'gpu-suites.json').write_text(json.dumps(records,indent=2)+'\n')
 print(name,records[-1]['tests'],'passed',flush=True)
assert collected and len(collected)==len(set(collected))==len(executed)
assert sorted(collected)==sorted(executed)
(out/'gpu-inventory.json').write_text(json.dumps({'collected':collected,'executed':executed,'complete':True},indent=2)+'\n')
ElementTree.ElementTree(combined).write(out/'gpu.xml',encoding='unicode',xml_declaration=True)
(out/'gpu.log').write_text(''.join('=== '+r['source']+' ===\n'+(out/r['log']).read_text() for r in records))
PY
"$PYTHON" tools/validate_df_source.py --probe "$BUILD/vibeqc_df_value_probe" \
  --directory "$OUTPUT_DIR/raw-source" --derivatives > "$OUTPUT_DIR/raw-source.log" 2>&1
for sanitizer in memcheck synccheck; do
  "$CUDA_HOME/bin/compute-sanitizer" --tool "$sanitizer" --error-exitcode 99 \
    "$PYTHON" -m pytest tests/python/test_df_derivatives_cuda.py \
    -k cooperative_sparse_weights_and_ragged_primitive_partitions -x -q \
    > "$OUTPUT_DIR/$sanitizer.log" 2>&1
done
