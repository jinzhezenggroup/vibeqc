#!/usr/bin/env bash
set -euo pipefail
repository_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
cd -- "$repository_root"
: "${SLURM_JOB_ID:?run through a finite Slurm allocation}"
: "${CUDA_VISIBLE_DEVICES:?preserve scheduler-assigned visibility}"
printf 'SLURM_JOB_ID=%s CUDA_VISIBLE_DEVICES=%s\n' "$SLURM_JOB_ID" "$CUDA_VISIBLE_DEVICES"
export PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VIBEQC_LIBRARY="${VIBEQC_LIBRARY:-$PWD/build-cuda/libvibeqc.so}"
if [[ -n "${CUDA_HOME:-}" ]]; then
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi
PYTHON=${PYTHON:-python}
# Fail if a run directory already exists; archived evidence is never overwritten.
archive143=${OUTPUT_DIR:-${TMPDIR:-/tmp}/vibeqc-df-${SLURM_JOB_ID}-components}
mkdir -p -- "$(dirname -- "$archive143")"
mkdir -- "$archive143"
printf 'OUTPUT_DIR=%s\n' "$archive143"
"$PYTHON" benchmarks/df_run_provenance.py "$archive143"

CXX=${CXX:-c++}
probe143=$(mktemp -d)
trap 'rm -rf -- "$probe143"' EXIT
library143=$(dirname -- "$VIBEQC_LIBRARY")
"$CXX" -std=c++20 -O2 -Iinclude -Isrc benchmarks/df_response_probe.cpp -L"$library143" -Wl,-rpath,"$library143" -lvibeqc -o "$probe143/probe"
for spin143 in rhf uhf; do
 for mode143 in reference_resident reference_source generated_resident generated_source; do
  for spec143 in '16384 3' '65536 7'; do
   read -r budget143 cap143 <<< "$spec143"
   "$probe143/probe" sp8 "$mode143" "$budget143" "$cap143" "$spin143" 5 > "$archive143/component-sp8-$spin143-$mode143-$budget143.json"
  done
 done
done
for mode143 in reference_resident reference_source generated_resident generated_source; do
 for spec143 in '65536 5' '131072 11'; do
  read -r budget143 cap143 <<< "$spec143"
  "$probe143/probe" sdf18 "$mode143" "$budget143" "$cap143" rhf 5 > "$archive143/component-sdf18-rhf-$mode143-$budget143.json"
 done
done
# Keep all measured records together so the evidence does not overwhelm code
# review file inventories. Keys retain each original per-command filename.
"$PYTHON" - "$archive143" <<'PY'
import json
import pathlib
import sys

archive = pathlib.Path(sys.argv[1])
files = sorted(archive.glob("component-*.json"))
if len(files) != 24:
    raise RuntimeError("expected all 24 component records")
payload = {"schema": "vibeqc.df_response_components", "records": {
    path.name: json.loads(path.read_text()) for path in files
}}
(archive / "components.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
for path in files:
    path.unlink()
PY
