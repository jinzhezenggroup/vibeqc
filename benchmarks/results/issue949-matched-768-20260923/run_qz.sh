#!/usr/bin/env bash
# One fixed-source H100 build or matched 768-AO endpoint campaign per Inspire Job.
set -Eeuo pipefail

phase=${1:?use probe, build or measure}
source_dir=${VIBEQC_SOURCE_DIR:?absolute fixed-source checkout required}
run_dir=${VIBEQC_RUN_DIR:?absolute evidence directory required}
source_sha=${VIBEQC_SOURCE_SHA:?full source commit required}
allocation=${VIBEQC_BENCHMARK_ALLOCATION:?actual Job name required}

case "$phase" in probe|build|measure) ;; *) exit 64 ;; esac
[[ "$source_dir" = /* && "$run_dir" = /* && "$source_sha" =~ ^[0-9a-f]{40}$ ]]
[[ "$(git -C "$source_dir" rev-parse HEAD)" = "$source_sha" ]]
[[ -z "$(git -C "$source_dir" status --porcelain -- src python benchmarks)" ]]
mkdir -p "$run_dir"
cd "$source_dir"

finish() {
    local status=$?
    printf '%s\n' "$status" > "$run_dir/$phase.exit.tmp"
    mv "$run_dir/$phase.exit.tmp" "$run_dir/$phase.exit"
    printf '%s phase=%s exit=%s\n' "$(date -Is)" "$phase" "$status"
}
trap finish EXIT

printf 'phase=%s source=%s allocation=%s started=%s\n' "$phase" "$source_sha" "$allocation" "$(date -Is)"
nvidia-smi --query-gpu=name,uuid,memory.total,compute_cap --format=csv,noheader
python3 -c 'import numpy; print("numpy", numpy.__version__)'
command -v cmake ninja nvcc python3

if [[ "$phase" = probe ]]; then
    python3 -m py_compile benchmarks/df_policy_endpoint.py \
        benchmarks/results/issue949-matched-768-20260923/reuse_cpu_oracle.py
    exit 0
fi

build_dir="$run_dir/build-sm90"
if [[ "$phase" = build ]]; then
    cmake -S "$source_dir" -B "$build_dir" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DVIBEQC_ENABLE_CUDA=ON \
        -DVIBEQC_CUDA_ARCHITECTURES=90 -DVIBEQC_CUDA_FAST_COMPILE=OFF \
        -DVIBEQC_CUDA_SEPARABLE_COMPILATION=OFF \
        -DVIBEQC_AOT_UNIT_MODE=stable-shards -DVIBEQC_AOT_PROFILE=auto
    cmake --build "$build_dir" --target vibeqc --parallel 4
    sha256sum "$build_dir/libvibeqc.so"
    exit 0
fi

[[ -f "$run_dir/build.exit" && "$(cat "$run_dir/build.exit")" = 0 ]]
export VIBEQC_LIBRARY="$build_dir/libvibeqc.so"
export VIBEQC_DF_RESPONSE_STORAGE=panel
export PYTHONPATH="$source_dir/python:$source_dir"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 benchmarks/results/issue949-matched-768-20260923/reuse_cpu_oracle.py \
    --aos 768 --repeats 1 --cpu-reference --expected-iterations 6 \
    --control VIBEQC_DF_RESPONSE_ALGEBRA --policies blas scalar \
    --cold-control VIBEQC_DF_RESPONSE_ALGEBRA=blas \
    --warm-checkpoint-out "$run_dir/frozen-warm.checkpoint" \
    --output "$run_dir/matched-768.json"
