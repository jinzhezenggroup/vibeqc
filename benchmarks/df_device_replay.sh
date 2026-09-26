#!/usr/bin/env bash
# Component counters and real CUDA transfer traces for #205. Run through Slurm.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
: "${SLURM_JOB_ID:?run through a finite Slurm allocation}"
: "${CUDA_VISIBLE_DEVICES:?preserve scheduler-assigned visibility}"
: "${CUDA_ROOT:?set the CUDA toolkit directory}"
: "${VIBEQC_LIBRARY:?set the tested native library}"
: "${PYTHON:?set the Python environment with VibeQC test dependencies}"
export PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:${LD_LIBRARY_PATH:-}"
replay_dir=${OUTPUT_DIR:-/tmp/vibeqc-df-device-${SLURM_JOB_ID}}
mkdir -- "$replay_dir"
"$PYTHON" benchmarks/df_run_provenance.py "$replay_dir"
replay_lib=$(dirname -- "$VIBEQC_LIBRARY")
${CXX:-c++} -std=c++20 -O2 -Iinclude -Isrc -I"$CUDA_ROOT/include" \
  -DVIBEQC_DF_PROBE_PROFILE benchmarks/df_response_probe.cpp \
  -L"$replay_lib" -Wl,-rpath,"$replay_lib" -lvibeqc \
  -L"$CUDA_ROOT/lib64" -Wl,-rpath,"$CUDA_ROOT/lib64" -lcudart -o "$replay_dir/probe"
for replay_spin in rhf uhf; do
  for replay_mode in generated_source generated_resident; do
    for replay_spec in '16384 3' '65536 7'; do
      read -r replay_budget replay_cap <<< "$replay_spec"
      "$replay_dir/probe" sp8 "$replay_mode" "$replay_budget" "$replay_cap" "$replay_spin" 5 \
        > "$replay_dir/component-sp8-$replay_spin-$replay_mode-$replay_budget.json"
    done
  done
done
for replay_spec in '32768 1' '65536 5' '131072 11'; do
  read -r replay_budget replay_cap <<< "$replay_spec"
  "$replay_dir/probe" sdf18 generated_source "$replay_budget" "$replay_cap" rhf 5 \
    > "$replay_dir/component-sdf18-rhf-generated_source-$replay_budget.json"
done
for replay_spin in rhf uhf; do
  replay_trace="$replay_dir/profile-force-$replay_spin"
  "$CUDA_ROOT/bin/nsys" profile --trace=cuda --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop --output "$replay_trace" \
    "$replay_dir/probe" sp8 generated_source 16384 3 "$replay_spin" 5 \
    > "$replay_trace.log" 2>&1
  "$CUDA_ROOT/bin/nsys" export --type=sqlite --output "$replay_trace.sqlite" "$replay_trace.nsys-rep"
done
# This includes SCF control, J/K and all force components for a homogeneous
# batch. The separate component windows above isolate the DF response itself.
replay_trace="$replay_dir/profile-complete-hf"
"$CUDA_ROOT/bin/nsys" profile --trace=cuda --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop --output "$replay_trace" \
  "$PYTHON" benchmarks/profile_one_electron_force.py --case sp8 --batch 3 \
  --mode generated_thread --fitted --df-budget 1048576 --repeats 5 \
  --output "$replay_trace.json" > "$replay_trace.log" 2>&1
"$CUDA_ROOT/bin/nsys" export --type=sqlite --output "$replay_trace.sqlite" "$replay_trace.nsys-rep"
"$PYTHON" benchmarks/summarize_df_device_replay.py "$replay_dir"
