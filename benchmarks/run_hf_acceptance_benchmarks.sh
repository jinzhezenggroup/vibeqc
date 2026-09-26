#!/usr/bin/env bash
# Run direct and fitted HF under the same measured scientific contract.
# Separate processes avoid charging either engine for the other's resident DF B.
set -euo pipefail
: "${SLURM_JOB_ID:?Run through finite srun on main with --gres=gpu:5090:1}"
: "${VIBEQC_LIBRARY:?Select the qualified Release native library}"
hf_python="${HF_BENCHMARK_PYTHON:-python}"
hf_output="${HF_BENCHMARK_OUTPUT:-.artifacts/hf-unified-acceptance}"
hf_sizes="${HF_BENCHMARK_AOS:-24 48 96 192 384 768}"
hf_repeats="${HF_BENCHMARK_REPEATS:-5}"
export PYTHONPATH=".:python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p "$hf_output"
for aos in $hf_sizes; do
    for route in direct df; do
        for mode in reference native; do
            if [[ "$mode" == reference ]]; then
                destination="$hf_output/$aos/reference-$route"
                extra=()
            else
                destination="$hf_output/$aos/$route"
                extra=(--reference "$hf_output/$aos/reference-$route/results.json")
            fi
            mkdir -p "$hf_output/$aos"
            "$hf_python" -m benchmarks.compare_df_direct_endpoint "$mode" \
                --nested-water --aos "$aos" --route "$route" --repeats "$hf_repeats" \
                "${extra[@]}" --output "$destination" > "$destination.log" 2>&1
            printf 'PASS %s %s %s AOs\n' "$mode" "$route" "$aos"
        done
    done
done
