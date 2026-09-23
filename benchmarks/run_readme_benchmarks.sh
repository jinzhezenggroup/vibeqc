#!/usr/bin/env bash
# Execute inside one finite Slurm allocation; preserve its device visibility.
set -euo pipefail
: "${SLURM_JOB_ID:?Run with srun --partition=main --gres=gpu:5090:1 --time=01:00:00}"
: "${VIBEQC_LIBRARY:?Select a Release native library built from the recorded source}"
readme_python="${README_BENCHMARK_PYTHON:-python}"
readme_output="${README_BENCHMARK_OUTPUT:-.artifacts/readme-benchmarks-20260922}"
readme_timeout="${README_BENCHMARK_POINT_TIMEOUT:-900}"
[[ "$readme_timeout" =~ ^[1-9][0-9]*$ ]] || { printf 'Invalid point timeout\n' >&2; exit 2; }
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONPATH=".:python${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$readme_output"/{hf,dft,dft-reference,cc}
readme_group="${1:-all}"
case "$readme_group" in
    all|hf|dft|dft-reference|dft-paired|cc|smoke) ;;
    *) printf 'Unknown benchmark group: %s\n' "$readme_group" >&2; exit 2 ;;
esac
readme_failed=0
if [[ "$readme_group" == dft* ]]; then
    readme_timeout="${README_BENCHMARK_POINT_TIMEOUT:-120}"
fi

# Each point gets a bounded lifetime and independent log. A failure is retained
# without suppressing the rest of the matrix; the script returns nonzero at end.
run_point() {
    local name="$1"
    local result
    shift
    if timeout "$readme_timeout" "$readme_python" "$@" > "$readme_output/$name.log" 2>&1; then
        result=0
        printf 'PASS %s\n' "$name"
    else
        result=$?
        printf 'FAIL %s (exit %s; see log)\n' "$name" "$result"
        readme_failed=1
    fi
    printf '{"exit_code":%s,"time_limit_seconds":%s}\n' "$result" "$readme_timeout" \
        > "$readme_output/$name.outcome"
}

# The reviewed paired subset stops immediately at an error or timeout. Large
# reference-only workloads are measured independently, so a slow native route
# cannot prevent collecting the corresponding GPU reference result.
if [[ "$readme_group" == dft-paired ]]; then
    for point in pbe:3 pbe:6 r2scan:3 r2scan:6 pbe:96; do
        method="${point%:*}"
        atoms="${point#*:}"
        run_point "dft/$method-direct-$atoms" -m benchmarks.readme_method_endpoints \
            --method "$method" --atoms "$atoms" --repeats 3 \
            --output "$readme_output/dft/$method-direct-$atoms.json"
        [[ "$readme_failed" == 0 ]] || exit 1
    done
fi

if [[ "$readme_group" == all || "$readme_group" == hf || "$readme_group" == smoke ]]; then
    for mode in direct df; do
        readme_extra=()
        if [[ "$mode" == df ]]; then
            readme_extra=(--density-fitting cuda --reference-full-fock
                --auxiliary-basis-file benchmarks/results/issue206-practical-auxiliary/identity/cc-pvdz-jkfit.json)
        fi
        readme_sizes="3 6 12 24 48 96"
        [[ "$readme_group" == smoke ]] && readme_sizes=3
        for atoms in $readme_sizes; do
            run_point "hf/$mode-$atoms" -m benchmarks.readme_hf_scaling \
                --case "water-$atoms" --batch 1 --repeats 3 --max-iterations 100 \
                --energy-tolerance 1e-10 --density-tolerance 1e-9 \
                --reference-gradient-tolerance 1e-8 --screening-tolerance 1e-12 \
                --maximum-energy-error 1e-8 --maximum-force-error 1e-7 \
                "${readme_extra[@]}" --output "$readme_output/hf/$mode-$atoms.json" \
                --progress-output "$readme_output/hf/$mode-$atoms.progress.jsonl"
        done
    done
fi

if [[ "$readme_group" == all || "$readme_group" == dft || "$readme_group" == dft-reference || "$readme_group" == smoke ]]; then
    readme_dft_directory=dft
    readme_dft_extra=()
    if [[ "$readme_group" == dft-reference ]]; then
        readme_dft_directory=dft-reference
        readme_dft_extra=(--reference-only)
    fi
    for method in pbe r2scan pbe0; do
        for mode in direct df; do
            readme_sizes="3 6 12 24 48 96"
            [[ "$readme_group" == dft-reference ]] && readme_sizes="96 48 24 12 6 3"
            [[ "$readme_group" == smoke ]] && readme_sizes=3
            for atoms in $readme_sizes; do
                run_point "$readme_dft_directory/$method-$mode-$atoms" -m benchmarks.readme_method_endpoints \
                    --method "$method" --mode "$mode" --atoms "$atoms" --repeats 3 \
                    "${readme_dft_extra[@]}" \
                    --output "$readme_output/$readme_dft_directory/$method-$mode-$atoms.json"
            done
        done
    done
fi

if [[ "$readme_group" == all || "$readme_group" == cc || "$readme_group" == smoke ]]; then
    readme_molecules="h2o nh3 ch4"
    [[ "$readme_group" == smoke ]] && readme_molecules=h2o
    for molecule in $readme_molecules; do
        run_point "cc/$molecule" -m benchmarks.readme_method_endpoints \
            --method 'ccsd(t)' --molecule "$molecule" --repeats 3 \
            --output "$readme_output/cc/$molecule.json"
    done
fi
exit "$readme_failed"
