#!/usr/bin/env bash
set -euo pipefail
root=/inspire/qb-ilm/project/chemicalreaction/diwenxi-CZXS25120072/vibeqc-workspace
source "$root/activate-311.sh"
task="$root/build-171/f-20260917"
evidence="$task/evidence"
trap 'echo $? > "$evidence/cuda.exit"' EXIT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$task/source"
export PYTHONPATH="$PWD/python" VIBEQC_LIBRARY="$task/cuda/libvibeqc.so" VIBEQC_ECP_CUDA_TEST=1
python -m pytest tests/python/test_ecp.py tests/python/test_ecp_ir.py tests/python/test_ecp_validation.py tests/python/test_ecp_f.py -q -k cuda > "$evidence/cuda-focused-python.log" 2>&1
python tools/benchmark_ecp.py --device cuda --repeats 3 --output "$evidence/candidate-benchmark.json" > "$evidence/candidate-benchmark.log" 2>&1
/usr/local/cuda-12.8/bin/compute-sanitizer --tool memcheck --error-exitcode 99 "$task/cuda/vibeqc_ecp_cuda_error_tests" > "$evidence/cuda-memcheck.log" 2>&1
/usr/local/cuda-12.8/bin/compute-sanitizer --tool memcheck --error-exitcode 99 python -m pytest tests/python/test_ecp_f.py -q -k 'raw_matrices and cuda' > "$evidence/cuda-f-memcheck.log" 2>&1
python "$root/evidence-171/f-endpoints.py" --source "$PWD" --device cuda --output "$evidence/cuda-f-endpoints.json" > "$evidence/cuda-f-endpoints.log" 2>&1
cp "$task/cuda/generated/generated_ecp_ao.cuh" "$evidence/candidate-generated.cuh"
cuobjdump --dump-resource-usage "$task/cuda/CMakeFiles/vibeqc.dir/src/integrals/ecp_cuda.cu.o" > "$evidence/candidate-resources.txt"
python "$root/evidence-171/f-provenance.py" candidate source
cd "$task/candidate"
export PYTHONPATH="$PWD/python" VIBEQC_LIBRARY="$task/cpu/libvibeqc.so"
python -m pytest tests/python/test_ecp_f.py -q -k 'prepared_replay and cpu' > "$evidence/cpu-focused-python.log" 2>&1
python "$root/evidence-171/f-provenance.py" cpu candidate
