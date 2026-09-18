#!/usr/bin/env bash
set -euo pipefail
root=/inspire/qb-ilm/project/chemicalreaction/diwenxi-CZXS25120072/vibeqc-workspace
cd "$root"
source activate-311.sh
task="$root/build-171/f-20260917"
evidence="$task/evidence"
mode="$1"
trap 'echo $? > "$evidence/$mode.exit"' EXIT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [ "$mode" = cpu ]; then
  cd "$task/candidate"
  export PYTHONPATH="$PWD/python"
  cmake -S . -B "$task/cpu" -G 'Unix Makefiles' -DCMAKE_BUILD_TYPE=Release \
    -DVIBEQC_ENABLE_CUDA=OFF -DPython3_EXECUTABLE="$(command -v python)" > "$evidence/cpu-configure.log" 2>&1
  cmake --build "$task/cpu" --parallel 2 > "$evidence/cpu-build.log" 2>&1
  export VIBEQC_LIBRARY="$task/cpu/libvibeqc.so"
  ctest --test-dir "$task/cpu" --output-on-failure --parallel 2 > "$evidence/cpu-ctest.log" 2>&1
  python -m pytest tests/python/test_ecp.py tests/python/test_ecp_ir.py tests/python/test_ecp_validation.py tests/python/test_ecp_f.py -q > "$evidence/cpu-python.log" 2>&1
  python "$root/evidence-171/f-endpoints.py" --source "$PWD" --device cpu --output "$evidence/cpu-f-endpoints.json" > "$evidence/cpu-f-endpoints.log" 2>&1
  python "$root/evidence-171/f-provenance.py" cpu candidate
elif [ "$mode" = baseline ]; then
  cd "$task/source"
  export PYTHONPATH="$PWD/python"
  cmake -S . -B "$task/cuda" -G 'Unix Makefiles' -DCMAKE_BUILD_TYPE=Release \
    -DVIBEQC_ENABLE_CUDA=ON -DVIBEQC_ENABLE_AOT_SHELLS=OFF \
    -DCMAKE_CUDA_ARCHITECTURES=89 -DPython3_EXECUTABLE="$(command -v python)" > "$evidence/baseline-configure.log" 2>&1
  cmake --build "$task/cuda" --target vibeqc --parallel 5 > "$evidence/baseline-build.log" 2>&1
  export VIBEQC_LIBRARY="$task/cuda/libvibeqc.so"
  cp "$VIBEQC_LIBRARY" "$task/baseline-libvibeqc.so"
  cp "$task/cuda/generated/generated_ecp_ao.cuh" "$evidence/baseline-generated.cuh"
  cp "$task/cuda/CMakeCache.txt" "$evidence/baseline-CMakeCache.txt"
  cuobjdump --dump-resource-usage "$task/cuda/CMakeFiles/vibeqc.dir/src/integrals/ecp_cuda.cu.o" > "$evidence/baseline-resources.txt"
  python tools/benchmark_ecp.py --device cuda --repeats 3 --output "$evidence/baseline-benchmark.json" > "$evidence/baseline-benchmark.log" 2>&1
  python "$root/evidence-171/f-provenance.py" baseline source
elif [ "$mode" = cuda ]; then
  test "$(cat "$evidence/baseline.exit")" = 0
  tar -xzf "$root/evidence-171/f-formatted.tar.gz" -C "$task/source"
  cd "$task/source"
  export PYTHONPATH="$PWD/python"
  python -I -S tools/generate_ecp_kernels.py --output "$task/cuda/generated/generated_ecp_ao.cuh"
  touch src/api/c_api_ecp.cpp tests/native/test_ecp_projector.cpp tests/native/test_ecp_capabilities.cpp
  cmake --build "$task/cuda" --target vibeqc vibeqc_ecp_projector_tests vibeqc_ecp_capability_tests vibeqc_ecp_cuda_error_tests --parallel 5 > "$evidence/cuda-build.log" 2>&1
  export VIBEQC_LIBRARY="$task/cuda/libvibeqc.so" VIBEQC_ECP_CUDA_TEST=1
  ctest --test-dir "$task/cuda" -R 'vibeqc_ecp_' --output-on-failure > "$evidence/cuda-ctest.log" 2>&1
  python -m pytest tests/python/test_ecp.py tests/python/test_ecp_ir.py tests/python/test_ecp_validation.py tests/python/test_ecp_f.py -q > "$evidence/cuda-python.log" 2>&1
  python tools/benchmark_ecp.py --device cuda --repeats 3 --output "$evidence/candidate-benchmark.json" > "$evidence/candidate-benchmark.log" 2>&1
  /usr/local/cuda-12.8/bin/compute-sanitizer --tool memcheck --error-exitcode 99 "$task/cuda/vibeqc_ecp_cuda_error_tests" > "$evidence/cuda-memcheck.log" 2>&1
  /usr/local/cuda-12.8/bin/compute-sanitizer --tool memcheck --error-exitcode 99 python -m pytest tests/python/test_ecp_f.py -q -k 'raw_matrices and cuda' > "$evidence/cuda-f-memcheck.log" 2>&1
  python "$root/evidence-171/f-endpoints.py" --source "$PWD" --device cuda --output "$evidence/cuda-f-endpoints.json" > "$evidence/cuda-f-endpoints.log" 2>&1
  cp "$task/cuda/generated/generated_ecp_ao.cuh" "$evidence/candidate-generated.cuh"
  cuobjdump --dump-resource-usage "$task/cuda/CMakeFiles/vibeqc.dir/src/integrals/ecp_cuda.cu.o" > "$evidence/candidate-resources.txt"
  python "$root/evidence-171/f-provenance.py" candidate source
else
  exit 2
fi
