#!/bin/bash
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:5090:1
#SBATCH --time=02:00:00
#SBATCH --output=/tmp/vibeqc-issue493-d4-%j.out
#SBATCH --error=/tmp/vibeqc-issue493-d4-%j.err
set -euo pipefail

ROOT="${ISSUE493_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}"
BUILD="${ISSUE493_BUILD:-${ROOT}/build-issue493-d4}"
OUTPUT="${ISSUE493_OUTPUT_DIR:-${ROOT}/.artifacts/issue493-d4-${SLURM_JOB_ID}}"
CUDA_ROOT="${CUDA_HOME:-/group/software/cuda-12.9.1}"
PYTHON="${ISSUE493_PYTHON:-python3}"

cd "${ROOT}"
mkdir -p "${OUTPUT}"
export CUDA_HOME="${CUDA_ROOT}"
export PATH="${CUDA_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

printf 'revision=%s\n' "$(git rev-parse HEAD)" | tee "${OUTPUT}/source.txt"
git status --short | tee -a "${OUTPUT}/source.txt"
printf 'SLURM_JOB_ID=%s CUDA_VISIBLE_DEVICES=%s\n' \
  "${SLURM_JOB_ID}" "${CUDA_VISIBLE_DEVICES:-}" | tee "${OUTPUT}/allocation.txt"
nvidia-smi | tee "${OUTPUT}/nvidia-smi.txt"

cmake -S . -B "${BUILD}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="${CUDA_ROOT}/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DVIBEQC_ENABLE_AOT_SHELLS=OFF
cmake --build "${BUILD}" -j8 --target \
  vibeqc_d4_reference_tests vibeqc_d4_reference_cuda_tests \
  vibeqc_d4_eeq_tests vibeqc_d4_eeq_cuda_tests vibeqc_d4_schedule_cuda_tests \
  vibeqc_d4_production_tests vibeqc_d4_ragged_tests vibeqc_dft_api_tests
export VIBEQC_LIBRARY="${BUILD}/libvibeqc.so"
export VIBEQC_GPU_ALLOCATION_ID="slurm:${SLURM_JOB_ID}"

ctest --test-dir "${BUILD}" --output-on-failure -R \
  'vibeqc_d4_(reference|eeq|production|ragged|schedule)'

"${BUILD}/vibeqc_d4_production_tests" cuda
"${BUILD}/vibeqc_d4_ragged_tests" cuda
CUDA_VISIBLE_DEVICES="" "${BUILD}/vibeqc_dft_api_tests"
"${PYTHON}" -m pytest -q tests/python/test_d4_production.py tests/python/test_d4_eeq.py

compute-sanitizer --tool memcheck --error-exitcode=99 \
  "${BUILD}/vibeqc_d4_production_tests" cuda \
  >"${OUTPUT}/memcheck-production.txt" 2>&1
compute-sanitizer --tool memcheck --error-exitcode=99 \
  "${BUILD}/vibeqc_d4_ragged_tests" cuda \
  >"${OUTPUT}/memcheck-ragged.txt" 2>&1

"${PYTHON}" benchmarks/d4_production_gate.py --device cpu --repeats 7 \
  --output "${OUTPUT}/cpu.json"
"${PYTHON}" benchmarks/d4_production_gate.py --device cuda --repeats 7 \
  --output "${OUTPUT}/cuda.json"
