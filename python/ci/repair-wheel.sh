#!/usr/bin/env bash
set -euo pipefail

destination=${1:?usage: repair-wheel.sh <destination> <wheel>}
wheel=${2:?usage: repair-wheel.sh <destination> <wheel>}

# Match xTBloom's distribution boundary: CUDA user-space providers remain
# separately distributed NVIDIA packages and the kernel driver remains a
# system dependency. prepare-wheel-build.sh stages provider wheels only into
# the ephemeral build toolkit; the installed-wheel test installs [cuda12] and
# VibeQC's Python loader registers those SONAMEs before loading libvibeqc.
auditwheel repair -w "$destination" "$wheel" \
  --exclude libcublas.so.12 \
  --exclude libcublasLt.so.12 \
  --exclude libcusolver.so.11 \
  --exclude libcusparse.so.12 \
  --exclude libcudart.so.12 \
  --exclude libnvJitLink.so.12 \
  --exclude libcufft.so.11 \
  --exclude libcurand.so.10 \
  --exclude libcuda.so.1

# Surface whether repeated wheel builds actually reuse C++/CUDA compilation.
ccache --show-stats
