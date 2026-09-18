#!/bin/bash
# Inspect qz environment for #150 B GPU validation. LF-only; run via:
#   inspire notebook exec general-copy --workspace CPU资源空间 "bash /inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/vibeqc/.../inspect_qz.sh"
set -u
echo "=== GPU visibility ==="
nvidia-smi -L 2>&1 || echo "no nvidia-smi"
echo "=== NVCC ==="
which nvcc 2>&1 || echo "no nvcc in PATH"
ls -d /usr/local/cuda* 2>/dev/null || echo "no /usr/local/cuda*"
nvcc --version 2>&1 | tail -2 || true
echo "=== gcc ==="
gcc --version | head -1
g++ --version | head -1
echo "=== env-assets ==="
ls /inspire/qb-ilm/project/chemicalreaction/czxs25220150/env-assets/
echo "=== recipe ==="
ls /inspire/qb-ilm/project/chemicalreaction/czxs25220150/env-assets/vibeqc-cpu-py311-v1/recipe/
echo "=== existing venv ==="
ls /inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/vibeqc/.venv/bin/ 2>/dev/null | head -10 || echo "no venv"
echo "=== git remotes ==="
git -C /inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/vibeqc remote -v
git -C /inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/vibeqc log --oneline -2
echo "=== disk ==="
df -h /inspire/qb-ilm/project/chemicalreaction/czxs25220150 2>/dev/null | tail -1
echo DONE