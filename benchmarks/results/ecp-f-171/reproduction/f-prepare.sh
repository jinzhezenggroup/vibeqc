#!/usr/bin/env bash
set -euo pipefail
root=/inspire/qb-ilm/project/chemicalreaction/diwenxi-CZXS25120072/vibeqc-workspace
cd "$root"
source activate-311.sh
task="$root/build-171/f-20260917"
mkdir "$task"
mkdir "$task/source" "$task/candidate" "$task/evidence"
tar -xzf evidence-171/f-baseline.tar.gz -C "$task/source"
tar -xzf evidence-171/f-baseline.tar.gz -C "$task/candidate"
tar -xzf evidence-171/f-changes.tar.gz -C "$task/candidate"
cd "$task/candidate"
ruff check python/vibeqc_compiler/integral/ecp.py python/vibeqc_compiler/integral/ecp_grid.py python/vibeqc_compiler/integral/ir.py python/vibeqc/ecp.py tests/python/test_ecp.py tests/python/test_ecp_f.py tests/python/test_ecp_ir.py tests/python/test_ecp_validation.py
ruff format python/vibeqc_compiler/integral/ecp.py python/vibeqc_compiler/integral/ecp_grid.py python/vibeqc_compiler/integral/ir.py python/vibeqc/ecp.py tests/python/test_ecp.py tests/python/test_ecp_f.py tests/python/test_ecp_ir.py tests/python/test_ecp_validation.py
clang-format -i src/api/c_api_ecp.cpp tests/native/test_ecp_projector.cpp tests/native/test_ecp_capabilities.cpp
python -I -S tools/generate_ecp_kernels.py --output "$task/generated_ecp_ao.cuh"
c++ -std=c++20 -O2 -Wall -Wextra -Werror -I "$task" tests/native/test_ecp_projector.cpp -o "$task/grid-test"
"$task/grid-test"
tar -czf "$root/evidence-171/f-formatted.tar.gz" python/vibeqc_compiler/integral/ecp.py python/vibeqc_compiler/integral/ecp_grid.py python/vibeqc_compiler/integral/ir.py python/vibeqc/ecp.py src/api/c_api_ecp.cpp tests/native/test_ecp_projector.cpp tests/native/test_ecp_capabilities.cpp cmake/VibeQCTests.cmake tests/python/test_ecp.py tests/python/test_ecp_f.py tests/python/test_ecp_ir.py tests/python/test_ecp_validation.py
