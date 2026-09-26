#!/usr/bin/env bash
set -euo pipefail
root=/inspire/qb-ilm/project/chemicalreaction/diwenxi-CZXS25120072/vibeqc-workspace
cd "$root"
source activate-311.sh
task="$root/build-171/fprojector-20260918"
mkdir "$task"
mkdir "$task/source" "$task/candidate" "$task/evidence"
tar -xzf evidence-171/fprojector-baseline.tar.gz -C "$task/source"
tar -xzf evidence-171/fprojector-baseline.tar.gz -C "$task/candidate"
tar -xzf evidence-171/fprojector-changes.tar.gz -C "$task/candidate"
cd "$task/candidate"
mapfile -t files < "$root/evidence-171/fprojector-files.txt"
for f in "${files[@]}"; do
  case "$f" in
    *.py) ruff check "$f"; ruff format "$f" ;;
    *.cpp|*.cu|*.hpp|*.h) clang-format -i "$f" ;;
  esac
done
python -I -S tools/generate_ecp_kernels.py --output "$task/generated_ecp_ao.cuh"
c++ -std=c++20 -O2 -Wall -Wextra -Werror -I "$task" tests/native/test_ecp_projector.cpp -o "$task/projector-test"
"$task/projector-test"
tar -czf "$root/evidence-171/fprojector-formatted.tar.gz" "${files[@]}"
