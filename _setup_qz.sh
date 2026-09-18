#!/bin/bash
# Setup script for #150 B validation on qz. LF-only.
set -eu
ROOT=/inspire/qb-ilm/project/chemicalreaction/czxs25220150
BASE=$ROOT/projects/vibeqc
WORK=$BASE/worktrees/issue-150-b
ASSETS=$ROOT/env-assets/vibeqc-cpu-py311-v1
PY=$ROOT/env-assets/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11

echo "=== base repo HEAD before ==="
git -C "$BASE" log --oneline -1

echo "=== fetch fork branch ==="
git -C "$BASE" remote set-url fork https://github.com/xshengrui/vibeqc.git 2>/dev/null || git -C "$BASE" remote add fork https://github.com/xshengrui/vibeqc.git
git -C "$BASE" fetch fork claude/issue-150-b-tiles

echo "=== create remote worktree ==="
mkdir -p "$BASE/worktrees"
if [ -d "$WORK" ]; then
  echo "worktree exists, reusing"
else
  git -C "$BASE" worktree add "$WORK" fork/claude/issue-150-b-tiles
fi
git -C "$WORK" log --oneline -3

echo "=== setup venv from wheelhouse ==="
if [ -f "$WORK/.venv/bin/python" ]; then
  echo "venv exists, reusing"
else
  export VIBEQC_UV="$ASSETS/bin/uv"
  export UV_LINK_MODE=copy
  bash "$ASSETS/recipe/setup_cpu.sh" "$WORK" "$ASSETS/wheels" "$PY"
fi
"$WORK/.venv/bin/python" -c "import numpy, scipy; print('numpy', numpy.__version__)"

echo "=== vibeqc imports ==="
cd "$WORK"
PYTHONPATH=python:. "$WORK/.venv/bin/python" -c "
from tools.vibeqc_cc import triples_energy, cpu_triples_tiles, TriplesTileConfig
print('imports OK')
"
echo SETUP_DONE