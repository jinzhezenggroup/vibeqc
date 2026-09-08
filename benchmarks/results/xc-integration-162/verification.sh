#!/usr/bin/env bash
set -euo pipefail
repo=/inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/vibeqc-issue-162
cd "$repo"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VIBEQC_LIBRARY="$repo/build/cpu/libvibeqc.so"
export PATH="$repo/.venv/bin:$PATH"
out=benchmarks/results/xc-integration-162
mkdir -p "$out"
trap 'status=$?; printf "%s\n" "$status" > build/issue-162-verification.exit' EXIT
files=(tools/vibeqc_xc/__init__.py tools/vibeqc_xc/potential.py tools/vibeqc_xc/integration.py tools/vibeqc_xc/integration_fixtures.py tools/generate_xc_integration_references.py tools/validate_xc_integration.py tests/python/test_xc_integration.py)
{
  date -u +%FT%TZ
  git rev-parse HEAD
  python --version
  python -m pip check
  python -c 'import pyscf,numpy,scipy,sympy; from pyscf.dft import libxc; print("PySCF",pyscf.__version__,"Libxc",libxc.__version__,"NumPy",numpy.__version__,"SciPy",scipy.__version__,"SymPy",sympy.__version__)'
  cmake --version
  c++ --version
  ruff --version
  sha256sum "$VIBEQC_LIBRARY" "${files[@]}"
} > "$out/versions.log" 2>&1
ruff check "${files[@]}" > "$out/format.log" 2>&1
ruff format --check "${files[@]}" >> "$out/format.log" 2>&1
git diff --check >> "$out/format.log" 2>&1
python tools/generate_xc_integration_references.py build/xc-reference-1 > "$out/reference-generation.log" 2>&1
python tools/generate_xc_integration_references.py build/xc-reference-2 >> "$out/reference-generation.log" 2>&1
python - <<'PY' >> "$out/reference-generation.log" 2>&1
import json
from pathlib import Path
for name in ("h2", "water", "f_cartesian", "f_spherical"):
    paths = [Path(p) / (name + ".json") for p in ("build/xc-reference-1", "build/xc-reference-2", "tests/reference_data/xc_integration")]
    values = [json.loads(p.read_text()) for p in paths]
    assert values[0] == values[1] == values[2], name
    print(name, "two generations and committed candidate: identical metadata and every array hash")
PY
python -m pytest tests/python/test_xc_integration.py tests/python/test_grid_cpu.py tests/python/test_grid_prepared.py tests/python/test_xc_expressions.py -q > "$out/focused-tests.log" 2>&1
cmake --build build/cpu --parallel 2 > "$out/build.log" 2>&1
ctest --test-dir build/cpu --output-on-failure > "$out/ctest.log" 2>&1
python -m pytest tests/python -q > "$out/python-tests.log" 2>&1
python tools/validate_xc_integration.py --output "$out/numerical.json" > "$out/numerical.log" 2>&1
printf 'All validation commands passed.\n'
