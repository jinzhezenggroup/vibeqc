import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.method.gfn2_es3_runtime import (
    build_gfn2_es3_kernel,
    build_gfn2_es3_primal,
)
from vibeqc_compiler.tensor import execute

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("charge", (-1.7, -0.2, 0.0, 0.4, 2.1))
def test_gfn2_es3_tensorir_energy_potential_and_ad_identity(charge: float) -> None:
    gamma3 = 0.73
    primal = build_gfn2_es3_primal()
    kernel = build_gfn2_es3_kernel()
    values = execute(
        kernel,
        {
            "gamma3": np.asarray(gamma3, dtype=np.float64),
            "charge": np.asarray(charge, dtype=np.float64),
            "d_charge": np.asarray(1.0, dtype=np.float64),
        },
    ).outputs
    direct = execute(
        primal,
        {
            "gamma3": np.asarray(gamma3, dtype=np.float64),
            "charge": np.asarray(charge, dtype=np.float64),
        },
    ).outputs
    assert values["energy"] == direct["energy"]
    assert values["potential"] == direct["potential"]
    assert values["energy"] == pytest.approx(gamma3 * charge**3 / 3.0, rel=2e-15)
    assert values["potential"] == pytest.approx(
        gamma3 * charge**2, rel=2e-15, abs=1e-16
    )
    assert values["energy_charge_derivative"] == pytest.approx(
        values["potential"], rel=2e-15, abs=1e-16
    )


def _generate(output: Path) -> str:
    subprocess.run(
        [
            sys.executable,
            "-S",
            str(ROOT / "tools" / "generate_gfn2_es3_native.py"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return output.read_text(encoding="utf-8")


def test_gfn2_es3_codegen_is_standalone_and_records_ad_identity(tmp_path: Path) -> None:
    source = _generate(tmp_path / "generated_gfn2_es3_native.cuh")
    assert "gfn2_es3_primal_logical_hash" in source
    assert "gfn2_es3_charge_jvp_hash" in source
    assert "evaluate_gfn2_es3_energy" in source
    assert "evaluate_gfn2_es3_potential" in source
    assert "VIBEQC_GFN2_ES3_HD" in source


def test_gfn2_es3_generated_host_recovers_representable_extremes(
    tmp_path: Path,
) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    header = tmp_path / "generated_gfn2_es3_native.cuh"
    _generate(header)
    source = tmp_path / "es3_smoke.cpp"
    source.write_text(
        r"""#include <cmath>
#include "generated_gfn2_es3_native.cuh"
int main() {
  using vibeqc::xtb::generated::evaluate_gfn2_es3_energy;
  using vibeqc::xtb::generated::evaluate_gfn2_es3_potential;
  double value = 0.0;
  if (!evaluate_gfn2_es3_potential(1.0e-300, 1.0e200, &value) ||
      !std::isfinite(value) || std::abs(value / 1.0e100 - 1.0) > 2.0e-15) return 1;
  if (!evaluate_gfn2_es3_potential(1.0e300, 1.0e-200, &value) ||
      !std::isfinite(value) || std::abs(value / 1.0e-100 - 1.0) > 2.0e-15) return 2;
  const long double energy_ref = 1.0e150L * 1.0e150L * 1.0e150L * 1.0e-200L / 3.0L;
  if (!evaluate_gfn2_es3_energy(1.0e-200, 1.0e150, &value) ||
      !std::isfinite(value) || std::abs(value / static_cast<double>(energy_ref) - 1.0) > 2.0e-15) return 3;
  return 0;
}
""",
        encoding="utf-8",
    )
    executable = tmp_path / "es3_smoke"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-I",
            str(tmp_path),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(executable)], check=True)


def test_gfn2_es3_runtime_has_no_duplicate_handwritten_shell_formula() -> None:
    paths = (
        ROOT / "src/xtb/native/src/model/gfn2/es3.cpp",
        ROOT / "src/xtb/native/src/backends/cuda/gfn2_es3.cu",
    )
    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert '#include "generated_gfn2_es3_native.cuh"' in source
        assert "bool shell_potential(" not in source
        assert "bool shell_energy(" not in source
        assert "is_normal_double(" not in source
