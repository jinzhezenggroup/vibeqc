"""Compile the actual host-retirement helper against its native data contract."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_resident_response_survives_value_phase_retirement(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    text = (ROOT / "src/scf/rhf.cpp").read_text()
    begin = text.index("[[maybe_unused]] void discard_density_fitting_tensor_storage(")
    end = text.index("\n}\n", begin) + 3
    helper = text[begin:end]
    source = (
        r"""
#include "scf/density_fitting.hpp"
#include <stdexcept>
namespace vibeqc::scf {
"""
        + helper
        + r"""
}
int main() {
  using namespace vibeqc::scf;
  DensityFittingScfData resident;
  resident.resolved_budget.value_bytes = 1U << 20;
  resident.raw.nbf = 2; resident.raw.naux = 2;
  resident.raw.three_center.assign(8, 0.5);
  resident.raw.three_center_derivative.assign(24, 0.25);
  resident.raw.metric_derivative.assign(12, 0.25);
  resident.three_center.values.assign(8, 0.75);
  resident.three_center.auxiliary_major_values.assign(8, 0.75);
  resident.df_gradient_orbital.emplace();
  resident.df_gradient_auxiliary.emplace();
  const auto* raw = resident.raw.three_center.data();
  discard_density_fitting_tensor_storage(resident);
  if (resident.raw.three_center.size() != 8 || resident.raw.three_center.data() != raw)
    throw std::runtime_error("resident response raw owner was retired before use");
  if (!resident.three_center.values.empty() || !resident.three_center.auxiliary_major_values.empty()
      || !resident.raw.three_center_derivative.empty() || !resident.raw.metric_derivative.empty())
    throw std::runtime_error("uploaded transformed or derivative storage was retained");
  discard_density_fitting_tensor_storage(resident);
  if (resident.raw.three_center.data() != raw)
    throw std::runtime_error("repeat retirement changed the bound raw owner");
  DensityFittingScfData energy_only;
  energy_only.raw.three_center.assign(8, 1.0);
  discard_density_fitting_tensor_storage(energy_only);
  if (energy_only.raw.three_center.capacity() != 0)
    throw std::runtime_error("energy-only host tensor was not released");
  DensityFittingScfData source_backed;
  source_backed.df_gradient_orbital.emplace();
  source_backed.df_gradient_auxiliary.emplace();
  discard_density_fitting_tensor_storage(source_backed);
  if (!source_backed.raw.three_center.empty())
    throw std::runtime_error("source-backed metadata materialized a raw tensor");
}
"""
    )
    path = tmp_path / "retirement.cpp"
    path.write_text(source)
    executable = tmp_path / "retirement"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O1",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
            str(path),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    completed = subprocess.run(
        [str(executable)], capture_output=True, text=True, check=False, timeout=10
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
