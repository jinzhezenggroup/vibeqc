"""Execute the native cache fixture's budget predicate against the real planner."""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def test_native_cache_budget_expectations_match_planner(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    root = Path(__file__).resolve().parents[2]
    native = (root / "tests/native/test_density_fitting.cpp").read_text()
    conditions = re.findall(r"if \((budget != 0 && budget <= [^\n]+)\) \{", native)
    assert len(conditions) == 1
    generated = tmp_path / "generated"
    generated.mkdir()
    subprocess.run(
        [
            sys.executable,
            str(root / "tools/generate_df_exchange_schedule.py"),
            "--output",
            str(generated / "generated_df_exchange_schedule.hpp"),
        ],
        check=True,
        cwd=root,
    )
    source = tmp_path / "budget.cpp"
    source.write_text(
        r"""
#include <iostream>
#include "src/scf/density_fitting.cpp"
#include "scf/df_preparation_budget.hpp"
int main() {
  using namespace generativeqc::scf;
  // Exact native fixture: two H2 sp systems, each with 8 orbital/auxiliary
  // AOs, four orbital/auxiliary shells and one primitive per shell. Include
  // each system's dummy source, transforms and the unchanged eight-slot DIIS.
  const auto source_bytes = density_fitting_source_metadata_bytes(2,4,18,34,18,256);
  const auto fixed = source_bytes + density_fitting_scf_diis_device_bytes(2,8,8);
  for (std::size_t rank : {0U,1U}) {
    for (std::size_t budget : {32768U,1U<<20,2U<<20,4U<<20,8U<<20,12U<<20,16U<<20}) {
      const auto resolved = resolve_df_budget({8,8,2,2,8,true},{},budget);
      if (resolved.total_bytes != budget ||
          resolved.value_bytes + resolved.response_bytes != budget) return 1;
      bool rejected = false;
      std::size_t peak = 0;
      try {
        const auto plan = plan_requested_density_fitting_tiles(
            DfPairStorageRequest::Automatic,2,8,8,1,1,resolved.value_bytes,fixed,true,rank);
        peak = plan.peak_workspace_bytes;
        if (!plan.stores_full_three_center || peak > resolved.value_bytes) return 2;
      } catch (const DensityFittingBudgetError&) {
        rejected = true;
      }
      const bool native_expects_rejection = NATIVE_PREDICATE;
      std::cout << rank << ' ' << budget << ' ' << resolved.value_bytes << ' '
                << source_bytes << ' ' << peak << ' ' << rejected << '\n';
      if (rejected != native_expects_rejection) return 3;
    }
  }
}
""".replace("NATIVE_PREDICATE", f"({conditions[0]})")
    )
    executable = tmp_path / "budget"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O0",
            "-ffunction-sections",
            "-fdata-sections",
            "-Wl,--gc-sections",
            f"-I{root}",
            f"-I{root / 'include'}",
            f"-I{root / 'src'}",
            f"-I{generated}",
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
    )
    # No device, large tensor or solver is allocated. Explicit external
    # experiment controls must not change this default-policy regression.
    env = os.environ.copy()
    for key in tuple(env):
        if key.startswith("GENERATIVEQC_DF_"):
            del env[key]
    subprocess.run([str(executable)], check=True, env=env)
