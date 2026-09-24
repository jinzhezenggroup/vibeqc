"""Generated native CPU DF-RCCSD virtual correction matches Slice-B oracle."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tools.vibeqc_cc.df_factorized import virtual_corrections


def _values(values: np.ndarray) -> str:
    return ",".join(format(float(value), ".17g") for value in values.ravel())


def test_generated_native_df_virtual_correction_matches_oracle(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")

    rng = np.random.default_rng(157)
    nocc, nvir, naux = 2, 3, 4
    bov = rng.normal(size=(naux, nocc, nvir))
    bvv = rng.normal(size=(naux, nvir, nvir))
    bvv = 0.5 * (bvv + bvv.transpose(0, 2, 1))
    t1 = rng.normal(size=(nocc, nvir))
    t2 = rng.normal(size=(nocc, nocc, nvir, nvir))
    expected_r1, expected_r2 = virtual_corrections(bov, bvv, t1, t2)

    root = Path(__file__).resolve().parents[2]
    header = tmp_path / "generated_rccsd_cpu.hpp"
    subprocess.run(
        [
            sys.executable,
            str(root / "tools/generate_rccsd_native.py"),
            "--cpu-header",
            str(header),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )

    source = tmp_path / "df_virtual.cpp"
    source.write_text(
        f"""
#include "generated_rccsd_cpu.hpp"
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

int main() {{
  constexpr std::size_t o={nocc}, v={nvir}, q={naux};
  std::vector<double> bov={{{_values(bov)}}};
  std::vector<double> bvv={{{_values(bvv)}}};
  std::vector<double> t1={{{_values(t1)}}};
  std::vector<double> t2={{{_values(t2)}}};
  std::vector<double> expected_r1={{{_values(expected_r1)}}};
  std::vector<double> expected_r2={{{_values(expected_r2)}}};

  using namespace vibeqc::cc::generated;
  const auto required=df_virtual_correction_arena_elements(o,v,q);
  if(required==0) return 1;
  std::vector<double> arena(required);
  const DFVirtualInputs inputs{{bov.data(),bvv.data(),t1.data(),t2.data()}};
  const auto actual=run_df_virtual_correction_cpu(
      o,v,q,inputs,arena.data(),arena.size());

  double error=0.0;
  for(std::size_t i=0;i<expected_r1.size();++i)
    error=std::max(error,std::abs(actual.r1[i]-expected_r1[i]));
  for(std::size_t i=0;i<expected_r2.size();++i)
    error=std::max(error,std::abs(actual.r2[i]-expected_r2[i]));
  if(!(error <= 8e-11)) {{
    std::cerr << "maximum error " << error << "\\n";
    return 2;
  }}
  if(df_virtual_correction_program_hash[0]=='\\0') return 3;
  std::cout << "generated DF correction parity passed: " << error << "\\n";
}}
""",
        encoding="utf-8",
    )
    executable = tmp_path / "df_virtual"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O0",
            "-I" + str(tmp_path),
            str(source),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=90,
    )
    result = subprocess.run(
        [str(executable)],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
