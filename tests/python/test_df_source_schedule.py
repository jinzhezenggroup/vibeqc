"""Compiler source schedules preserve explicit controls and distinct consumers."""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

import pytest
from vibeqc_compiler.integral.df_policy import emit_df_value_source_schedule_cuda

if TYPE_CHECKING:
    from pathlib import Path


def test_compiled_raw_and_transformed_schedule_contract(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    driver = tmp_path / "schedule.cpp"
    driver.write_text(
        emit_df_value_source_schedule_cuda()
        + """
using S=ValueSourceSchedule;
static_assert(S::resolve(S::automatic_mapping, false)==S::auxiliary_mapping);
static_assert(S::resolve(S::automatic_mapping, true)==S::primitive_mapping);
static_assert(32 % S::primitive_lanes == 0);
int main() {
  for (unsigned requested: {S::auxiliary_mapping,S::component_mapping,S::primitive_mapping})
    for (bool transformed: {false,true})
      if (S::resolve(requested,transformed)!=requested) return 1;
}
"""
    )
    executable = tmp_path / "schedule"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-include",
            "initializer_list",
            str(driver),
            "-o",
            str(executable),
        ],
        check=True,
        timeout=30,
    )
    subprocess.run([str(executable)], check=True, timeout=10)
