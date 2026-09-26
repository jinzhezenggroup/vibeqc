"""Test emitted worker adaptation independently of the functional algebra."""

import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from vibeqc_compiler.xc import libxc_bulk
from vibeqc_compiler.xc.libxc_maple import MapleImportError
from vibeqc_compiler.xc.split_hybrid_domain import stabilize_m06_2x_stoll

from tools.generate_xc_split_hybrid_cuda import (
    SplitHybridWorkPolicy,
    _component_work_policy,
    _emit_work_wrapper,
    emit_split_hybrid_device,
)
from tools.libxc_split_hybrid import (
    _bound_component,
    _catalog,
    _root,
    build_split_global_hybrid,
)


def test_pinned_mgga_components_use_global_fhc_and_direct_spin_screens() -> None:
    for identifier in ("M06-2X", "MN15"):
        program = build_split_global_hybrid(identifier)
        for component in (program.exchange, program.correlation):
            assert _component_work_policy(component).enforce_fhc
        source = emit_split_hybrid_device(identifier)
        assert "if (rho_a <= " in source
        assert "if (rho_b <= " in source
        assert "work_sigma_aa = fmin(work_sigma_aa" in source


def test_stoll_rewrite_fails_closed_on_changed_modified_pw_parameters() -> None:
    record = _bound_component("MGGA_C_M06_2X", allow_hybrid_exchange=False)
    catalog, source_root = _catalog(), _root()
    raw = libxc_bulk.build_record(record, catalog["source_files"], source_root)
    module = libxc_bulk.import_record(record, catalog["source_files"], source_root)
    altered = replace(
        module,
        assignments=tuple(
            (name, "[0, 0, 0]" if name == "params_a_a" else value)
            for name, value in module.assignments
        ),
    )
    with pytest.raises(MapleImportError, match="modified PW parameter policy"):
        stabilize_m06_2x_stoll(raw, altered, record)


def test_emitted_mgga_work_adapter(tmp_path: Path) -> None:
    plain = SplitHybridWorkPolicy(1e-15, 1e-40, 1e-20, False, "test")
    fhc = SplitHybridWorkPolicy(1e-15, 1e-40, 1e-20, True, "test")
    source = r"""
#include <cmath>
#include <cassert>
#define __device__
struct Value {double energy_density{}; double feature_derivative[7]{};};
Value point_interior(double a,double b,double c,double d,double e,double f,double g) {
  return {2*(a+b),{a,b,c,d,e,f,g}};
}
Value fhc_interior(double a,double b,double c,double d,double e,double f,double g) {
  return point_interior(a,b,c,d,e,f,g);
}
"""
    source += _emit_work_wrapper("Value", "point", plain)
    source += _emit_work_wrapper("Value", "fhc", fhc)
    source += r"""
int main() {
  auto a = point(0,0,0,0,0,0,0);
  assert(a.energy_density == 0);
  for(double value : a.feature_derivative) assert(value == 0);
  a = point(0.1,0.2,0,0,0,0,0);
  assert(std::abs(a.energy_density-0.6)<1e-15);
  assert(a.feature_derivative[2]==1e-40);
  assert(a.feature_derivative[5]==1e-20);
  a = point(0.1,0,0.2,0.2,0,0.3,0);
  assert(std::abs(a.energy_density-0.2)<1e-15);
  assert(a.feature_derivative[1]==1e-15);
  assert(a.feature_derivative[3]==0.1);
  a = fhc(0.1,0.2,0.02,0,0.03,0.001,0.002);
  assert(std::abs(a.feature_derivative[2]-0.0008)<1e-15);
  assert(std::abs(a.feature_derivative[4]-0.0032)<1e-15);
}
"""
    path = tmp_path / "adapter.cpp"
    path.write_text(source, encoding="utf-8")
    executable = tmp_path / "adapter"
    compiler = shutil.which("c++")
    assert compiler is not None
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(path),
            "-o",
            str(executable),
        ],
        check=True,
        timeout=120,
    )
    subprocess.run([str(executable)], check=True, timeout=30)
