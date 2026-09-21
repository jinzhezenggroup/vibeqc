"""Automatic owner selection must retain the bounded planner's factor decision."""

import shutil
import subprocess
from pathlib import Path

import pytest
from vibeqc import _native

ROOT = Path(__file__).resolve().parents[2]


def _prefix(source: str, name: str, end: str) -> str:
    start = source.index(name + "(")
    start = source.index(") {", start) + 3
    return source[start : source.index(end, start)]


def test_actual_single_and_batch_routing_preserve_admitted_factor_rank(
    tmp_path: Path,
) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    library = Path(_native.load_library()._name).resolve()
    source = (ROOT / "src/scf/rhf.cpp").read_text()
    begin = source.index("automatic_dense_resident_df_owner(")
    begin = source.rfind("\n", 0, begin) + 1
    helper = source[begin : source.index("\n}\n", begin) + 3]
    single = _prefix(
        source, "make_cuda_density_fitting_plan", "  CudaDensityFittingJkPlan* raw_plan"
    )
    batch = _prefix(
        source, "make_cuda_density_fitting_batch_plan", "  std::vector<double> metrics;"
    )
    driver = tmp_path / "admission.cpp"
    driver.write_text(
        "#include <cstdlib>\n#include <iostream>\n#include <utility>\n"
        '#include "scf/density_fitting.hpp"\nnamespace vibeqc::scf {\n'
        + helper
        + "\nstd::pair<size_t,size_t> single(const DensityFittingScfData& data, size_t occupied, bool unrestricted, unsigned history=0) { struct { unsigned diis_history; } options{history};"
        + single
        + "\nreturn {planning_budget,automatic_rhf_rank};}\n"
        + "std::pair<size_t,size_t> batch(const std::vector<DensityFittingScfData>& data, size_t occupied, bool unrestricted, unsigned history=0) { struct { unsigned diis_history; } options{history};"
        + batch
        + "\nreturn {planning_budget,automatic_rhf_rank};}\n}\n"
        + r"""
int main() {
  using namespace vibeqc::scf;
  setenv("VIBEQC_DF_EXCHANGE", "auto", 1);
  unsetenv("VIBEQC_DF_RESIDENT_EXCHANGE");
  unsetenv("VIBEQC_DF_PAIR_STORAGE");
  for (size_t n : {24u,96u,384u}) {
    const size_t a=n, r=n/4;
    const auto dense=plan_density_fitting_tiles(1,n,a,r,1ULL<<40,0,false,0);
    const auto factors=plan_density_fitting_tiles(1,n,a,r,1ULL<<40,0,false,r);
    if (!(factors.peak_workspace_bytes > dense.peak_workspace_bytes)) return 1;
    DensityFittingScfData data{}; data.raw.nbf=n; data.raw.naux=a;
    data.resolved_budget.value_bytes=dense.peak_workspace_bytes;
    const auto admitted=plan_density_fitting_tiles(1,n,a,r,data.resolved_budget.value_bytes,0,false,r);
    if (!admitted.stores_full_three_center || admitted.automatic_rhf_rank != 0) return 2;
    for (auto result : {single(data,r,false),batch({data},r,false)}) {
      if (result.first != 0 || result.second != 0) {
        std::cerr << "dense-only admission restored unbudgeted occupied factors\n"; return 3;
      }
    }
"""
        + r"""
    data.resolved_budget.value_bytes=factors.peak_workspace_bytes;
    for (auto result : {single(data,r,false),batch({data},r,false)})
      if (result.first != 0 || result.second != r) return 4;
    data.resolved_budget.requested_bytes=1;
    if (single(data,r,false).first != data.resolved_budget.value_bytes) return 5;
    data.resolved_budget.requested_bytes=0;
    if (single(data,r,true).first != data.resolved_budget.value_bytes) return 6;
    if (batch({data,data},r,false).first != data.resolved_budget.value_bytes) return 7;
  }
  {
    const size_t n=96, a=96, rank=24;
    const auto bare=plan_density_fitting_tiles(1,n,a,rank,1ULL<<40,0,false,0);
    const auto diis=density_fitting_scf_diis_device_bytes(1,n,8);
    if(diis==0) return 8;
    DensityFittingScfData data{};data.raw.nbf=n;data.raw.naux=a;
    data.resolved_budget.value_bytes=bare.peak_workspace_bytes;
    const auto bounded=plan_density_fitting_tiles(1,n,a,rank,data.resolved_budget.value_bytes,diis,false,rank);
    if(bounded.stores_full_three_center) return 9;
    for(auto result:{single(data,rank,false,8),batch({data},rank,false,8)})
      if(result.first==0) {std::cerr<<"resident admission omitted fixed DIIS storage\n";return 10;}
    const auto full=plan_density_fitting_tiles(1,n,a,rank,1ULL<<40,diis,false,rank);
    data.resolved_budget.value_bytes=full.peak_workspace_bytes;
    for(auto result:{single(data,rank,false,8),batch({data},rank,false,8)})
      if(result.first!=0 || result.second!=rank) return 11;
  }
  std::cout << "all bounded rank and routing checks passed\n";
}
"""
    )
    executable = tmp_path / "admission"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O0",
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
            str(driver),
            str(library),
            "-Wl,-rpath," + str(library.parent),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = subprocess.run(
        [str(executable)], check=False, capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_batch_owner_keeps_positive_resolved_budget_identity() -> None:
    source = (ROOT / "src/scf/rhf.cpp").read_text()
    start = source.index(
        "CudaDensityFittingPlanPtr make_cuda_density_fitting_batch_plan("
    )
    body = source[start : source.index("\n}\n", start)]
    assert (
        "set_cuda_density_fitting_scf_value_budget(owned_plan.get(), planning_budget)"
        not in body
    )


def test_preparation_routes_use_their_bound_diis_history() -> None:
    source = (ROOT / "src/scf/rhf.cpp").read_text()
    assert source.count("resident_occupied, unrestricted, diis_history") == 2
    assert "resident_occupied, unrestricted, options.diis_history" not in source
