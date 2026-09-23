"""Automatic RI-K selection shares a work policy and checks actual resident capacity."""

import os
import subprocess
import typing
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)


def test_auto_requires_rank_reference_residency_and_reservation(
    tmp_path: typing.Any,
) -> None:
    """Query a shape-only plan; no large tensors or SCF solve are necessary."""
    assert os.environ.get("SLURM_JOB_ID")
    root = Path(__file__).resolve().parents[2]
    library = Path(os.environ["VIBEQC_LIBRARY"]).resolve()
    source = tmp_path / "selector.cpp"
    source.write_text(
        r"""
#include <cstdlib>
#include <stdexcept>
#include "scf/cuda/df_plan_internal.hpp"
#include "scf/cuda/df_scf_factor.hpp"
#include "scf/df_exchange_policy.hpp"
#include "scf/density_fitting.hpp"
int main() {
  using namespace vibeqc::scf;
  CudaDensityFittingJkPlan p;
  p.device_id=-1; p.nbf=768; p.naux=768; p.batch_size=1;
  p.row_tile=768; p.auxiliary_tile=768; p.occupied_scf_reserved=true;
  // The selector uses workload/capacity only, so even an invalid device ID must
  // not cause a CUDA query. Real numerical tests validate the execution owner.
  p.resident_exchange_enabled=true;
  p.three_center=reinterpret_cast<double*>(1);
  p.auxiliary_tile_values=reinterpret_cast<double*>(1);
  p.projection_capacity=768ULL*768*768;
  const auto check = [&](bool expected, int rank=160, bool uhf=false) {
    bool enabled=false;
    std::string detail;
    std::vector<std::int32_t> alpha{rank}, beta=uhf ? alpha : std::vector<std::int32_t>{};
    const auto status=cuda_df::occupied_scf_policy(p, enabled, detail, alpha, beta);
    if(status!=VIBEQC_STATUS_SUCCESS || enabled!=expected)
      throw std::runtime_error("nbf=" + std::to_string(p.nbf) +
          " rank=" + std::to_string(rank) +
          " capacity=" + std::to_string(p.value_storage.rank_capacity) +
          " expected=" + std::to_string(expected) +
          " enabled=" + std::to_string(enabled) + " " + detail);
  };
  const auto check_packed_capacity = [&](int rank) {
    // Logical rank and actual allocated projection capacity must both fit.
    p.integral_source=reinterpret_cast<CudaDensityFittingIntegralSource*>(1);
    p.packed_raw=reinterpret_cast<double*>(1);
    p.value_storage.pairs=DfPairStorage::SymmetricLower;
    p.value_storage.rank_capacity=rank;
    p.projection_capacity=p.nbf*p.naux*rank;
    check(true, rank);
    --p.projection_capacity; check(false, rank); ++p.projection_capacity;
    p.value_storage.rank_capacity=rank-1; check(false, rank);
    p.value_storage.rank_capacity=rank;
    p.packed_raw=nullptr; check(false, rank);
    p.integral_source=nullptr;
    p.value_storage={};
    p.projection_capacity=p.nbf*p.nbf*p.naux;
  };
  setenv("VIBEQC_DF_EXCHANGE", "auto", 1);
  check(true);
  check_packed_capacity(160);
  check(true, 159); check(true, 161); check(false, 160, true);
  p.naux=767; p.auxiliary_tile=767; check(true); p.naux=768; p.auxiliary_tile=768;
  p.nbf=384; p.naux=384; p.row_tile=384; p.auxiliary_tile=384;
  check(true, 80);
  check_packed_capacity(80);
  check(true, 79); check(true, 81); check(false, 80, true);
  p.naux=383; p.auxiliary_tile=383; check(true, 80);
  for (int n : {96, 192, 512, 513}) {
    p.nbf=n; p.naux=n+17; p.row_tile=n; p.auxiliary_tile=n+17;
    p.projection_capacity=p.nbf*p.nbf*p.naux;
    check(true, n/4); check_packed_capacity(n/4);
    check(true, n/2); check(false, n/2+1);
    check(false, 0); check(false, -1); check(false, n+1);
  }
  p.nbf=768; p.naux=768; p.row_tile=768; p.auxiliary_tile=768;
  p.projection_capacity=768ULL*768*768;
  p.resident_exchange_enabled=false; check(false); p.resident_exchange_enabled=true;
  p.three_center=nullptr; check(false); p.three_center=reinterpret_cast<double*>(1);
  p.auxiliary_tile_values=nullptr; check(false);
  p.auxiliary_tile_values=reinterpret_cast<double*>(1);
  p.batch_size=2; check(false); p.batch_size=1;
  p.row_tile=384; check(false); p.row_tile=768;
  p.auxiliary_tile=384; check(false); p.auxiliary_tile=768;
  p.streamed=true; check(false); p.streamed=false;
  // Streamed value eligibility is separate from resident final/response
  // borrowing. A full-rank source, four buffers and profitable work are needed.
  p.streamed=true;
  p.integral_source=reinterpret_cast<CudaDensityFittingIntegralSource*>(1);
  p.exchange_intermediate=p.exchange_contributions=p.exchange_tile_output=
      reinterpret_cast<double*>(1);
  p.metric_full_rank={1}; p.panel_capacity=768ULL*768*128;
  check(true);
  if (cuda_df::qualified_resident_rhf_exchange(p,160))
    throw std::runtime_error("streamed value factor authorized resident response");
  p.metric_full_rank={0}; check(false); p.metric_full_rank={1};
  p.exchange_tile_output=nullptr; check(false);
  p.exchange_tile_output=reinterpret_cast<double*>(1);
  p.panel_capacity=1; check(false);
  p.streamed=false; p.integral_source=nullptr;
  p.integral_source=reinterpret_cast<CudaDensityFittingIntegralSource*>(1);
  check(false); p.integral_source=nullptr;
  p.occupied_scf_reserved=false; check(false); p.occupied_scf_reserved=true;
  setenv("VIBEQC_DF_EXCHANGE", "dense", 1); check(false);
  setenv("VIBEQC_DF_EXCHANGE", "occupied", 1); check(true, 159, true);

  // Real allocation must preserve the method-aware planner's decision. The
  // rank-zero hint denotes UHF/unknown reference, not a rank-zero RHF guess.
  std::vector<double> metric(64, 0), raw(512, 0);
  for (int i=0; i<8; ++i) metric[i*8+i]=1;
  setenv("VIBEQC_DF_EXCHANGE", "dense", 1);
  const auto dense_budget=plan_density_fitting_tiles(1,8,8,2,0).peak_workspace_bytes;
  const auto allocated = [&](std::size_t hint, std::size_t budget, bool expected) {
    const auto tile=plan_density_fitting_tiles(1,8,8,2,budget,0,false,hint);
    if (!tile.stores_full_three_center)
      throw std::runtime_error("optional factors forced a resident plan to stream");
    CudaDensityFittingJkPlan* actual=nullptr;
    std::vector<CudaDensityFittingMetricDiagnostic> diagnostics;
    std::string detail;
    const auto status=create_cuda_density_fitting_jk_plan_tiled(
        0,1,8,8,metric,raw,1e-10,tile.auxiliary_tile,tile.ao_pair_tile,
        &actual,diagnostics,detail,tile.automatic_rhf_rank);
    if (status!=VIBEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
    const bool reserved=actual->occupied_scf_reserved;
    const auto peak=diagnostics[0].peak_device_bytes;
    destroy_cuda_density_fitting_jk_plan(actual);
    if(reserved!=expected) throw std::runtime_error("native reservation differs from planner");
    return peak;
  };
  const auto dense_peak=allocated(2,dense_budget,false);
  setenv("VIBEQC_DF_EXCHANGE", "auto", 1);
  for (std::size_t hint : {0U,6U,2U})
    if(allocated(hint,dense_budget,false)!=dense_peak)
      throw std::runtime_error("ineligible or constrained auto changed dense reservation");
  constexpr auto factor_bytes=2*8*8*sizeof(double)+12;
  if(allocated(2,dense_budget+factor_bytes,true)!=dense_peak+factor_bytes)
    throw std::runtime_error("admitted factor reservation was not charged exactly");
}
"""
    )
    executable = tmp_path / "selector"
    # The same supported toolkit that builds the native library supplies the
    # private plan's type declarations; the query itself uses the linked owner.
    cuda = Path(os.environ.get("CUDA_HOME", "/group/software/cuda-12.9.1"))
    subprocess.run(
        [
            "c++",
            "-std=c++20",
            "-I" + str(root / "src"),
            "-I" + str(root / "include"),
            "-I" + str(cuda / "include"),
            str(source),
            str(library),
            "-Wl,-rpath," + str(library.parent),
            "-o",
            str(executable),
        ],
        check=True,
    )
    subprocess.run([str(executable)], check=True)
