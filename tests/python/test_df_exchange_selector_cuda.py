"""Automatic RI-K selection requires the entire measured execution domain."""

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)


def test_auto_requires_rank_reference_residency_and_reservation(tmp_path):
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
int main() {
  using namespace vibeqc::scf;
  CudaDensityFittingJkPlan p;
  p.device_id=0; p.nbf=768; p.naux=768; p.batch_size=1;
  p.row_tile=768; p.auxiliary_tile=768; p.occupied_scf_reserved=true;
  const auto check = [&](bool expected, int rank=160, bool uhf=false) {
    bool enabled=false;
    std::string detail;
    std::vector<std::int32_t> alpha{rank}, beta=uhf ? alpha : std::vector<std::int32_t>{};
    const auto status=cuda_df::occupied_scf_policy(p, enabled, detail, alpha, beta);
    if(status!=VIBEQC_STATUS_SUCCESS || enabled!=expected) throw std::runtime_error(detail);
  };
  setenv("VIBEQC_DF_EXCHANGE", "auto", 1);
  check(true);
  check(false, 159); check(false, 161); check(false, 160, true);
  p.naux=767; check(false); p.naux=768;
  p.nbf=384; check(false); p.nbf=768;
  p.batch_size=2; check(false); p.batch_size=1;
  p.row_tile=384; check(false); p.row_tile=768;
  p.auxiliary_tile=384; check(false); p.auxiliary_tile=768;
  p.streamed=true; check(false); p.streamed=false;
  p.integral_source=reinterpret_cast<CudaDensityFittingIntegralSource*>(1);
  check(false); p.integral_source=nullptr;
  p.occupied_scf_reserved=false; check(false); p.occupied_scf_reserved=true;
  setenv("VIBEQC_DF_EXCHANGE", "dense", 1); check(false);
  setenv("VIBEQC_DF_EXCHANGE", "occupied", 1); check(true, 159, true);
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
