"""The native publication gate must not certify incomplete failed-stage counts."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def publication_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    source = (ROOT / "src/scf/cuda_rhf.cpp").read_text()
    begin = source.index("    result.precision.operator_work_counters_valid")
    end = source.index("    const std::size_t density_stride", begin)
    publication = source[begin:end]
    folder = tmp_path_factory.mktemp("precision-publication")
    path = folder / "probe.cpp"
    path.write_text(
        r"""#include "scf/types.hpp"
#include <cstdint>
#include <cstdlib>
#include <vector>
int main(int argc, char** argv) {
  if(argc != 2) return 90;
  const int scenario = std::atoi(argv[1]);
  const bool failed = scenario == 1, audited = scenario == 2;
  const std::size_t system = 0;
  const bool precision_item_mixed = false;
  const bool scf_force_ready_state = false, reuse_converged_fock = false;
  const std::vector<std::uint32_t> host_iterations{5}, host_mixed_iterations{0};
  const std::vector<std::uint64_t> host_mixed_item_census{0};
  const std::vector<double> host_mixed_item_threshold{0};
  const std::vector<std::uint8_t> host_failed{static_cast<std::uint8_t>(failed)};
  const std::vector<std::uint8_t> host_converged{static_cast<std::uint8_t>(scenario == 0)};
  const std::vector<std::uint8_t> host_final_audit_mask{static_cast<std::uint8_t>(audited)};
  const std::vector<std::uint8_t> host_final_fock_reuse_mask{0};
  const struct { double item_budget_error; } requested_precision_policy{0};
  vibeqc::scf::ScfResult result;
"""
        + publication
        + r"""
  // A final eigensolver can reject an item after its physical Fock was built.
  // Final-state iteration/status counters cannot reconstruct that failed work.
  if(bool(result.precision.operator_work_counters_valid) != !failed) return 1;
  if(result.precision.strict_stage_fock_builds != 5) return 2;
  if(!failed && result.precision.post_scf_fock_builds != (audited ? 2U : scenario == 0 ? 1U : 0U)) return 3;
  if(result.precision.final_residual_audits != (audited ? 1U : 0U)) return 4;
  return 0;
}
"""
    )
    binary = folder / "probe"
    compiled = subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O0",
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
            str(path),
            "-o",
            str(binary),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert compiled.returncode == 0, compiled.stderr
    return binary


@pytest.mark.parametrize(
    "scenario",
    [0, 1, 2, 3],
    ids=[
        "success",
        "late-numerical-failure",
        "audit-rejection",
        "iteration-exhaustion",
    ],
)
def test_precision_work_validity_tracks_complete_item_accounting(
    publication_binary: Path, scenario: int
) -> None:
    result = subprocess.run(
        [str(publication_binary), str(scenario)],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, (scenario, result.returncode, result.stderr)
