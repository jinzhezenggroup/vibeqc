"""Exercise bounded preparation directly, without hiding it behind an SCF timeout."""

import json
import os
import shutil
import subprocess
import typing
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)


@pytest.fixture(scope="module")
def preparation_probe(tmp_path_factory: typing.Any) -> typing.Any:
    assert os.environ.get("SLURM_JOB_ID")
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("host C++ compiler unavailable")
    library = Path(os.environ["VIBEQC_LIBRARY"]).resolve()
    root = Path(__file__).resolve().parents[2]
    directory = tmp_path_factory.mktemp("df-preparation")
    source = directory / "prepare.cpp"
    source.write_text(
        r"""
#include "molecule/basis.hpp"
#include "scf/density_fitting.hpp"
#include "scf/df_preparation_budget.hpp"
#include <iostream>
#include <string>

namespace vibeqc::scf {
// Internal native preparation entry: isolate its live-set policy from SCF/J/K.
std::vector<std::optional<DensityFittingScfData>> prepare_cuda_density_fitting_batch(
    const std::vector<core::System>&, const std::optional<core::System>&,
    double, std::size_t, int, std::vector<vibeqc_status>&, bool);
}
int main(int argc, char** argv) {
  using namespace vibeqc;
  if (argc != 4 || !std::getenv("SLURM_JOB_ID")) return 1;
  const bool spherical = std::stoi(argv[1]), forces = std::stoi(argv[2]);
  const std::string mode = argv[3];
  core::System system;
  system.basis_representation = spherical ? VIBEQC_BASIS_SPHERICAL : VIBEQC_BASIS_CARTESIAN;
  for (std::size_t atom = 0; atom < 8; ++atom) {
    system.atoms.push_back({2, {0.0, 0.0, 3.0 * atom}});
    for (unsigned l = 0; l < 3; ++l)
      system.shells.push_back({atom, l, {{0.7 + 0.2 * l, 1.0}}});
  }
  std::string detail;
  if (molecule::validate_and_normalize(system, detail) != VIBEQC_STATUS_SUCCESS) return 2;
  const auto c = molecule::cartesian_ao_count(system), n = molecule::ao_count(system);
  scf::DfPreparationShape shape{c, n, 8, 24, 24, 24, 24, forces, true};
  const auto storage = scf::df_preparation_storage(shape);
  const std::size_t count = mode == "partial" ? 4 : 1;
  std::size_t budget = count * storage.metadata_bytes + storage.peak_bytes;
  if (mode == "partial") budget += storage.retained_bytes;  // Fits two of four.
  if (mode == "reject") --budget;
  if (mode == "small") budget = 2U << 20;
  if (mode == "zero") budget = 0;
  std::vector<vibeqc_status> statuses;
  const auto prepared = scf::prepare_cuda_density_fitting_batch(
      std::vector<core::System>(count, system), std::nullopt, 1e-10, budget, 0, statuses, forces);
  std::size_t accepted = 0, derivative_values = 0, nuclear_values = 0, generated_owners = 0;
  for (std::size_t i = 0; i < count; ++i) {
    if (!prepared[i]) {
      if (statuses[i] != VIBEQC_STATUS_OUT_OF_MEMORY) return 3;
      continue;
    }
    ++accepted;
    const auto& data = *prepared[i];
    if (data.one_electron.overlap.size() != n*n || data.one_electron.hcore.size() != n*n) return 4;
    derivative_values += data.one_electron.overlap_derivative.size() + data.one_electron.hcore_derivative.size();
    nuclear_values += data.one_electron.nuclear_repulsion_derivative.size();
    generated_owners += data.one_electron_gradient_system.has_value();
  }
  std::cout << "[" << accepted << "," << derivative_values << "," << nuclear_values
            << "," << generated_owners << "," << n << "," << budget << "]\n";
}
"""
    )
    executable = directory / "prepare"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-I" + str(root / "src"),
            "-I" + str(root / "include"),
            str(source),
            str(library),
            "-Wl,-rpath," + str(library.parent),
            "-o",
            str(executable),
        ],
        check=True,
    )
    return executable


@pytest.mark.parametrize("spherical", [False, True])
@pytest.mark.parametrize("forces", [False, True])
@pytest.mark.parametrize("mode", ["exact", "reject", "partial", "small", "zero"])
def test_preparation_owns_generated_response_and_prior_items(
    preparation_probe: typing.Any,
    spherical: typing.Any,
    forces: typing.Any,
    mode: typing.Any,
) -> None:
    values = json.loads(
        subprocess.check_output(
            [
                str(preparation_probe),
                str(int(spherical)),
                str(int(forces)),
                mode,
            ],
            text=True,
        )
    )
    accepted, derivatives, nuclear, owners, n, _ = values
    expected = 0 if mode == "reject" else 2 if mode == "partial" else 1
    assert accepted == expected
    assert derivatives == 0
    assert nuclear == (accepted * 24 if forces else 0)
    assert owners == (accepted if forces else 0)
