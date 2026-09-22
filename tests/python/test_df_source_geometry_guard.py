"""Execute the real force-response entry guard without running CUDA work."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_generated_response_rejects_stale_geometry_before_work(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = (ROOT / "src/scf/cuda/df_force_response.cpp").read_text()
    start = source.index(
        "vibeqc_status execute_cuda_density_fitting_generated_force_response("
    )
    start = source.index("{\n", start) + 2
    end = source.index("  const auto elements =", start)
    guard = source[start:end]
    program = (
        r"""
#include <cstddef>
#include <string>
constexpr int VIBEQC_STATUS_INVALID_ARGUMENT = 1;
struct System { std::size_t count{2}; int geometry{}; };
namespace molecule { std::size_t ao_count(const System& s) { return s.count; } }
struct Source { int orbital{}, auxiliary{}; };
struct Plan { std::size_t batch_size{1}, nbf{2}, naux{2}; Source* integral_source{}; };
bool cuda_density_fitting_integral_source_geometry_matches(
    const Source* source, std::size_t system, const System& orbital, const System& auxiliary) {
  return source && system == 0 && source->orbital == orbital.geometry &&
         source->auxiliary == auxiliary.geometry;
}
int execute(Plan* plan, std::size_t system, System orbital, System auxiliary) {
  int* resources = nullptr;
  std::string detail;
"""
        + guard
        + r"""
  return 0;
}
int main() {
  Source source{}; Plan plan; plan.integral_source = &source;
  if (execute(&plan, 0, {}, {}) != 0) return 10;
  if (execute(&plan, 0, {2, 1}, {}) != VIBEQC_STATUS_INVALID_ARGUMENT) return 11;
  if (execute(&plan, 0, {}, {2, 1}) != VIBEQC_STATUS_INVALID_ARGUMENT) return 12;
  if (execute(&plan, 1, {}, {}) != VIBEQC_STATUS_INVALID_ARGUMENT) return 13;
  plan.integral_source = nullptr;
  if (execute(&plan, 0, {}, {}) != 0) return 14;
}
"""
    )
    path = tmp_path / "guard.cpp"
    path.write_text(program)
    exe = tmp_path / "guard"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(path),
            "-o",
            str(exe),
        ],
        check=True,
    )
    result = subprocess.run([str(exe)], check=False)
    assert result.returncode == 0, (
        "stale orbital/auxiliary geometry reached response work"
    )


def test_exact_source_identity_detects_geometry_and_basis_changes(
    tmp_path: Path,
) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = tmp_path / "identity.cpp"
    source.write_text(r"""
#include "molecule/basis_geometry_identity.hpp"
#include <cassert>
#include <cmath>
int main() {
  using namespace vibeqc;
  core::System original;
  original.atoms = {{1, {0.0, 0.1, 0.2}, 0}, {8, {0.5, -0.4, 1.0}, 0}};
  original.shells = {{0, 0, {{1.0, 0.7}, {0.3, 0.4}}}, {1, 1, {{2.0, 0.9}}}};
  molecule::BasisGeometryIdentity identity(original);
  assert(identity.matches(original));
  auto copy = original;
  assert(identity.matches(copy)); // Equal value, distinct allocations.
  copy.charge = 1; copy.multiplicity = 2; copy.electron_count = 8;
  assert(identity.matches(copy)); // DF integral identity is not occupation policy.
  for (int change = 0; change < 11; ++change) {
    auto modified = original;
    switch (change) {
      case 0: modified.atoms[0].position[0] = std::nextafter(0.0, 1.0); break;
      case 1: modified.atoms[0].atomic_number = 2; break;
      case 2: modified.atoms[0].ecp_core = 2; break;
      case 3: modified.atoms.push_back(original.atoms[0]); break;
      case 4: modified.shells[0].atom_index = 1; break;
      case 5: modified.shells[0].angular_momentum = 1; break;
      case 6: modified.shells[0].primitives[0].exponent = 1.01; break;
      case 7: modified.shells[0].primitives[0].coefficient = 0.701; break;
      case 8: modified.shells[0].primitives.pop_back(); break;
      case 9: modified.basis_representation = VIBEQC_BASIS_SPHERICAL; break;
      case 10: std::swap(modified.shells[0], modified.shells[1]); break;
    }
    assert(!identity.matches(modified));
  }
  original.atoms[0].position[0] = 10.0; // Mutating the source object cannot rebind identity.
  assert(!identity.matches(original));
  assert(identity.storage_bytes() >= 8U * (3U + 10U + 1U + 6U + 6U));
}
""")
    executable = tmp_path / "identity"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
    )
    subprocess.run([str(executable)], check=True)


def test_exact_identity_allocations_are_in_both_source_ledgers() -> None:
    source = (ROOT / "src/scf/cuda/df_source_setup.cpp").read_text()
    assert "capacity_bytes(candidate->orbital_identities)" in source
    assert "capacity_bytes(candidate->auxiliary_identities)" in source
    assert "identity_bytes += identity.storage_bytes()" in source
    assert "host_peak += identity_bytes" in source
    retained = source.split("const long double retained_host =", 1)[1].split(";", 1)[0]
    assert "identity_bytes" in retained
