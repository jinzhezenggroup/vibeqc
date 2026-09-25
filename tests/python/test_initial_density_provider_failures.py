"""Execute the production seed guards with a fixed, device-free core frame."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def seed_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    source = (ROOT / "src/scf/initial_guess/density.cpp").read_text(encoding="utf-8")
    normalization = source.split("Matrix normalized_warm_density(", 1)[1].split(
        "std::pair<Matrix, Matrix> normalized_warm_uhf_density", 1
    )[0]
    preparation = source.split("Matrix prepare_initial_density(", 1)[1].split(
        "}  // namespace vibeqc::scf::initial_guess", 1
    )[0]
    prefix = r"""
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
namespace vibeqc {
namespace core { struct System { int electron_count = 2; }; }
namespace integrals {
struct IntegralData {
  std::size_t nbf = 2;
  std::vector<double> overlap{1, 0, 0, 1}, hcore{-1, 0, 0, 1};
};
}
namespace scf::reference {
using Matrix = std::vector<double>;
struct EigenResult { Matrix values{-1, 1}, vectors{1, 0, 0, 1}; };
unsigned solves = 0;
EigenResult generalized_eigen(const Matrix&, const Matrix&, std::size_t) {
  ++solves; return {};
}
Matrix density_from_orbitals(const Matrix&, std::size_t, std::size_t) {
  return {2, 0, 0, 0};
}
std::size_t index(std::size_t i, std::size_t j, std::size_t n) { return i*n+j; }
namespace observation {
enum class EigenReason { core_guess };
struct Reason { explicit Reason(EigenReason) {} };
struct Scope { Scope(const char*, std::size_t) {} };
}
}
namespace scf::initial_guess {
using namespace reference;
using EigenOperation = std::function<EigenResult(const Matrix&, const Matrix*, const Matrix*, std::size_t)>;
struct RestrictedInitialDensityRequest {
  const core::System& system;
  const integrals::IntegralData& integrals;
  const Matrix& orthogonalizer;
  const Matrix& core_density;
  std::size_t occupied;
};
using RestrictedInitialDensityProvider = std::function<std::optional<Matrix>(const RestrictedInitialDensityRequest&)>;
enum class InitialOrbitalRequest { ColdDensityOnly, RequireCoreFrame };
"""
    driver = r"""
}}
int main(int argc, char** argv) {
  if (argc != 2) return 100;
  using namespace vibeqc;
  using namespace scf::initial_guess;
  const std::string mode = argv[1];
  core::System system;
  integrals::IntegralData ints;
  const Matrix x{1, 0, 0, 1}, good{1, .125, .125, 1};
  const Matrix tiny{1e-310, 0, 0, 1e-310};
  unsigned calls = 0;
  RestrictedInitialDensityProvider provider = [&](const RestrictedInitialDensityRequest& r) -> std::optional<Matrix> {
    ++calls;
    if (&r.system != &system || &r.integrals != &ints || r.core_density != Matrix{2,0,0,0})
      throw std::bad_alloc();
    if (mode == "none") return std::nullopt;
    if (mode == "exception") throw std::runtime_error("provider failure");
    if (mode == "allocation") throw std::bad_alloc();
    if (mode == "shape") return Matrix{1};
    if (mode == "zero") return Matrix(4, 0);
    if (mode == "nonfinite") return Matrix(4, std::numeric_limits<double>::infinity());
    if (mode == "tiny") return tiny;
    if (mode == "scaled-overflow") return Matrix{1e-100, 1e300, 1e300, 1e-100};
    return good;
  };
  const bool warm = mode == "warm" || mode == "warm-overflow";
  const Matrix* initial = !warm ? nullptr : mode == "warm" ? &good : &tiny;
  std::optional<EigenResult> frame;
  const auto request = mode == "frame" ? InitialOrbitalRequest::RequireCoreFrame
                                      : InitialOrbitalRequest::ColdDensityOnly;
  Matrix result;
  try {
    result = prepare_initial_density(system, ints, x, 1, initial, frame, request, {}, provider);
  } catch (const std::bad_alloc&) {
    return mode == "allocation" && calls == 1 ? 0 : 101;
  } catch (const std::invalid_argument&) {
    return mode == "warm-overflow" && calls == 0 && !frame ? 0 : 102;
  }
  if (mode == "allocation" || mode == "warm-overflow") return 103;
  if (!std::all_of(result.begin(), result.end(), [](double v) { return std::isfinite(v); })) return 104;
  if (calls != (warm ? 0u : 1u) || scf::reference::solves != (warm ? 0u : 1u)) return 105;
  const bool accepted = mode == "valid" || mode == "frame" || warm;
  if (result != (accepted ? good : Matrix{2,0,0,0})) return 106;
  if (frame.has_value() != (!accepted || mode == "frame")) return 107;
}
"""
    directory = tmp_path_factory.mktemp("seed-provider")
    cpp, executable = directory / "probe.cpp", directory / "probe"
    cpp.write_text(
        prefix
        + "Matrix normalized_warm_density("
        + normalization
        + "Matrix prepare_initial_density("
        + preparation
        + driver,
        encoding="utf-8",
    )
    subprocess.run(
        [compiler, "-std=c++20", "-O2", str(cpp), "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return executable


@pytest.mark.parametrize(
    "mode",
    [
        "valid",
        "frame",
        "none",
        "exception",
        "allocation",
        "shape",
        "zero",
        "nonfinite",
        "tiny",
        "scaled-overflow",
        "warm",
        "warm-overflow",
    ],
)
def test_seed_provider_failure_and_frame_contract(seed_probe: Path, mode: str) -> None:
    completed = subprocess.run(
        [str(seed_probe), mode], capture_output=True, text=True, timeout=10
    )
    assert completed.returncode == 0, (mode, completed.returncode, completed.stderr)
