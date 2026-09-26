"""A failed KS proposal cannot keep an intermediate orbital frame eligible."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _method(source: str, signature: str) -> str:
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


@pytest.fixture(scope="module")
def orbital_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a C++ compiler")
    source = (ROOT / "src/dft/cuda_ks.cpp").read_text()
    methods = "\n".join(
        _method(source, signature)
        for signature in (
            "  void invalidate_warm_orbitals()",
            "  void clear_warm_state()",
            "  void enqueue()",
            "  bool finish()",
        )
    )
    legacy = _method(source, "  bool finish_legacy()")
    begin = legacy.index("    try {\n      if (output.converged && warm_updates)")
    end = legacy.index("    previous_energy =", begin)
    # Compile the actual accepted-proposal copy/exception block and entry
    # guards. Only the CUDA calls and unrelated solver stages are test doubles.
    publication = legacy[begin:end]
    harness = (
        r"""
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
constexpr int cudaMemcpyDeviceToDevice = 3;
int copies = 0, fail_copy = 0, fences = 0;
int cudaMemcpyAsync(void* target, const void* source, std::size_t bytes, int kind, void*) {
  assert(kind == cudaMemcpyDeviceToDevice);
  if (++copies == fail_copy) return 1;
  std::memcpy(target, source, bytes);
  return 0;
}
int cudaStreamSynchronize(void*) { ++fences; return 0; }
void check(int status) { if (status) throw std::runtime_error("injected copy failure"); }
struct Owner {
  bool device_chunk_mode = false, is_active = true, is_failed = false;
  bool warm_ready = true, warm_orbitals_ready = false, warm_updates = true;
  bool warm_energy_baseline = false, final_state_ready = false;
  double warm_energy = std::numeric_limits<double>::infinity();
  std::size_t elements = 8;
  std::uint64_t final_generation = 0, generation = 7;
  struct { bool converged = false; } output;
  struct {
    std::uint64_t warm_orbital_frames_retained = 0, warm_orbital_frame_invalidations = 0;
  } movement;
  double density[8], proposal[8], warm[8], warm_orbitals[8], tmp2[8];
  void* stream = nullptr;
  std::string scenario;
  Owner() {
    std::fill_n(density, 8, 10.0);
    std::fill_n(warm_orbitals, 8, 10.0);
    std::fill_n(proposal, 8, 20.0);
    std::fill_n(tmp2, 8, 20.0);
    std::fill_n(warm, 8, 42.0);
  }
"""
        + methods
        + r"""
  void enqueue_legacy() {
    if (scenario == "enqueue") {
      is_failed = true;
      is_active = false;
      throw std::runtime_error("injected submission failure");
    }
  }
  void enqueue_device() { enqueue_legacy(); }
  bool finish_device() { return finish_legacy(); }
  bool finish_legacy() {
    if (scenario == "completion") {
      is_failed = true;
      is_active = false;
      throw std::runtime_error("injected completion failure");
    }
    if (scenario == "physical") {
      is_failed = true;
      is_active = false;
      return false;
    }
    if (scenario == "retry") return true;
"""
        + publication
        + r"""
    return is_active;
  }
};
int main(int argc, char** argv) {
  assert(argc == 4);
  Owner owner;
  owner.scenario = argv[1];
  const bool initially_ready = std::atoi(argv[2]) != 0;
  owner.warm_orbitals_ready = initially_ready;
  owner.elements = 4 * std::atoi(argv[3]);
  const auto& scenario = owner.scenario;
  if (scenario == "density") fail_copy = 1;
  if (scenario == "orbitals") fail_copy = 2;
  if (scenario == "converged") { owner.is_active = false; owner.output.converged = true; }
  bool thrown = false;
  try {
    owner.enqueue();
    owner.finish();
  } catch (const std::runtime_error&) { thrown = true; }
  const bool failed = scenario == "density" || scenario == "orbitals" ||
      scenario == "enqueue" || scenario == "completion" || scenario == "physical";
  assert(thrown == (failed && scenario != "physical"));
  if (failed) {
    assert(owner.is_failed && !owner.is_active);
    assert(!owner.warm_orbitals_ready);
    assert(owner.movement.warm_orbital_frame_invalidations == initially_ready);
    assert(owner.movement.warm_orbital_frames_retained == 0);
    assert(owner.warm_ready && owner.warm[0] == 42.0);
    if (scenario == "orbitals") {
      assert(owner.density[0] == 20.0 && owner.warm_orbitals[0] == 10.0);
      assert(fences == 1);
    }
    // A new successful trajectory may bind a fresh frame after rejection.
    owner.scenario = "success";
    owner.is_active = true;
    owner.is_failed = false;
    fail_copy = 0;
    owner.enqueue();
    assert(owner.finish());
    assert(owner.warm_orbitals_ready && owner.movement.warm_orbital_frames_retained == 1);
  } else if (scenario == "retry") {
    assert(owner.warm_orbitals_ready == initially_ready);
    assert(copies == 0 && owner.density[0] == 10.0);
  } else if (scenario == "converged") {
    assert(owner.warm_ready && owner.warm[0] == 10.0);
    assert(owner.warm_orbitals_ready == initially_ready);
    assert(owner.final_state_ready && owner.final_generation == owner.generation);
  } else {
    assert(owner.warm_orbitals_ready && owner.movement.warm_orbital_frames_retained == 1);
    for (std::size_t i = 0; i < owner.elements; ++i)
      assert(owner.density[i] == 20.0 && owner.warm_orbitals[i] == 20.0);
  }
  const auto before_clear = owner.movement.warm_orbital_frame_invalidations;
  const auto was_ready = owner.warm_orbitals_ready;
  owner.clear_warm_state();
  owner.clear_warm_state();
  assert(!owner.warm_ready && !owner.warm_orbitals_ready);
  assert(owner.movement.warm_orbital_frame_invalidations == before_clear + was_ready);
}
"""
    )
    directory = tmp_path_factory.mktemp("ks-orbital-failure")
    cpp, executable = directory / "probe.cpp", directory / "probe"
    cpp.write_text(harness)
    subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Werror", str(cpp), "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return executable


@pytest.mark.parametrize("spins", [1, 2])
@pytest.mark.parametrize("initially_ready", [False, True])
@pytest.mark.parametrize(
    "scenario",
    [
        "density",
        "orbitals",
        "enqueue",
        "completion",
        "physical",
        "success",
        "retry",
        "converged",
    ],
)
def test_failure_invalidates_only_the_intermediate_frame(
    orbital_probe: Path, scenario: str, initially_ready: bool, spins: int
) -> None:
    result = subprocess.run(
        [str(orbital_probe), scenario, str(int(initially_ready)), str(spins)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
