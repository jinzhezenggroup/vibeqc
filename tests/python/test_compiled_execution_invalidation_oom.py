"""Shared execution invalidation must survive exhausted diagnostic allocation."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def invalidation_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    directory = tmp_path_factory.mktemp("compiled-invalidation")
    source = directory / "probe.cpp"
    source.write_text(
        r'''
#include <cstdlib>
#include <exception>
#include <new>
#include "runtime/compiled_execution_region.hpp"
static bool exhausted = false;
void* operator new(std::size_t n) {
  if (exhausted) throw std::bad_alloc();
  if (void* p = std::malloc(n ? n : 1)) return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
int main(int argc, char** argv) {
  if (argc != 3) return 1;
  std::set_terminate([] { std::_Exit(91); });
  using namespace vibeqc::runtime;
  const int stage = std::atoi(argv[1]);
  CompiledExecutionRegion region;
  CompiledExecutionBinding binding{"valid", 0, nullptr, nullptr, nullptr};
  if (stage > 0) region.bind(binding);
  if (stage == 2) region.mark_success();
  if (stage == 3) region.mark_failure("short failure");
  exhausted = std::atoi(argv[2]) != 0;
  region.invalidate();
  exhausted = false;
  if (region.bound() || region.warmed() || region.failed() || region.matches(binding)) return 2;
  if (region.metrics().invalidations != (stage > 0 ? 1U : 0U)) return 3;
  if (region.metrics().executions != (stage == 2 ? 1U : 0U)) return 4;
  if (region.metrics().failures != (stage == 3 ? 1U : 0U)) return 5;
  if (std::atoi(argv[2]) == 0 && region.reason() != "compiled execution region invalidated") return 6;
  if (!region.reason().empty() && region.reason() != "compiled execution region invalidated") return 7;
  region.invalidate();
  if (region.metrics().invalidations != (stage > 0 ? 1U : 0U)) return 8;
  region.bind(binding);
  region.mark_success();
  if (!region.bound() || !region.warmed() || region.failed()) return 9;
  return 0;
}
''',
        encoding="utf-8",
    )
    executable = directory / "probe"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-I" + str(ROOT / "src"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return executable


@pytest.mark.parametrize(
    "stage", range(4), ids=["unbound", "bound", "warmed", "failed"]
)
@pytest.mark.parametrize("exhausted", [False, True])
def test_invalidation_cannot_terminate_on_diagnostic_allocation(
    invalidation_probe: Path, stage: int, exhausted: bool
) -> None:
    result = subprocess.run(
        [str(invalidation_probe), str(stage), str(int(exhausted))],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, (
        f"native invalidation exited {result.returncode}; 91 indicates std::terminate\n"
        + result.stdout
        + result.stderr
    )
