"""A failed native region must recover explicitly before recording success."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("warmed", [False, True])
def test_success_cannot_clear_sticky_failure(tmp_path: Path, warmed: bool) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a C++ compiler")
    source = tmp_path / "sticky.cpp"
    source.write_text(
        '#include "runtime/compiled_execution_region.hpp"\n'
        "#include <cassert>\n"
        "int main() {\n"
        "  using namespace vibeqc::runtime;\n"
        "  CompiledExecutionRegion region;\n"
        '  CompiledExecutionBinding binding{"qualified"};\n'
        "  region.bind(binding);\n"
        + ("  region.mark_success();\n" if warmed else "")
        + r"""
  const auto executions = region.metrics().executions;
  region.mark_failure("injected");
  assert(!region.bind(binding));
  bool rejected = false;
  try { region.mark_success(); }
  catch (const std::logic_error&) { rejected = true; }
  assert(rejected);
  assert(region.failed() && region.reason() == "injected");
  assert(region.metrics().executions == executions);
  assert(region.metrics().failures == 1 && region.metrics().recoveries == 0);
  region.recover();
  assert(!region.failed() && !region.warmed());
  assert(region.metrics().recoveries == 1);
  region.mark_success();
  assert(region.warmed() && region.metrics().executions == executions + 1);
  region.mark_failure("again");
  binding.qualification = "replacement";
  assert(region.bind(binding));
  region.mark_success();
  assert(!region.failed() && region.metrics().invalidations == 1);
}
"""
    )
    binary = tmp_path / "sticky"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-I",
            str(ROOT / "src"),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
