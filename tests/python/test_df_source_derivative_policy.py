"""Host-only source/derivative policy regression; no GPU or runtime import."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_source_response_derivative_policy_matrix(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    root = Path(__file__).resolve().parents[2]
    source = tmp_path / "source_policy.cpp"
    source.write_text(r"""
#include "scf/df_derivative_policy.hpp"
int main() {
  using namespace vibeqc::scf;
  for (unsigned bits = 0; bits != 32; ++bits) {
    const bool source = bits & 1, raw = bits & 2, fitted = bits & 4;
    const bool validated = bits & 8, qualify = bits & 16;
    const bool expected = !source || raw || fitted || (qualify && validated);
    if (df_response_shell_source_eligible(source, raw, fitted, validated, qualify)
        != expected) return 1;
    if (!qualify &&
        df_response_shell_source_eligible(source, raw, fitted, validated, false)
          != (!source || raw || fitted)) return 2;
  }
  if (df_response_shell_source_eligible(true, false, false, true, false)) return 3;
  if (df_response_shell_source_eligible(true, false, false, false, true)) return 4;
  if (!df_response_shell_source_eligible(true, false, false, true, true)) return 5;
  // Representation eligibility does not manufacture a target profile.
  if (df_shell_execution_preferred(768, 3712, 0)) return 6;
  if (!df_shell_execution_preferred(768, 3712, 120)) return 7;
}
""")
    executable = tmp_path / "source_policy"
    subprocess.run([compiler, "-std=c++20", "-Wall", "-Wextra", "-Werror",
                    f"-I{root / 'src'}", str(source), "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True)
