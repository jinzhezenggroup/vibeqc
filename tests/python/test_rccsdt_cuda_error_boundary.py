"""Execute the emitted CUDA error adapter without allocating a real GPU."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def adapter(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    source = (ROOT / "tools/generate_rccsdt_cuda.py").read_text(encoding="utf-8")
    begin = source.index("void cuda_check(cudaError_t error, const char* what)")
    end = source.index("std::size_t checked_add", begin)
    body = source[begin:end].replace("{{", "{").replace("}}", "}")
    directory = tmp_path_factory.mktemp("triples-cuda-error")
    unit, executable = directory / "error.cpp", directory / "error"
    unit.write_text(
        "#include <cstdlib>\n#include <new>\n#include <stdexcept>\n#include <string>\n"
        "using cudaError_t = int;\n"
        "constexpr int cudaSuccess = 0, cudaErrorMemoryAllocation = 2;\n"
        'const char* cudaGetErrorString(int) { return "injected CUDA error"; }\n'
        + body
        + r"""
int main(int argc, char** argv) {
  if (argc != 2) return 99;
  try {
    cuda_check(std::atoi(argv[1]), "triples allocation");
    return 0;
  } catch (const std::bad_alloc&) {
    return 1;
  } catch (const std::runtime_error& error) {
    return std::string(error.what()) ==
        "triples allocation: injected CUDA error" ? 2 : 3;
  }
}
""",
        encoding="utf-8",
    )
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(unit),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return executable


@pytest.mark.parametrize("status,expected", [(0, 0), (2, 1), (10, 2), (999, 2)])
def test_cuda_error_adapter_preserves_oom_and_other_diagnostics(
    adapter: Path, status: int, expected: int
) -> None:
    result = subprocess.run(
        [str(adapter), str(status)], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == expected, result.stderr
