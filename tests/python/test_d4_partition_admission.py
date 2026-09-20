"""Serial host execution of the actual D4 partition guards, not GPU numerics."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _definition(source: str, name: str) -> str:
    match = re.search(
        r"(?:__global__|__device__) [^{;]*\b" + name + r"\([^;]*?\)\s*\{", source
    )
    assert match is not None, name
    start = match.start()
    depth = 1
    end = match.end()
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return (
        source[start:end]
        .replace("__global__", "")
        .replace("__device__", "")
        .replace("__shared__", "static")
    )


def test_invalid_partition_is_rejected_before_any_member_work(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler required")
    source = (ROOT / "src/dft/dispersion/d4_cuda.cu").read_text()
    # Real math/table types; replace only CUDA execution mechanics by a serial lane.
    prefix = r"""
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include "dft/dispersion/d4_reference.hpp"
using namespace vibeqc::dft::dispersion;
struct Dim { unsigned x; } blockIdx{0}, blockDim{1}, threadIdx{0};
void __syncthreads() {}
int atomicCAS(int* p, int compare, int value) { int old=*p; if (old==compare) *p=value; return old; }
int atomicExch(int* p, int value) { int old=*p; *p=value; return old; }
double atomic_add_fp64(double* p, double value) { double old=*p; *p+=value; return old; }
"""
    header = (ROOT / "src/dft/dispersion/d4_cuda.hpp").read_text()
    prefix += header[
        header.index("struct D4CudaBatch") : header.index("inline constexpr")
    ]
    names = ["record_status", "active_member", "unpack_pair"]
    partition = "validate_partition_kernel" in source
    if partition:
        names.append("validate_partition_kernel")
    names += ["validate_and_coordination_kernel", "finalize_kernel"]
    program = prefix + "\n".join(_definition(source, name) for name in names)
    program += r"""
int main() {
  const std::int32_t z[6]{1,1,1,1,1,1};
  const double xyz[18]{0,0,0, 2,0,0, 0,2,0, 4,0,0, 6,0,0, 4,2,0};
  const double q[6]{};
"""
    program += r"""
  const std::array<std::array<std::uint32_t,5>,4> cases{{
    {{0,3,1,4,6}}, {{1,3,4,4,6}}, {{0,3,4,4,5}}, {{0,3,UINT32_MAX,4,6}}
  }};
  for (const auto& offsets: cases) {
    std::array<D4Status,4> statuses{};
    std::array<double,8> energies{};
    std::array<double,18> gradients{};
    std::array<double,6> dedq{};
    std::array<double,162> workspace{};
    D4CudaBatch batch{4,6,offsets.data(),z,xyz,q,nullptr};
    D4CudaResult result{statuses.data(),energies.data(),gradients.data(),dedq.data()};
    PARTITION_GUARD
    for (blockIdx.x=0;blockIdx.x<4;++blockIdx.x)
      validate_and_coordination_kernel(batch,gfn2_d4_parameters(),gfn2_d4_host_tables(),workspace.data(),result);
    for (auto status: statuses) if (status!=D4Status::invalid_argument) return 10;
    for (blockIdx.x=0;blockIdx.x<4;++blockIdx.x) finalize_kernel(batch,workspace.data(),result);
    for (auto value: gradients) if (value!=0) return 11;
  }
}
""".replace(
        "PARTITION_GUARD",
        "validate_partition_kernel(batch,result);" if partition else "",
    )
    generated = tmp_path / "generated_method_parameters.hpp"
    subprocess.run(
        [
            "python3",
            str(ROOT / "tools/generate_method_parameters.py"),
            "--source",
            str(ROOT / "python/vibeqc_compiler/method/method_parameters.json"),
            "--cpp-output",
            str(generated),
        ],
        check=True,
        cwd=ROOT,
    )
    path = tmp_path / "partition.cpp"
    path.write_text(program)
    binary = tmp_path / "partition"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O1",
            "-I" + str(ROOT / "src"),
            "-I" + str(tmp_path),
            str(path),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run([str(binary)], check=False, capture_output=True, text=True)
    assert result.returncode == 0, "malformed partition admitted a member: " + str(
        result.returncode
    )
