"""Host-simulate the flat kernel ABI and pair ownership, not CUDA math."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_flat_one_electron_launch_preserves_fields_and_pair_ownership(
    tmp_path: Path,
) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = (ROOT / "src/scf/cuda/one_electron_values.cu").read_text()
    header = (ROOT / "src/scf/cuda/one_electron_values.cuh").read_text()
    view = (
        "struct OneElectronDeviceView {"
        + header.split("struct OneElectronDeviceView {", 1)[1].split("};", 1)[0]
        + "};"
    )
    abi = (ROOT / "src/scf/cuda/one_electron_kernel_abi.cuh").read_text()
    abi = abi.replace('#include "scf/cuda/one_electron_values.cuh"', "")
    runtime = (ROOT / "src/runtime/cuda_ao_pairs.cuh").read_text()
    template = (
        "template <class Policy, std::size_t TermCapacity, class... OutputPointers>\n"
    )
    begin = runtime.index(
        template + "__device__ __forceinline__ void thread_pairs_body"
    )
    end = runtime.index("}  // namespace vibeqc::runtime::cuda_ao_pairs", begin)
    bodies = runtime[begin:end].replace(
        "evaluate_pair<Policy, TermCapacity>", "::capture_pair"
    )
    kernel = (
        "__global__ void thread_pairs_flat"
        + source.split("__global__ void thread_pairs_flat", 1)[1].split(
            "}  // namespace", 1
        )[0]
    )
    launches = []
    for name in ("shell_warp_pairs_flat", "thread_pairs_flat"):
        match = re.search(rf"{name}<<<[^>]+>>>\((.*?)\);", source, re.DOTALL)
        assert match is not None
        launches.append(f"{name}({match.group(1)});")
    harness = (
        r"""
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>
#define __global__
#define __device__
#define __forceinline__ inline
constexpr std::size_t kTerms=3;
struct Dim { unsigned x{}; } blockIdx, blockDim, threadIdx;
struct Policy { static constexpr unsigned channels = 4; };
"""
        + view
        + r"""
OneElectronDeviceView expected;
double* expected_outputs[4]{};
std::vector<unsigned> seen;
void check(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
void capture_pair(
    std::int32_t nbf, const std::int64_t* atom_offsets, const std::int32_t* atomic_numbers,
    const double* positions, const std::int32_t* shell_atoms,
    const std::int64_t* shell_primitive_offsets, const std::int32_t* ao_shells,
    const std::uint8_t* ao_term_counts, const std::uint8_t* ao_term_angular,
    const double* ao_term_coefficients, const double* primitive_exponents,
    const double* primitive_coefficients, std::int64_t i, std::int64_t j,
    double* overlap, double* hcore, double* kinetic, double* attraction) {
  check(nbf == expected.nbf, "nbf");
  check(atom_offsets == expected.atom_offsets, "atom_offsets");
  check(atomic_numbers == expected.atomic_numbers, "atomic_numbers");
  check(positions == expected.positions, "positions");
  check(shell_atoms == expected.shell_atoms, "shell_atoms");
  check(shell_primitive_offsets == expected.shell_primitive_offsets, "shell_primitive_offsets");
  check(ao_shells == expected.ao_shells, "ao_shells");
  check(ao_term_counts == expected.ao_term_counts, "ao_term_counts");
  check(ao_term_angular == expected.ao_term_angular, "ao_term_angular");
  check(ao_term_coefficients == expected.ao_term_coefficients, "ao_term_coefficients");
  check(primitive_exponents == expected.primitive_exponents, "primitive_exponents");
  check(primitive_coefficients == expected.primitive_coefficients, "primitive_coefficients");
  check(overlap == expected_outputs[0] && hcore == expected_outputs[1], "required outputs");
  check(kinetic == expected_outputs[2] && attraction == expected_outputs[3], "optional outputs");
  check(i>=0 && j>=0 && i<26 && j<26 && i/13==j/13 && i>=j, "pair domain");
  check(++seen[i*26+j]==1, "duplicate pair");
}
"""
        + "\nnamespace vibeqc::scf { using ::OneElectronDeviceView; }\n"
        + abi
        + "\nnamespace pairs {\n"
        + bodies
        + "\n}\n"
        + kernel
        + r"""
int main() {
  std::int64_t atoms[]{0,2,4}, ao_offsets[]{0,6,13,19,26}, primitives[]{0,1,3,4,6};
  std::int32_t numbers[]{1,2,3,4}, shell_atoms[]{0,1,2,3};
  std::int32_t first_shell[]{0,1,1,2,3,3}, second_shell[]{0,0,1,2,2,3}, ao_shells[26]{};
  std::uint8_t term_counts[26]{}, term_angular[26*3*3]{};
  double positions[12]{}, coefficients[26*3]{}, exponents[6]{}, primitive_coefficients[6]{};
  OneElectronDeviceView batch{2,13,6,atoms,numbers,positions,shell_atoms,ao_offsets,primitives,
      first_shell,second_shell,ao_shells,term_counts,term_angular,coefficients,exponents,
      primitive_coefficients};
  expected=batch;
  std::vector<std::int32_t> first,second;
  for(int i=12;i>=0;--i) for(int j=i;j>=0;--j) { first.push_back(i);second.push_back(j); }
  const auto* pair_first=first.data();const auto* pair_second=second.data();
  const std::size_t pair_count=first.size();
  double output[4][338]{};
  for(bool shell:{false,true}) for(bool optional:{false,true}) {
    double* overlap=output[0];double* hcore=output[1];
    double* kinetic=optional?output[2]:nullptr;double* attraction=optional?output[3]:nullptr;
    expected_outputs[0]=overlap;expected_outputs[1]=hcore;
    expected_outputs[2]=kinetic;expected_outputs[3]=attraction;seen.assign(26*26,0);
    blockDim.x=128;
    for(blockIdx.x=0;blockIdx.x<3;++blockIdx.x) for(threadIdx.x=0;threadIdx.x<128;++threadIdx.x) {
      if(shell) { TRUE_LAUNCH } else { FALSE_LAUNCH }
    }
    for(unsigned i=0;i<26;++i) for(unsigned j=0;j<26;++j)
      check(seen[i*26+j]==unsigned(i/13==j/13 && i>=j), "missing or excess pair");
  }
}
"""
    )
    harness = harness.replace("TRUE_LAUNCH", launches[0]).replace(
        "FALSE_LAUNCH", launches[1]
    )
    source_file, executable = tmp_path / "launch.cpp", tmp_path / "launch"
    source_file.write_text(harness)
    subprocess.run(
        [compiler, "-std=c++20", "-O2", str(source_file), "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    subprocess.run([str(executable)], check=True, capture_output=True, timeout=10)


def test_device_call_graph_has_no_one_electron_view_aggregate() -> None:
    source = (ROOT / "src/scf/cuda/one_electron_values.cu").read_text()
    runtime = (ROOT / "src/runtime/cuda_ao_pairs.cuh").read_text()
    policy = (
        ROOT / "python/vibeqc_compiler/integral/one_electron_policy_cuda.py"
    ).read_text()
    derivatives = (ROOT / "src/scf/cuda/one_electron_derivatives.cu").read_text()
    derivative_device = derivatives.split(
        "}  // namespace\n\ncudaError_t launch_generated_one_electron_gradient", 1
    )[0]
    assert "OneElectronDeviceView batch" not in source
    assert "const View& batch" not in runtime
    assert "Outputs<" not in runtime
    assert "const View& batch" not in policy
    assert "OneElectronDeviceView" not in derivative_device
    assert "OneElectronWeightView" not in derivative_device