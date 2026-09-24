"""The actual append boundary must retire borrowed host inputs on errors too."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def append_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    source = (ROOT / "src/integrals/first_gradient_runtime.cuh").read_text()
    start = source.index("template <class Program>\nvoid append(")
    append = source[start : source.index("template <class Operation>", start)]
    # Compile production control flow verbatim, replacing only CUDA launch
    # syntax with a submission stub. This is not a generated-math/GPU test.
    append, replacements = re.subn(
        r"execute<Program><<<.*?>>>", "submit<Program>", append, flags=re.DOTALL
    )
    assert replacements == 1
    harness = r"""
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
struct DeviceError : std::runtime_error {
  int status;
  explicit DeviceError(int value): std::runtime_error("injected"), status(value) {}
};
int failure=0, drain_failure=0, copies=0, launches=0, event_waits=0, drains=0;
bool host_pending=false, kernel_pending=false;
constexpr int cudaMemcpyHostToDevice=1;
int cudaMemcpyAsync(void*, const void*, std::size_t, int, int) {
  ++copies;
  if(failure==1) return 101;
  host_pending=true; return 0;
}
int cudaEventRecord(int,int) { return failure==2 ? 102 : 0; }
int cudaGetLastError() { return failure==3 ? 103 : 0; }
int cudaEventSynchronize(int) {
  ++event_waits;
  if(failure==4) return 104;
  host_pending=false; return 0;
}
int cudaStreamSynchronize(int) {
  ++drains; host_pending=kernel_pending=false;
  return drain_failure ? 999 : 0;
}
namespace vibeqc_tensor {
void cuda_check(int status) { if(status) throw DeviceError(status); }
}
struct Mapping { std::size_t offsets[4]{}, atoms[4]{}; };
constexpr std::size_t stride=17;
struct Plan {
  struct { std::mutex mutex; int begin=1, stream=1; int* error=nullptr; } context;
  bool valid=true;
  std::size_t capacity=2, weight_slots=1, nbf=4, natoms=4, record_offset=0, output_offset=34;
  double arena[64]{};
  double* data() { return arena; }
  void check(const char*) {}
};
struct Program {
  static constexpr unsigned components=3, weight_slots=1, exponents=2, centers=2;
  static std::size_t extent(unsigned) { return 1; }
};
template<class T> void submit(const double*,std::size_t,Mapping,std::size_t,
                              const double*,double*,int*) {
  ++launches; kernel_pending=true;
}
""" + append + r"""
int main(int argc, char** argv) {
  assert(argc==4);
  failure=std::atoi(argv[1]); drain_failure=std::atoi(argv[2]);
  const auto count=static_cast<std::size_t>(std::atoi(argv[3]));
  Plan plan; Mapping mapping; double records[51]; std::fill_n(records,51,1.0);
  bool rejected=false;
  try { append<Program>(plan,records,count,mapping,"identity"); }
  catch(const DeviceError& error) {
    rejected=true; assert(error.status==100+failure);
  } catch(const std::invalid_argument&) { rejected=true; assert(count>plan.capacity); }
  assert(!host_pending);
  if(failure) {
    assert(rejected && !plan.valid && drains==1 && !kernel_pending);
    // Failure stays sticky until an explicit reset, not a second append.
    const int before=copies;
    try { append<Program>(plan,records,1,mapping,"identity"); assert(false); }
    catch(const std::runtime_error&) {}
    assert(copies==before);
  } else if(count>plan.capacity) {
    assert(rejected && !plan.valid && copies==0 && drains==0);
  } else {
    assert(!rejected && plan.valid && drains==0);
    assert(copies==(count!=0) && launches==(count!=0) && event_waits==(count!=0));
    assert(kernel_pending==(count!=0));
  }
}
"""
    directory = tmp_path_factory.mktemp("first-gradient-append")
    cpp, executable = directory / "append.cpp", directory / "append"
    cpp.write_text(harness)
    subprocess.run(
        [compiler, "-std=c++17", str(cpp), "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return executable


@pytest.mark.parametrize("failure", [1, 2, 3, 4])
@pytest.mark.parametrize("drain_failure", [False, True])
def test_failed_append_drains_before_releasing_host_input(
    append_probe: Path, failure: int, drain_failure: bool
) -> None:
    subprocess.run(
        [str(append_probe), str(failure), str(int(drain_failure)), "1"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )


@pytest.mark.parametrize("count", [0, 1, 2, 3])
def test_success_and_capacity_rejection_preserve_fence_scope(
    append_probe: Path, count: int
) -> None:
    subprocess.run(
        [str(append_probe), "0", "0", str(count)],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
