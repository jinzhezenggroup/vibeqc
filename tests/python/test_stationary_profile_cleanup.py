"""Execute the production profiling entry with deterministic CUDA allocation faults."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_failed_event_creation_is_transactional(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a C++ compiler")
    root = Path(__file__).resolve().parents[2]
    source = (root / "src/dft/stationary_gradient_cuda.cuh").read_text()
    entry = source[
        source.index("int stationary_profile(") : source.index("int stationary_reset(")
    ]
    path = tmp_path / "profile.cpp"
    path.write_text(PREFIX + entry + MAIN)
    binary = tmp_path / "profile"
    subprocess.run(
        [compiler, "-std=c++20", str(path), "-o", str(binary)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    result = subprocess.run(
        [str(binary)], check=False, capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stdout + result.stderr


PREFIX = r"""
#include <stdexcept>
#include <set>
#include <iostream>
using cudaEvent_t = int;
"""
PREFIX += r"""
int next_id=0, calls=0, fail_at=0;
std::set<int> live;
int cudaEventCreate(cudaEvent_t* event) {
  if (++calls == fail_at) return 1;
  *event=++next_id; live.insert(*event); return 0;
}
int cudaEventDestroy(cudaEvent_t event) { live.erase(event); return 0; }
void cuda_check(int status) { if(status) throw std::runtime_error("injected"); }
namespace vibeqc_stationary_cuda {
struct Context { void check_device() {} };
struct Owner { Context context; bool profile=false; int stage0=0,stage1=0,stage2=0,stage3=0; };
template<class F> int guarded(Owner*,char*,size_t,F f) { try { f();return 0; } catch(...) {return 1;} }
}
"""

MAIN = r"""
int main() {
  using namespace vibeqc_stationary_cuda;
  for(int failure=1; failure<=4; ++failure) {
    Owner owner; calls=0; fail_at=failure; char error[128]{};
    if(stationary_profile(&owner,error,sizeof(error))!=1) return 1;
    if(!live.empty() || owner.profile) {std::cerr<<"partial event leak "<<failure;return 2;}
    fail_at=0;
    if(stationary_profile(&owner,error,sizeof(error))!=0 || live.size()!=4) return 3;
    for(int event:{owner.stage0,owner.stage1,owner.stage2,owner.stage3}) cudaEventDestroy(event);
    if(!live.empty()) return 4;
  }
}
"""
