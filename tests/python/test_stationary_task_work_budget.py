"""Execute native host admission with CUDA side effects explicitly stubbed out."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _block(source: str, marker: str) -> str:
    start = source.index(marker)
    opening = source.index("{", start)
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if not depth:
                return source[start : index + 1]
    raise AssertionError(f"unterminated native block: {marker}")


def test_native_task_budget_is_per_reset_not_cumulative(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler required")
    header = (ROOT / "src/dft/stationary_gradient_cuda.cuh").read_text()
    pieces = [
        _block(header, "struct Owner {") + ";",
        "template <class F>\n" + _block(header, "int guarded("),
        _block(header, "void check("),
    ]
    launches = 0
    for name in ("stationary_reset", "stationary_tasks", "stationary_nuclear"):
        body, count = re.subn(
            r"\b\w+<<<.*?>>>\(.*?\);",
            "/* CUDA kernel execution is outside this host-admission test. */",
            _block(header, f"int {name}("),
            flags=re.DOTALL,
        )
        launches += count
        pieces.append(body)
    assert launches == 4
    source = tmp_path / "admission.cpp"
    source.write_text(PREAMBLE + "\n".join(pieces) + MAIN)
    binary = tmp_path / "admission"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = subprocess.run(
        [str(binary)], check=False, capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_task_metrics_are_reported_per_execution() -> None:
    from vibeqc import _stationary_cuda as runtime

    before = {
        "owned_device_bytes": 1024,
        "h2d_bytes": 100,
        "d2h_bytes": 20,
        "launches": 8,
        "primitive_records": 50,
        "xc_points": 11,
        "grid_pair_visits": 17,
        "task_descriptors": 9,
        "task_batches": 3,
    }
    after = {
        "owned_device_bytes": 1024,
        "h2d_bytes": 140,
        "d2h_bytes": 28,
        "launches": 12,
        "primitive_records": 67,
        "xc_points": 16,
        "grid_pair_visits": 29,
        "task_descriptors": 14,
        "task_batches": 5,
    }

    delta = runtime._metric_delta(after, before)

    assert delta["task_descriptors"] == 5
    assert delta["task_batches"] == 2
    assert delta["primitive_records"] == 17
    assert delta["owned_device_bytes"] == 1024


PREAMBLE = r"""
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
using std::size_t;
namespace vibeqc_stationary_cuda {}
constexpr size_t task_stride=9;
struct Context { void* stream{}; int* error{}; void check_device() {} };
void error_text(char* out,size_t size,const char* message) {
  if(out && size) std::snprintf(out,size,"%s",message);
}
void cuda_check(int) {}
int cudaMemsetAsync(void*,int,size_t,void*) { return 0; }
bool fail_finish=false;
template<class T, class... A> void upload(T&,A...) {}
template<class T> void finished(T&,void*) {
  if(fail_finish) {fail_finish=false; throw std::runtime_error("injected completion failure");}
}
"""

MAIN = r"""
int main() {
  Owner p; p.atoms=2; p.aos=2; p.spin_blocks=1; p.task_capacity=1; p.max_primitive_work=7;
  p.topology_ready=true;
  double xyz[6]{0,0,0,1,0,0}, density[4]{}, weighted[4]{}, charges[1]{1}; char error[256]{};
  int64_t task[9]{0,0,2,-1,0,1,-1,-1,7};
  auto reset=[&](){return stationary_reset(&p,xyz,density,weighted,0,error,sizeof(error));};
  auto page=[&](){return stationary_tasks(&p,task,charges,1,error,sizeof(error));};
  for(int epoch=0; epoch<2; ++epoch) {
    if(reset() || page()) {std::fprintf(stderr,"legal replay rejected: %s\n",error); return 1;}
    if(p.primitive_count!=uint64_t((epoch+1)*7)) return 2;
    if(page()==0) return 3;
  }
  if(reset()) return 4;
  task[8]=3; fail_finish=true;
  if(page()==0 || !p.failed || p.primitive_count!=17) return 5;
  task[8]=7;
  if(reset() || page() || p.primitive_count!=24) return 6;
  if(reset()) return 7;
  for(int i=0;i<7;++i)
    if(stationary_nuclear(&p,0,0,1,1,1,error,sizeof(error))) return 8;
  if(stationary_nuclear(&p,0,0,1,1,1,error,sizeof(error))==0) return 9;
  if(p.primitive_count!=31) return 10;
  p.primitive_count=std::numeric_limits<uint64_t>::max()-1;
  if(reset()) return 11;
  task[8]=2;
  if(page()==0 || p.primitive_count!=std::numeric_limits<uint64_t>::max()-1) return 12;
}
"""
