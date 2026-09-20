"""Run the actual CUDA entry preflight before a sentinel allocation owner."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _function(source: str, signature: str) -> str:
    start = source.index(signature)
    cursor = source.index("{", start)
    depth = 1
    end = cursor + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def test_all_problem_fields_rejected_before_cuda_owner(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    cpu = (ROOT / "src/cc/solver.cpp").read_text()
    cuda = (ROOT / "src/cc/cuda_solver.cu").read_text()
    # Extract current production validators and the entry through construction.
    # CUDA kernels never execute in this device-free boundary regression.
    validators = _function(cpu, "void validate_problem(")
    validators += _function(cpu, "void validate_options(")
    if "void validate_problem_cuda(" in cuda:
        validators += _function(cuda, "void validate_problem_cuda(")
    start = cuda.index("SolverResult solve_cuda(")
    stop = cuda.index("Owner owner(p, options, device);", start)
    entry = cuda[start : stop + len("Owner owner(p, options, device);")]
    source = tmp_path / "preflight.cpp"
    source.write_text(PREFIX + validators + entry + "return {}; }\n}\n" + MAIN)
    exe = tmp_path / "preflight"
    subprocess.run(
        [compiler, "-std=c++20", "-I" + str(ROOT / "src"), str(source), "-o", str(exe)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = subprocess.run(
        [str(exe)], capture_output=True, text=True, timeout=10, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


PREFIX = r"""
#include "cc/solver.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
namespace vibeqc::cc {
int owners = 0;
std::size_t checked_mul(std::size_t a, std::size_t b) {
  if (a && b > std::numeric_limits<std::size_t>::max()/a) throw std::length_error("overflow");
  return a*b;
}
struct Owner {
  Owner(const Problem&, const SolverOptions&, int) { ++owners; }
};
"""
MAIN = r"""
int main() {
  using namespace vibeqc::cc;
  Problem valid; valid.nocc=1; valid.nvir=2;
  valid.foo.resize(1); valid.fov.resize(2); valid.fvv.resize(4);
  valid.ovov.resize(4); valid.ovvo.resize(4); valid.oovv.resize(4);
  valid.ovvv.resize(8); valid.ovoo.resize(2); valid.oooo.resize(1); valid.vvvv.resize(16);
  valid.d1.assign(2,-2); valid.d2.assign(4,-4);
  valid.initial_t1.resize(2); valid.initial_t2.resize(4);
  SolverOptions options;
  solve_cuda(valid,options,0);
  if (owners != 1) return 1;
  int checked=0;
  auto refuse = [&](const Problem& p) {
    owners=0;
    try { solve_cuda(p,options,0); }
    catch (const std::invalid_argument&) { ++checked; return owners==0; }
    return false;
  };
  std::vector<double> Problem::* fields[]={&Problem::foo,&Problem::fov,&Problem::fvv,
      &Problem::ovov,&Problem::ovvo,&Problem::oovv,&Problem::ovvv,&Problem::ovoo,
      &Problem::oooo,&Problem::vvvv,&Problem::d1,&Problem::d2,
      &Problem::initial_t1,&Problem::initial_t2};
  int field=0;
  for (auto member: fields) {
    for (int mode=0;mode<4;++mode) {
      auto p=valid; auto& values=p.*member;
      if (mode==0) values.pop_back();
      if (mode==1) values.push_back(0);
      if (mode==2) values[0]=std::numeric_limits<double>::quiet_NaN();
      if (mode==3) values[0]=std::numeric_limits<double>::infinity();
      if (!refuse(p)) { std::cerr<<"field "<<field<<" mode "<<mode<<" reached owner\n"; return 2; }
    }
    ++field;
  }
  for (int mode=0;mode<4;++mode) {
    auto p=valid;
    if (mode==0) p.nocc=0;
    if (mode==1) p.nvir=0;
    if (mode==2) p.reference_energy=std::numeric_limits<double>::quiet_NaN();
    if (mode==3) p.reference_energy=std::numeric_limits<double>::infinity();
    if (!refuse(p)) return 3;
  }
  std::cout << checked << " invalid inputs rejected before construction\n";
}
"""
