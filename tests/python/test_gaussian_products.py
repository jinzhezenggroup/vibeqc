"""Check generic lane ownership independently of any integral evaluator."""

import ctypes
import math
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def product_probe(tmp_path_factory):
    """A separable policy has a closed-form tensor-product sum as its oracle.

    Unequal primitive and sparse-term lengths exercise recursion, empty factors,
    and cooperative remainders. Actual integral mathematics has its separate
    libcint tests; this fixture isolates runtime scheduling and coefficients.
    """
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    directory = tmp_path_factory.mktemp("gaussian-products")
    (directory / "cuda_runtime.h").write_text("")
    source = directory / "probe.cpp"
    source.write_text(
        r"""
#define __device__
inline void atomicAdd(double* out, double value) { *out += value; }
#include "runtime/cuda_gaussian_products.cuh"
namespace gp = vibeqc::runtime::cuda_gaussian_products;
struct Policy {
  struct Vec3 { double x, y, z; };
  struct Angular { unsigned x, y, z; };
  using Accumulator = double;
  template <unsigned Rank>
  static void accumulate(double& out, const double* e, const Vec3* r,
                         const Angular* a, double weight) {
    for (unsigned i=0; i<Rank; ++i) weight *= e[i]*(r[i].x+1)*(a[i].x+1);
    out += weight;
  }
};
template <unsigned Rank>
double evaluate(unsigned lane, unsigned lanes, bool empty) {
  std::int32_t atoms[]{0,1,2}, shells[]{0,1,2};
  std::int64_t offsets[]{0,9,16,empty ? 16 : 21};
  std::uint8_t counts[]{2,3,1}, angular[27]{};
  double terms[9], exponents[21], coefficients[21];
  for (unsigned i=0;i<9;++i) { terms[i]=i%3+1; angular[3*i]=i%3; }
  for (unsigned i=0;i<21;++i) { exponents[i]=i+1; coefficients[i]=i%2+1; }
  gp::BasisView view{3,atoms,shells,offsets,counts,angular,terms,exponents,coefficients};
  gp::Factor factors[Rank]{};
  for (unsigned i=0;i<Rank;++i) factors[i]={view,i};
  const double positions[]{1,0,0,2,0,0,3,0,0};
  return gp::contract<Policy,3>(factors,positions,lane,lanes);
}
extern "C" double probe(unsigned rank, unsigned lane, unsigned lanes, bool empty) {
  return rank==2 ? evaluate<2>(lane,lanes,empty) : evaluate<3>(lane,lanes,empty);
}
"""
    )
    output = directory / "probe.so"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-shared",
            "-fPIC",
            "-I",
            str(directory),
            "-I",
            str(Path(__file__).resolve().parents[2] / "src"),
            str(source),
            "-o",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    library = ctypes.CDLL(str(output))
    library.probe.argtypes = [ctypes.c_uint] * 3 + [ctypes.c_bool]
    library.probe.restype = ctypes.c_double
    return library.probe


@pytest.mark.parametrize("rank,empty", [(2, False), (3, False), (3, True)])
@pytest.mark.parametrize("lanes", [1, 2, 3, 32, 64])
def test_lane_partitions_preserve_complete_sparse_product(
    product_probe, rank, empty, lanes
):
    offsets = (0, 9, 16, 16 if empty else 21)
    terms = (2, 3, 1)
    expected = math.prod(
        sum((p + 1) * (p % 2 + 1) for p in range(offsets[i], offsets[i + 1]))
        * sum(t * t for t in range(1, terms[i] + 1))
        * (i + 2)
        for i in range(rank)
    )
    actual = sum(product_probe(rank, lane, lanes, empty) for lane in range(lanes))
    assert actual == expected
