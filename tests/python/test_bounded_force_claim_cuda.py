"""Stress persistent claims for triangular products and indexed empty pages.

Preserve the master skip regression and PR #1767's indexed-page extension of
njzjz-bot commit 0b99c6ce298f2726373f1909ad10a37b5acffc43. The host invariant
checks control flow; optional CUDA stress does not replace integral oracles.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _claim_parts() -> tuple[str, str, str]:
    source = (ROOT / "src/scf/cuda/direct_bounded_fallback.cu").read_text()
    start = source.index("  constexpr auto full_block_candidates")
    stop = source.index("\n\n  while (true)", start)
    preparation = source[start:stop]
    start = source.index("  while (true) {", stop)
    stop = source.index("    if (block_quartet >= total) return;", start)
    claim = source[start:stop]
    start = source.index("    const std::size_t page_begin =", stop)
    stop = source.index("    for (std::size_t candidate_begin", start)
    return preparation, claim, source[start:stop]


def test_claim_consumption_barrier_precedes_every_leader_overwrite() -> None:
    """A publication-only barrier cannot protect an empty previous page."""
    _, claim, _ = _claim_parts()
    statements = "\n".join(
        line for line in claim.splitlines() if not line.strip().startswith("//")
    )
    assert statements.split("{", 1)[1].lstrip().startswith("__syncthreads();")
    assert statements.count("__syncthreads();") == 2
    assert statements.index("__syncthreads();") < statements.index("atomicAdd(")
    assert statements.rindex("__syncthreads();") > statements.index("atomicAdd(")


@pytest.fixture(scope="module")
def claim_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if os.environ.get("GENERATIVEQC_CLAIM_CUDA_TEST") != "1":
        pytest.skip("opt-in finite Slurm CUDA queue qualification")
    assert os.environ.get("SLURM_JOB_ID"), "CUDA qualification requires finite Slurm"
    cache = shutil.which("ccache")
    compiler = Path(os.environ["CUDA_PATH"]) / "bin/nvcc"
    assert cache and compiler.is_file()
    subprocess.run([cache, "--version"], check=True, timeout=10)
    preparation, claim, page = _claim_parts()
    directory = tmp_path_factory.mktemp("bounded-claim")
    source = directory / "probe.cu"
    source.write_text(
        r"""
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "scf/direct_block_domain.hpp"
#include "scf/direct_task_layout.hpp"
namespace detail = generativeqc::scf::detail;
constexpr unsigned threads = 128;
constexpr unsigned workers = 4;
constexpr unsigned products_count = 7;
constexpr unsigned counts[products_count] = {528, 1024, 1, 17, 0, 63, 256};
struct Batch { std::size_t total_shell_pair_block_quartets; };
__global__ void probe(Batch batch, detail::BoundedDirectBlockDomain block_domain,
                      const unsigned* candidate_counts, unsigned long long* global_cursor,
                      unsigned* visits, unsigned skip_mode) {
  __shared__ unsigned long long block_quartet;
"""
        + preparation
        + "\n"
        + claim
        + r"""
    // Delay a nonleader warp before its first read, not merely a later reload.
    if (threadIdx.x / 32 == 1) {
      const auto began = clock64();
      while (clock64() - began < 100000ULL) {}
    }
    if (block_quartet >= total) return;
    const auto observed = block_quartet;
    const auto product = observed / pages;
    if ((skip_mode & 1U) && product % 3 == 0) continue;
    if ((skip_mode & 2U) && product % 3 == 1) continue;
    const std::size_t candidate_count = candidate_counts[product];
"""
        + page
        + r"""
    for (std::size_t candidate_begin = page_begin; candidate_begin < page_end;
         candidate_begin += detail::kBoundedDirectQueueCapacity) {
      __syncthreads();
      atomicAdd(visits + observed * threads + threadIdx.x, 1U);
      __syncthreads();
    }
  }
}
void checked(cudaError_t status) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
    std::abort();
  }
}
int main() {
  unsigned *device_counts{}, *visits{};
  unsigned long long* cursor{};
  std::uint64_t* prefix{};
  constexpr auto maximum_claims = products_count * detail::kBoundedDirectIndexedCandidatePages;
  checked(cudaMalloc(&device_counts, sizeof(counts)));
  checked(cudaMalloc(&visits, maximum_claims * threads * sizeof(unsigned)));
  checked(cudaMalloc(&cursor, sizeof(*cursor)));
  checked(cudaMalloc(&prefix, sizeof(*prefix)));
  checked(cudaMemcpy(device_counts, counts, sizeof(counts), cudaMemcpyHostToDevice));
  for (unsigned replay = 0; replay < 16; ++replay) {
    for (bool indexed : {false, true}) {
      const auto pages = indexed ? detail::kBoundedDirectIndexedCandidatePages : 1U;
      const auto total = products_count * pages;
      for (unsigned skip_mode = 0; skip_mode < 4; ++skip_mode) {
        for (unsigned long long initial : {0ULL, 1ULL, 15ULL}) {
          if (initial >= total) continue;
          checked(cudaMemset(visits, 0, maximum_claims * threads * sizeof(unsigned)));
          checked(cudaMemcpy(cursor, &initial, sizeof(initial), cudaMemcpyHostToDevice));
          detail::BoundedDirectBlockDomain domain{indexed ? prefix : nullptr, products_count, products_count};
          probe<<<workers, threads>>>({products_count}, domain, device_counts, cursor, visits, skip_mode);
          checked(cudaGetLastError());
          checked(cudaDeviceSynchronize());
          std::vector<unsigned> actual(maximum_claims * threads);
          checked(cudaMemcpy(actual.data(), visits, actual.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));
          unsigned long long claimed{};
          checked(cudaMemcpy(&claimed, cursor, sizeof(claimed), cudaMemcpyDeviceToHost));
          assert(claimed == total + workers);
          for (unsigned ordinal = 0; ordinal < maximum_claims; ++ordinal) {
            unsigned expected = 0;
            if (ordinal >= initial && ordinal < total) {
              const auto product = ordinal / pages;
              const bool skipped = ((skip_mode & 1U) && product % 3 == 0) ||
                                   ((skip_mode & 2U) && product % 3 == 1);
              const std::size_t begin = indexed ? (ordinal % pages) * 64 : 0;
              const auto end = indexed ? std::min<std::size_t>(counts[product], begin + 64) : counts[product];
              if (!skipped && end > begin)
                expected = (end - begin + detail::kBoundedDirectQueueCapacity - 1) / detail::kBoundedDirectQueueCapacity;
            }
            for (unsigned lane = 0; lane < threads; ++lane)
              assert(actual[ordinal * threads + lane] == expected);
          }
        }
      }
    }
  }
  checked(cudaFree(prefix)); checked(cudaFree(cursor));
  checked(cudaFree(visits)); checked(cudaFree(device_counts));
  std::puts("empty diagonal/tail, inactive and screened claims preserve every reader");
}
"""
    )
    executable = directory / "probe"
    subprocess.run(
        [
            cache,
            str(compiler),
            "-std=c++20",
            "-arch=sm_120",
            "-I",
            str(ROOT / "src"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        timeout=180,
    )
    return executable


@pytest.mark.parametrize("sanitizer", [None, "synccheck", "racecheck"])
def test_cuda_claim_readers_survive_empty_and_skipped_pages(
    claim_probe: Path,
    sanitizer: str | None,
) -> None:
    """Use real producer/page code; independent force oracles are separate gates."""
    command = [str(claim_probe)]
    if sanitizer:
        tool = shutil.which("compute-sanitizer")
        assert tool, "requested synchronization qualification needs compute-sanitizer"
        command = [tool, "--tool", sanitizer, "--error-exitcode", "91", *command]
    result = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=180
    )
    assert "preserve every reader" in result.stdout
    if sanitizer == "racecheck":
        assert "RACECHECK SUMMARY: 0 hazards" in result.stdout + result.stderr
    elif sanitizer:
        assert "ERROR SUMMARY: 0 errors" in result.stdout + result.stderr
