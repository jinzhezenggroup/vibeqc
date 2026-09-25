"""Exercise actual shared DIIS history under host allocation failures."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

_DRIVER = r"""
#include <cstdlib>
#include <iostream>
#include <new>
#include "solver/diis_history.hpp"

int remaining = -1;
void* operator new(std::size_t bytes) {
  if (remaining == 0) throw std::bad_alloc();
  if (remaining > 0) --remaining;
  if (void* p = std::malloc(bytes ? bytes : 1)) return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
using vibeqc::solver::detail::DiisHistory;

int main(int argc, char** argv) {
  if (argc != 3) return 99;
  const int entries = std::atoi(argv[1]), fail = std::atoi(argv[2]);
  DiisHistory history(2);
  for (int i = 0; i < entries; ++i)
    history.push({double(i), 2, 3}, {double(i)+1, 4, 5});
  const auto vectors = history.vectors(), errors = history.errors();
  const auto elements = history.elements(), bytes = history.numeric_capacity_bytes();
  const std::vector<double> value{10, 11, 12};
  std::vector<double> residual{20, 21, 22};
  bool failed = false;
  remaining = fail;
  try { history.push(value, std::move(residual)); }
  catch (const std::bad_alloc&) { failed = true; }
  remaining = -1;
  if (failed) {
    if (history.vectors() != vectors || history.errors() != errors ||
        history.elements() != elements || history.numeric_capacity_bytes() != bytes) {
      std::cerr << "failed insertion changed history: vectors=" << history.size()
                << " errors=" << history.errors().size() << '\n';
      return 1;
    }
    // Failed initial admission must not bind a dimension that never committed.
    if (entries == 0) history.push({7, 8}, {9, 10});
    else history.push(value, {20, 21, 22});
  }
  if (history.size() != history.errors().size() || history.size() > 2) return 2;
  std::size_t observed = 0;
  for (const auto& v : history.vectors()) observed += v.capacity()*sizeof(double);
  for (const auto& e : history.errors()) observed += e.capacity()*sizeof(double);
  if (observed != history.numeric_capacity_bytes()) return 3;
  if (entries == 2 && history.vectors().front()[0] != 1) return 4;
  history.retire_oldest();
  if (history.size() != history.errors().size()) return 5;
  history.clear();
  if (history.size() || history.numeric_capacity_bytes()) return 6;
}
"""


@pytest.fixture(scope="module")
def history_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    directory = tmp_path_factory.mktemp("diis-history-transaction")
    source, executable = directory / "history.cpp", directory / "history"
    source.write_text(_DRIVER, encoding="utf-8")
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O0",
            f"-I{ROOT / 'src'}",
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return executable


@pytest.mark.parametrize("entries", [0, 1, 2])
@pytest.mark.parametrize("failure", [0, 1, 2, 3])
def test_failed_push_preserves_paired_history(
    history_probe: Path, entries: int, failure: int
) -> None:
    completed = subprocess.run(
        [str(history_probe), str(entries), str(failure)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
