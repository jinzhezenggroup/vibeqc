"""Run the real diagnostic host-staging helper against exact byte boundaries."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_actual_screening_staging_reuse_and_budget(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler required")
    source = (ROOT / "src/scf/cuda/df_gradient_bridge.cu").read_text()
    start = source.index("std::span<double> prepare_screening_buffer(")
    end = source.index("\n}\n", start) + 3
    helper = source[start:end]
    harness = tmp_path / "screening.cpp"
    harness.write_text(
        "#include <memory>\n#include <span>\n#include <stdexcept>\n#include <limits>\n"
        "struct Arena { std::size_t budget; struct { std::size_t host_bytes; } stats; };\n"
        + helper
        + r"""
int main() {
  Arena arena{80, {16}};
  std::unique_ptr<double[]> storage;
  std::size_t capacity = 0;
  auto first = prepare_screening_buffer(storage, capacity, 4, arena);
  if(first.size()!=4 || arena.stats.host_bytes!=48 || capacity!=4) return 1;
  auto* pointer = first.data();
  auto reused = prepare_screening_buffer(storage, capacity, 2, arena);
  if(reused.data()!=pointer || arena.stats.host_bytes!=48 || capacity!=4) return 2;
  auto full = prepare_screening_buffer(storage, capacity, 8, arena);
  if(full.size()!=8 || arena.stats.host_bytes!=80 || capacity!=8) return 3;
  pointer = storage.get();
  try { prepare_screening_buffer(storage, capacity, 9, arena); return 4; }
  catch(const std::bad_alloc&) {}
  if(storage.get()!=pointer || arena.stats.host_bytes!=80 || capacity!=8) return 5;
  try { prepare_screening_buffer(storage, capacity, std::numeric_limits<std::size_t>::max(), arena); return 6; }
  catch(const std::bad_alloc&) {}
  Arena short_budget{79, {16}};
  std::unique_ptr<double[]> other;
  std::size_t other_capacity=0;
  try { prepare_screening_buffer(other, other_capacity, 8, short_budget); return 7; }
  catch(const std::bad_alloc&) {}
  if(other || other_capacity || short_budget.stats.host_bytes!=16) return 8;
}
"""
    )
    binary = tmp_path / "screening"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(harness),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, timeout=10)


def test_diagnostic_transfers_reconcile_general_metrics() -> None:
    source = (ROOT / "src/scf/cuda/df_gradient_bridge.cu").read_text()
    block = source.split("if (screening_features) {", 1)[1].split(
        "if (shell_diagnostics)", 1
    )[0]
    assert block.index("prepare_screening_buffer") < block.index("cudaMemcpyAsync")
    assert "arena.stats.device_to_host_bytes += count * sizeof(double)" in block
    assert "++arena.stats.stream_synchronizations" in block
