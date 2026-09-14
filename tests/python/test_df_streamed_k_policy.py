"""Raw work and physical panel capacity are testable without a CUDA context."""

import itertools
import json
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def policy_query(tmp_path_factory):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("host C++ compiler unavailable")
    root = Path(__file__).resolve().parents[2]
    output = tmp_path_factory.mktemp("streamed-policy")
    source = output / "query.cpp"
    source.write_text(
        r"""
#include "scf/df_streamed_k_policy.hpp"
#include <iostream>
int main() {
  std::size_t n, a, capacity;
  while (std::cin >> n >> a >> capacity) {
    const auto p = vibeqc::scf::df_streamed_k_panel(n, a, capacity);
    std::cout << "[" << p.rows << "," << p.output_auxiliaries << ","
              << p.raw_auxiliaries << "," << p.row_tiles << ","
              << p.output_tiles << "]\n";
  }
}
"""
    )
    binary = output / "query"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-I" + str(root / "src"),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
    )

    def query(shapes):
        result = subprocess.run(
            [str(binary)],
            input="".join(f"{n} {a} {cap}\n" for n, a, cap in shapes),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return [json.loads(line) for line in result.stdout.splitlines()]

    return query


def test_policy_minimizes_actual_source_work_and_handles_tails(policy_query):
    # Exhaustively compare both row and Q choices for many nondivisible shapes.
    # This independently enumerates the traversal, rather than repeating the
    # production quotient search or using VibeQC-to-VibeQC numeric parity.
    shapes = [
        (n, a, cap)
        for n in range(1, 16)
        for a in range(1, 14)
        for cap in range(n, n * n * a + 1, max(1, n * a // 5))
    ]
    for (n, a, cap), (rows, q, raw, nr, nq) in zip(
        shapes, policy_query(shapes), strict=True
    ):
        assert rows * n * q <= cap
        assert rows * n * raw <= cap
        assert 1 <= q <= a and 1 <= raw <= a
        work = sum(
            min(rows, n - begin) * n * a
            for _ in range(0, a, q)
            for _ in range(0, n, rows)
            for begin in range(0, n, rows)
        )
        assert work == n * n * a * nr * nq
        possible = (
            ((n + r - 1) // r) * ((a + s - 1) // s)
            for r in range(1, n + 1)
            for s in range(1, a + 1)
            if r * n * s <= cap
        )
        assert nr * nq == min(possible)


def test_reported_768_shape_uses_raw_panel_for_multiple_outputs(policy_query):
    ns = [96, 192, 384, 768]
    shapes = [(n, n, (min(n * n, 8192) // n) * n * min(n, 128)) for n in ns]
    panels = policy_query(shapes)
    assert panels == [
        [96, 85, 85, 1, 2],
        [192, 28, 28, 1, 7],
        [384, 7, 7, 1, 55],
        [256, 5, 5, 3, 154],
    ]
    assert panels[-1][3] * panels[-1][4] == 462
    # Admission failure has no executable zero-stride traversal. Large queries
    # cover the quotient search without allocating tensors or iterating n rows.
    assert policy_query([(0, 3, 3), (3, 0, 3), (3, 3, 2)]) == [[0] * 5] * 3
    large = policy_query([(10**9, 10**9, 10**18)])[0]
    assert large[0] * 10**9 * large[1] <= 10**18


def test_generated_positive_budget_reduces_raw_passes_monotonically(policy_query):
    from vibeqc import Calculator
    from vibeqc.resources_df import density_fitting_tile_plan

    library = Calculator()._library
    plans = [
        density_fitting_tile_plan(
            library,
            1,
            768,
            768,
            160,
            budget_bytes=gib << 30,
            fixed_device_bytes=1 << 20,
            generated_source=True,
        )
        for gib in [1, 2, 4, 8, 12]
    ]
    assert [p.stores_full_three_center for p in plans] == [
        False,
        False,
        True,
        True,
        True,
    ]
    assert all(p.peak_workspace_bytes <= p.budget_bytes for p in plans)
    shapes = [
        (768, 768, (p.ao_pair_tile // 768) * 768 * p.auxiliary_tile) for p in plans
    ]
    panels = policy_query(shapes)
    passes = [
        0 if plan.stores_full_three_center else panel[3] * panel[4]
        for plan, panel in zip(plans, panels, strict=True)
    ]
    assert all(a >= b for a, b in itertools.pairwise(passes))
    assert passes[0] < 32 and passes[2] <= 4 and passes[3] <= 2
    assert all(p.ao_pair_tile == 768**2 for p in plans[2:])
    assert plans[2].auxiliary_tile < 128
    assert plans[3].auxiliary_tile == plans[4].auxiliary_tile == 128
    assert plans[3].peak_workspace_bytes == plans[4].peak_workspace_bytes < 6 << 30
    # Explicit zero compatibility still allocates the established 8192/128
    # plan; only its execution shape follows the shared source-work policy.
    default = density_fitting_tile_plan(
        library,
        1,
        768,
        768,
        160,
        budget_bytes=0,
        fixed_device_bytes=1 << 20,
        generated_source=True,
    )
    assert (default.ao_pair_tile, default.auxiliary_tile) == (8192, 128)
