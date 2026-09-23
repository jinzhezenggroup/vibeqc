"""Execute emitted schedules against an independent enumeration of source work."""

import dataclasses
import shutil
import subprocess
import typing
from pathlib import Path

import pytest
from vibeqc_compiler.method.df_exchange_schedule import (
    native_header,
    projected_exchange_schedule,
)


@pytest.fixture(scope="module")
def schedule_query(tmp_path_factory: typing.Any) -> typing.Any:
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("host C++ compiler unavailable")
    root = Path(__file__).resolve().parents[2]
    directory = tmp_path_factory.mktemp("projected-exchange")
    (directory / "generated_df_exchange_schedule.hpp").write_text(native_header())
    source = directory / "query.cpp"
    source.write_text(
        r"""
#include <iostream>
#include <vector>
#include "scf/df_projected_exchange_schedule.hpp"
int main() {
  std::size_t n, a, rank, capacity;
  bool triangular;
  while (std::cin >> n >> a >> rank >> capacity >> triangular) {
    const auto dense = vibeqc::scf::df_streamed_k_panel(n, a, capacity);
    const auto p = vibeqc::scf::df_projected_exchange_schedule(n, a, rank, capacity, triangular);
    std::size_t loaded = 0;
    if (p.rows) {
      // Independent storage simulator: a contraction must read precisely the
      // rows currently resident in each slot, and write every required AO pair
      // exactly once. This catches an invalid lifetime even if census matches.
      std::size_t begins[2] = {n, n}, counts[2] = {0, 0};
      std::vector<int> coverage(n * n, 0);
      const bool ok = vibeqc::scf::generated::visit_projected_exchange(
          n, p.rows, triangular,
          [&](std::size_t begin, std::size_t count, std::size_t slot) {
            if (slot >= 2 || !count || begin + count > n ||
                count * a * rank > capacity || count * n > capacity) return false;
            begins[slot] = begin;
            counts[slot] = count;
            loaded += count;
            return true;
          },
          [&](std::size_t r, std::size_t nr, std::size_t c, std::size_t nc,
              std::size_t left, std::size_t right, bool) {
            if (left >= 2 || right >= 2 || begins[left] != r || counts[left] != nr ||
                begins[right] != c || counts[right] != nc) return false;
            for (auto i = r; i < r + nr; ++i)
              for (auto j = c; j < c + nc; ++j) {
                if (++coverage[i * n + j] != 1) return false;
                if (triangular && r != c && ++coverage[j * n + i] != 1) return false;
              }
            return true;
          });
      if (!ok || std::any_of(coverage.begin(), coverage.end(), [](auto v) { return v != 1; }))
        return 2;
      // Both callback failures must stop immediately, before subsequent work.
      for (bool fail_projection : {false, true}) {
        int calls = 0;
        const auto aborted = vibeqc::scf::generated::visit_projected_exchange(
            n, p.rows, triangular,
            [&](auto...) { ++calls; return !fail_projection; },
            [&](auto...) { ++calls; return false; });
        if (aborted || calls != (fail_projection ? 1 : 2)) return 3;
      }
    }
    std::cout << p.rows << ' ' << p.blocks << ' ' << p.generated_rows << ' '
              << dense.row_tiles << ' ' << dense.output_tiles << ' ' << loaded << '\n';
  }
}
"""
    )
    binary = directory / "query"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-I" + str(root / "src"),
            "-I" + str(directory),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
    )

    def query(shapes: typing.Any) -> typing.Any:
        result = subprocess.run(
            [str(binary)],
            input="".join(" ".join(map(str, shape)) + "\n" for shape in shapes),
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return [tuple(map(int, line.split())) for line in result.stdout.splitlines()]

    return query


def generated_rows(n: int, rows: int, triangular: bool) -> int:
    """Count required loads, allowing diagonal and adjacent triangular reuse."""
    result = 0
    for begin in range(0, n, rows):
        result += min(rows, n - begin)
        for column in range(0, begin + 1 if triangular else n, rows):
            if column != begin and not (triangular and column + rows == begin):
                result += min(rows, n - column)
    return result


def test_emitted_schedule_minimizes_raw_work(schedule_query: typing.Any) -> None:
    shapes = [
        (n, a, rank, cap, triangular)
        for n in range(2, 18)
        for a in (3, 11, 19)
        for rank in {1, n // 2, n}
        for cap in range(n, n * n * a, max(n, n * a // 3))
        for triangular in (0, 1)
    ]
    for shape, result in zip(shapes, schedule_query(shapes), strict=True):
        n, a, rank, capacity, triangular = shape
        rows, blocks, count, dense_rows, dense_q, actual_count = result
        assert count == actual_count
        # Python and emitted native policy share the contract, while this
        # independent census qualifies its optimum against every legal width.
        expected = projected_exchange_schedule(
            n, a, rank, capacity, dense_rows, dense_q, bool(triangular)
        )
        assert (rows, blocks, count) == dataclasses.astuple(expected)
        choices = [
            (generated_rows(n, r, triangular), (n + r - 1) // r, r)
            for r in range(1, n + 1)
            if r * a * rank <= capacity and r * n <= capacity
        ]
        if not choices or min(choices)[0] >= n * dense_rows * dense_q:
            assert rows == 0
        else:
            # One and two retained blocks both need one tensor pass. Prefer
            # fewer products at equal work, then the smallest balanced width.
            assert (count, blocks, rows) == min(choices)
            assert blocks == len(range(0, n, rows))


def test_practical_96_atom_capacity_and_rejection(schedule_query: typing.Any) -> None:
    # Exact observed n/naux/rank with the original four Q=580 buffers.
    shape = (768, 3712, 160, 768 * 768 * 580, 1)
    rows, blocks, count, dense_rows, dense_q, actual_count = schedule_query([shape])[0]
    assert (rows, blocks, count, dense_rows, dense_q) == (384, 2, 768, 1, 7)
    assert count == actual_count
    assert count * 768 * 3712 == 2_189_426_688
    invalid = [
        (0, 3, 1, 10, 1),
        (8, 0, 1, 64, 1),
        (8, 3, 0, 64, 1),
        (8, 3, 9, 64, 1),
        (8, 3, 1, 7, 1),
        (65536, 2, 1, 65536, 1),
    ]
    assert all(row[:3] == (0, 0, 0) for row in schedule_query(invalid))
