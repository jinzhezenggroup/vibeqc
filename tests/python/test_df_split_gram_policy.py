"""Split Gram views must cover L exactly without overflowing charged scratch."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def query(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    directory = tmp_path_factory.mktemp("split-gram-policy")
    source = directory / "query.cpp"
    source.write_text(
        r"""
#include "scf/df_exchange_policy.hpp"
#include <iostream>
int main() {
  std::size_t n, a, rank, capacity;
  while (std::cin >> n >> a >> rank >> capacity) {
    const auto p = vibeqc::scf::df_occupied_gram_split(n, a, rank, capacity);
    std::cout << "[" << p.count << "," << p.segment << "," << p.tail
              << "," << p.partial_elements << "]\n";
  }
}
"""
    )
    binary = directory / "query"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-I" + str(Path(__file__).resolve().parents[2] / "src"),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
    )

    def run(shapes):
        result = subprocess.run(
            [str(binary)],
            input="".join(" ".join(map(str, shape)) + "\n" for shape in shapes),
            text=True,
            capture_output=True,
            check=True,
            timeout=10,
        )
        return [json.loads(line) for line in result.stdout.splitlines()]

    return run


def test_disjoint_views_cover_all_reduction_rows_and_fit_scratch(query):
    shapes = [
        (n, a, rank, capacity)
        for n in range(1, 12)
        for a in range(1, 15)
        for rank in (0, 1, n)
        for capacity in (0, 3 * n * n, 4 * n * n - 1, 4 * n * n, a * n * n)
    ]
    for (n, a, rank, capacity), (count, segment, tail, partial) in zip(
        shapes, query(shapes), strict=True
    ):
        if not rank or a * rank < 4 or capacity < 4 * n * n:
            assert [count, segment, tail, partial] == [0, 0, 0, 0]
            continue
        covered = [
            i
            for b in range(count)
            for i in range(
                b * segment, b * segment + (tail if b == count - 1 else segment)
            )
        ]
        assert covered == list(range(a * rank))
        assert partial <= capacity
        outputs = [{b * n * n + i for i in range(n * n)} for b in range(count)]
        assert len(set.union(*outputs)) == partial


def test_blas_and_byte_overflow_fail_before_any_allocation(query):
    maximum = 2**64 - 1
    invalid = [
        (0, 5, 1, maximum),
        (2, 5, 3, maximum),
        (2**31, 4, 1, maximum),
        (2, 2**31 - 1, 2, maximum),
        (2**31 - 1, 4, 1, maximum),
        (maximum, maximum, maximum, maximum),
    ]
    assert query(invalid) == [[0, 0, 0, 0]] * len(invalid)
    assert query([(768, 768, 160, 768**3)]) == [[4, 30720, 30720, 4 * 768**2]]
    assert query([(97, 193, 37, 193 * 97**2)]) == [[4, 1785, 1786, 4 * 97**2]]
