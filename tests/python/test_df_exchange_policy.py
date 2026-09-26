"""The occupied work policy is CPU-safe and independent of benchmark endpoints."""

import os
import shutil
import subprocess
import typing
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def exchange_policy(tmp_path_factory: typing.Any) -> typing.Any:
    """Compile the production policy without CUDA or a native library."""
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("host C++ compiler unavailable")
    root = Path(__file__).resolve().parents[2]
    directory = tmp_path_factory.mktemp("exchange-policy")
    source = directory / "query.cpp"
    source.write_text(
        r"""
#include <iostream>
#include "scf/df_exchange_policy.hpp"
int main() {
  std::size_t n, a, batch, rank;
  while (std::cin >> n >> a >> batch >> rank)
    std::cout << vibeqc::scf::df_occupied_exchange_preferred(n,a,batch,rank) << " "
              << vibeqc::scf::df_occupied_exchange_requested(n,a,batch,rank) << "\n";
}
"""
    )
    binary = directory / "query"
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

    def query(shapes: typing.Any, policy: typing.Any = "auto") -> typing.Any:
        env = dict(os.environ)
        env.pop("VIBEQC_DF_EXCHANGE", None)
        if policy is not None:
            env["VIBEQC_DF_EXCHANGE"] = policy
        result = subprocess.run(
            [str(binary)],
            input="".join(" ".join(map(str, shape)) + "\n" for shape in shapes),
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return [tuple(map(int, line.split())) for line in result.stdout.splitlines()]

    return query


def test_work_policy_tracks_contraction_costs(
    exchange_policy: typing.Any,
) -> None:
    """Compare modeled operation counts, including odd sizes and rank boundaries."""
    shapes = [
        (n, a, 1, rank)
        for n in (2, 7, 24, 96, 192, 384, 512, 513, 768)
        for a in (1, n - 1, n, 2 * n + 3)
        for rank in {0, 1, n // 5, n // 2, n // 2 + 1, n, n + 1}
    ]
    for (n, a, _, rank), (selected, reserved) in zip(
        shapes, exchange_policy(shapes), strict=True
    ):
        dense_flops = 2 * a * n**3 + 2 * a * n**3
        occupied_flops = 2 * a * n * n * rank + 2 * n * n * a * rank
        assert selected == (rank > 0 and 2 * occupied_flops <= dense_flops)
        assert reserved == selected  # Known RHF rank authorizes the reservation.


def test_invalid_indexing_batch_and_explicit_controls(
    exchange_policy: typing.Any,
) -> None:
    """Invalid/overflowing dimensions reject before products; overrides survive."""
    invalid = [
        (0, 3, 1, 1),
        (3, 0, 1, 1),
        (1, 1, 1, 1),
        (384, 384, 0, 80),
        (384, 384, 2, 80),
        (1 << 32, 1, 1, 1),
        (2, 1 << 32, 1, 1),
        ((1 << 64) - 1, (1 << 64) - 1, 1, 1),
    ]
    assert exchange_policy(invalid) == [(0, 0)] * len(invalid)
    points = [(384, 384, 1, 80), (768, 768, 1, 160), (512, 777, 1, 123)]
    assert exchange_policy(points) == [(1, 1)] * len(points)
    assert exchange_policy(points, None) == exchange_policy(points)
    assert exchange_policy(points, "dense") == [(1, 0)] * len(points)
    assert exchange_policy(invalid, "occupied") == [(0, 1)] * len(invalid)
