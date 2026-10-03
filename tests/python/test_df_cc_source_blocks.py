"""Independent source-factor sector and retained-integral contraction gates."""

from __future__ import annotations

import numpy as np
import pytest
from generativeqc_compiler.cc.df_source import (
    BLOCK_FACTORS,
    block_program,
    factor_program,
)
from generativeqc_compiler.tensor import execute


@pytest.mark.parametrize("o,v,q", [(1, 1, 2), (2, 3, 4), (3, 2, 1)])
@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("project_pairs", [True, False])
def test_factor_sectors_and_retained_blocks_match_dense_gram(
    o: int, v: int, q: int, symmetric: bool, project_pairs: bool
) -> None:
    n = o + v
    b = np.random.default_rng(1765 + o + v + q).normal(size=(n, n, q))
    if symmetric:
        b = (b + b.transpose(1, 0, 2)) / 2
    factors = execute(
        factor_program(o, v, q, symmetric_pairs=project_pairs), {"bmo": b}
    ).outputs
    if project_pairs:
        b = (b + b.transpose(1, 0, 2)) / 2
    for name in ("boo", "bov", "bvo", "bvv"):
        first = slice(0, o) if name[1] == "o" else slice(o, n)
        second = slice(0, o) if name[2] == "o" else slice(o, n)
        np.testing.assert_array_equal(
            factors[name], b[first, second, :].transpose(2, 0, 1)
        )
    dense = np.einsum("pqQ,rsQ->pqrs", b, b, optimize=False)
    for name, (left, right) in BLOCK_FACTORS.items():
        program = block_program(o, v, q, name)
        actual = execute(program, {key: factors[key] for key in {left, right}}).outputs[
            name
        ]
        slices = tuple(slice(0, o) if sector == "o" else slice(o, n) for sector in name)
        np.testing.assert_allclose(actual, dense[slices], atol=2e-13, rtol=2e-14)
    if project_pairs:
        np.testing.assert_array_equal(factors["boo"], factors["boo"].transpose(0, 2, 1))
        np.testing.assert_array_equal(factors["bvv"], factors["bvv"].transpose(0, 2, 1))
        np.testing.assert_array_equal(factors["bov"], factors["bvo"].transpose(0, 2, 1))


@pytest.mark.parametrize("name", ["ovvv", "vvvv", "anything"])
def test_source_rejects_unretained_blocks(name: str) -> None:
    with pytest.raises(ValueError, match="only retained"):
        block_program(2, 3, 4, name)
