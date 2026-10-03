"""DF primal and response schedules against the dense conventional inventory."""

from __future__ import annotations

import numpy as np
import pytest
from generativeqc_compiler.cc.df_equations import (
    build_df_virtual_correction_program,
    build_df_virtual_response_programs,
)
from generativeqc_compiler.cc.doubles import build_ccsd_program
from generativeqc_compiler.tensor import analyze_complexity, execute

from tools.generativeqc_cc.df_factorized import virtual_corrections


def _case(o: int, v: int, q: int = 3, seed: int = 1753) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed + 10 * o + v)
    bov = rng.normal(scale=0.2, size=(q, o, v))
    bvv = rng.normal(scale=0.2, size=(q, v, v))
    bvv = 0.5 * (bvv + bvv.transpose(0, 2, 1))
    t1 = rng.normal(scale=0.03, size=(o, v))
    t2 = rng.normal(scale=0.02, size=(o, o, v, v))
    t2 = 0.5 * (t2 + t2.transpose(1, 0, 3, 2))
    return {"bov": bov, "bvv": bvv, "t1": t1, "t2": t2}


def _dense_correction(feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Reconstruct only in this small independent four-index test oracle.

    Set retained smaller integrals and Fock to zero. Linearity of RCCSD in its
    integral parameters then isolates exactly the virtual-factor contribution.
    """
    o, v = feeds["t1"].shape
    program = build_ccsd_program(o, v, form="expanded", diagnostics=False)
    arrays = {
        node.attrs["name"]: np.zeros(node.spec.shape)
        for node in program.live_nodes
        if node.op == "input"
    }
    arrays.update(t1=feeds["t1"], t2=feeds["t2"])
    arrays["ovvv"] = np.einsum("Qia,Qbc->iabc", feeds["bov"], feeds["bvv"])
    arrays["vvvv"] = np.einsum("Qab,Qcd->abcd", feeds["bvv"], feeds["bvv"])
    result = execute(program, arrays).outputs
    return {
        "df_virtual_singles": result["singles_residual"],
        "df_virtual_doubles": result["doubles_residual"],
    }


def _factorized(feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    program = build_df_virtual_correction_program(*feeds["t1"].shape)
    rows = [
        execute(
            program,
            {
                "bov": ov,
                "bvv": vv,
                "t1": feeds["t1"],
                "t2": feeds["t2"],
            },
        ).outputs
        for ov, vv in zip(feeds["bov"], feeds["bvv"], strict=True)
    ]
    return {key: sum(row[key] for row in rows) for key in program.outputs}


@pytest.mark.parametrize("o,v", [(1, 1), (2, 3), (3, 2)])
def test_factorized_inventory_matches_dense_and_independent_numpy(
    o: int, v: int
) -> None:
    feeds = _case(o, v)
    expected = _dense_correction(feeds)
    actual = _factorized(feeds)
    numpy_reference = virtual_corrections(**feeds)
    for key, numpy_value in zip(
        ("df_virtual_singles", "df_virtual_doubles"), numpy_reference, strict=True
    ):
        value = actual[key]
        np.testing.assert_allclose(value, expected[key], atol=2e-12, rtol=0)
        np.testing.assert_allclose(value, numpy_value, atol=2e-12, rtol=0)


def test_auxiliary_rotation_preserves_correction() -> None:
    feeds = _case(2, 3)
    q, _ = np.linalg.qr(np.random.default_rng(157).normal(size=(3, 3)))
    rotated = {
        **feeds,
        **{key: np.einsum("QR,Rij->Qij", q, feeds[key]) for key in ("bov", "bvv")},
    }
    for key, expected in _factorized(feeds).items():
        np.testing.assert_allclose(
            _factorized(rotated)[key], expected, atol=2e-12, rtol=0
        )


def test_generated_amplitude_and_factor_pullbacks_match_dense_differences() -> None:
    feeds = _case(2, 3)
    programs = build_df_virtual_response_programs(2, 3)
    directions = _case(2, 3, seed=1853)
    # Independent seeds need not have doubles pair symmetry. AD must implement
    # the declared input projections under the dense Frobenius inner product.
    rng = np.random.default_rng(158)
    seeds = {
        key: rng.normal(size=node.spec.shape)
        for key, node in programs.primal.outputs.items()
    }
    scalar_ad = 0.0
    for index in range(len(feeds["bov"])):
        primal = {
            key: value[index] if key.startswith("b") else value
            for key, value in feeds.items()
        }
        cotangents = {f"bar_{key}": value for key, value in seeds.items()}
        forward = execute(
            programs.amplitude_jvp.program,
            {
                **primal,
                **{f"d_{key}": directions[key] for key in ("t1", "t2")},
            },
        ).outputs
        amplitude = execute(
            programs.amplitude_vjp.program, {**primal, **cotangents}
        ).outputs
        factor = execute(programs.factor_vjp.program, {**primal, **cotangents}).outputs
        left = sum(np.sum(seeds[key] * forward[f"d_{key}"]) for key in seeds)
        right = sum(
            np.sum(amplitude[f"bar_{key}"] * directions[key]) for key in ("t1", "t2")
        )
        np.testing.assert_allclose(left, right, atol=2e-12, rtol=0)
        scalar_ad += right + sum(
            np.sum(factor[f"bar_{key}"] * directions[key][index])
            for key in ("bov", "bvv")
        )
        np.testing.assert_allclose(
            factor["bar_bvv"], factor["bar_bvv"].T, atol=2e-13, rtol=0
        )
        np.testing.assert_allclose(
            amplitude["bar_t2"],
            amplitude["bar_t2"].transpose(1, 0, 3, 2),
            atol=2e-13,
            rtol=0,
        )
    for step in (1e-3, 3e-4, 1e-4):
        plus = _dense_correction(
            {key: value + step * directions[key] for key, value in feeds.items()}
        )
        minus = _dense_correction(
            {key: value - step * directions[key] for key, value in feeds.items()}
        )
        finite_difference = sum(
            np.sum(seeds[key] * (plus[key] - minus[key])) for key in seeds
        ) / (2 * step)
        np.testing.assert_allclose(scalar_ad, finite_difference, atol=2e-7, rtol=1e-6)


@pytest.mark.parametrize("o,v", [(1, 1), (2, 3), (20, 5), (13, 189), (50, 190)])
def test_shape_contract_never_reconstructs_three_or_four_virtual_axes(
    o: int, v: int
) -> None:
    programs = build_df_virtual_response_programs(o, v)
    for program in (
        programs.primal,
        programs.amplitude_jvp.program,
        programs.amplitude_vjp.program,
        programs.factor_vjp.program,
    ):
        assert (
            analyze_complexity(program).max_work_degree <= 5
        )  # one Q, not the complete Q sum
        for node in program.live_nodes:
            assert (
                sum(index.space.kind == "virtual" for index in node.spec.indices) <= 2
            )
            assert node.spec.size <= o * o * v * v
