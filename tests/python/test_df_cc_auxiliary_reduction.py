"""Auxiliary-reduction schedule against independent complete RCCSD equations."""

from __future__ import annotations

from math import prod

import numpy as np
import pytest
from generativeqc_compiler.cc.df_hoist import (
    HOISTED_INTERMEDIATES,
    build_df_auxiliary_reduction_programs,
)
from generativeqc_compiler.cc.doubles import build_ccsd_program
from generativeqc_compiler.tensor import Program, execute
from test_df_cc_native_solver import _case

from tools.generate_df_ccsd_native import programs as old_programs
from tools.generativeqc_cc.df_factorized import virtual_corrections
from tools.generativeqc_cc.oracle import DeterminantOracle


def _evaluate(feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    o, v = feeds["t1"].shape
    pipeline = build_df_auxiliary_reduction_programs(o, v)
    prepared = execute(pipeline.prepare, feeds).outputs
    reduced = {
        key: np.zeros(node.spec.shape)
        for key, node in pipeline.auxiliary.outputs.items()
    }
    for ov, vv in zip(feeds["bov"], feeds["bvv"], strict=True):
        row = execute(
            pipeline.auxiliary, {**feeds, **prepared, "bov": ov, "bvv": vv}
        ).outputs
        for key, value in row.items():
            reduced[key] += value
    return execute(pipeline.core, {**feeds, **reduced}).outputs


@pytest.mark.parametrize("o,v,q", [(1, 3, 2), (2, 3, 4), (3, 2, 3)])
def test_shared_auxiliary_reduction_matches_expanded_and_determinants(
    o: int, v: int, q: int
) -> None:
    f, g, feeds = _case(o, v, q)
    rng = np.random.default_rng(1766 + o)
    feeds["t1"] = rng.normal(scale=0.06, size=(o, v))
    t2 = rng.normal(scale=0.08, size=(o, o, v, v))
    feeds["t2"] = (t2 + t2.transpose(1, 0, 3, 2)) / 2
    actual = _evaluate(feeds)
    expanded = execute(
        build_ccsd_program(o, v, form="expanded", diagnostics=False), feeds
    ).outputs
    determinant = DeterminantOracle(f, g, o).evaluate_full(feeds["t1"], feeds["t2"])
    for key, ref in zip(
        ("correlation_energy", "singles_residual", "doubles_residual"),
        determinant,
        strict=True,
    ):
        np.testing.assert_allclose(actual[key], ref, atol=2e-12, rtol=0)
        np.testing.assert_allclose(actual[key], expanded[key], atol=2e-12, rtol=0)
    # Independently written NumPy virtual equations isolate the new Q pipeline
    # by zeroing only the retained Hamiltonian, keeping every amplitude intact.
    zeroed = {
        key: value if key in ("t1", "t2", "bov", "bvv") else np.zeros_like(value)
        for key, value in feeds.items()
    }
    virtual = _evaluate(zeroed)
    r1, r2 = virtual_corrections(feeds["bov"], feeds["bvv"], feeds["t1"], feeds["t2"])
    np.testing.assert_allclose(virtual["singles_residual"], r1, atol=2e-12, rtol=0)
    np.testing.assert_allclose(virtual["doubles_residual"], r2, atol=2e-12, rtol=0)


def test_auxiliary_gauge_rotation_leaves_complete_core_invariant() -> None:
    _, _, feeds = _case()
    rotation, _ = np.linalg.qr(np.random.default_rng(1766).normal(size=(4, 4)))
    rotated = {
        **feeds,
        **{
            key: np.einsum("QR,Rab->Qab", rotation, feeds[key])
            for key in ("bov", "bvv")
        },
    }
    actual, expected = _evaluate(rotated), _evaluate(feeds)
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], atol=2e-12, rtol=0)


def _contraction_terms(program: Program, o: int, v: int) -> int:
    """Count scalar summands, including output positions and reduction labels."""
    total = 0
    for node in program.live_nodes:
        if node.op == "einsum":
            dimensions = {
                label: o if index.space.kind == "occupied" else v
                for child, labels in zip(node.inputs, node.attrs["labels"], strict=True)
                for label, index in zip(labels, child.spec.indices, strict=True)
            }
            total += prod(dimensions.values())
    return total


def test_runtime_shape_schedule_hoists_work_before_t2_contraction() -> None:
    # Native code uses this fixed representative schedule with runtime extents.
    # Count the complete Q loop plus prepare/core, never one slice alone.
    pipeline = build_df_auxiliary_reduction_programs(2, 3)

    def names(program: Program) -> set[str]:
        return {n.attrs["name"] for n in program.live_nodes if n.op == "input"}

    assert names(pipeline.prepare) == {"t1", "t2"}
    assert "df_tau" in names(pipeline.auxiliary)
    assert {f"df_{name}" for name in HOISTED_INTERMEDIATES} <= names(pipeline.core)
    assert not names(pipeline.core) & {"bov", "bvv", "ovvv", "vvvv"}
    for program in (pipeline.prepare, pipeline.auxiliary, pipeline.core):
        assert all(
            sum(i.space.kind == "virtual" for i in n.spec.indices) <= 2
            for n in program.live_nodes
        )
    old = old_programs("cuda")["virtual"]
    for o, v, q in [(9, 221, 488), (21, 243, 666)]:
        old_virtual = q * _contraction_terms(old, o, v)
        complete_new = (
            _contraction_terms(pipeline.prepare, o, v)
            + q * _contraction_terms(pipeline.auxiliary, o, v)
            + _contraction_terms(pipeline.core, o, v)
        )
        # Even charging the entire new retained core against just the old
        # virtual correction leaves a conservative 3x scalar-work reduction.
        assert 3 * complete_new < old_virtual
