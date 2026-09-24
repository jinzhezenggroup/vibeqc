# ruff: noqa: ANN201
"""TensorIR ownership checks for #157 factorized DF virtual corrections."""

import numpy as np

from vibeqc_compiler.tensor import execute

from tools.vibeqc_cc.df_equations import build_df_virtual_correction_program
from tools.vibeqc_cc.df_factorized import virtual_corrections


def _problem(seed=157, nocc=2, nvir=3, naux=5):
    rng = np.random.default_rng(seed)
    bov = rng.normal(size=(naux, nocc, nvir))
    bvv = rng.normal(size=(naux, nvir, nvir))
    bvv = 0.5 * (bvv + bvv.transpose(0, 2, 1))
    t1 = rng.normal(size=(nocc, nvir))
    t2 = rng.normal(size=(nocc, nocc, nvir, nvir))
    return bov, bvv, t1, t2


def test_tensorir_df_virtual_correction_matches_slice_b_oracle():
    bov, bvv, t1, t2 = _problem()
    expected_r1, expected_r2 = virtual_corrections(bov, bvv, t1, t2)

    program = build_df_virtual_correction_program(2, 3, 5)
    actual = execute(
        program,
        {"bov": bov, "bvv": bvv, "t1": t1, "t2": t2},
    ).outputs

    np.testing.assert_allclose(
        actual["df_virtual_singles"], expected_r1, atol=5e-12, rtol=0
    )
    np.testing.assert_allclose(
        actual["df_virtual_doubles"], expected_r2, atol=5e-11, rtol=0
    )


def test_tensorir_df_virtual_correction_has_no_four_index_virtual_input():
    program = build_df_virtual_correction_program(2, 3, 5)
    names = {
        node.attrs["name"] for node in program.live_nodes if node.op == "input"
    }

    assert names == {"bov", "bvv", "t1", "t2"}
    assert "ovvv" not in names
    assert "vvvv" not in names
    assert program.provenance["resident_ovvv"] is False
    assert program.provenance["resident_vvvv"] is False
    assert program.provenance["auxiliary_reduction"] == "direct-in-output-einsums"


def test_tensorir_df_virtual_correction_is_auxiliary_gauge_invariant():
    bov, bvv, t1, t2 = _problem(seed=158)
    q, _ = np.linalg.qr(np.random.default_rng(159).normal(size=(5, 5)))
    program = build_df_virtual_correction_program(2, 3, 5)

    reference = execute(
        program,
        {"bov": bov, "bvv": bvv, "t1": t1, "t2": t2},
    ).outputs
    rotated = execute(
        program,
        {
            "bov": np.einsum("PQ,Qia->Pia", q, bov),
            "bvv": np.einsum("PQ,Qab->Pab", q, bvv),
            "t1": t1,
            "t2": t2,
        },
    ).outputs

    np.testing.assert_allclose(
        rotated["df_virtual_singles"],
        reference["df_virtual_singles"],
        atol=5e-12,
        rtol=0,
    )
    np.testing.assert_allclose(
        rotated["df_virtual_doubles"],
        reference["df_virtual_doubles"],
        atol=5e-11,
        rtol=0,
    )
