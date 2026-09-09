"""RCCSD A: independent fermion algebra, off-shell and per-group gates."""

import numpy as np
import pytest

from tools.vibeqc_cc import amplitude_layouts, build_program
from tools.vibeqc_cc.oracle import DeterminantOracle, dense_feeds, random_case
from tools.vibeqc_tensor import Program, execute, optimize


def check(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-10)


@pytest.mark.parametrize(
    "o,v,seed", [(1, 1, 148), (1, 3, 149), (2, 2, 150), (2, 3, 151)]
)
def test_random_unconverged_against_exact_fermionic_projection(o, v, seed):
    f, g, t1, t2 = random_case(o, v, seed)
    reference = DeterminantOracle(f, g, o)
    energy, singles = reference.evaluate(t1, t2)
    assert np.max(np.abs(singles)) > 1e-3  # deliberately far from a CC root
    program = build_program(o, v)
    feeds = dense_feeds(f, g, t1, t2)
    for p in (program, Program.loads(program.dumps()), optimize(program)):
        result = execute(p, feeds).outputs
        check(result["correlation_energy"], energy)
        check(result["singles_residual"], singles)
    assert program.logical_hash == build_program(o, v).logical_hash


def test_homogeneous_groups_against_independent_polynomial_projections():
    f, g, x, y = random_case()
    o, v = x.shape
    # f-only and g-only Hamiltonians separate the one-/two-body contributions.
    f_oracle = DeterminantOracle(f, g * 0, o)
    g_oracle = DeterminantOracle(f * 0, g, o)
    result = execute(build_program(o, v), dense_feeds(f, g, x, y)).outputs
    ef, rf = f_oracle.evaluate(x, y)
    e1, r1 = g_oracle.evaluate(x, y * 0)
    em1, rm1 = g_oracle.evaluate(-x, y * 0)
    _, r2 = g_oracle.evaluate(2 * x, y * 0)
    _, rm2 = g_oracle.evaluate(-2 * x, y * 0)
    e_y, r_y = g_oracle.evaluate(x * 0, y)
    _, r_xy = g_oracle.evaluate(x, y)
    odd1, odd2 = (r1 - rm1) / 2, (r2 - rm2) / 2
    cubic = (odd2 - 2 * odd1) / 6
    groups = {
        "energy_t1": ef,
        "energy_t2": e_y,
        "energy_t1t1": (e1 + em1) / 2,
        "singles_fock": rf,
        "singles_g_t1": odd1 - cubic,
        "singles_g_t2": r_y,
        "singles_g_t1t1": (r1 + rm1) / 2,
        "singles_g_t1t2": r_xy - r1 - r_y,
        "singles_g_t1t1t1": cubic,
    }
    for key, value in groups.items():
        check(result[key], value)
        assert np.max(np.abs(value)) > 1e-6, key


def test_missing_disconnected_terms_and_exchange_factors_are_detected():
    f, g, t1, t2 = random_case()
    result = execute(build_program(2, 2), dense_feeds(f, g, t1, t2)).outputs
    e, r = DeterminantOracle(f, g, 2).evaluate(t1, t2)
    # Mutate the actual DAG by removing each original contribution. This is
    # not a negative check against a separately fabricated formula.
    original = build_program(2, 2)
    from tools.vibeqc_tensor import add

    for key, target, ref in (
        ("energy_t1t1", "correlation_energy", e),
        ("E03", "correlation_energy", e),
        ("S07", "singles_residual", r),
        ("singles_g_t1t1", "singles_residual", r),
    ):
        mutant = Program(
            {
                "bad": add(
                    original.outputs[target],
                    original.outputs[key],
                    coefficients=(1, -1),
                )
            }
        )
        bad = execute(mutant, dense_feeds(f, g, t1, t2)).outputs["bad"]
        with pytest.raises(AssertionError):
            check(bad, ref)
        assert np.max(np.abs(result[key])) > 1e-8


def test_zero_amplitudes_and_full_offdiagonal_fock():
    f, g, x, y = random_case()
    p = build_program(2, 2)
    zero = execute(p, dense_feeds(f, g, x * 0, y * 0)).outputs
    assert zero["correlation_energy"] == 0
    check(zero["singles_residual"], f[:2, 2:])
    # Exercise oo, vv and ov independently; canonical simplification would
    # destroy at least one of these arbitrary-F response dependencies.
    for block in ("oo", "vv", "ov"):
        perturbation = np.zeros_like(f)
        if block == "oo":
            perturbation[0, 1] = perturbation[1, 0] = 0.3
        elif block == "vv":
            perturbation[2, 3] = perturbation[3, 2] = 0.3
        else:
            perturbation[0, 3] = perturbation[3, 0] = 0.3
        expected = DeterminantOracle(perturbation, g * 0, 2).evaluate(x, y)[1]
        actual = execute(p, dense_feeds(perturbation, g * 0, x, y)).outputs[
            "singles_residual"
        ]
        check(actual, expected)
        assert np.max(np.abs(actual)) > 1e-3


def test_packed_metric_pair_symmetry_and_invalid_inputs():
    one, two = amplitude_layouts(2, 3)
    assert one.size == 6 and two.size == 21
    f, g, x, y = random_case(2, 3)
    for layout, tensor in ((one, x), (two, y)):
        packed = layout.pack(tensor)
        check(layout.unpack(packed), tensor)
        check(layout.inner_product(packed, packed), np.sum(tensor * tensor))
    assert not np.allclose(y, -y.transpose(1, 0, 2, 3))
    oracle = DeterminantOracle(f, g, 2)
    excitation = oracle.cluster(x, y) @ oracle.ket
    physical_norm = 2 * np.sum(x * x) + np.sum(y * (2 * y - y.transpose(0, 1, 3, 2)))
    check(excitation @ excitation, physical_norm)
    assert not np.isclose(np.sum(x * x) + np.sum(y * y), physical_norm)
    feeds = dense_feeds(f, g, x, y)
    p = build_program(2, 3)
    for bad in (
        x.astype(complex),
        x.astype(np.float32),
        np.full_like(x, np.nan),
        x[:, :2],
    ):
        with pytest.raises((ValueError, TypeError)):
            execute(p, {**feeds, "t1": bad})
    broken = y.copy()
    broken[0, 1, 0, 1] += 1
    with pytest.raises(ValueError):
        execute(p, {**feeds, "t2": broken})
    with pytest.raises(ValueError, match="budget"):
        execute(p, feeds, max_bytes=1)
    with pytest.raises(ValueError):
        build_program(1, 0)


def test_total_energy_cancellation_cannot_hide_component_failure():
    f, g, x, y = random_case()
    program = build_program(2, 2)
    initial = execute(program, dense_feeds(f, g, x, y)).outputs
    # Force the three energy groups to cancel while each remains nonzero.
    scale = -(initial["energy_t2"] + initial["energy_t1t1"]) / initial["energy_t1"]
    f[:2, 2:] *= scale
    f[2:, :2] = f[:2, 2:].T
    result = execute(program, dense_feeds(f, g, x, y)).outputs
    assert abs(result["correlation_energy"]) < 1e-14
    for key in ("energy_t1", "energy_t2", "energy_t1t1"):
        assert abs(result[key]) > 1e-6
        with pytest.raises(AssertionError):
            check(0.0, result[key])


@pytest.mark.parametrize("key", ["foo", "fvv", "ovov", "oovv", "ovvv", "ovoo", "ovvo"])
def test_declared_fock_and_eri_symmetries_reject_corruption(key):
    feeds = dense_feeds(*random_case())
    corrupt = feeds[key].copy()
    index = (0,) * (corrupt.ndim - 1) + (1,)
    corrupt[index] += 1
    with pytest.raises(ValueError, match="symmetr"):
        execute(build_program(2, 2), {**feeds, key: corrupt})
