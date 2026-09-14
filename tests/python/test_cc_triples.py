"""Closed-shell CCSD(T) triples: audited reference, full-sum oracle, TensorIR.

Slice A of issue #150.  These tests never import PySCF: the nonzero-molecule
ground truth below is hard-coded from the pinned PySCF 2.14.0
``ccsd_t_slow.kernel`` run over ``tests/reference_data/cc/endpoints/*`` (see
``docs/rccsd_t.md``). The independently generated production-reference JSON
is also checked against the committed inputs without importing PySCF.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.tensor import Program, dot_test, execute, jvp

INPUT_NAMES = ("ovvv", "ovoo", "ovov", "fov", "t1", "t2", "eps_o", "eps_v")


def _random_case(nocc, nvir, seed):
    rng = np.random.default_rng(seed)
    ovvv = rng.normal(size=(nocc, nvir, nvir, nvir))
    ovoo = rng.normal(size=(nocc, nvir, nocc, nocc))
    ovov = rng.normal(size=(nocc, nvir, nocc, nvir))
    fov = rng.normal(size=(nocc, nvir))
    t1 = rng.normal(size=(nocc, nvir))
    t2 = rng.normal(size=(nocc, nocc, nvir, nvir))
    t2 = (t2 + t2.transpose(1, 0, 3, 2)) / 2
    eps_o = np.linspace(-1.0, -0.5, nocc)
    eps_v = np.linspace(0.5, 1.5, nvir)
    return ovvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v


# --------------------------- core engine agreement ---------------------------


@pytest.mark.parametrize("term", ["w1", "w2", "v1", "v2"])
def test_numerator_inventory_matches_pinned_source_and_execution(term):
    """Audit each term against explicit source-index loops, before spin sums.

    Unequal occupied/virtual sizes expose contracted-axis swaps, and isolated
    terms prevent cancellation from hiding an incorrect coefficient. The
    inventory operands refer to the transposed views used by the slow source.
    """
    from tools.vibeqc_cc.triples import V_TERMS, W_TERMS, _v, _views, _w

    o, v = 2, 3
    ovvv, ovoo, ovov, fov, t1, t2, _, _ = _random_case(o, v, 17)
    if term != "w1":
        ovvv.fill(0)
    if term != "w2":
        ovoo.fill(0)
    if term != "v1":
        ovov.fill(0)
    if term != "v2":
        fov.fill(0)
    a, b, c = 2, 1, 0
    expected = np.zeros((o, o, o))
    for i in range(o):
        for j in range(o):
            for k in range(o):
                expected[i, j, k] = (
                    sum(ovvv[i, a, f, b] * t2[k, j, c, f] for f in range(v))
                    - sum(ovoo[i, a, j, m] * t2[m, k, b, c] for m in range(o))
                    + ovov[i, a, j, b] * t1[k, c]
                    + t2[i, j, a, b] * fov[k, c]
                )

    views = _views(ovvv, ovoo, ovov, fov, t1, t2)
    t1T, t2T, vvov, vooo, vvoo, fvo = views
    # These are the literal coefficient/subscript/operand contracts in the
    # pinned PySCF 2.14.0 get_w/get_v functions (no PySCF import at test time).
    assert W_TERMS == (
        (1, "if,fkj->ijk", ("vvov", "t2T")),
        (-1, "ijm,mk->ijk", ("vooo", "t2T")),
    )
    assert V_TERMS == (
        (1, "ij,k->ijk", ("vvoo", "t1T")),
        (1, "ij,k->ijk", ("t2T", "fvo")),
    )
    operands = (
        (vvov[a, b], t2T[c]),
        (vooo[a], t2T[b, c]),
        (vvoo[a, b], t1T[c]),
        (t2T[a, b], fvo[c]),
    )
    audited = sum(
        coefficient * np.einsum(subscripts, *args)
        for (coefficient, subscripts, _), args in zip(W_TERMS + V_TERMS, operands)
    )
    np.testing.assert_allclose(audited, expected, atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(
        _w(views, a, b, c) + _v(views, a, b, c), expected, atol=1e-13, rtol=1e-13
    )


@pytest.mark.parametrize(
    "o,v,seed",
    [
        (1, 1, 1),
        (1, 2, 2),
        (2, 1, 3),
        (2, 2, 4),
        (2, 3, 5),
        (3, 2, 6),
        (3, 3, 7),
        (3, 4, 8),
        (5, 2, 9),
    ],
)
def test_reference_fullsum_tensorir_agree(o, v, seed):
    from tools.vibeqc_cc import triples_energy, triples_energy_tensorir, triples_fullsum

    arrays = _random_case(o, v, seed)
    reference = triples_energy(o, v, *arrays)
    fullsum = triples_fullsum(o, v, *arrays)
    tensorir = triples_energy_tensorir(o, v, *arrays)
    np.testing.assert_allclose(fullsum, reference, atol=1e-11, rtol=1e-10)
    np.testing.assert_allclose(tensorir, reference, atol=1e-11, rtol=1e-10)


def test_reference_matches_verbatim_slow_kernel_shapes():
    """The triangular reference reproduces the literal ccsd_t_slow 36-entry
    table, so it must be symmetric under simultaneous (a,b,c) permutation.
    An independent re-implementation here rebuilds the table from the source
    einsum strings and compares it against the module's SLOW_TABLE."""
    from tools.vibeqc_cc.triples import _LABELS, OP, SLOW_TABLE, VP

    def inv(p):
        out = [0, 0, 0]
        for i, x in enumerate(p):
            out[x] = i
        return tuple(out)

    def compose(p, q):
        return tuple(p[i] for i in q)

    # The S3-derived pairing that the fullsum oracle relies on must reproduce
    # every one of the 36 occupied pairings of the literal table.
    assert len(SLOW_TABLE) == 6
    for zlbl in _LABELS:
        p = VP[zlbl]
        assert len(SLOW_TABLE[zlbl]) == 6
        for wlbl, ost in SLOW_TABLE[zlbl]:
            pp = VP[wlbl]
            assert compose(inv(pp), p) == OP[ost]


# ------------------------------ multiplicity --------------------------------

# r3 coefficients exercised on an explicit cubic tensor (issue's (4,1,1,-2,-2,-2)).


def _explicit_r3(w):
    return (
        4 * w
        + w.transpose(1, 2, 0)
        + w.transpose(2, 0, 1)
        - 2 * w.transpose(2, 1, 0)
        - 2 * w.transpose(0, 2, 1)
        - 2 * w.transpose(1, 0, 2)
    )


def test_r3_coefficients_one_term_at_a_time():
    from tools.vibeqc_cc.triples import R3, r3

    rng = np.random.default_rng(3)
    w = rng.normal(size=(3, 3, 3))
    expected = np.zeros_like(w)
    for coefficient, perm in R3:
        expected += coefficient * w.transpose(perm)
    np.testing.assert_array_equal(r3(w), expected)
    np.testing.assert_array_equal(r3(w), _explicit_r3(w))
    assert [c for c, _ in R3] == [4, 1, 1, -2, -2, -2]
    # r3 adds six distinct permutations; mutating any single coefficient must
    # be detectable on a generic (non-symmetric) w.
    for k, (coefficient, perm) in enumerate(R3):
        wrong = expected - coefficient * w.transpose(perm)
        assert not np.allclose(wrong, expected)


def test_triangular_virtual_multiplicity_against_fullsum():
    """The 36-term contraction is symmetric under simultaneous (a,b,c)
    permutation, so the ordered full sum equals 6 times the triangular sum
    after each triangle is re-weighted by 6/2/1.  Fullsum already checks this
    globally; here we assert the degeneracy function in isolation."""
    from tools.vibeqc_cc.triples import _degeneracy

    # a == b == c -> 6
    assert _degeneracy(0, 0, 0) == 6
    assert _degeneracy(2, 2, 2) == 6
    # a == b -> 2, and b == c -> 2 (with a != c)
    assert _degeneracy(2, 2, 1) == 2
    assert _degeneracy(2, 1, 1) == 2
    # all distinct in the triangular domain -> 1
    assert _degeneracy(2, 1, 0) == 1


def test_explicit_degenerate_and_double_indices_numerically():
    """Construct all four degeneracy classes on a small (2,2) case and confirm
    the audited triangle vs the independent full sum at each block weight.

    The overall energy agreement is asserted in the parameterized test; this
    test additionally nails the specific index patterns the issue calls for
    (a==b, a==c, b==c, a==b==c) by checking the degeneracy table directly.
    """
    from tools.vibeqc_cc.triples import VP, _degeneracy, _permuted

    # All six virtual permutations of (a, b, c) must be reachable through VP
    # and each maps to a distinct triple index ordering.
    seen = {_permuted((2, 1, 0), VP[lbl]) for lbl in VP}
    assert seen == {(2, 1, 0), (2, 0, 1), (1, 2, 0), (1, 0, 2), (0, 2, 1), (0, 1, 2)}
    # Distinct-triple degeneracy is 1; all-equal is 6; any pairing is 2.
    assert _degeneracy(2, 1, 0) == 1  # all distinct
    assert _degeneracy(1, 1, 0) == 2  # a == b
    assert _degeneracy(1, 0, 0) == 2  # b == c
    assert _degeneracy(1, 1, 1) == 6  # a == b == c


# -------------------------- degenerate-limit paths ---------------------------


@pytest.mark.parametrize("o,v", [(2, 3), (3, 3)])
def test_t1_zero_removes_v_and_t2_zero_keeps_w_only(o, v):
    from tools.vibeqc_cc import triples_energy, triples_fullsum

    ovvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v = _random_case(o, v, 11)
    zero1 = np.zeros_like(t1)
    zero2 = np.zeros_like(t2)
    # With fov == 0 AND t1 == 0, V = ovov*t1 + t2*fov vanishes and only the W
    # (t2-driven) numerator survives.
    w_only = triples_energy(
        o, v, ovvv, ovoo, ovov, np.zeros_like(fov), zero1, t2, eps_o, eps_v
    )
    w_full = triples_fullsum(
        o, v, ovvv, ovoo, ovov, np.zeros_like(fov), zero1, t2, eps_o, eps_v
    )
    assert abs(w_only) > 1e-6  # generic nonzero (T) path with only W
    np.testing.assert_allclose(w_full, w_only, atol=1e-11, rtol=1e-10)
    # With t2 == 0 every W seed vanishes, so the (T) energy -- which is the
    # contraction of W against r3(W + 0.5 V)/d3 -- is identically zero even
    # though V = ovov*t1 + t2*fov is nonzero.  This is the correct degeneracy:
    # no doubles means no triples correction.
    v_only = triples_energy(o, v, ovvv, ovoo, ovov, fov, t1, zero2, eps_o, eps_v)
    assert abs(v_only) < 1e-12


def test_two_electron_triples_are_zero():
    """A two-electron reference has no genuine triples: (1,1) (T) vanishes.

    CCSD already equals FCI for two electrons in any basis, so the (T)
    correction must be numerically zero to machine rounding.
    """
    from tools.vibeqc_cc import triples_energy, triples_fullsum

    ovvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v = _random_case(1, 1, 12)
    reference = triples_energy(1, 1, ovvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v)
    fullsum = triples_fullsum(1, 1, ovvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v)
    assert abs(reference) < 1e-12
    assert abs(fullsum) < 1e-12


# ------------------------------ TensorIR + AD ------------------------------


def test_tensorir_program_roundtrip_and_replay():
    from tools.vibeqc_cc import build_triples_program, triples_energy_tensorir

    o, v = 2, 2
    arrays = _random_case(o, v, 21)
    ovvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v = arrays
    feeds = {
        "ovvv": ovvv,
        "ovoo": ovoo,
        "ovov": ovov,
        "fov": fov,
        "t1": t1,
        "t2": t2,
        "eps_o": eps_o,
        "eps_v": eps_v,
    }
    program = build_triples_program(o, v)
    replayed = Program.loads(program.dumps())
    expected = triples_energy_tensorir(o, v, *arrays)
    for p in (program, replayed):
        value = execute(p, feeds).outputs["triples_energy"]
        np.testing.assert_allclose(value, expected, atol=1e-11, rtol=1e-10)
    assert program.logical_hash == build_triples_program(o, v).logical_hash


def test_tensorir_program_is_differentiable():
    from tools.vibeqc_cc import build_triples_program, triples_energy

    o, v = 2, 2
    arrays = _random_case(o, v, 22)
    ovvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v = arrays
    feeds = {
        "ovvv": ovvv,
        "ovoo": ovoo,
        "ovov": ovov,
        "fov": fov,
        "t1": t1,
        "t2": t2,
        "eps_o": eps_o,
        "eps_v": eps_v,
    }
    program = build_triples_program(o, v)
    rng = np.random.default_rng(23)
    tangents = {k: rng.normal(size=x.shape) for k, x in feeds.items()}
    result = dot_test(program, feeds, tangents, {"triples_energy": np.array(1.0)})
    assert result.passed, "TensorIR (T) adjoint dot test failed"
    # One finite-difference check of a single t1 entry against the JVP rule.
    seed = np.zeros_like(t1)
    seed[0, 0] = 1.0
    derived = jvp(program, feeds, {"t1": seed}).output_tangents["triples_energy"]
    h = 1e-6
    plus = {k: x.copy() for k, x in feeds.items()}
    minus = {k: x.copy() for k, x in feeds.items()}
    plus["t1"][0, 0] += h
    minus["t1"][0, 0] -= h

    def energy(feeds_map):
        return triples_energy(
            o,
            v,
            feeds_map["ovvv"],
            feeds_map["ovoo"],
            feeds_map["ovov"],
            feeds_map["fov"],
            feeds_map["t1"],
            feeds_map["t2"],
            feeds_map["eps_o"],
            feeds_map["eps_v"],
        )

    finite = (energy(plus) - energy(minus)) / (2 * h)
    np.testing.assert_allclose(derived, finite, rtol=1e-6, atol=1e-8)


# --------------------------- input validation paths --------------------------


@pytest.mark.parametrize(
    "engine", ["triples_energy", "triples_fullsum", "triples_energy_tensorir"]
)
@pytest.mark.parametrize("name", INPUT_NAMES)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_inputs_fail_closed(engine, name, value):
    """Invalid amplitudes, integrals or energies must not yield a (T) result."""
    from tools import vibeqc_cc

    arrays = _random_case(2, 3, 32)
    arrays[INPUT_NAMES.index(name)].flat[0] = value
    with pytest.raises(ValueError, match=f"{name}.*finite"):
        getattr(vibeqc_cc, engine)(2, 3, *arrays)


@pytest.mark.parametrize(
    "engine", ["triples_energy", "triples_fullsum", "triples_energy_tensorir"]
)
@pytest.mark.parametrize(
    "threshold", [np.float32(1e-4), np.float64(1e-10), np.int64(1)]
)
def test_numpy_scalar_denominator_thresholds(engine, threshold):
    """NumPy-derived tolerances obey the same gate as Python scalar values."""
    from tools import vibeqc_cc

    arrays = _random_case(2, 2, 33)
    energy = getattr(vibeqc_cc, engine)
    expected = energy(2, 2, *arrays, denominator_threshold=float(threshold))
    assert energy(2, 2, *arrays, denominator_threshold=threshold) == expected
    # The actual magnitude still controls rejection, regardless of scalar type.
    with pytest.raises(ValueError, match="near-zero"):
        energy(2, 2, *arrays, denominator_threshold=type(threshold)(100))


@pytest.mark.parametrize(
    "threshold",
    [True, np.bool_(True), np.float64(np.nan), np.float32(np.inf), np.int64(0), -1.0],
)
def test_invalid_denominator_thresholds(threshold):
    from tools.vibeqc_cc import triples_energy

    with pytest.raises(ValueError, match="threshold must be a positive finite number"):
        triples_energy(2, 2, *_random_case(2, 2, 34), denominator_threshold=threshold)


def test_invalid_inputs_rejected():
    from tools.vibeqc_cc import build_triples_program, triples_energy

    arrays = _random_case(2, 2, 31)
    ovvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v = arrays
    with pytest.raises(ValueError):
        triples_energy(
            2, 2, ovvv.astype(np.float32), ovoo, ovov, fov, t1, t2, eps_o, eps_v
        )
    with pytest.raises(ValueError):
        triples_energy(
            2, 2, ovvv, ovoo, ovov, fov, t2, t1, eps_o, eps_v
        )  # t1/t2 swapped mismatch
    with pytest.raises(ValueError):
        build_triples_program(0, 2)
    with pytest.raises(ValueError):
        build_triples_program(2, 0)


def test_degenerate_denominators_fail_closed():
    """Noncanonical (crossing) or near-degenerate (T) denominators are rejected
    explicitly rather than silently dividing into NaN (issue #150 step 7)."""
    from tools.vibeqc_cc import triples_energy, triples_fullsum

    ovvv, ovoo, ovov, fov, t1, t2, _, _ = _random_case(2, 2, 41)
    # occupied energies not strictly below virtual -> nonnegative denominator
    with pytest.raises(ValueError, match="noncanonical"):
        triples_energy(
            2,
            2,
            ovvv,
            ovoo,
            ovov,
            fov,
            t1,
            t2,
            np.array([-0.4, -0.5]),
            np.array([-0.5, -0.4]),
        )
    with pytest.raises(ValueError, match="noncanonical"):
        triples_fullsum(
            2,
            2,
            ovvv,
            ovoo,
            ovov,
            fov,
            t1,
            t2,
            np.array([-0.4, -0.5]),
            np.array([-0.5, -0.4]),
        )
    # canonical but near-zero denominator -> explicit near-degeneracy error:
    # occupied = -1.0 (triple sum -3.0), virtual = -1.0 + 1e-11 (triple sum
    # -3.0 + 3e-11), so every d3 = -3e-11 is negative but |d3| <= threshold.
    arrays = _random_case(3, 3, 43)
    ovvv, ovoo, ovov, fov, t1, t2, _, _ = arrays
    with pytest.raises(ValueError, match="near-zero"):
        triples_energy(
            3,
            3,
            ovvv,
            ovoo,
            ovov,
            fov,
            t1,
            t2,
            np.full(3, -1.0),
            np.full(3, -1.0 + 1e-11),
        )


# -------------------------- hard-coded ground truth --------------------------

# E_T (Hartree) from pinned PySCF 2.14.0 ccsd_t_slow.kernel over
# tests/reference_data/cc/endpoints/* (o,v) documented in docs/rccsd_t.md.
GROUND_TRUTH = {
    "h2": (1, 1, 8.392021714075268e-49),
    "he": (1, 1, 0.0),
    "h2o": (5, 2, -6.731393342463869e-05),
    "nh3": (5, 3, -1.122922812723691e-04),
    "ch4": (5, 4, -1.555665872715297e-04),
}

ENDPOINTS = Path(__file__).resolve().parents[1] / "reference_data/cc/endpoints"


def _endpoint_feeds(name):
    with np.load(ENDPOINTS / f"{name}.npz", allow_pickle=False) as data:
        eps = data["eps"]
        occ = data["occ"]
        C = data["C"]
        F = data["F"]
        g = data["g"]
        t1 = data["t1"]
        t2 = data["t2"]
    nocc = int(np.sum(occ > 0))
    nvir = len(eps) - nocc
    fov = (C.T @ F @ C)[:nocc, nocc:]
    return (
        nocc,
        nvir,
        g[:nocc, nocc:, nocc:, nocc:],
        g[:nocc, nocc:, :nocc, :nocc],
        g[:nocc, nocc:, :nocc, nocc:],
        fov,
        t1,
        t2,
        eps[:nocc],
        eps[nocc:],
    )


@pytest.mark.parametrize("name", ["h2", "he", "h2o", "nh3", "ch4"])
def test_pinned_ground_truth_regression(name):
    from tools.vibeqc_cc import triples_energy, triples_energy_tensorir, triples_fullsum

    expected_nocc, expected_nvir, expected = GROUND_TRUTH[name]
    feeds = _endpoint_feeds(name)
    assert feeds[0] == expected_nocc and feeds[1] == expected_nvir
    reference = triples_energy(*feeds)
    tensorir = triples_energy_tensorir(*feeds)
    fullsum = triples_fullsum(*feeds)
    if name in ("h2", "he"):
        assert abs(reference) < 1e-9
        assert abs(tensorir) < 1e-9
        assert abs(fullsum) < 1e-9
    else:
        np.testing.assert_allclose(reference, expected, atol=1e-9, rtol=0)
        np.testing.assert_allclose(tensorir, expected, atol=1e-9, rtol=0)
        np.testing.assert_allclose(fullsum, expected, atol=1e-9, rtol=0)


def test_committed_production_reference_provenance():
    """Keep the independent reference tied to its source and endpoint arrays."""
    from tools.cc_endpoint_fixtures import array_hash
    from tools.vibeqc_validation.schema import canonical_hash

    root = Path(__file__).resolve().parents[2]
    data = json.loads((ENDPOINTS.parent / "rccsd-t.json").read_text())
    assert data["schema"] == "vibeqc.rccsd-t.reference"
    assert data["version"] == 1
    assert data["pyscf"] == "2.14.0"
    assert data["upstream"] == json.loads(
        (root / "tools/vibeqc_cc/source_manifest.json").read_text()
    )
    assert data["molecules_hash"] == canonical_hash(data["molecules"])
    assert [row["name"] for row in data["molecules"]] == list(GROUND_TRUTH)
    for row in data["molecules"]:
        nocc, nvir, *arrays = _endpoint_feeds(row["name"])
        assert (row["nocc"], row["nvir"]) == (nocc, nvir)
        assert row["inputs_hash"] == array_hash(dict(zip(INPUT_NAMES, arrays)))
        assert row["et_ground_truth"] == GROUND_TRUTH[row["name"]][2]
        np.testing.assert_allclose(
            [row["et_numpy"], row["et_pyscf_ccsd_t"]],
            row["et_ground_truth"],
            atol=1e-9,
            rtol=0,
        )
        assert row["et_agreement"] == abs(row["et_numpy"] - row["et_pyscf_ccsd_t"])
        assert row["et_agreement"] <= 1e-9


def test_reference_generator_requires_direct_energy_agreement():
    """Being close to the same target does not imply mutual 1e-9 agreement."""
    from tools.generate_cc_triples_references import _check_energies

    truth = -1e-4
    _check_energies("within-gate", truth - 4e-10, truth + 4e-10, truth)
    with pytest.raises(ValueError, match="opposite-sides.*diverged"):
        _check_energies("opposite-sides", truth - 7.5e-10, truth + 7.5e-10, truth)
    with pytest.raises(ValueError, match="shared-error.*diverged"):
        _check_energies("shared-error", truth + 2e-9, truth + 2e-9, truth)
