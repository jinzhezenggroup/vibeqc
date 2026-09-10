"""Independent snapshot, convention, dense transform and MP2 bridge gates."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tools.vibeqc_posthf.conventions import MOBlock
from tools.vibeqc_posthf.fixtures import fixture_snapshot, load_fixture
from tools.vibeqc_posthf.mp2 import restricted_mp2, spin_orbital_mp2
from tools.vibeqc_posthf.oracle import dense_ao_to_mo
from tools.vibeqc_posthf.providers import BlockResult


@pytest.mark.parametrize("name", ["h2", "water", "lih", "f_heh"])
def test_dense_elements_and_independent_mp2(name):
    meta, a = load_fixture(name)
    s = fixture_snapshot(meta, a)
    mo = dense_ao_to_mo(a["ao"], s.coefficients)
    np.testing.assert_allclose(mo, a["conventional_mo"], atol=1e-11, rtol=1e-10)
    block = MOBlock.from_spaces(s, "ovov")
    g = mo[np.ix_(*block.slots)]
    provider = SimpleNamespace(
        snapshot=s,
        get=lambda request: BlockResult(request, g, s.identity, s.hamiltonian_id, {}),
    )
    result = restricted_mp2(s, provider)
    assert (
        abs(
            result.correlation_energy
            - meta["records"]["conventional"]["correlation_energy"]
        )
        <= 1e-9
    )
    np.testing.assert_allclose(
        result.amplitudes, a["conventional_t2"], atol=1e-11, rtol=1e-10
    )
    assert abs(spin_orbital_mp2(s, g) - result.correlation_energy) < 1e-12
    np.testing.assert_allclose(
        result.amplitudes, result.amplitudes.transpose(1, 0, 3, 2), atol=1e-12
    )
    # The numerator is unantisymmetrized spatial (ia|jb): dropping exchange
    # or the restricted factor fails independently of total HF energy.
    wrong = float(np.sum(result.amplitudes * g.transpose(0, 2, 1, 3)) * 2)
    assert abs(wrong - result.correlation_energy) > 1e-7


def test_complete_tiny_transform_explicit_eight_loops():
    rng = np.random.default_rng(147)
    g = rng.normal(size=(2,) * 4)
    c = rng.normal(size=(2, 2))
    expected = np.zeros_like(g)
    for p, q, r, s in np.ndindex((2,) * 4):
        for u, v, w, x in np.ndindex((2,) * 4):
            expected[p, q, r, s] += (
                g[u, v, w, x] * c[u, p] * c[v, q] * c[w, r] * c[x, s]
            )
    np.testing.assert_allclose(dense_ao_to_mo(g, c), expected, atol=1e-12)
    with pytest.raises(ValueError, match="at most 12"):
        dense_ao_to_mo(None, np.eye(13))


def test_snapshot_deep_ownership_and_invalidation():
    meta, a = load_fixture("h2")
    s = fixture_snapshot(meta, a)
    before = s.coefficients.copy()
    a["conventional_C"][:] = 100
    np.testing.assert_array_equal(s.coefficients, before)
    with pytest.raises(ValueError):
        s.coefficients.setflags(write=True)
    with pytest.raises(ValueError):
        s.coefficients[0, 0] = 0
    for key in ("generation_id", "geometry_hash", "basis_hash", "hamiltonian_id"):
        assert replace(s, **{key: "changed"}).identity != s.identity
    # Even a reused generation identifier cannot conceal newly phased C.
    assert replace(s, coefficients=-s.coefficients).identity != s.identity


@pytest.mark.parametrize(
    "change,match",
    [
        ({"algorithm": "UHF"}, "RHF"),
        ({"algorithm": "ROHF"}, "RHF"),
        ({"converged": False}, "unconverged"),
        ({"frozen_mask": (0,)}, "frozen"),
        ({"scf_residual": 1e-2}, "invalid RHF"),
        ({"screening_tolerance": -1}, "invalid"),
        ({"electron_count": 3}, "even"),
        ({"validation_tolerance": 1}, "tolerances"),
    ],
)
def test_invalid_reference_metadata(change, match):
    meta, a = load_fixture("h2")
    s = fixture_snapshot(meta, a)
    with pytest.raises(ValueError, match=match):
        replace(s, **change)


@pytest.mark.parametrize(
    "field", ["coefficients", "overlap", "fock", "occupations", "orbital_energies"]
)
def test_invalid_reference_arrays(field):
    meta, a = load_fixture("h2")
    s = fixture_snapshot(meta, a)
    bad = getattr(s, field).copy()
    bad.flat[0] += 0.1
    with pytest.raises(ValueError):
        replace(s, **{field: bad})
    with pytest.raises(ValueError, match="complex"):
        replace(s, coefficients=s.coefficients.astype(complex))
    with pytest.raises(ValueError, match="linear"):
        replace(s, overlap=np.zeros_like(s.overlap))


def test_small_denominators_are_not_clamped():
    meta, a = load_fixture("h2")
    s = fixture_snapshot(meta, a)
    eps = np.array([-1.0, -1.0 + 1e-12])
    f = (s.overlap @ s.coefficients * eps) @ s.coefficients.T @ s.overlap
    s = replace(s, fock=f, orbital_energies=eps)
    provider = SimpleNamespace(
        snapshot=s,
        get=lambda request: BlockResult(
            request, np.ones((1,) * 4), s.identity, s.hamiltonian_id, {}
        ),
    )
    with pytest.raises(ValueError, match="no regularization"):
        restricted_mp2(s, provider)


def test_restricted_mp2_rejects_ks_reference():
    meta, a = load_fixture("h2")
    s = fixture_snapshot(meta, a)
    ks = replace(
        s,
        algorithm="KS",
        functional_identity="test-functional",
        grid_identity="test-grid",
        hf_backend="test-ks",
    )
    provider = SimpleNamespace(
        snapshot=ks,
        get=lambda request: pytest.fail("KS reference must be rejected before reads"),
    )
    with pytest.raises(ValueError, match="RHF"):
        restricted_mp2(ks, provider)


def test_explicit_slot_order_and_invalid_indices():
    meta, a = load_fixture("water")
    s = fixture_snapshot(meta, a)
    b = MOBlock.from_spaces(s, "ovov")
    assert b.slots == (tuple(range(5)), (5, 6), tuple(range(5)), (5, 6))
    with pytest.raises(ValueError):
        MOBlock(((0, 0), (), (), ()))
    with pytest.raises(ValueError):
        MOBlock(((True,), (), (), ()))
    with pytest.raises(ValueError):
        MOBlock(((s.nmo,), (), (), ())).validate(s)
