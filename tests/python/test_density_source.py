"""Current D/C parity, independent pinned features and rejected-factor semantics."""

import typing
from dataclasses import FrozenInstanceError, replace
from itertools import combinations

import numpy as np
import pytest
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.dft import DensitySource, density_features, orbital_features
from vibeqc_compiler.dft.fixtures import NAMES, load_fixture

BASIS = canonical_hash({"basis": "synthetic", "geometry": [0, 0, 0]})
KEYS = ("rho", "gradient", "sigma", "tau")


def assert_features(actual: typing.Any, expected: typing.Any) -> None:
    assert actual.keys() == expected.keys()
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-11, rtol=1e-10)


def sample(counts: typing.Any = (3, 2), points: typing.Any = 11) -> typing.Any:
    """Nonorthogonal factors exercise arbitrary PSD densities, not SCF assumptions."""
    rng = np.random.default_rng(235)
    c = tuple(rng.normal(size=(5, n)) / 3 for n in counts)
    occ = tuple(np.linspace(0.8, 0.2, n) for n in counts)
    d = np.stack([(cs * f) @ cs.T for cs, f in zip(c, occ, strict=True)])
    jets = rng.normal(size=(20, points, 5)) / 2
    source = DensitySource(d, basis_identity=BASIS, density_generation=7)
    return source, c, occ, jets


@pytest.mark.parametrize("name", NAMES)
def test_current_routes_against_independent_pyscf_features(
    name: typing.Any,
) -> None:
    meta, data = load_fixture(name)
    source = DensitySource(
        data["density"], basis_identity=canonical_hash(meta["inputs"])
    )
    checked = source.with_orbitals(
        data["coefficients"], data["occupations"], stamp=source.stamp, validation_rows=3
    )
    assert checked.source_kind == "orbitals"
    assert checked.validation_status == "validated_external"
    assert checked.validation_max_abs_error < 1e-11
    assert source.fallback_reason == "missing_orbitals"
    for route in ("density_matrix", "orbitals", "auto"):
        actual = checked.features(data["ao_jets"], stamp=source.stamp, route=route)
        assert_features(actual, {key: data[key] for key in KEYS})


@pytest.mark.parametrize("counts", [(3, 2), (3, 0), (0, 0)])
@pytest.mark.parametrize("points", [0, 11])
def test_fractional_ragged_and_empty_spin_channels(
    counts: typing.Any, points: typing.Any
) -> None:
    source, c, occ, jets = sample(counts, points)
    checked = source.with_orbitals(c, occ, stamp=source.stamp, validation_rows=2)
    actual = checked.features(jets, stamp=source.stamp, route="orbitals")
    assert_features(actual, density_features(jets, source.density))
    for spin, count in enumerate(counts):
        if not count:
            for key in ("rho", "gradient", "tau"):
                assert np.count_nonzero(actual[key][spin]) == 0


REQUESTS = [r for size in range(1, 5) for r in combinations(KEYS, size)]


@pytest.mark.parametrize("ingredients", REQUESTS)
@pytest.mark.parametrize("njet", [4, 10, 20])
def test_both_routes_prune_requested_outputs_and_accept_jet_orders(
    ingredients: typing.Any, njet: typing.Any
) -> None:
    source, c, occ, jets = sample()
    checked = source.with_orbitals(c, occ, stamp=source.stamp)
    full = density_features(jets, source.density)
    # An LDA request must work without computing or supplying AO derivatives.
    jets = jets[: 1 if ingredients == ("rho",) else njet]
    for route in ("density_matrix", "orbitals"):
        actual = checked.features(
            jets, stamp=source.stamp, route=route, ingredients=ingredients
        )
        assert_features(actual, {key: full[key] for key in ingredients})


def test_total_density_occupations_are_split_once() -> None:
    source, c, occ, jets = sample()
    total = 2 * source.density[0]
    source = DensitySource(total, basis_identity=BASIS)
    checked = source.with_orbitals((c[0], c[0]), (occ[0], occ[0]), stamp=source.stamp)
    assert source.stamp.layout == "total"
    assert_features(
        checked.features(jets, stamp=source.stamp), density_features(jets, total)
    )
    doubled = source.with_orbitals(
        (c[0], c[0]), (2 * occ[0], 2 * occ[0]), stamp=source.stamp
    )
    assert doubled.fallback_reason == "incompatible_orbitals"
    separate = DensitySource(source.density, basis_identity=BASIS)
    assert separate.stamp.density_identity != source.stamp.density_identity


def test_signs_and_equal_occupation_rotations_preserve_density() -> None:
    _, c, _, jets = sample((3, 3))
    occ = (np.array([0.7, 0.7, 0.2]),) * 2
    d = np.stack([(cs * f) @ cs.T for cs, f in zip(c, occ, strict=True)])
    source = DensitySource(d, basis_identity=BASIS)
    checked = source.with_orbitals(c, occ, stamp=source.stamp)
    q = np.array([[0.6, -0.8, 0], [0.8, 0.6, 0], [0, 0, -1]])
    rotated = source.with_orbitals(tuple(cs @ q for cs in c), occ, stamp=source.stamp)
    assert rotated.factor_identity != checked.factor_identity
    assert_features(
        rotated.features(jets, stamp=source.stamp, route="orbitals"),
        checked.features(jets, stamp=source.stamp, route="orbitals"),
    )


@pytest.mark.parametrize("ids", [[4, 0, 2], [3], []])
def test_local_ao_maps_retain_cross_terms_and_all_orbitals(
    ids: typing.Any,
) -> None:
    source, c, occ, jets = sample()
    checked = source.with_orbitals(c, occ, stamp=source.stamp)
    masked = np.zeros_like(jets)
    masked[:, :, ids] = jets[:, :, ids]
    expected = density_features(masked, source.density)
    for route in ("density_matrix", "orbitals"):
        assert_features(
            checked.features(
                jets[:, :, ids], stamp=source.stamp, route=route, ao_ids=ids
            ),
            expected,
        )
    if len(ids) > 1:
        diagonal = np.stack([np.diag(np.diag(d)) for d in source.density])
        assert (
            np.max(abs(density_features(masked, diagonal)["rho"] - expected["rho"]))
            > 1e-3
        )
        truncated = orbital_features(
            masked, tuple(cs[:, :1] for cs in c), tuple(f[:1] for f in occ)
        )
        assert np.max(abs(truncated["rho"] - expected["rho"])) > 1e-3


def test_orbital_density_directions_include_sigma_cross_terms() -> None:
    source, c, occ, jets = sample()
    rng = np.random.default_rng(33)
    dc = tuple(rng.normal(size=cs.shape) / 5 for cs in c)
    delta = np.stack(
        [
            (change * f) @ cs.T + (cs * f) @ change.T
            for cs, change, f in zip(c, dc, occ, strict=True)
        ]
    )
    linear = density_features(jets, delta)
    base = density_features(jets, source.density)
    g, dg = base["gradient"], linear["gradient"]
    expected = dict(linear)
    expected["sigma"] = np.stack(
        [
            np.sum(dg[a] * g[b] + g[a] * dg[b], axis=1)
            for a, b in ((0, 0), (0, 1), (1, 1))
        ]
    )
    errors = []
    for step in (1e-3, 1e-4, 1e-5):
        plus = orbital_features(
            jets, tuple(cs + step * ds for cs, ds in zip(c, dc, strict=True)), occ
        )
        minus = orbital_features(
            jets, tuple(cs - step * ds for cs, ds in zip(c, dc, strict=True)), occ
        )
        for key in KEYS:
            fd = (plus[key] - minus[key]) / (2 * step)
            np.testing.assert_allclose(
                fd, expected[key], atol=20 * step**2 + 1e-9, rtol=1e-9
            )
        errors.append(
            np.max(
                abs((plus["sigma"] - minus["sigma"]) / (2 * step) - expected["sigma"])
            )
        )
    assert errors[-1] < errors[0] / 100
    assert np.max(abs(expected["sigma"] - linear["sigma"])) > 1e-3


def test_ao_node_keeps_tau_and_offdiagonal_gradient() -> None:
    # At a p-like AO node phi_0=0 but d_x phi_0=1. A second AO preserves
    # cross terms in grad rho, while the node alone contributes nonzero tau.
    jets = np.zeros((4, 1, 2))
    jets[0, 0, 1], jets[1, 0, 0] = 1, 1
    c = (np.ones((2, 1)), np.empty((2, 0)))
    occ = (np.array([0.6]), np.empty(0))
    source = DensitySource(
        np.array([np.full((2, 2), 0.6), np.zeros((2, 2))]), basis_identity=BASIS
    )
    checked = source.with_orbitals(c, occ, stamp=source.stamp)
    value = checked.features(jets, stamp=source.stamp, route="orbitals")
    np.testing.assert_allclose(
        [value["rho"][0, 0], value["gradient"][0, 0, 0], value["tau"][0, 0]],
        [0.6, 1.2, 0.3],
        atol=1e-15,
        rtol=0,
    )
    assert_features(value, density_features(jets, source.density))


@pytest.mark.parametrize(
    "field,value",
    [
        ("basis_identity", "0" * 64),
        ("basis_generation", 1),
        ("density_generation", 8),
        ("density_identity", "0" * 64),
        ("layout", "total"),
        ("role", "response"),
    ],
)
def test_stale_candidate_falls_back_but_stale_source_cannot_replay(
    field: typing.Any, value: typing.Any
) -> None:
    source, c, occ, jets = sample()
    stale = replace(source.stamp, **{field: value})
    rejected = source.with_orbitals(c, occ, stamp=stale)
    assert rejected.fallback_reason == "stale_orbitals"
    assert_features(
        rejected.features(jets, stamp=source.stamp),
        density_features(jets, source.density),
    )
    with pytest.raises(ValueError, match="stale density source"):
        source.features(jets, stamp=stale)


@pytest.mark.parametrize(
    "bad", ["changed", "negative", "complex", "nan", "shape", "overflow"]
)
def test_invalid_candidates_clear_old_factors_and_preserve_original_d(
    bad: typing.Any,
) -> None:
    source, c, occ, jets = sample()
    checked = source.with_orbitals(c, occ, stamp=source.stamp)
    c, occ = list(c), list(occ)
    if bad == "changed":
        c[0] = c[0] * 1.01
    elif bad == "negative":
        occ[0] = -occ[0]
    elif bad == "complex":
        c[0] = c[0].astype(complex) + 1j
    elif bad == "nan":
        c[0][0, 0] = np.nan
    elif bad == "shape":
        c[0] = c[0][:-1]
    else:
        c[0] = np.full_like(c[0], 1e308)
    rejected = checked.with_orbitals(c, occ, stamp=source.stamp)
    assert rejected.source_kind == "density_matrix"
    assert rejected.coefficients is None and rejected.factor_identity is None
    assert checked.source_kind == "orbitals"
    assert_features(
        rejected.features(jets, stamp=source.stamp),
        density_features(jets, source.density),
    )
    with pytest.raises(ValueError, match="orbital route unavailable"):
        rejected.features(jets, stamp=source.stamp, route="orbitals")


def test_indefinite_and_response_densities_are_not_repaired() -> None:
    source, c, occ, jets = sample()
    indefinite = DensitySource(np.diag([-0.2, 0.1, 0.3, 0, 0]), basis_identity=BASIS)
    rejected = indefinite.with_orbitals(c, occ, stamp=indefinite.stamp)
    assert rejected.fallback_reason == "incompatible_orbitals"
    assert_features(
        rejected.features(jets, stamp=indefinite.stamp),
        density_features(jets, indefinite.density),
    )
    response = DensitySource(source.density, basis_identity=BASIS, role="response")
    rejected = response.with_orbitals(c, occ, stamp=response.stamp)
    assert rejected.fallback_reason == "response_density"
    assert_features(
        rejected.features(jets, stamp=response.stamp),
        density_features(jets, source.density),
    )


def test_zero_and_tiny_occupations_are_weighted_before_collocation() -> None:
    c = (np.full((2, 1), 1e200), np.full((2, 1), 1e200))
    occ = (np.array([0.0]), np.array([1e-300]))
    density = np.stack((np.zeros((2, 2)), np.full((2, 2), 1e100)))
    source = DensitySource(density, basis_identity=BASIS)
    checked = source.with_orbitals(c, occ, stamp=source.stamp)
    jets = np.full((4, 1, 2), 1e-50)
    with np.errstate(over="raise", invalid="raise"):
        actual = checked.features(jets, stamp=source.stamp, route="orbitals")
    assert_features(actual, density_features(jets, density))


def test_content_identity_and_ownership_and_replay_without_reconstruction(
    monkeypatch: typing.Any,
) -> None:
    source, c, occ, jets = sample()
    raw = source.density.copy()
    rebuilt = DensitySource(raw, basis_identity=BASIS, density_generation=7)
    assert rebuilt.stamp == source.stamp
    checked = rebuilt.with_orbitals(c, occ, stamp=rebuilt.stamp, validation_rows=1)
    before = checked.features(jets, stamp=source.stamp)
    c[0][:] = 0
    occ[0][:] = 0
    raw[:] = 0
    with pytest.raises(FrozenInstanceError):
        checked.stamp = source.stamp
    for array in (checked.density, *checked.coefficients, *checked.occupations):
        with pytest.raises(ValueError):
            array.setflags(write=True)

    # External compatibility scans must not reappear in orbital tile replay.
    def no_reconstruction(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        raise AssertionError("density compatibility was rechecked on replay")

    with monkeypatch.context() as m:
        m.setattr(np, "allclose", no_reconstruction)
        after = checked.features(jets, stamp=source.stamp, route="orbitals")
    assert_features(after, before)
    changed = DensitySource(np.eye(5), basis_identity=BASIS, density_generation=7)
    assert changed.stamp != source.stamp


@pytest.mark.parametrize("ids", [[0, 0], [-1], [5], [1.5], [[1]]])
def test_invalid_local_maps_fail(ids: typing.Any) -> None:
    source, _, _, jets = sample()
    with pytest.raises(ValueError, match="local AO IDs"):
        source.features(jets, stamp=source.stamp, ao_ids=ids)


@pytest.mark.parametrize("ingredients", [(), ("rho", "rho"), ("unknown",)])
def test_invalid_requests_fail_on_both_routes(ingredients: typing.Any) -> None:
    source, c, occ, jets = sample()
    for function, args in (
        (density_features, (source.density,)),
        (orbital_features, (c, occ)),
    ):
        with pytest.raises(ValueError, match="ingredient"):
            function(jets, *args, ingredients=ingredients)
    with pytest.raises(ValueError, match="derivative domain"):
        orbital_features(jets[:1], c, occ)
