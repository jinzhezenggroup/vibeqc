"""Small independent-energy/gradient gates for the migrated D3 baseline (#492)."""

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.method import (
    METHOD_CATALOG,
    D3Spec,
    DispersionCorrectionPrimitive,
    MethodIR,
    resolve_method,
)

from tools.vibeqc_d3.reference import (
    NativeD3,
    build_reference,
    gfn1_compatibility,
    make_spec,
)

_GOLDENS = json.loads(
    (Path(__file__).parents[1] / "data/d3_bj_reference.json").read_text()
)["fixtures"]


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    path = os.environ.get("VIBEQC_D3_LIBRARY")
    if path is None:
        path = build_reference(tmp_path_factory.mktemp("d3-native"))
    engine = NativeD3(path)
    assert engine.backend == os.environ.get("VIBEQC_D3_EXPECT_BACKEND", "cpu")
    return engine


@pytest.mark.parametrize("case", _GOLDENS, ids=lambda case: case["name"])
def test_independent_simple_dftd3_golden(native, case):
    spec = make_spec(**case["parameters"])
    energy, gradient = native.evaluate(spec, case["numbers"], case["positions"])
    assert energy == pytest.approx(case["energy"], abs=2e-14, rel=0)
    np.testing.assert_allclose(gradient, case["gradient"], atol=3e-13, rtol=0)


@pytest.mark.parametrize("step", [2e-4, 7e-5, 2e-5])
def test_complete_cn_response_finite_differences(native, step):
    case = _GOLDENS[1]
    spec = make_spec(**case["parameters"])
    x = np.array(case["positions"])
    _, gradient = native.evaluate(spec, case["numbers"], x)
    numerical = np.zeros_like(x)
    for axis in np.ndindex(x.shape):
        plus, minus = x.copy(), x.copy()
        plus[axis] += step
        minus[axis] -= step
        ep, _ = native.evaluate(spec, case["numbers"], plus)
        em, _ = native.evaluate(spec, case["numbers"], minus)
        numerical[axis] = (ep - em) / (2 * step)
    np.testing.assert_allclose(gradient, numerical, atol=1e-9, rtol=0)


def test_invariance_and_atom_pair_transposition(native):
    case = _GOLDENS[1]
    z = np.array(case["numbers"])
    x = np.array(case["positions"])
    spec = make_spec(**case["parameters"])
    energy, gradient = native.evaluate(spec, z, x)
    perm = np.array([3, 1, 0, 2])
    rotation = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    e, g = native.evaluate(
        spec, z[perm], x[perm] @ rotation + np.array([3.0, 2.0, -1.0])
    )
    assert e == pytest.approx(energy, abs=2e-14, rel=0)
    np.testing.assert_allclose(g, gradient[perm] @ rotation, atol=3e-13, rtol=0)
    np.testing.assert_allclose(gradient.sum(axis=0), 0, atol=1e-14)


def test_pair_switch_and_energy_force_sign(native):
    spec = gfn1_compatibility()
    z = [6, 8]
    x = np.array([[0.0, 0.0, 0.0], [49.975, 0.0, 0.0]])
    e, g = native.evaluate(spec, z, x)
    step = 1e-5
    xp = x.copy()
    xm = x.copy()
    xp[1, 0] += step
    xm[1, 0] -= step
    ep, _ = native.evaluate(spec, z, xp)
    em, _ = native.evaluate(spec, z, xm)
    assert g[1, 0] == pytest.approx((ep - em) / (2 * step), abs=1e-12, rel=0)
    assert e < 0 and g[1, 0] > 0  # Force would be negative, toward the partner.
    x[1, 0] = 50.0
    e, g = native.evaluate(spec, z, x)
    assert e == 0
    np.testing.assert_array_equal(g, 0.0)
    unscreened = replace(spec, pair_cutoff=None, pair_switch_width=0.0, cn_cutoff=None)
    assert native.evaluate(unscreened, z, x)[0] < 0


def test_scaling_parameters_are_not_gfn1_constants(native):
    case = _GOLDENS[1]
    spec = make_spec(**case["parameters"])
    e, g = native.evaluate(spec, case["numbers"], case["positions"])
    e6, g6 = native.evaluate(replace(spec, s8=0.0), case["numbers"], case["positions"])
    e8, g8 = native.evaluate(replace(spec, s6=0.0), case["numbers"], case["positions"])
    assert e == pytest.approx(e6 + e8, abs=1e-15, rel=0)
    np.testing.assert_allclose(g, g6 + g8, atol=1e-14, rtol=0)
    ez, gz = native.evaluate(
        replace(spec, s6=0.0, s8=0.0), case["numbers"], case["positions"]
    )
    assert ez == 0
    np.testing.assert_array_equal(gz, 0.0)


@pytest.mark.parametrize(
    "change",
    [
        {"s9": 1.0},
        {"damping": "zero"},
        {"a1": float("nan")},
        {"s8": float("inf")},
        {"s6": -1.0},
        {"a2": True},
        {"a1": 0.0, "a2": 0.0},
        {"cn_cutoff": 0.0},
        {"pair_cutoff": 1.0, "pair_switch_width": 1.0},
        {"pair_switch_width": 0.1},
        {"table_sha256": "unknown"},
        {"radii_sha256": "a" * 63},
    ],
)
def test_unsupported_parameters_fail_closed(change):
    spec = make_spec(s6=1.0, s8=1.0, a1=0.4, a2=4.0)
    with pytest.raises((ValueError, TypeError)):
        replace(spec, **change)


def test_guardrails_and_unchanged_inputs(native):
    spec = gfn1_compatibility()
    x = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    original = x.copy()
    with pytest.raises(MemoryError):
        native.evaluate(spec, [6, 8], x, memory_budget_bytes=1)
    with pytest.raises(ValueError):
        native.evaluate(spec, [6, 8], x * np.nan)
    with pytest.raises(ValueError):
        native.evaluate(spec, [6.0, 8.0], x)
    with pytest.raises(ValueError):
        native.evaluate(spec, [6, 87], x)
    with pytest.raises(ValueError):
        native.evaluate(replace(spec, table_sha256="0" * 64), [6, 8], x)
    with pytest.raises(RuntimeError):
        native.evaluate(spec, [6, 8], np.zeros((2, 3)))
    native.evaluate(spec, [6, 8], x)
    np.testing.assert_array_equal(x, original)
    e, g = native.evaluate(spec, [6], [[0, 0, 0]])
    assert e == 0
    np.testing.assert_array_equal(g, 0.0)


def test_canonical_methodir_composition_and_identity():
    spec = gfn1_compatibility()
    assert D3Spec(**spec.to_payload()) == spec
    plain = resolve_method("PBE")
    assert "dispersion" not in METHOD_CATALOG["PBE"].to_payload()
    mixed = resolve_method(replace(METHOD_CATALOG["PBE"], dispersion=spec))
    assert mixed.identity != plain.identity
    assert mixed.primitives[-1] == DispersionCorrectionPrimitive(spec)
    assert mixed.requirements["ingredients"] == plain.requirements["ingredients"]
    assert mixed.requirements["operators"] == ("semilocal-xc", "geometry-d3-bj")
    alias = replace(mixed, identifier="descriptive-alias")
    assert (
        alias.identity == mixed.identity
        and alias.manifest_identity != mixed.manifest_identity
    )
    other = resolve_method(
        replace(METHOD_CATALOG["PBE"], dispersion=replace(spec, s8=1.0))
    )
    assert other.identity != mixed.identity
    with pytest.raises(ValueError):
        MethodIR("duplicate", mixed.spin, mixed.primitives + (mixed.primitives[-1],))
    hybrid = resolve_method(replace(METHOD_CATALOG["PBE0"], dispersion=spec))
    assert len(hybrid.primitives) == 3


@pytest.mark.parametrize(
    "method,spin", [("pbe-rks", "unpolarized"), ("pbe-uks", "polarized")]
)
def test_native_ks_cannot_silently_omit_correction(monkeypatch, method, spin):
    from vibeqc import ks

    graph = resolve_method(
        replace(METHOD_CATALOG["PBE"], dispersion=gfn1_compatibility()), spin=spin
    )
    monkeypatch.setattr(ks, "resolve_method", lambda *args, **kwargs: graph)
    with pytest.raises(NotImplementedError, match="exactly one supported semilocal"):
        ks.resolve_ks_method(method)
