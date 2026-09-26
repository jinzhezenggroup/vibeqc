"""Native CPU Hessian chain; PySCF is an external optional oracle only."""

import os
import subprocess
import sys
import typing
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tools.vibeqc_hessian.analytic import (
    analytic_hessian,
    build_reference,
    cphf_relaxation,
)
from tools.vibeqc_hessian.native import NativeRHFState
from tools.vibeqc_posthf.sources import NativeSource
from tools.vibeqc_validation.hessian_fixtures import fixture_inputs, oracle_system


@pytest.fixture(scope="module", params=("h2", "water"))
def case(request: typing.Any) -> typing.Any:
    with NativeSource(**fixture_inputs(request.param)) as source:
        state = NativeRHFState.from_source(source)
        # One complete native Hessian per immutable state for all checks.
        yield request.param, state, analytic_hessian(state)


def test_first_order_sources_match_independent_native_derivatives(
    case: typing.Any,
) -> None:
    _, state, _ = case
    h1, s1 = state.first_order_inputs
    reference = state.source.integral_derivatives()  # oracle, never the live path
    expected = reference["hcore"] + np.einsum(
        "apqrs,rs->apq", reference["eri"], state.P0
    )
    expected -= 0.5 * np.einsum("aprqs,rs->apq", reference["eri"], state.P0)
    np.testing.assert_allclose(
        h1.reshape(expected.shape), expected, atol=2e-10, rtol=2e-10
    )
    np.testing.assert_allclose(
        s1.reshape(reference["overlap"].shape), reference["overlap"], atol=2e-11
    )
    assert not h1.flags.writeable and not s1.flags.writeable


def test_reference_is_the_native_converged_state(
    case: typing.Any, monkeypatch: typing.Any
) -> None:
    _, state, _ = case

    def unexpected_scf(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        raise AssertionError("Hessian must not rerun SCF")

    monkeypatch.setattr(state.source, "rhf_density", unexpected_scf)
    assert build_reference(state) is state.reference
    assert state.reference.hf_backend == "native-cpu"
    assert state.reference.scf_residual < 1e-8
    raw = cphf_relaxation(state)
    np.testing.assert_allclose(raw, raw.transpose(1, 0, 3, 2), atol=2e-10, rtol=0)


def test_total_matches_independent_pyscf_hessian(case: typing.Any) -> None:
    pytest.importorskip("pyscf")
    from pyscf import scf

    _, state, comp = case
    oracle = oracle_system(state.source)
    mf = scf.RHF(oracle.mol).run(conv_tol=1e-13, conv_tol_grad=1e-10)
    np.testing.assert_allclose(comp["total"], mf.Hessian().kernel(), atol=5e-7, rtol=0)


def test_components_match_independent_finite_difference_oracle(
    case: typing.Any,
) -> None:
    pytest.importorskip("pyscf")
    from tools.vibeqc_hessian.reference import hessian_components

    _, state, comp = case
    oracle = oracle_system(state.source)
    oracle.derive()
    expected = hessian_components(oracle)
    for key in ("core", "pulay", "two_electron", "relaxation", "nuclear"):
        np.testing.assert_allclose(comp[key], expected[key], atol=5e-5, rtol=0)


def test_raw_total_invariants(case: typing.Any) -> None:
    _, _, comp = case
    raw = comp["total"]
    assert np.isfinite(raw).all()
    np.testing.assert_allclose(raw, raw.transpose(1, 0, 3, 2), atol=2e-9, rtol=0)
    # Sum over atoms separately for each translation axis, not all coordinates.
    np.testing.assert_allclose(raw.sum(axis=0), 0, atol=2e-8, rtol=0)
    np.testing.assert_allclose(raw.sum(axis=1), 0, atol=2e-8, rtol=0)
    assert np.max(np.abs(comp["relaxation"])) > 1e-4


@pytest.mark.parametrize(
    "relax",
    [
        1.0,
        np.zeros((1, 1, 3, 3)),
        np.full((2, 2, 3, 3), np.nan),
        np.full((2, 2, 3, 3), np.inf),
        np.ones((2, 2, 3, 3), dtype=complex),
        np.full((2, 2, 3, 3), "x", dtype=object),
        np.full((2, 2, 3, 3), "x"),
        np.full((2, 2, 3, 3), np.datetime64("2026-09-19")),
    ],
)
def test_invalid_relaxation_rejected_before_provider_work(
    relax: typing.Any, monkeypatch: typing.Any
) -> None:
    with NativeSource(**fixture_inputs("h2")) as source:
        state = NativeRHFState.from_source(source)

        def unexpected(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            raise AssertionError("invalid input reached the derivative provider")

        monkeypatch.setattr(
            "tools.vibeqc_hessian.analytic.provider_components", unexpected
        )
        with pytest.raises(ValueError, match="relaxation"):
            analytic_hessian(state, relax=relax)


def test_source_lifetime_and_reference_identity_fail_closed() -> None:
    with NativeSource(**fixture_inputs("h2")) as source:
        state = NativeRHFState.from_source(source)
        with pytest.raises(ValueError, match="geometry"):
            replace(
                state, reference=replace(state.reference, geometry_hash="different")
            )
        with pytest.raises(ValueError, match="native CPU"):
            replace(
                state, reference=replace(state.reference, hf_backend="cpu-reference")
            )
    with pytest.raises(RuntimeError, match="closed"):
        analytic_hessian(state)
    with pytest.raises(TypeError, match="NativeRHFState"):
        analytic_hessian(object())


def test_native_hessian_does_not_import_or_call_an_oracle() -> None:
    # Fresh process is essential: collection by other tests may import PySCF.
    code = r"""
import importlib.abc
import sys
class BlockOracles(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pyscf' or fullname.startswith('pyscf.') or fullname == 'tools.vibeqc_hessian.reference':
            raise AssertionError('native Hessian imported an oracle: ' + fullname)
sys.meta_path.insert(0, BlockOracles())
import numpy as np
from tools.vibeqc_posthf.sources import NativeSource
from tools.vibeqc_response.backends import DenseAOResponseBackend
from tools.vibeqc_hessian import NativeRHFState, analytic_hessian

def forbidden(*args, **kwargs):
    raise AssertionError('native Hessian called a dense derivative/response oracle')
NativeSource.integral_derivatives = forbidden
DenseAOResponseBackend.__init__ = forbidden
with NativeSource([(1, [0, 0, 0]), (1, [0, 0, 1.4])]) as source:
    state = NativeRHFState.from_source(source)
    result = analytic_hessian(state)['total']
    assert result.shape == (2, 2, 3, 3) and np.isfinite(result).all()
    assert not any(k == 'pyscf' or k.startswith('pyscf.') for k in sys.modules)
    print('native SCF -> generated first/second derivatives -> native response: no oracle')
"""
    root = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(root / "python"), str(root))),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("name", ["h2", "water", "water_sdf"])
def test_reduced_response_matches_full_space(name: typing.Any) -> None:
    pytest.importorskip("pyscf")
    from tools.vibeqc_hessian.reference import (
        _first_order_mo1_e1,
        _first_order_mo1_e1_vir_only,
        h1ao,
        hessian_total,
    )

    with NativeSource(**fixture_inputs(name)) as source:
        oracle = oracle_system(source)
    oracle.derive()
    full = _first_order_mo1_e1(oracle, h1ao(oracle))
    reduced = _first_order_mo1_e1_vir_only(oracle, h1ao(oracle))
    for actual, expected in zip(reduced, full, strict=True):
        np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=1e-9)
    np.testing.assert_allclose(
        hessian_total(oracle, mo1e1_fn=_first_order_mo1_e1_vir_only),
        hessian_total(oracle),
        atol=2e-9,
        rtol=1e-9,
    )


@pytest.mark.skipif(
    os.environ.get("VIBEQC_HESSIAN_SLOW") != "1",
    reason="explicit 12-AO d-shell generated Hessian qualification",
)
def test_native_d_shell_hessian_matches_external_oracle() -> None:
    pytest.importorskip("pyscf")
    from pyscf import scf

    with NativeSource(**fixture_inputs("water_sdf")) as source:
        state = NativeRHFState.from_source(source)
        oracle = oracle_system(source)
        mf = scf.RHF(oracle.mol).run(conv_tol=1e-13, conv_tol_grad=1e-10)
        np.testing.assert_allclose(
            analytic_hessian(state)["total"], mf.Hessian().kernel(), atol=5e-5, rtol=0
        )


def test_three_step_directional_differences_of_native_forces(
    case: typing.Any,
) -> None:
    from vibeqc import Calculator

    _, state, comp = case
    calculator = Calculator(
        method="rhf",
        basis=state.source.shells,
        device="cpu",
        energy_tolerance=1e-13,
        density_tolerance=1e-12,
        max_iterations=400,
    )
    direction = np.random.default_rng(449).normal(size=(state.nat, 3))
    direction /= np.linalg.norm(direction)
    expected = np.einsum("abxy,by->ax", comp["total"], direction)
    errors = []
    for step in (3e-3, 1e-3, 3e-4):
        gradients = []
        for sign in (1, -1):
            xyz = state.coords + sign * step * direction
            result = calculator.singlepoint(
                [
                    (atom.atomic_number, position)
                    for atom, position in zip(state.source.atoms, xyz, strict=True)
                ],
                charge=state.source.charge,
                multiplicity=1,
            )
            assert result.forces is not None
            gradients.append(-np.asarray(result.forces))
        numeric = (gradients[0] - gradients[1]) / (2 * step)
        errors.append(float(np.max(np.abs(numeric - expected))))
    assert errors[-1] < 2e-5, errors
    assert errors[-1] < max(errors[0] * 0.2, 1e-7), errors


def test_large_domain_is_rejected_before_native_scf(
    monkeypatch: typing.Any,
) -> None:
    from vibeqc import Primitive, Shell

    shells = tuple(
        Shell(i % 2, 0, (Primitive(0.2 + i * 0.13, 1.0),)) for i in range(13)
    )
    with NativeSource([(1, (0, 0, 0)), (1, (0, 0, 1.4))], basis=shells) as source:

        def unexpected(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            raise AssertionError("out-of-domain state reached SCF")

        monkeypatch.setattr(source, "rhf_density", unexpected)
        with pytest.raises(ValueError, match="12 AOs"):
            NativeRHFState.from_source(source)
