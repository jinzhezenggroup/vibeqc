"""Real native LDA/PBE RKS nuclear response for the DFT Hessian path."""

import typing
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from vibeqc import Calculator, GridSpec, KsOptions, Primitive, Shell
from vibeqc._dft_gradient import _native_ao_atoms
from vibeqc._stationary_cpu import complete_rks_gradient_diagnostic
from vibeqc_compiler.dft import NativeAO
from vibeqc_compiler.xc.contractions import ExternalPointContraction
from vibeqc_compiler.xc.grid_response import partition_response

from tools.vibeqc_hessian import (
    directional_rks_response,
    directional_rks_responses,
    native_rks_xc_hvp_components,
    rks_hessian,
    rks_hvp,
    rks_hvp_many,
)
from tools.vibeqc_response import GMRESOptions, NativeRKSResponse

H2 = [("H", (0.0, 0.0, -0.72)), ("H", (0.08, -0.03, 0.71))]
LARGE_HE = [("He", (0.13, -0.21, 0.17))]
LARGE_HE_BASIS = (
    Shell(0, 0, (Primitive(1.5, 1.0),)),
    Shell(0, 2, (Primitive(0.8, 1.0),)),
    Shell(0, 2, (Primitive(0.35, 1.0),)),
)
GRID = GridSpec(radial_points=10, angular_polar=4, angular_azimuth=8)


def _calculator(method: str) -> Calculator:
    return Calculator(
        method=method,
        device="cpu",
        ks_options=KsOptions(grid=GRID),
        max_iterations=200,
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
    )


def _moved(atoms: typing.Any, direction: np.ndarray, scale: float) -> list:
    return [
        (symbol, np.asarray(position) + scale * delta)
        for (symbol, position), delta in zip(atoms, direction, strict=True)
    ]


def _native_xc_gradient(operator: NativeRKSResponse) -> dict[str, np.ndarray]:
    """Re-evaluate only the native semilocal XC first-gradient sources."""
    operator.validate_current()
    state = operator.state
    source = state._source
    basis = operator.xc_kernel.basis
    spec = operator.xc_kernel.spec
    grid = state.grid
    points = np.asarray(grid.points)
    owners = np.asarray(grid.owners, dtype=np.int64)
    contraction = ExternalPointContraction(spec, "geometry")
    jets = basis.evaluate(points, contraction.contract.ao_order)
    features = contraction.features(jets, state.density[0])
    zero_gradient = np.zeros((2, len(points), 3))
    values = source.evaluate_xc_points(
        spec,
        features["rho"],
        features.get("gradient", zero_gradient),
    )
    partials = contraction.geometry_from_cartesian_coefficients(
        jets,
        state.density[0],
        grid.weights,
        values["energy"],
        values["rho"],
        values["gradient"] if "sigma" in spec.ingredients else None,
        ao_atoms=_native_ao_atoms(basis),
        natom=basis.natom,
    )
    result = {
        "xc_ao": np.array(partials.centers),
        "xc_grid": np.zeros((basis.natom, 3)),
        "xc_weight": np.zeros((basis.natom, 3)),
    }
    np.add.at(result["xc_grid"], owners, partials.points)

    centers = np.asarray([atom.position for atom in basis.atoms], dtype=np.float64)
    atomic_weights = np.asarray(source.atomic_weights)
    grid_spec = source.grid_spec
    assert grid_spec is not None
    selected = (np.arange(len(points)), owners)
    for atom in range(basis.natom):
        for axis in range(3):
            motion = np.zeros((basis.natom, 3))
            motion[atom, axis] = 1.0
            response = partition_response(
                points,
                centers,
                point_motion=motion[owners],
                center_motion=motion,
                iterations=grid_spec.partition_iterations,
                coincident_tolerance=grid_spec.coincident_tolerance,
            )
            result["xc_weight"][atom, axis] = np.dot(
                partials.weights,
                atomic_weights * response.directional[selected],
            )
    return result


@pytest.fixture(params=("lda-rks", "pbe-rks"), scope="module")
def case(request: typing.Any) -> typing.Iterator[typing.Any]:
    with (
        _calculator(request.param).prepare_batch([H2]) as batch,
        NativeAO(H2) as basis,
    ):
        batch.execute(strict=True)
        with NativeRKSResponse.from_native(batch, basis, tile_points=257) as operator:
            direction = np.array(
                [[0.17, -0.09, 0.31], [-0.13, 0.07, -0.26]], dtype=np.float64
            )
            direction /= np.linalg.norm(direction)
            result = directional_rks_response(
                operator,
                direction,
                solver_options=GMRESOptions(atol=1e-12, rtol=1e-11),
            )
            yield request.param, operator, direction, result


def test_real_rks_geometry_direction_solves_shared_cpks(case: typing.Any) -> None:
    _, _operator, direction, result = case
    assert result.response.solve_result.converged
    assert result.response.solve_result.residual_norm < 1e-9
    np.testing.assert_array_equal(result.direction, direction)
    np.testing.assert_allclose(
        result.frozen_fock_derivative,
        result.integral_frozen_fock_derivative + result.xc_frozen_fock_derivative,
        atol=2e-14,
        rtol=0,
    )
    assert np.max(np.abs(result.xc_frozen_fock_derivative)) > 1e-8
    assert result.diagnostics["nuclear_response_solves"] == 1
    assert result.diagnostics["response_operator"] == "shared-native-rks-cpks"
    for value in (
        result.direction,
        result.integral_frozen_fock_derivative,
        result.xc_frozen_fock_derivative,
        result.frozen_fock_derivative,
        result.overlap_derivative,
        result.response.density_derivative,
        result.response.energy_weighted_density_derivative,
    ):
        assert not value.flags.writeable


def test_rks_nuclear_response_multi_rhs_matches_single(
    case: typing.Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The RKS response layer must use one shared solve_many call for all RHSs."""
    import tools.vibeqc_hessian.rks_directional as rks_directional_module

    _, operator, direction, single = case
    other = np.roll(direction.reshape(-1), 1).reshape(direction.shape)
    other /= np.linalg.norm(other)

    calls = []
    original = rks_directional_module.solve_stationary_nuclear_perturbations

    def counted(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        rks_directional_module, "solve_stationary_nuclear_perturbations", counted
    )
    result = directional_rks_responses(
        operator,
        np.stack((direction, other)),
        strategy="blocked",
        solver_options=GMRESOptions(atol=1e-12, rtol=1e-11),
    )
    assert len(calls) == 1
    assert result.solve_result.converged
    assert result.diagnostics["multi_rhs_calls"] == 1
    assert result.diagnostics["nrhs"] == 2
    assert result.diagnostics["strategy"] == "blocked"
    np.testing.assert_allclose(
        result.responses[0].response.density_derivative,
        single.response.density_derivative,
        atol=2e-10,
        rtol=2e-9,
    )
    np.testing.assert_allclose(
        result.responses[0].response.energy_weighted_density_derivative,
        single.response.energy_weighted_density_derivative,
        atol=2e-10,
        rtol=2e-9,
    )
    for response, expected in zip(result.responses, (direction, other), strict=True):
        assert response.response.solve_result.converged
        np.testing.assert_array_equal(response.direction, expected)


def test_rks_nuclear_response_matches_reconverged_density_and_weighted_density(
    case: typing.Any,
) -> None:
    method, _, direction, result = case
    expected = (
        result.response.density_derivative,
        result.response.energy_weighted_density_derivative,
    )
    errors = []
    for step in (2e-3, 7e-4, 2e-4):
        displaced = []
        for sign in (1, -1):
            atoms = _moved(H2, direction, sign * step)
            with (
                _calculator(method).prepare_batch([atoms]) as batch,
                NativeAO(atoms) as basis,
            ):
                batch.execute(strict=True)
                with NativeRKSResponse.from_native(batch, basis) as current:
                    displaced.append(
                        (
                            np.array(current.state.density[0]),
                            np.array(current.state.weighted_density[0]),
                        )
                    )
        numeric = tuple(
            (plus - minus) / (2 * step)
            for plus, minus in zip(displaced[0], displaced[1], strict=True)
        )
        errors.append(
            [
                float(np.max(np.abs(actual - target)))
                for actual, target in zip(numeric, expected, strict=True)
            ]
        )
    errors = np.asarray(errors)
    assert np.all(errors[-1] < 4e-5), errors
    assert np.all(errors[-1] < np.maximum(0.3 * errors[0], 8e-7)), errors


def test_native_rks_xc_hvp_matches_reconverged_xc_gradient(
    case: typing.Any,
) -> None:
    method, operator, direction, directional = case
    actual = native_rks_xc_hvp_components(operator, directional)
    assert actual.diagnostics["additional_response_solves"] == 0
    assert actual.diagnostics["source_names"] == ("xc_ao", "xc_grid", "xc_weight")
    np.testing.assert_allclose(
        actual.total,
        actual.xc_ao + actual.xc_grid + actual.xc_weight,
        atol=0,
        rtol=0,
    )
    assert np.max(np.abs(actual.total)) > 1e-7
    for value in (actual.xc_ao, actual.xc_grid, actual.xc_weight, actual.total):
        assert not value.flags.writeable

    errors = []
    for step in (1.5e-3, 5e-4, 1.7e-4):
        displaced = []
        for sign in (1, -1):
            atoms = _moved(H2, direction, sign * step)
            with (
                _calculator(method).prepare_batch([atoms]) as batch,
                NativeAO(atoms) as basis,
            ):
                batch.execute(strict=True)
                with NativeRKSResponse.from_native(batch, basis) as current:
                    displaced.append(_native_xc_gradient(current))
        numeric = {
            name: (displaced[0][name] - displaced[1][name]) / (2 * step)
            for name in ("xc_ao", "xc_grid", "xc_weight")
        }
        numeric["total"] = sum(numeric.values())
        errors.append(
            [
                float(np.max(np.abs(numeric["xc_ao"] - actual.xc_ao))),
                float(np.max(np.abs(numeric["xc_grid"] - actual.xc_grid))),
                float(np.max(np.abs(numeric["xc_weight"] - actual.xc_weight))),
                float(np.max(np.abs(numeric["total"] - actual.total))),
            ]
        )
    errors = np.asarray(errors)
    assert np.all(errors[-1] < 3e-4), errors
    assert np.all(errors[-1] < np.maximum(0.45 * errors[0], 2e-5)), errors

    with pytest.raises(ValueError, match="does not belong"):
        native_rks_xc_hvp_components(
            operator,
            replace(directional, identity="not-the-current-response"),
        )


def test_global_translation_has_zero_rks_nuclear_rhs(case: typing.Any) -> None:
    _, operator, _, _ = case
    direction = np.tile([0.13, -0.21, 0.31], (2, 1))
    result = directional_rks_response(
        operator,
        direction,
        solver_options=GMRESOptions(atol=1e-12, rtol=1e-11),
    )
    np.testing.assert_allclose(result.overlap_derivative, 0, atol=3e-11, rtol=0)
    np.testing.assert_allclose(result.frozen_fock_derivative, 0, atol=3e-9, rtol=0)
    np.testing.assert_allclose(result.response.density_derivative, 0, atol=3e-8, rtol=0)
    np.testing.assert_allclose(
        result.response.energy_weighted_density_derivative, 0, atol=3e-8, rtol=0
    )


def test_complete_rks_hvp_matches_reconverged_analytic_gradient(
    case: typing.Any, tmp_path: typing.Any
) -> None:
    """Gate the first complete LDA/PBE HVP against displaced analytic gradients."""
    method, operator, direction, _ = case
    result = rks_hvp(
        operator,
        direction,
        cache=tmp_path / "hvp",
        solver_options=GMRESOptions(atol=1e-12, rtol=1e-11),
    )
    assert result.directional_response.response.solve_result.converged
    assert result.diagnostics["nuclear_response_solves"] == 1
    assert result.diagnostics["complete_source_coverage"]
    assert not result.diagnostics["full_molecular_hessian_allocated"]
    assert not result.diagnostics["full_ao_rank_four_weights"]
    assert tuple(result.components) == (
        "one_electron",
        "coulomb",
        "xc_ao",
        "xc_grid",
        "xc_weight",
        "overlap_pulay",
        "nuclear",
    )
    np.testing.assert_allclose(
        sum(result.components.values(), start=np.zeros_like(result.value)),
        result.value,
        atol=2e-13,
        rtol=0,
    )

    errors = []
    for step in (1.2e-3, 4e-4, 1.3e-4):
        gradients = []
        for sign in (1, -1):
            atoms = _moved(H2, direction, sign * step)
            with (
                _calculator(method).prepare_batch([atoms]) as batch,
                NativeAO(atoms) as basis,
            ):
                batch.execute(strict=True)
                with NativeRKSResponse.from_native(batch, basis) as current:
                    gradients.append(
                        np.array(
                            complete_rks_gradient_diagnostic(
                                current.state,
                                basis,
                                cache=tmp_path / "gradient",
                                execution="reference",
                            ).gradient,
                            copy=True,
                        )
                    )
        numeric = (gradients[0] - gradients[1]) / (2 * step)
        errors.append(float(np.max(np.abs(result.value - numeric))))
    assert errors[-1] < 4e-4, errors
    assert errors[-1] < max(0.35 * errors[0], 2e-5), errors


def test_complete_rks_hvp_many_reuses_one_multi_rhs_response(
    case: typing.Any, tmp_path: typing.Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Complete HVP blocks must not fall back to one CPKS solve per direction."""
    import tools.vibeqc_hessian.rks_directional as rks_directional_module

    method, operator, direction, _ = case
    if method != "lda-rks":
        pytest.skip("one semilocal method is sufficient for multi-RHS orchestration")

    other = np.roll(direction.reshape(-1), 2).reshape(direction.shape)
    other /= np.linalg.norm(other)
    calls = []
    original = rks_directional_module.solve_stationary_nuclear_perturbations

    def counted(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        rks_directional_module, "solve_stationary_nuclear_perturbations", counted
    )
    result = rks_hvp_many(
        operator,
        np.stack((direction, other)),
        cache=tmp_path / "hvp-many",
        strategy="blocked",
        solver_options=GMRESOptions(atol=1e-12, rtol=1e-11),
    )

    assert len(calls) == 1
    assert result.diagnostics["multi_rhs_calls"] == 1
    assert result.diagnostics["nrhs"] == 2
    assert result.diagnostics["strategy"] == "blocked"
    assert result.directional_responses.solve_result.converged
    assert not result.diagnostics["full_molecular_hessian_allocated"]
    assert not result.diagnostics["full_ao_rank_four_weights"]
    np.testing.assert_array_equal(result.directions, np.stack((direction, other)))
    np.testing.assert_allclose(
        result.values,
        np.stack([item.value for item in result.results]),
        atol=0,
        rtol=0,
    )
    for item in result.results:
        assert item.diagnostics["nuclear_response_solves"] == 0
        assert item.diagnostics["complete_source_coverage"]
        assert tuple(item.components) == (
            "one_electron",
            "coulomb",
            "xc_ao",
            "xc_grid",
            "xc_weight",
            "overlap_pulay",
            "nuclear",
        )
        np.testing.assert_allclose(
            sum(item.components.values(), start=np.zeros_like(item.value)),
            item.value,
            atol=2e-13,
            rtol=0,
        )


def test_rks_hessian_assembles_raw_columns_in_blocks(
    case: typing.Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full assembly must preserve raw HVP columns without symmetrization."""
    import tools.vibeqc_hessian.rks_molecular as rks_molecular_module

    _, operator, _, _ = case
    coordinates = 3 * operator.xc_kernel.basis.natom
    expected = np.arange(coordinates * coordinates, dtype=np.float64).reshape(
        coordinates, coordinates
    )
    calls = []

    def fake_many(
        _operator: typing.Any, directions: typing.Any, **kwargs: typing.Any
    ) -> typing.Any:
        vectors = np.asarray(directions).reshape(len(directions), coordinates)
        values = (vectors @ expected.T).reshape(len(directions), -1, 3)
        calls.append((np.array(directions, copy=True), kwargs))
        return SimpleNamespace(
            values=values,
            identity=f"block-{len(calls)}",
            diagnostics={"multi_rhs_calls": 1, "nrhs": len(directions)},
        )

    monkeypatch.setattr(rks_molecular_module, "rks_hvp_many", fake_many)
    result = rks_hessian(
        operator,
        block_size=2,
        strategy="blocked",
        output_budget_bytes=1 << 20,
    )

    np.testing.assert_array_equal(result.matrix, expected)
    assert len(calls) == (coordinates + 1) // 2
    assert result.diagnostics["block_size"] == 2
    assert result.diagnostics["block_count"] == len(calls)
    assert result.diagnostics["strategy"] == "blocked"
    assert not result.diagnostics["posthoc_symmetrization"]
    assert not result.diagnostics["public_calculator_endpoint"]
    assert not result.diagnostics["complete_resource_bound"]
    assert result.diagnostics["raw_symmetry_error"] == float(
        np.max(np.abs(expected - expected.T))
    )


def test_rks_hessian_output_budget_fails_before_hvp(
    case: typing.Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Insufficient full-output storage must fail before any HVP is evaluated."""
    import tools.vibeqc_hessian.rks_molecular as rks_molecular_module

    _, operator, _, _ = case
    coordinates = 3 * operator.xc_kernel.basis.natom
    output_bytes = coordinates * coordinates * np.dtype(np.float64).itemsize

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        raise AssertionError("HVP work started before the output-budget gate")

    monkeypatch.setattr(rks_molecular_module, "rks_hvp_many", forbidden)
    with pytest.raises(ValueError, match="output_budget_bytes"):
        rks_hessian(
            operator,
            block_size=2,
            output_budget_bytes=2 * output_bytes - 1,
        )

def test_rks_hvp_integral_budget_fails_before_response(
    case: typing.Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An undersized integral budget must reject work before CPKS starts."""
    import tools.vibeqc_hessian.rks_molecular as rks_molecular_module

    _, operator, direction, _ = case

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        raise AssertionError("CPKS started before the integral-budget gate")

    monkeypatch.setattr(rks_molecular_module, "directional_rks_response", forbidden)
    with pytest.raises(MemoryError, match="integral_budget_bytes"):
        rks_hvp(operator, direction, integral_budget_bytes=1)


def test_rks_hvp_admits_resource_bounded_domain_above_12_aos(
    tmp_path: typing.Any,
) -> None:
    """The former 12-AO admission limit is replaced by explicit resource gates."""
    calculator = Calculator(
        method="lda-rks",
        basis=LARGE_HE_BASIS,
        device="cpu",
        ks_options=KsOptions(grid=GRID),
        max_iterations=200,
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
    )
    direction = np.array([[0.31, -0.27, 0.19]])
    direction /= np.linalg.norm(direction)
    with (
        calculator.prepare_batch([LARGE_HE]) as batch,
        NativeAO(LARGE_HE, basis=LARGE_HE_BASIS) as basis,
    ):
        assert basis.nao == 13
        batch.execute(strict=True)
        with NativeRKSResponse.from_native(batch, basis, tile_points=257) as operator:
            result = rks_hvp(
                operator,
                direction,
                cache=tmp_path / "large-hvp",
                integral_budget_bytes=64 << 20,
                solver_options=GMRESOptions(atol=1e-12, rtol=1e-11),
            )

    assert result.diagnostics["complete_source_coverage"]
    assert result.diagnostics["integral_budget_bytes"] == 64 << 20
    assert (
        result.diagnostics["plan_weight_workspace_bound_bytes"]
        < result.diagnostics["integral_budget_bytes"]
    )
    assert np.isfinite(result.value).all()
    np.testing.assert_allclose(result.value, 0.0, atol=2e-7, rtol=0)
    for diagnostic in result.diagnostics["integral_providers"].values():
        assert diagnostic["budget_bytes"] == 64 << 20
        assert diagnostic["output_accumulator_bytes"] <= diagnostic["budget_bytes"]

