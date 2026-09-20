"""Binding contracts with an exact oracle, independent of native HF convergence."""

import typing
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vibeqc.torch import batched_energy, energy


class QuadraticCalculator:
    """Record backend entry and return E=sum(R**2), forces=-2*R exactly."""

    def __init__(self) -> None:
        self.calls = []

    def singlepoint(
        self,
        atoms: typing.Any,
        *,
        charge: typing.Any,
        multiplicity: typing.Any,
        properties: typing.Any = ("energy", "forces"),
    ) -> typing.Any:
        # The autograd binding explicitly requests analytic forces, including
        # when the public calculator's method would default to energy only.
        assert properties == ("energy", "forces")
        self.calls.append((tuple(a[0] for a in atoms), charge, multiplicity))
        xyz = np.asarray([a[1] for a in atoms], dtype=float)
        return SimpleNamespace(energy=float((xyz**2).sum()), forces=-2 * xyz)

    def batch_singlepoint(
        self,
        systems: typing.Any,
        *,
        charges: typing.Any,
        multiplicities: typing.Any,
        strict: typing.Any,
    ) -> typing.Any:
        assert strict
        items = tuple(
            self.singlepoint(atoms, charge=c, multiplicity=m)
            for atoms, c, m in zip(systems, charges, multiplicities, strict=True)
        )
        return SimpleNamespace(
            items=items, energies=np.array([r.energy for r in items])
        )


class QuadraticPreparedBatch:
    """Exercise the prepared callback boundary without requiring a native handle."""

    atomic_numbers = ((1, 1),)
    charges = (0,)
    multiplicities = (1,)

    def __init__(self, calculator: typing.Any) -> None:
        self.calculator = calculator

    def execute(self, coordinates: typing.Any, *, strict: typing.Any) -> typing.Any:
        return self.calculator.batch_singlepoint(
            [
                list(zip(numbers, xyz, strict=True))
                for numbers, xyz in zip(self.atomic_numbers, coordinates, strict=True)
            ],
            charges=self.charges,
            multiplicities=self.multiplicities,
            strict=strict,
        )


def evaluate(
    route: typing.Any,
    xyz: typing.Any,
    calculator: typing.Any,
    numbers: typing.Any = (1, 1),
    charge: typing.Any = 0,
    multiplicity: typing.Any = 1,
) -> typing.Any:
    """Apply the same molecular request to each public binding route."""
    if route == "single":
        return energy(
            xyz, numbers, calculator, charge=charge, multiplicity=multiplicity
        )
    prepared = QuadraticPreparedBatch(calculator) if route == "prepared" else None
    return batched_energy(
        [xyz],
        [numbers],
        calculator,
        charges=[charge],
        multiplicities=[multiplicity],
        prepared_batch=prepared,
    )[0]


@pytest.fixture(params=["single", "batch", "prepared"])
def route(request: typing.Any) -> typing.Any:
    return request.param


@pytest.fixture
def xyz() -> typing.Any:
    return torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )


@pytest.mark.parametrize("numbers", [(1,), (1, 1, 8)])
def test_atom_count_mismatch_never_reaches_backend(
    route: typing.Any, xyz: typing.Any, numbers: typing.Any
) -> None:
    calculator = QuadraticCalculator()
    with pytest.raises(ValueError, match="atomic_numbers.*coordinates"):
        evaluate(route, xyz, calculator, numbers=numbers)
    assert calculator.calls == []


@pytest.mark.parametrize("field", ["numbers", "charge", "multiplicity"])
@pytest.mark.parametrize(
    "value", [True, np.bool_(False), 1.9, np.float64(1.0), "1", 2**40]
)
def test_lossy_integer_inputs_never_reach_backend(
    route: typing.Any, xyz: typing.Any, field: typing.Any, value: typing.Any
) -> None:
    calculator = QuadraticCalculator()
    argument = (value, 1) if field == "numbers" else value
    with pytest.raises(ValueError, match="must be an integer"):
        evaluate(route, xyz, calculator, **{field: argument})
    assert calculator.calls == []


@pytest.mark.parametrize(
    "arguments",
    [
        {"numbers": (0, 1)},
        {"numbers": (119, 1)},
        {"charge": -(2**31) - 1},
        {"multiplicity": 0},
    ],
)
def test_native_integer_ranges_checked_before_backend(
    route: typing.Any, xyz: typing.Any, arguments: typing.Any
) -> None:
    calculator = QuadraticCalculator()
    with pytest.raises(ValueError, match="must be an integer"):
        evaluate(route, xyz, calculator, **arguments)
    assert calculator.calls == []


@pytest.mark.parametrize("integer", [int, np.int32, np.int64, np.uint64])
def test_exact_integers_keep_molecular_identity(
    route: typing.Any, xyz: typing.Any, integer: typing.Any
) -> None:
    calculator = QuadraticCalculator()
    value = evaluate(
        route,
        xyz,
        calculator,
        numbers=(integer(1), integer(1)),
        charge=integer(0),
        multiplicity=integer(1),
    )
    assert value.item() == 1.0
    assert calculator.calls == [((1, 1), 0, 1)]
    assert all(
        type(v) is int for v in (*calculator.calls[0][0], *calculator.calls[0][1:])
    )


@pytest.mark.parametrize("route", ["single", "batch"])
def test_signed_charge_is_preserved(route: typing.Any, xyz: typing.Any) -> None:
    calculator = QuadraticCalculator()
    evaluate(route, xyz, calculator, charge=np.int64(-1), multiplicity=np.int32(2))
    assert calculator.calls == [((1, 1), -1, 2)]


def test_invalid_input_does_not_construct_default_calculator(
    monkeypatch: typing.Any, route: typing.Any, xyz: typing.Any
) -> None:
    def unexpected_calculator(**kwargs: typing.Any) -> None:
        pytest.fail("invalid input constructed a native calculator")

    monkeypatch.setattr("vibeqc.torch.Calculator", unexpected_calculator)
    with pytest.raises(ValueError, match="ionic charge must be an integer"):
        evaluate(route, xyz, None, charge=0.9)


@pytest.mark.parametrize("field", ["charges", "multiplicities"])
@pytest.mark.parametrize("values", [[], [0, 0]])
def test_batch_metadata_count_never_reaches_backend(
    xyz: typing.Any, field: typing.Any, values: typing.Any
) -> None:
    calculator = QuadraticCalculator()
    if field == "multiplicities":
        values = [1] * len(values)
    with pytest.raises(ValueError, match="must match the ragged batch size"):
        batched_energy([xyz], [[1, 1]], calculator, **{field: values})
    assert calculator.calls == []


def test_first_order_composition_and_gradcheck(
    route: typing.Any, xyz: typing.Any
) -> None:
    calculator = QuadraticCalculator()
    result = evaluate(route, xyz, calculator)
    (gradient,) = torch.autograd.grad(result.square(), xyz)
    torch.testing.assert_close(gradient, 4 * xyz.detach())
    assert torch.autograd.gradcheck(lambda r: evaluate(route, r, calculator), (xyz,))


def test_ragged_first_order_keeps_independent_cotangents(xyz: typing.Any) -> None:
    other = torch.tensor([[0.5, -2.0, 1.0]], dtype=torch.float64, requires_grad=True)
    values = batched_energy([xyz, other], [[1, 1], [2]], QuadraticCalculator())
    gradients = torch.autograd.grad(values, (xyz, other), torch.tensor([2.0, -3.0]))
    torch.testing.assert_close(gradients[0], 4 * xyz.detach())
    torch.testing.assert_close(gradients[1], -6 * other.detach())


@pytest.mark.parametrize("nonlinear", [False, True])
def test_differentiable_backward_is_explicitly_unsupported(
    route: typing.Any, xyz: typing.Any, nonlinear: typing.Any
) -> None:
    value = evaluate(route, xyz, QuadraticCalculator())
    loss = value.square() if nonlinear else value
    with pytest.raises(RuntimeError, match="higher-order derivatives.*unsupported"):
        torch.autograd.grad(loss, xyz, create_graph=True)


@pytest.mark.parametrize("operation", ["hessian", "hessian_vectorized", "hvp", "vhp"])
@pytest.mark.parametrize("nonlinear", [False, True])
def test_functional_second_derivatives_fail_with_default_non_strict_options(
    route: typing.Any,
    xyz: typing.Any,
    operation: typing.Any,
    nonlinear: typing.Any,
) -> None:
    def loss(r: typing.Any) -> typing.Any:
        value = evaluate(route, r, QuadraticCalculator())
        return value.square() if nonlinear else value

    # Default strict=False must not turn a missing native Hessian into zeros.
    with pytest.raises(RuntimeError, match="higher-order derivatives.*unsupported"):
        if operation == "hessian_vectorized":
            torch.autograd.functional.hessian(loss, xyz, vectorize=True)
        elif operation == "hessian":
            torch.autograd.functional.hessian(loss, xyz)
        else:
            getattr(torch.autograd.functional, operation)(
                loss, xyz, torch.ones_like(xyz)
            )


def test_all_torch_oracle_contains_both_composite_hessian_terms(
    xyz: typing.Any,
) -> None:
    """Pin the reported failure: a detached native force previously returned 8."""
    hessian = torch.autograd.functional.hessian(
        lambda r: r.square().sum().square(), xyz
    )
    assert hessian[0, 0, 0, 0].item() == 12.0


@pytest.mark.parametrize(
    "dtype",
    [torch.bool, torch.int32, torch.int64, torch.complex64, torch.complex128],
)
def test_non_real_floating_coordinates_never_reach_backend(
    route: typing.Any, dtype: typing.Any
) -> None:
    coordinates = torch.tensor([[0, 0, -1], [0, 0, 1]], dtype=dtype)
    calculator = QuadraticCalculator()
    with pytest.raises(TypeError, match="real floating-point tensor"):
        evaluate(route, coordinates, calculator)
    assert calculator.calls == []


def test_coordinate_dtype_is_checked_before_default_calculator(
    monkeypatch: typing.Any, route: typing.Any
) -> None:
    def unexpected_calculator(**kwargs: typing.Any) -> None:
        pytest.fail("invalid coordinate dtype constructed a native calculator")

    monkeypatch.setattr("vibeqc.torch.Calculator", unexpected_calculator)
    coordinates = torch.ones((2, 3), dtype=torch.int64)
    with pytest.raises(TypeError, match="real floating-point tensor"):
        evaluate(route, coordinates, None)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_real_floating_coordinate_dtypes_preserve_energy_and_backward(
    route: typing.Any, dtype: typing.Any
) -> None:
    coordinates = torch.tensor(
        [[0.25, -0.5, 1.0], [0.0, 0.75, -1.5]],
        dtype=dtype,
        requires_grad=True,
    )
    result = evaluate(route, coordinates, QuadraticCalculator())
    assert result.dtype == dtype
    assert result.device == coordinates.device
    torch.testing.assert_close(
        result, coordinates.detach().double().square().sum().to(dtype)
    )
    result.backward()
    torch.testing.assert_close(coordinates.grad, 2 * coordinates.detach())
