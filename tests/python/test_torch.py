import pytest

torch = pytest.importorskip("torch")

from vibeqc import Calculator
from vibeqc.torch import batched_energy, energy


def test_torch_backward_matches_native_force():
    coordinates = torch.tensor(
        [[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]],
        dtype=torch.float64,
        requires_grad=True,
    )
    value = energy(coordinates, [1, 1])
    value.backward()
    assert value.item() == pytest.approx(-1.11671432506255, abs=2.0e-9)
    assert coordinates.grad is not None
    assert torch.max(torch.abs(coordinates.grad.sum(dim=0))).item() < 2.0e-10


def test_ragged_batched_torch_backward_and_warm_start():
    h2 = torch.tensor(
        [[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]],
        dtype=torch.float64,
        requires_grad=True,
    )
    h3 = torch.tensor(
        [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    calculator = Calculator()
    systems = [
        [(1, position.tolist()) for position in h2.detach()],
        [(1, position.tolist()) for position in h3.detach()],
    ]
    with calculator.prepare_batch(systems, charges=[0, 1]) as prepared:
        values = batched_energy(
            [h2, h3],
            [[1, 1], [1, 1, 1]],
            calculator,
            charges=[0, 1],
            prepared_batch=prepared,
        )
        values.sum().backward()
        assert values.shape == (2,)
        assert values[0].item() == pytest.approx(-1.11671432506255, abs=2.0e-9)
        assert h2.grad is not None and h3.grad is not None
        assert torch.max(torch.abs(h2.grad.sum(dim=0))).item() < 2.0e-10
        assert torch.max(torch.abs(h3.grad.sum(dim=0))).item() < 2.0e-10


def test_cuda_uhf_torch_backward_uses_native_analytic_force():
    coordinates = torch.tensor(
        [[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]],
        dtype=torch.float64,
        requires_grad=True,
    )
    try:
        calculator = Calculator(
            method="uhf",
            device="cuda",
            energy_tolerance=1.0e-12,
            density_tolerance=1.0e-10,
        )
        value = energy(
            coordinates,
            [1, 1],
            calculator,
            charge=1,
            multiplicity=2,
        )
        value.backward()
    except RuntimeError as error:
        pytest.skip(f"CUDA device unavailable: {error}")

    assert value.item() == pytest.approx(-0.53851134832246783, abs=2.0e-10)
    assert coordinates.grad is not None
    assert coordinates.grad[1, 2].item() == pytest.approx(
        -0.19038408686848290, abs=3.0e-9
    )
    assert torch.max(torch.abs(coordinates.grad.sum(dim=0))).item() < 3.0e-9


def test_cuda_uhf_batched_torch_backward_reuses_spin_warm_state():
    first = torch.tensor(
        [[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]],
        dtype=torch.float64,
        requires_grad=True,
    )
    second = torch.tensor(
        [[0.0, 0.0, -0.9], [0.0, 0.0, 0.9]],
        dtype=torch.float64,
        requires_grad=True,
    )
    systems = [
        [(1, position.tolist()) for position in first.detach()],
        [(1, position.tolist()) for position in second.detach()],
    ]
    try:
        calculator = Calculator(method="uhf", device="cuda")
        with calculator.prepare_batch(
            systems,
            charges=[1, 1],
            multiplicities=[2, 2],
            warm_start=True,
        ) as prepared:
            values = batched_energy(
                [first, second],
                [[1, 1], [1, 1]],
                calculator,
                charges=[1, 1],
                multiplicities=[2, 2],
                prepared_batch=prepared,
            )
            values.sum().backward()
    except RuntimeError as error:
        pytest.skip(f"CUDA device unavailable: {error}")

    assert values[0].item() == pytest.approx(-0.53851134832246783, abs=2.0e-10)
    assert first.grad is not None and second.grad is not None
    assert torch.max(torch.abs(first.grad.sum(dim=0))).item() < 3.0e-9
    assert torch.max(torch.abs(second.grad.sum(dim=0))).item() < 3.0e-9
