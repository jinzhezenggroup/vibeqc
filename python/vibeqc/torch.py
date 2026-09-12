"""First-order PyTorch backward backed by native analytic HF gradients."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .batch import PreparedBatch
from .calculator import Calculator
from .elements import checked_integer


def _validated_atomic_numbers(coordinates, atomic_numbers):
    """Preserve molecular identity before any transfer or native evaluation."""
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must have shape (natoms, 3)")
    if len(atomic_numbers) != coordinates.shape[0]:
        raise ValueError("atomic_numbers does not match its coordinates")
    return tuple(
        checked_integer(value, "atomic number", low=1, high=118)
        for value in atomic_numbers
    )


def _require_first_order_backward():
    """Reject differentiable backward before detached forces lose Hessian terms.

    Checking grad mode also covers functional Hessian/HVP APIs whose default
    non-strict behavior can turn a detached backward into silent zero entries.
    """
    if torch.is_grad_enabled():
        raise RuntimeError(
            "VibeQC energy supports only first-order coordinate derivatives; "
            "higher-order derivatives through native forces are unsupported"
        )


class _EnergyFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx, coordinates, atomic_numbers, calculator, charge, multiplicity
    ):
        if coordinates.device.type != "cpu":
            # The C ABI can accept device buffers in a later version; the MVP
            # makes this transfer explicit instead of hiding it in native code.
            coordinate_values = coordinates.detach().cpu()
        else:
            coordinate_values = coordinates.detach()
        atoms = [
            (atomic_numbers[index], coordinate_values[index].tolist())
            for index in range(coordinates.shape[0])
        ]
        result = calculator.singlepoint(atoms, charge=charge, multiplicity=multiplicity)
        force_tensor = torch.as_tensor(
            result.forces, dtype=coordinates.dtype, device=coordinates.device
        )
        ctx.save_for_backward(force_tensor)
        return coordinates.new_tensor(result.energy)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[override]
        _require_first_order_backward()
        (forces,) = ctx.saved_tensors
        # Native forces are -dE/dR, while autograd requests dE/dR.
        return -forces * grad_output, None, None, None, None


def energy(
    coordinates: torch.Tensor,
    atomic_numbers: Sequence[int],
    calculator: Calculator | None = None,
    *,
    charge: int = 0,
    multiplicity: int = 1,
) -> torch.Tensor:
    """Return HF energy with a native analytic first-order coordinate backward.

    Atomic numbers, charge and multiplicity must be exact integers (not bools).
    Differentiable backward (``create_graph=True``), Hessians and HVPs raise an
    error because the native force callback does not provide force derivatives.
    """

    atomic_numbers = _validated_atomic_numbers(coordinates, atomic_numbers)
    charge = checked_integer(charge, "ionic charge", low=-(2**31))
    multiplicity = checked_integer(multiplicity, "multiplicity", low=1)
    if calculator is None:
        calculator = Calculator(method="rhf", basis="sto-3g", device="cpu")
    return _EnergyFunction.apply(
        coordinates,
        atomic_numbers,
        calculator,
        charge,
        multiplicity,
    )


class _BatchedEnergyFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        calculator,
        prepared_batch,
        atomic_numbers,
        charges,
        multiplicities,
        *coordinates,
    ):
        if not coordinates:
            raise ValueError("batched energy requires at least one system")
        reference = coordinates[0]
        for item in coordinates:
            if item.device != reference.device or item.dtype != reference.dtype:
                raise ValueError(
                    "all ragged coordinate tensors must share device and dtype"
                )

        cpu_coordinates = [item.detach().cpu().numpy() for item in coordinates]
        if prepared_batch is not None:
            if prepared_batch.atomic_numbers != atomic_numbers:
                raise ValueError(
                    "prepared batch topology does not match atomic numbers"
                )
            if prepared_batch.charges != charges:
                raise ValueError("prepared batch charges do not match")
            if prepared_batch.multiplicities != multiplicities:
                raise ValueError("prepared batch multiplicities do not match")
            result = prepared_batch.execute(cpu_coordinates, strict=True)
        else:
            systems = [
                [
                    (atomic_numbers[system][atom], cpu_coordinates[system][atom])
                    for atom in range(len(atomic_numbers[system]))
                ]
                for system in range(len(coordinates))
            ]
            result = calculator.batch_singlepoint(
                systems,
                charges=charges,
                multiplicities=multiplicities,
                strict=True,
            )

        force_tensors = tuple(
            torch.as_tensor(item.forces, dtype=reference.dtype, device=reference.device)
            for item in result.items
        )
        ctx.save_for_backward(*force_tensors)
        return torch.as_tensor(
            result.energies, dtype=reference.dtype, device=reference.device
        )

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[override]
        _require_first_order_backward()
        coordinate_gradients = tuple(
            -force * grad_output[index] for index, force in enumerate(ctx.saved_tensors)
        )
        return None, None, None, None, None, *coordinate_gradients


def batched_energy(
    coordinates: Sequence[torch.Tensor],
    atomic_numbers: Sequence[Sequence[int]],
    calculator: Calculator | None = None,
    *,
    charges: Sequence[int] | None = None,
    multiplicities: Sequence[int] | None = None,
    prepared_batch: PreparedBatch | None = None,
) -> torch.Tensor:
    """Evaluate a ragged native HF batch with analytic coordinate backward.

    Systems remain separate tensors, so the interface never pads all molecules
    to the largest atom count. Passing a `PreparedBatch` enables native
    topology-aware warm starts across repeated forward calls.

    Each atomic-number list must match its coordinates. Atomic numbers, charges
    and multiplicities must be exact integers (not bools). Only first-order
    derivatives are supported; differentiable backward, Hessians and HVPs raise
    an error, including when a prepared batch supplies the detached forces.
    """

    if len(coordinates) != len(atomic_numbers) or not coordinates:
        raise ValueError(
            "coordinates and atomic_numbers must have equal nonzero length"
        )
    normalized_atomic_numbers = tuple(
        _validated_atomic_numbers(item, numbers)
        for item, numbers in zip(coordinates, atomic_numbers, strict=True)
    )
    normalized_charges = (
        tuple(0 for _ in coordinates)
        if charges is None
        else tuple(
            checked_integer(value, "ionic charge", low=-(2**31)) for value in charges
        )
    )
    normalized_multiplicities = (
        tuple(1 for _ in coordinates)
        if multiplicities is None
        else tuple(
            checked_integer(value, "multiplicity", low=1) for value in multiplicities
        )
    )
    if len(normalized_charges) != len(coordinates) or len(
        normalized_multiplicities
    ) != len(coordinates):
        raise ValueError("charges and multiplicities must match the ragged batch size")
    if calculator is None:
        calculator = Calculator(method="rhf", basis="sto-3g", device="cpu")
    return _BatchedEnergyFunction.apply(
        calculator,
        prepared_batch,
        normalized_atomic_numbers,
        normalized_charges,
        normalized_multiplicities,
        *tuple(coordinates),
    )
