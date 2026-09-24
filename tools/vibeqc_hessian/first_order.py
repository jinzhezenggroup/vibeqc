"""Generated shell-local first-integral derivatives for RHF Hessian consumers.

The same S/T/V and weighted-ERI DAGs supply directional H1/S1 matrices and
scalar first-integral contractions used by molecular HVP relaxation. Primitive
derivatives are bounded to shell/component tiles; neither path materializes a
molecular ERI Jacobian.
"""

import shutil
import typing
from itertools import product
from pathlib import Path

import numpy as np
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.integral.first_derivatives_execute import (
    FirstDerivativeEvaluator,
    compile_first_derivative,
)
from vibeqc_compiler.integral.first_derivatives_native import first_component_identity
from vibeqc_compiler.integral.first_directional import DirectionalMatrixTerm
from vibeqc_compiler.integral.one_electron_derivatives import (
    build_one_electron_derivative_ir,
)
from vibeqc_compiler.integral.weight_pullback import normalized_cartesian_components
from vibeqc_compiler.integral.weighted_eri import build_weighted_eri_ir

from .native import NativeRHFState

# One declared conventional RHF frozen-Fock contraction, shared by the CPU
# reference traversal and generated CUDA matrix consumer.
COULOMB_FIRST_ERI_TERMS = (
    DirectionalMatrixTerm(0, (0, 1), (2, 3), 1.0),
)

RHF_FIRST_ERI_TERMS = (
    *COULOMB_FIRST_ERI_TERMS,
    DirectionalMatrixTerm(0, (0, 2), (1, 3), -0.5),
)


def checked_direction(direction: typing.Any, natoms: typing.Any) -> typing.Any:
    """Own a finite real Cartesian direction; do not normalize its magnitude."""
    value = np.asarray(direction)
    if (
        value.shape != (natoms, 3)
        or value.dtype.kind not in "iuf"
        or not np.isfinite(value).all()
    ):
        raise ValueError("direction must be finite real with shape (natoms, 3)")
    value = np.array(value, dtype=np.float64, copy=True)
    if not np.isfinite(value).all():
        raise ValueError("direction must be representable in FP64")
    return value


def _checked_ao_weight(
    value: typing.Any, nbf: typing.Any, name: typing.Any
) -> typing.Any:
    array = np.asarray(value)
    if (
        array.shape != (nbf, nbf)
        or array.dtype.kind not in "iuf"
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must be a finite real AO matrix")
    array = np.array(array, dtype=np.float64, copy=True)
    if not np.allclose(array, array.T, atol=2e-10, rtol=2e-12):
        raise ValueError(f"{name} must be symmetric")
    return array


class _FirstDerivativeProvider:
    """Shared bounded CPU evaluator for matrix and weighted scalar consumers."""

    def __init__(self, state: typing.Any) -> None:
        if not isinstance(state, NativeRHFState):
            raise TypeError("generated Hessian sources require NativeRHFState")
        state.validate()
        self.state = state
        self.compiler = CppCompilerAdapter(Path(shutil.which("c++") or "c++"))
        self.cache = state.cache / "first-cache"
        self.evaluators = {}
        self.shells = state.source.shells
        self.offsets = state.offsets
        self.primitives = state.primitives
        self.coords = state.coords

    def raw_tiles(
        self, ir: typing.Any, slots: typing.Any, atom_indices: typing.Any
    ) -> typing.Any:
        angular = ir.signature.angular
        count = ir.signature.component_count
        component_shape = ir.signature.component_shape
        # Radial coefficients and angular double-factorial factors each occur once.
        scales = np.array(
            [
                weight
                for _, weight in normalized_cartesian_components(
                    angular, np.ones(component_shape)
                )
            ]
        )
        for start in range(0, count, 64):
            indices = tuple(range(start, min(start + 64, count)))
            key = first_component_identity(ir, indices)
            if key not in self.evaluators:
                artifact = compile_first_derivative(
                    ir, self.compiler, self.cache, component_indices=indices
                )
                self.evaluators[key] = FirstDerivativeEvaluator(artifact)
            values = self.evaluators[key].contract(
                tuple(self.primitives[i] for i in slots),
                self.coords[list(atom_indices)],
            )
            values *= scales[list(indices), None]
            for row, index in enumerate(indices):
                yield (
                    np.unravel_index(index, component_shape),
                    values[row, 1:].reshape(-1, 3),
                )


def generated_first_order(state: typing.Any) -> typing.Any:
    """Return frozen-Fock and overlap derivatives in (atom,xyz,AO,AO) order."""
    return _generated_first_order(state)


def generated_directional_first_order(
    state: typing.Any, direction: typing.Any
) -> typing.Any:
    """Contract the conventional RHF frozen Fock direction into H1(v)/S1(v)."""
    if not isinstance(state, NativeRHFState):
        raise TypeError("generated Hessian sources require NativeRHFState")
    return _generated_first_order(
        state,
        checked_direction(direction, state.nat),
        eri_terms=RHF_FIRST_ERI_TERMS,
    )


def generated_coulomb_directional_first_order(
    state: typing.Any, direction: typing.Any
) -> typing.Any:
    """Contract one pure-Coulomb stationary frozen-Fock direction.

    This shares the exact S/T/V and weighted-ERI derivative provider with RHF,
    but admits only the Coulomb density-to-Fock term. Semilocal XC geometry is
    a separate source owned by the DFT contraction layer.
    """
    if not isinstance(state, NativeRHFState):
        raise TypeError("generated Hessian sources require NativeRHFState")
    return _generated_first_order(
        state,
        checked_direction(direction, state.nat),
        eri_terms=COULOMB_FIRST_ERI_TERMS,
    )


def _generated_first_order(
    state: typing.Any,
    direction: typing.Any = None,
    *,
    eri_terms: typing.Any = RHF_FIRST_ERI_TERMS,
) -> typing.Any:
    provider = _FirstDerivativeProvider(state)
    shells, offsets = provider.shells, provider.offsets
    density = state.P0
    shape = (state.nbf, state.nbf)
    if direction is None:
        shape = (state.nat, 3, *shape)
    overlap, frozen = np.zeros(shape), np.zeros(shape)
    if direction is not None and not np.any(direction):
        return frozen, overlap

    def accumulate(
        out: typing.Any,
        atoms: typing.Any,
        derivative: typing.Any,
        u: typing.Any,
        v: typing.Any,
        coefficient: typing.Any = 1.0,
    ) -> None:
        if direction is not None:
            out[u, v] += coefficient * np.einsum(
                "ca,ca->", derivative, direction[list(atoms)]
            )
        else:
            for center, atom in enumerate(atoms):
                out[atom, :, u, v] += coefficient * derivative[center]

    for a, b in product(range(len(shells)), repeat=2):
        angular = (shells[a].angular_momentum, shells[b].angular_momentum)
        atoms = (shells[a].atom_index, shells[b].atom_index)
        for family, out in (("overlap", overlap), ("kinetic", frozen)):
            ir = build_one_electron_derivative_ir(family, angular)
            for (u, v), gradient in provider.raw_tiles(ir, (a, b), atoms):
                accumulate(out, atoms, gradient, offsets[a] + u, offsets[b] + v)
        for nucleus, charge in enumerate(state.Z):
            ir = build_one_electron_derivative_ir(
                "nuclear_attraction", angular, charge=float(charge)
            )
            centers = (*atoms, nucleus)
            for (u, v), gradient in provider.raw_tiles(ir, (a, b), centers):
                accumulate(frozen, centers, gradient, offsets[a] + u, offsets[b] + v)

    for slots in product(range(len(shells)), repeat=4):
        angular = tuple(shells[i].angular_momentum for i in slots)
        atoms = tuple(shells[i].atom_index for i in slots)
        ir = build_weighted_eri_ir(angular)
        for component, gradient in provider.raw_tiles(ir, slots, atoms):
            u, v, w, x = (
                offsets[shell] + c for shell, c in zip(slots, component, strict=True)
            )
            # Ordered AO traversal: no orbit multiplicities or energy prefactors.
            ao = (u, v, w, x)
            for term in eri_terms:
                i, j = (ao[k] for k in term.output_pair)
                k, l = (ao[k] for k in term.weight_pair)
                accumulate(
                    frozen,
                    atoms,
                    gradient,
                    i,
                    j,
                    term.coefficient * density[k, l],
                )
    if not np.isfinite(overlap).all() or not np.isfinite(frozen).all():
        raise FloatingPointError("generated nuclear-perturbation source is nonfinite")
    return frozen, overlap


def _generated_relaxation_components(
    state: typing.Any,
    density_response: typing.Any,
    energy_weighted_density_response: typing.Any,
    *,
    exchange_energy_coefficient: float,
) -> dict[str, np.ndarray]:
    """Contract stationary first-integral HVP terms by physical source.

    exchange_energy_coefficient multiplies the symmetric derivative of the
    ordered exchange energy weight. Pure-Coulomb KS passes zero; conventional
    closed-shell RHF passes -0.25. The shared one-electron, Coulomb and
    overlap/Pulay terms are therefore generated once without method-specific
    derivative kernels.
    """
    provider = _FirstDerivativeProvider(state)
    density_response = _checked_ao_weight(
        density_response, state.nbf, "density response"
    )
    energy_weighted_density_response = _checked_ao_weight(
        energy_weighted_density_response,
        state.nbf,
        "energy-weighted density response",
    )
    if not np.isfinite(exchange_energy_coefficient):
        raise ValueError("exchange energy coefficient must be finite")
    shells, offsets = provider.shells, provider.offsets
    density = state.P0
    components = {
        "one_electron": np.zeros((state.nat, 3), dtype=np.float64),
        "coulomb": np.zeros((state.nat, 3), dtype=np.float64),
        "exchange": np.zeros((state.nat, 3), dtype=np.float64),
        "overlap_pulay": np.zeros((state.nat, 3), dtype=np.float64),
    }

    def accumulate(
        target: str,
        atoms: typing.Any,
        derivative: typing.Any,
        coefficient: typing.Any,
    ) -> None:
        if coefficient == 0:
            return
        for center, atom in enumerate(atoms):
            components[target][atom] += coefficient * derivative[center]

    for a, b in product(range(len(shells)), repeat=2):
        angular = (shells[a].angular_momentum, shells[b].angular_momentum)
        atoms = (shells[a].atom_index, shells[b].atom_index)
        ir = build_one_electron_derivative_ir("kinetic", angular)
        for (u, v), gradient in provider.raw_tiles(ir, (a, b), atoms):
            i, j = offsets[a] + u, offsets[b] + v
            accumulate("one_electron", atoms, gradient, density_response[i, j])

        ir = build_one_electron_derivative_ir("overlap", angular)
        for (u, v), gradient in provider.raw_tiles(ir, (a, b), atoms):
            i, j = offsets[a] + u, offsets[b] + v
            accumulate(
                "overlap_pulay",
                atoms,
                gradient,
                -energy_weighted_density_response[i, j],
            )

        for nucleus, charge in enumerate(state.Z):
            ir = build_one_electron_derivative_ir(
                "nuclear_attraction", angular, charge=float(charge)
            )
            centers = (*atoms, nucleus)
            for (u, v), gradient in provider.raw_tiles(ir, (a, b), centers):
                i, j = offsets[a] + u, offsets[b] + v
                accumulate(
                    "one_electron", centers, gradient, density_response[i, j]
                )

    for slots in product(range(len(shells)), repeat=4):
        angular = tuple(shells[i].angular_momentum for i in slots)
        atoms = tuple(shells[i].atom_index for i in slots)
        ir = build_weighted_eri_ir(angular)
        for component, gradient in provider.raw_tiles(ir, slots, atoms):
            u, v, w, x = (
                offsets[shell] + c for shell, c in zip(slots, component, strict=True)
            )
            coulomb = 0.5 * (
                density_response[u, v] * density[w, x]
                + density[u, v] * density_response[w, x]
            )
            accumulate("coulomb", atoms, gradient, coulomb)
            exchange = exchange_energy_coefficient * (
                density_response[u, w] * density[v, x]
                + density[u, w] * density_response[v, x]
            )
            accumulate("exchange", atoms, gradient, exchange)

    if not all(np.isfinite(value).all() for value in components.values()):
        raise FloatingPointError("nonfinite stationary relaxation contraction")
    return {name: np.array(value, copy=True) for name, value in components.items()}


def generated_coulomb_relaxation_components(
    state: typing.Any,
    density_response: typing.Any,
    energy_weighted_density_response: typing.Any,
) -> dict[str, np.ndarray]:
    """Return pure-J stationary first-integral HVP terms by plan source."""
    result = _generated_relaxation_components(
        state,
        density_response,
        energy_weighted_density_response,
        exchange_energy_coefficient=0.0,
    )
    result.pop("exchange")
    return result


def generated_rhf_relaxation_contraction(
    state: typing.Any,
    density_response: typing.Any,
    energy_weighted_density_response: typing.Any,
) -> typing.Any:
    """Return the complete first-integral relaxation term for RHF HVPs."""
    components = _generated_relaxation_components(
        state,
        density_response,
        energy_weighted_density_response,
        exchange_energy_coefficient=-0.25,
    )
    result = sum(components.values(), np.zeros((state.nat, 3), dtype=np.float64))
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite RHF relaxation contraction")
    return result
