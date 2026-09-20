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
RHF_FIRST_ERI_TERMS = (
    DirectionalMatrixTerm(0, (0, 1), (2, 3), 1.0),
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
    """Contract a direction shell-locally into H1(v)/S1(v), each (AO,AO).

    The same generated primitive derivatives supply the full and directional
    callers. Only bounded shell-component gradients are formed; no molecular
    coordinate-indexed H1/S1 or ERI derivative tensor is allocated here.
    Integral arithmetic is native CPU; caller-side direction/density reduction
    is explicit host work, not a generated CUDA derivative contraction.
    """
    if not isinstance(state, NativeRHFState):
        raise TypeError("generated Hessian sources require NativeRHFState")
    return _generated_first_order(state, checked_direction(direction, state.nat))


def _generated_first_order(
    state: typing.Any, direction: typing.Any = None
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
            for term in RHF_FIRST_ERI_TERMS:
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


def generated_rhf_relaxation_contraction(
    state: typing.Any,
    density_response: typing.Any,
    energy_weighted_density_response: typing.Any,
) -> typing.Any:
    """Return the first-integral part of a complete RHF molecular HVP.

    For a solved directional response this contracts, for every output nuclear
    coordinate R,

        Tr[H1_R D1(v)] - Tr[S1_R W1(v)]

    directly against generated first-integral derivatives. The two-electron
    contribution is the directional derivative of the frozen RHF energy weight,
    evaluated shell-locally, so no all-coordinate H1/S1 or molecular ERI
    derivative tensor is formed.
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
    shells, offsets = provider.shells, provider.offsets
    density = state.P0
    result = np.zeros((state.nat, 3), dtype=np.float64)

    def accumulate(
        atoms: typing.Any, derivative: typing.Any, coefficient: typing.Any
    ) -> None:
        if coefficient == 0:
            return
        for center, atom in enumerate(atoms):
            result[atom] += coefficient * derivative[center]

    for a, b in product(range(len(shells)), repeat=2):
        angular = (shells[a].angular_momentum, shells[b].angular_momentum)
        atoms = (shells[a].atom_index, shells[b].atom_index)
        ir = build_one_electron_derivative_ir("kinetic", angular)
        for (u, v), gradient in provider.raw_tiles(ir, (a, b), atoms):
            i, j = offsets[a] + u, offsets[b] + v
            accumulate(atoms, gradient, density_response[i, j])

        ir = build_one_electron_derivative_ir("overlap", angular)
        for (u, v), gradient in provider.raw_tiles(ir, (a, b), atoms):
            i, j = offsets[a] + u, offsets[b] + v
            accumulate(
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
                accumulate(centers, gradient, density_response[i, j])

    for slots in product(range(len(shells)), repeat=4):
        angular = tuple(shells[i].angular_momentum for i in slots)
        atoms = tuple(shells[i].atom_index for i in slots)
        ir = build_weighted_eri_ir(angular)
        for component, gradient in provider.raw_tiles(ir, slots, atoms):
            u, v, w, x = (
                offsets[shell] + c for shell, c in zip(slots, component, strict=True)
            )
            # d/dP of W2(P) = 1/2 P_uv P_wx - 1/4 P_uw P_vx,
            # evaluated along D1(v). This is exactly Tr[D1 G_R(P0)].
            coefficient = 0.5 * (
                density_response[u, v] * density[w, x]
                + density[u, v] * density_response[w, x]
            )
            coefficient -= 0.25 * (
                density_response[u, w] * density[v, x]
                + density[u, w] * density_response[v, x]
            )
            accumulate(atoms, gradient, coefficient)

    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite RHF relaxation contraction")
    return result
