"""Generated shell-local H1/S1 sources for the existing RHF nuclear RHS.

First integral derivatives are emitted from the compiler's S/T/V and weighted
ERI DAGs and evaluated in native CPU code. Each ERI component is immediately
contracted into two AO indices with the frozen density; no molecular ERI
Jacobian, finite differences, or reference-engine derivative call is used.
"""

import shutil
from itertools import product
from pathlib import Path

import numpy as np
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.integral.first_derivatives_execute import (
    FirstDerivativeEvaluator,
    compile_first_derivative,
)
from vibeqc_compiler.integral.first_derivatives_native import first_component_identity
from vibeqc_compiler.integral.one_electron_derivatives import (
    build_one_electron_derivative_ir,
)
from vibeqc_compiler.integral.weight_pullback import normalized_cartesian_components
from vibeqc_compiler.integral.weighted_eri import build_weighted_eri_ir

from .native import NativeRHFState


def generated_first_order(state):
    """Return frozen-Fock and overlap derivatives in (atom,xyz,AO,AO) order."""
    if not isinstance(state, NativeRHFState):
        raise TypeError("generated Hessian sources require NativeRHFState")
    state.validate()
    compiler = CppCompilerAdapter(Path(shutil.which("c++") or "c++"))
    cache = state.cache / "first-cache"
    evaluators = {}
    shells, offsets, prims = state.source.shells, state.offsets, state.primitives
    coords, density = state.coords, state.P0
    shape = (state.nat, 3, state.nbf, state.nbf)
    overlap, frozen = np.zeros(shape), np.zeros(shape)

    def raw_tiles(ir, slots, atom_indices):
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
            if key not in evaluators:
                artifact = compile_first_derivative(
                    ir, compiler, cache, component_indices=indices
                )
                evaluators[key] = FirstDerivativeEvaluator(artifact)
            values = evaluators[key].contract(
                tuple(prims[i] for i in slots), coords[list(atom_indices)]
            )
            values *= scales[list(indices), None]
            for row, index in enumerate(indices):
                yield (
                    np.unravel_index(index, component_shape),
                    values[row, 1:].reshape(-1, 3),
                )

    for a, b in product(range(len(shells)), repeat=2):
        angular = (shells[a].angular_momentum, shells[b].angular_momentum)
        atoms = (shells[a].atom_index, shells[b].atom_index)
        for family, out in (("overlap", overlap), ("kinetic", frozen)):
            ir = build_one_electron_derivative_ir(family, angular)
            for (u, v), gradient in raw_tiles(ir, (a, b), atoms):
                for center, atom in enumerate(atoms):
                    out[atom, :, offsets[a] + u, offsets[b] + v] += gradient[center]
        for nucleus, charge in enumerate(state.Z):
            ir = build_one_electron_derivative_ir(
                "nuclear_attraction", angular, charge=float(charge)
            )
            centers = (*atoms, nucleus)
            for (u, v), gradient in raw_tiles(ir, (a, b), centers):
                for center, atom in enumerate(centers):
                    frozen[atom, :, offsets[a] + u, offsets[b] + v] += gradient[center]

    for slots in product(range(len(shells)), repeat=4):
        angular = tuple(shells[i].angular_momentum for i in slots)
        atoms = tuple(shells[i].atom_index for i in slots)
        ir = build_weighted_eri_ir(angular)
        for component, gradient in raw_tiles(ir, slots, atoms):
            u, v, w, x = (
                offsets[shell] + c for shell, c in zip(slots, component, strict=True)
            )
            # Ordered AO traversal: no orbit multiplicities or energy prefactors.
            for center, atom in enumerate(atoms):
                frozen[atom, :, u, v] += density[w, x] * gradient[center]
                frozen[atom, :, u, w] -= 0.5 * density[v, x] * gradient[center]
    if not np.isfinite(overlap).all() or not np.isfinite(frozen).all():
        raise FloatingPointError("generated nuclear-perturbation source is nonfinite")
    return frozen, overlap
