"""Generated CUDA directional first-integral/Fock sources for small native RHF.

Host work prepares shell geometry and radial records. Direction contractions,
Cartesian factors, external density contraction and global AO accumulation are
emitted once by the common compiler and execute on the selected device.
"""

from itertools import product

import numpy as np
from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.integral.first_directional import (
    DirectionalMatrixTerm,
    directional_identity,
)
from vibeqc_compiler.integral.first_directional_execute import (
    DirectionalFirstAccumulator,
    compile_directional_first,
)
from vibeqc_compiler.integral.first_gradient import (
    FirstGradientTerm,
    FirstGradientWeight,
    first_gradient_identity,
)
from vibeqc_compiler.integral.first_gradient_execute import (
    FirstGradientAccumulator,
    compile_first_gradient,
    first_gradient_storage,
)
from vibeqc_compiler.integral.one_electron_derivatives import (
    build_one_electron_derivative_ir,
)
from vibeqc_compiler.integral.weighted_eri import build_weighted_eri_ir

from .first_order import RHF_FIRST_ERI_TERMS, _checked_ao_weight, checked_direction
from .native import NativeRHFState


def generated_directional_first_order_cuda(
    state,
    direction,
    compiler,
    *,
    device_id=0,
    budget_bytes=64 << 20,
    record_capacity=128,
    component_tile=8,
):
    """Return frozen H1(v), overlap S1(v) and truthful execution diagnostics.

    All ordered shell contributions share a device accumulator; intermediate
    integral derivatives and per-shell matrices never return to the host.
    This is not a device-resident CPHF solve, complete HVP or performance claim.
    """
    if not isinstance(state, NativeRHFState):
        raise TypeError("CUDA directional sources require NativeRHFState")
    state.validate()
    direction = checked_direction(direction, state.nat)
    if not isinstance(compiler, CudaCompilerAdapter):
        raise TypeError(
            "CUDA directional sources require an explicit CudaCompilerAdapter"
        )
    if type(component_tile) is not int or not 1 <= component_tile <= 8:
        raise ValueError("directional component_tile must be between one and eight")
    shells, offsets, prims = state.source.shells, state.offsets, state.primitives
    coords = state.coords
    programs = {}
    cache = state.cache / "directional-first-cuda"
    one = (DirectionalMatrixTerm(0, (0, 1)),)
    overlap = (DirectionalMatrixTerm(1, (0, 1)),)

    def compiled(ir, indices, terms):
        key = directional_identity(ir, indices, terms)
        if key not in programs:
            programs[key] = compile_directional_first(
                ir, compiler, cache, component_indices=indices, terms=terms
            )
        return programs[key]

    # Prepare a real (possibly later reused) program rather than a schema-only
    # capability. The generic owner accepts all compatible compiled shell plans.
    seed = compiled(build_one_electron_derivative_ir("overlap", (0, 0)), (0,), overlap)
    with DirectionalFirstAccumulator(
        seed,
        nbf=state.nbf,
        natoms=state.nat,
        outputs=2,
        capacity=record_capacity,
        device_id=device_id,
        budget_bytes=budget_bytes,
    ) as owner:
        owner.reset(state.P0, direction)

        def append(ir, slots, atoms, terms):
            count = ir.signature.component_count
            for start in range(0, count, component_tile):
                indices = tuple(range(start, min(start + component_tile, count)))
                owner.append_shell(
                    compiled(ir, indices, terms),
                    tuple(prims[s] for s in slots),
                    coords[list(atoms)],
                    offsets=tuple(int(offsets[s]) for s in slots),
                    atoms=tuple(int(a) for a in atoms),
                )

        if np.any(direction):
            for a, b in product(range(len(shells)), repeat=2):
                angular = (shells[a].angular_momentum, shells[b].angular_momentum)
                atoms = (shells[a].atom_index, shells[b].atom_index)
                for family, terms in (("overlap", overlap), ("kinetic", one)):
                    append(
                        build_one_electron_derivative_ir(family, angular),
                        (a, b),
                        atoms,
                        terms,
                    )
                for nucleus, charge in enumerate(state.Z):
                    append(
                        build_one_electron_derivative_ir(
                            "nuclear_attraction", angular, charge=float(charge)
                        ),
                        (a, b),
                        (*atoms, nucleus),
                        one,
                    )
            for slots in product(range(len(shells)), repeat=4):
                angular = tuple(shells[s].angular_momentum for s in slots)
                atoms = tuple(shells[s].atom_index for s in slots)
                append(
                    build_weighted_eri_ir(angular), slots, atoms, RHF_FIRST_ERI_TERMS
                )
        matrices = owner.finish()
        state.validate()
        diagnostics = {
            "backend": "cuda-generated",
            "direction_density_reduction": "cuda-generated",
            "matrix_accumulation": "cuda",
            "matrix_downloads": owner.statistics["matrix_downloads"],
            "primitive_records": owner.statistics["primitive_records"],
            "chunks": owner.statistics["chunks"],
            "compiled_programs": len(programs),
            "storage": dict(owner.storage),
            "program_identities": tuple(sorted(programs)),
            "native_artifacts": tuple(
                sorted(a.native.metadata["key"] for a in programs.values())
            ),
            "raw_derivative_downloads": 0,
            "source_identity": state.source.identity,
            "reference_identity": state.reference.identity,
            "runtime_identity": seed.runtime_identity,
            "device_id": device_id,
            "scope": "Cartesian RHF directional H1/S1; not complete HVP or resident CPHF",
        }
    return matrices[0], matrices[1], diagnostics


def generated_rhf_relaxation_contraction_cuda(
    state,
    density_response,
    energy_weighted_density_response,
    compiler,
    *,
    device_id=0,
    budget_bytes=64 << 20,
    record_capacity=128,
    component_tile=8,
):
    """Contract RHF first-integral relaxation in generated CUDA code.

    The compiler owns the generic AO-weight pullback. RHF only declares the
    matrix factors for Tr[H1 D1] - Tr[S1 W1], including the density derivative
    of the conventional two-electron weight.
    """
    if not isinstance(state, NativeRHFState):
        raise TypeError("CUDA RHF relaxation requires NativeRHFState")
    state.validate()
    if not isinstance(compiler, CudaCompilerAdapter):
        raise TypeError("CUDA RHF relaxation requires an explicit CudaCompilerAdapter")
    if type(component_tile) is not int or not 1 <= component_tile <= 8:
        raise ValueError("relaxation component_tile must be between one and eight")
    from vibeqc_compiler.common.resources import checked_bytes

    checked_bytes(budget_bytes)
    storage = first_gradient_storage(state.nbf, state.nat, 3, record_capacity)
    if storage["numeric_peak_bytes"] > budget_bytes:
        raise MemoryError("CUDA relaxation numeric storage exceeds budget_bytes")
    d1 = _checked_ao_weight(density_response, state.nbf, "density response")
    w1 = _checked_ao_weight(
        energy_weighted_density_response,
        state.nbf,
        "energy-weighted density response",
    )
    weights = np.stack((d1, w1, state.P0))
    shells, offsets, prims = state.source.shells, state.offsets, state.primitives
    coords = state.coords
    programs = {}
    cache = state.cache / "relaxation-first-cuda"

    weight = FirstGradientWeight
    kinetic = (FirstGradientTerm((weight(0, (0, 1)),)),)
    overlap = (FirstGradientTerm((weight(1, (0, 1)),), coefficient=-1.0),)
    eri = (
        FirstGradientTerm((weight(0, (0, 1)), weight(2, (2, 3))), coefficient=0.5),
        FirstGradientTerm((weight(2, (0, 1)), weight(0, (2, 3))), coefficient=0.5),
        FirstGradientTerm((weight(0, (0, 2)), weight(2, (1, 3))), coefficient=-0.25),
        FirstGradientTerm((weight(2, (0, 2)), weight(0, (1, 3))), coefficient=-0.25),
    )

    def compiled(ir, indices, terms):
        key = first_gradient_identity(ir, indices, terms)
        if key not in programs:
            programs[key] = compile_first_gradient(
                ir, compiler, cache, component_indices=indices, terms=terms
            )
        return programs[key]

    seed = compiled(
        build_one_electron_derivative_ir("overlap", (0, 0)),
        (0,),
        overlap,
    )
    with FirstGradientAccumulator(
        seed,
        nbf=state.nbf,
        natoms=state.nat,
        weight_slots=3,
        capacity=record_capacity,
        device_id=device_id,
        budget_bytes=budget_bytes,
    ) as owner:
        owner.reset(weights)

        def append(ir, slots, atoms, terms):
            count = ir.signature.component_count
            for start in range(0, count, component_tile):
                indices = tuple(range(start, min(start + component_tile, count)))
                owner.append_shell(
                    compiled(ir, indices, terms),
                    tuple(prims[s] for s in slots),
                    coords[list(atoms)],
                    offsets=tuple(int(offsets[s]) for s in slots),
                    atoms=tuple(int(atom) for atom in atoms),
                )

        for a, b in product(range(len(shells)), repeat=2):
            angular = (shells[a].angular_momentum, shells[b].angular_momentum)
            atoms = (shells[a].atom_index, shells[b].atom_index)
            append(
                build_one_electron_derivative_ir("kinetic", angular),
                (a, b),
                atoms,
                kinetic,
            )
            append(
                build_one_electron_derivative_ir("overlap", angular),
                (a, b),
                atoms,
                overlap,
            )
            for nucleus, charge in enumerate(state.Z):
                append(
                    build_one_electron_derivative_ir(
                        "nuclear_attraction", angular, charge=float(charge)
                    ),
                    (a, b),
                    (*atoms, nucleus),
                    kinetic,
                )
        for slots in product(range(len(shells)), repeat=4):
            angular = tuple(shells[s].angular_momentum for s in slots)
            atoms = tuple(shells[s].atom_index for s in slots)
            append(build_weighted_eri_ir(angular), slots, atoms, eri)

        result = owner.finish()
        state.validate()
        diagnostics = {
            "backend": "cuda-generated-weighted-gradient",
            "weight_matrices": (
                "density_response",
                "energy_weighted_density_response",
                "density",
            ),
            "matrix_weight_products": "cuda-generated",
            "gradient_accumulation": "cuda",
            "weight_uploads": owner.statistics["weight_uploads"],
            "gradient_downloads": owner.statistics["gradient_downloads"],
            "primitive_records": owner.statistics["primitive_records"],
            "chunks": owner.statistics["chunks"],
            "compiled_programs": len(programs),
            "storage": dict(owner.storage),
            "program_identities": tuple(
                sorted(artifact.program_identity for artifact in programs.values())
            ),
            "native_artifacts": tuple(
                sorted(
                    artifact.native.metadata["key"] for artifact in programs.values()
                )
            ),
            "raw_derivative_downloads": 0,
            "intermediate_matrix_downloads": 0,
            "source_identity": state.source.identity,
            "reference_identity": state.reference.identity,
            "runtime_identity": seed.runtime_identity,
            "device_id": device_id,
            "scope": (
                "RHF first-integral relaxation contraction only; "
                "not second-integral HVP"
            ),
        }
    return result, diagnostics
