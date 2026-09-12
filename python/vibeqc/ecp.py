"""Scalar semilocal Gaussian ECPs in the BSE/NWChem convention.

The highest angular channel is the local residual potential; lower channels
are projector differences, not absolute potentials. Radial coefficients
multiply r**(n-2) exp(-alpha*r*r). The separate Coulomb tail uses Z-core.
No runtime reference-package dependency or implicit parameter download.
"""

from __future__ import annotations

import ctypes
import json
from dataclasses import dataclass

import numpy as np

from .basis import BasisSet
from .elements import checked_integer


def resolve_ecp(basis, atoms):
    """Resolve immutable source records to atom-mapped scalar residual terms."""
    cores, terms = [], []
    metadata = basis.by_element if isinstance(basis, BasisSet) else {}
    for atom_index, atom in enumerate(atoms):
        element = metadata.get(atom.atomic_number)
        core = element.ecp_core_electrons if element else 0
        cores.append(core)
        if not core:
            continue
        if core >= atom.atomic_number:
            raise ValueError("ECP core count must leave a positive ionic charge")
        potentials = json.loads(element.ecp_data)
        channels = [p["angular_momentum"][0] for p in potentials]
        if len(set(channels)) != len(channels):
            raise ValueError("duplicate ECP angular channel")
        local = max(channels)
        if local > 3:
            raise NotImplementedError(
                "ECP local channel must be s/p/d/f; projectors at most d"
            )
        if any(s.angular_momentum > 2 for s in element.shells):
            raise NotImplementedError("ECP execution supports orbital s/p/d shells")
        for potential in potentials:
            if set(potential) - {
                "ecp_type",
                "angular_momentum",
                "r_exponents",
                "gaussian_exponents",
                "coefficients",
            }:
                raise NotImplementedError(
                    "unknown ECP parameter fields require an explicit format conversion"
                )
            if len(potential["coefficients"]) != 1:
                raise NotImplementedError(
                    "ECP requires exactly one scalar coefficient row"
                )
            channel = potential["angular_momentum"][0]
            for power, exponent, coefficient in zip(
                potential["r_exponents"],
                potential["gaussian_exponents"],
                potential["coefficients"][0],
                strict=True,
            ):
                if power > 4:
                    raise NotImplementedError(
                        "ECP radial powers are supported from 0 through 4"
                    )
                terms.append(
                    (
                        atom_index,
                        -1 if channel == local else channel,
                        power,
                        float(exponent),
                        float(coefficient),
                    )
                )
    if terms:
        # Applies to mixed all-electron/ECP atoms too.
        for atom in atoms:
            if any(s.angular_momentum > 2 for s in metadata[atom.atomic_number].shells):
                raise NotImplementedError("ECP execution supports orbital s/p/d shells")
    return tuple(cores), tuple(terms)


@dataclass(frozen=True)
class ECPIntegrals:
    """Separate residual matrices and atom/xyz energy derivatives (Hartree/bohr)."""

    local: np.ndarray
    nonlocal_: np.ndarray
    local_derivative: np.ndarray | None
    nonlocal_derivative: np.ndarray | None
    radial_points: int
    polar_points: int
    backend: str
    model_identity: str

    def quadrature_difference(self, refined):
        """Empirical discretization evidence; not a rigorous error bound."""
        if (
            not isinstance(refined, ECPIntegrals)
            or refined.local.shape != self.local.shape
            or refined.model_identity != self.model_identity
        ):
            raise ValueError(
                "quadrature comparisons require matching ECP models, geometry and matrix layouts"
            )
        result = {
            "local_matrix_max_abs": float(np.max(np.abs(self.local - refined.local))),
            "nonlocal_matrix_max_abs": float(
                np.max(np.abs(self.nonlocal_ - refined.nonlocal_))
            ),
        }
        if self.local_derivative is not None and refined.local_derivative is not None:
            result.update(
                local_derivative_max_abs=float(
                    np.max(np.abs(self.local_derivative - refined.local_derivative))
                ),
                nonlocal_derivative_max_abs=float(
                    np.max(
                        np.abs(self.nonlocal_derivative - refined.nonlocal_derivative)
                    )
                ),
            )
        return result

    @property
    def matrix(self):
        return self.local + self.nonlocal_

    def contract(self, weights):
        """Contract fixed arbitrary real AO weights, including nonsymmetric ones."""
        if self.local_derivative is None:
            raise ValueError("ECP derivatives were not requested")
        if np.iscomplexobj(weights):
            raise TypeError("ECP weights must be real")
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != self.local.shape or not np.isfinite(weights).all():
            raise ValueError("ECP weights must be finite and match the full AO matrix")
        return np.einsum(
            "axij,ij->ax", self.local_derivative + self.nonlocal_derivative, weights
        )


def ecp_integrals(
    atoms,
    basis,
    *,
    charge=0,
    multiplicity=1,
    device="cpu",
    derivatives=True,
    radial_points=160,
    polar_points=32,
):
    """Evaluate native ECP residuals, without SCF or evaluating electron repulsion.

    Quadrature settings are explicit. Compare a refined grid to assess
    discretization error independently from floating-point/backend differences.
    """
    from . import _native
    from .calculator import Atom, Calculator
    from .profiles import canonical_hash

    if not isinstance(basis, BasisSet):
        raise TypeError("ECP evaluation requires an owned canonical BasisSet")
    atoms = tuple(Atom.from_value(a) for a in atoms)
    radial_points = checked_integer(radial_points, "radial points", low=16, high=512)
    polar_points = checked_integer(polar_points, "polar points", low=8, high=96)
    if type(derivatives) is not bool:
        raise TypeError("derivatives must be boolean")
    resolve_ecp(basis, atoms)
    calculator = Calculator(basis=basis, device=device)
    context = ctypes.c_void_p()
    _native.check(
        calculator._library,
        calculator._library.vibeqc_context_create(
            ctypes.byref(calculator._context_descriptor()), ctypes.byref(context)
        ),
    )
    system = None
    try:
        system = calculator._create_native_system(context, atoms, charge, multiplicity)
        n = sum(
            (s.angular_momentum + 1) * (s.angular_momentum + 2) // 2
            if basis.representation == "cartesian"
            else 2 * s.angular_momentum + 1
            for s in basis.shells_for(atoms)
        )
        ncoord = 3 * len(atoms) if derivatives else 0
        output = np.empty((2, 1 + ncoord, n, n), dtype=np.float64)
        _native.check(
            calculator._library,
            calculator._library.vibeqc_system_ecp_integrals(
                context,
                system,
                radial_points,
                polar_points,
                derivatives,
                output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                output.size,
            ),
        )
        if not np.isfinite(output).all():
            raise ArithmeticError("nonfinite ECP integral output")
        output.setflags(write=False)
        return ECPIntegrals(
            output[0, 0],
            output[1, 0],
            output[0, 1:].reshape(len(atoms), 3, n, n) if derivatives else None,
            output[1, 1:].reshape(len(atoms), 3, n, n) if derivatives else None,
            radial_points,
            polar_points,
            device,
            canonical_hash(
                {
                    "basis": calculator.basis_metadata(
                        atoms, charge=charge, multiplicity=multiplicity
                    )["orbital"]["mathematical_identity"],
                    "geometry": [(a.atomic_number, a.position) for a in atoms],
                }
            ),
        )
    finally:
        if system is not None:
            calculator._library.vibeqc_system_destroy(system)
        calculator._library.vibeqc_context_destroy(context)
