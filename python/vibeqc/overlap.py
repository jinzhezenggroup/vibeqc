"""Bounded native CPU overlap matrices for independent AO bases."""

import ctypes

import numpy as np

from . import _native
from .basis_capabilities import require_basis
from .calculator import Atom


def cross_overlap(
    target,
    source,
    atoms,
    *,
    source_atoms=None,
    charge=0,
    multiplicity=1,
    maximum_bytes=128 << 20,
):
    """Return row-major ``<target AO | source AO>`` in the public AO conventions.

    ``target`` and ``source`` are calculators supplying their orbital bases.
    Geometry defaults to the same ordered atoms; independent source positions
    are allowed for raw-overlap diagnostics. The native CPU evaluator shares
    normalized Gaussian mathematics and spherical expansions with the existing
    integral oracle and allocates no ERIs or nuclear derivatives. Accelerator
    calculators still use this explicit host setup operation.
    """
    atoms = tuple(Atom.from_value(a) for a in atoms)
    source_atoms = (
        atoms
        if source_atoms is None
        else tuple(Atom.from_value(a) for a in source_atoms)
    )
    if type(maximum_bytes) is not int or maximum_bytes < 1:
        raise ValueError("maximum_bytes must be a positive integer")
    if target._library._handle != source._library._handle:
        raise ValueError(
            "cross overlap requires calculators using the same native library"
        )
    dimensions = []
    for calculator, geometry in ((target, atoms), (source, source_atoms)):
        require_basis(
            calculator._basis,
            geometry,
            backend="cpu",
            operator="overlap",
            derivative_order=0,
            role="orbital",
            representation=calculator._representation_name,
        )
        shells = calculator._shells_for_atoms(geometry)
        dimensions.append(
            sum(
                2 * shell.angular_momentum + 1
                if calculator._basis_representation == _native.BASIS_SPHERICAL
                else (shell.angular_momentum + 1) * (shell.angular_momentum + 2) // 2
                for shell in shells
            )
        )
    if 8 * dimensions[0] * dimensions[1] > maximum_bytes:
        raise MemoryError("rectangular overlap exceeds maximum_bytes")
    output = np.empty(dimensions, dtype=np.float64)
    library = target._library
    context, left, right = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    descriptor = _native.ContextDescriptor(
        ctypes.sizeof(_native.ContextDescriptor),
        _native.ABI_VERSION,
        0,
        _native.BACKEND_CPU_REFERENCE,
    )
    _native.check(
        library,
        library.vibeqc_context_create(ctypes.byref(descriptor), ctypes.byref(context)),
    )
    try:
        left = target._create_native_system(context, atoms, charge, multiplicity)
        right = source._create_native_system(
            context, source_atoms, charge, multiplicity
        )
        status = library.vibeqc_system_cross_overlap_cpu(
            context,
            left,
            right,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            output.size,
        )
        _native.check(library, status)
    finally:
        if right:
            library.vibeqc_system_destroy(right)
        if left:
            library.vibeqc_system_destroy(left)
        library.vibeqc_context_destroy(context)
    return output
