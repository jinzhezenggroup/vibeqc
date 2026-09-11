"""Call the generic public CUDA gradient ABI and independent libcint oracle."""

import ctypes

import numpy as np
from vibeqc import _native
from vibeqc.calculator import Atom

from tools.generate_validation_references import pyscf_molecule


class GradientResources(ctypes.Structure):
    """Mirror the additive descriptor without assuming another result ABI size."""

    _fields_ = [("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32)] + [
        (name, ctypes.c_uint64)
        for name in (
            "device_bytes",
            "host_numeric_bytes",
            "host_to_device_bytes",
            "device_to_host_bytes",
            "synchronous_uploads",
            "stream_synchronizations",
        )
    ]


def execute_gradient(
    calculator,
    atoms,
    weights,
    *,
    schedule=0,
    maximum_bytes=128 << 20,
    charge=0,
    multiplicity=1,
):
    """Measure the synchronous host bridge; each supplied weight is held fixed."""
    library = calculator._library
    pointer = ctypes.POINTER(ctypes.c_double)
    function = library.vibeqc_system_one_electron_gradient_cuda
    function.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        pointer,
        pointer,
        pointer,
        ctypes.c_size_t,
        ctypes.c_uint,
        ctypes.c_size_t,
        pointer,
        ctypes.c_size_t,
        ctypes.POINTER(GradientResources),
    ]
    function.restype = ctypes.c_int
    blocks = np.ascontiguousarray(weights, dtype=np.float64)
    if blocks.ndim != 3 or blocks.shape[0] != 3 or blocks.shape[1] != blocks.shape[2]:
        raise ValueError("weights require full [S/T/V, AO, AO] matrices")
    atoms = tuple(Atom.from_value(a) for a in atoms)
    output = np.empty((len(atoms), 3))
    context, system = ctypes.c_void_p(), ctypes.c_void_p()
    descriptor = _native.ContextDescriptor(
        ctypes.sizeof(_native.ContextDescriptor),
        _native.ABI_VERSION,
        0,
        _native.BACKEND_CUDA,
    )
    _native.check(
        library,
        library.vibeqc_context_create(ctypes.byref(descriptor), ctypes.byref(context)),
    )
    try:
        system = calculator._create_native_system(context, atoms, charge, multiplicity)
        resources = GradientResources(
            ctypes.sizeof(GradientResources), _native.ABI_VERSION
        )
        status = function(
            context,
            system,
            *(b.ctypes.data_as(pointer) for b in blocks),
            blocks.shape[1] ** 2,
            schedule,
            maximum_bytes,
            output.ctypes.data_as(pointer),
            output.size,
            ctypes.byref(resources),
        )
        if status != _native.STATUS_SUCCESS:
            detail = library.vibeqc_context_get_last_detail(context).decode()
            raise RuntimeError(f"generated gradient status {status}: {detail}")
        return output, {
            name: getattr(resources, name) for name, _ in resources._fields_[2:]
        }
    finally:
        if system:
            library.vibeqc_system_destroy(system)
        library.vibeqc_context_destroy(context)


def reference_matrices(inputs):
    """Independent full AO derivatives including every moving nuclear center."""
    mol, scales, _ = pyscf_molecule(inputs)
    ip = [
        mol.intor(name, comp=3)
        for name in ("int1e_ipovlp", "int1e_ipkin", "int1e_ipnuc")
    ]
    n = mol.nao_nr()
    derivative = np.zeros((len(inputs["coordinates"]), 3, 3, n, n))
    for atom, (_, _, begin, end) in enumerate(mol.aoslice_by_atom()):
        for operator, matrix in enumerate(ip):
            derivative[atom, :, operator, begin:end, :] -= matrix[:, begin:end, :]
            derivative[atom, :, operator, :, begin:end] -= matrix[
                :, begin:end, :
            ].transpose(0, 2, 1)
        with mol.with_rinv_origin(mol.atom_coord(atom)):
            external = mol.intor("int1e_iprinv", comp=3)
        derivative[atom, :, 2] -= mol.atom_charge(atom) * (
            external + external.transpose(0, 2, 1)
        )
    derivative *= scales[None, None, None, :, None] * scales[None, None, None, None, :]
    values = np.array(
        [mol.intor(name) for name in ("int1e_ovlp", "int1e_kin", "int1e_nuc")]
    )
    values *= scales[None, :, None] * scales[None, None, :]
    return values, derivative
