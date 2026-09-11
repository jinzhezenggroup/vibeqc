"""Generic DF gradient ABI and independent full public-AO libcint response."""

import ctypes

import numpy as np
from vibeqc import _native
from vibeqc.calculator import Atom

from tools.generate_validation_references import pyscf_molecule


class DfGradientResources(ctypes.Structure):
    """Additive resource descriptor; byte counts exclude caller-owned inputs."""

    _fields_ = [("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32)] + [
        (name, ctypes.c_uint64)
        for name in (
            "host_bytes",
            "device_bytes",
            "host_to_device_bytes",
            "device_to_host_bytes",
            "weight_tile_elements",
            "tiles",
            "uploads",
            "stream_synchronizations",
        )
    ]


def execute_df_gradient(
    orbital,
    auxiliary,
    atoms,
    bar_a,
    bar_m,
    *,
    schedule=0,
    maximum_bytes=128 << 20,
    maximum_tile_elements=0,
    charge=0,
    multiplicity=1,
):
    """Keep weights fixed while measuring the standalone synchronous bridge."""
    library = orbital._library
    pointer = ctypes.POINTER(ctypes.c_double)
    function = library.vibeqc_system_df_gradient_cuda
    function.argtypes = [ctypes.c_void_p] * 3 + [
        pointer,
        ctypes.c_size_t,
        pointer,
        ctypes.c_size_t,
        ctypes.c_uint,
        ctypes.c_size_t,
        ctypes.c_size_t,
        pointer,
        ctypes.c_size_t,
        ctypes.POINTER(DfGradientResources),
    ]
    function.restype = ctypes.c_int
    a, m = (
        np.ascontiguousarray(bar_a, dtype=np.float64),
        np.ascontiguousarray(bar_m, dtype=np.float64),
    )
    if a.ndim != 3 or a.shape[0] != a.shape[1] or m.shape != (a.shape[2], a.shape[2]):
        raise ValueError("weights require full A[mu,nu,P] and M[P,Q]")
    atoms = tuple(Atom.from_value(atom) for atom in atoms)
    result = np.empty((len(atoms), 3))
    context, o, x = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
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
        o = orbital._create_native_system(context, atoms, charge, multiplicity)
        x = auxiliary._create_native_system(context, atoms, charge, multiplicity)
        resources = DfGradientResources(
            ctypes.sizeof(DfGradientResources), _native.ABI_VERSION
        )
        status = function(
            context,
            o,
            x,
            a.ctypes.data_as(pointer),
            a.size,
            m.ctypes.data_as(pointer),
            m.size,
            schedule,
            maximum_bytes,
            maximum_tile_elements,
            result.ctypes.data_as(pointer),
            result.size,
            ctypes.byref(resources),
        )
        if status != _native.STATUS_SUCCESS:
            raise RuntimeError(
                f"DF gradient status {status}: {library.vibeqc_context_get_last_detail(context).decode()}"
            )
        return result, {
            name: getattr(resources, name) for name, _ in resources._fields_[2:]
        }
    finally:
        if x:
            library.vibeqc_system_destroy(x)
        if o:
            library.vibeqc_system_destroy(o)
        library.vibeqc_context_destroy(context)


def reference_df_matrices(orbital_inputs, auxiliary_inputs):
    """All physical-atom derivatives, including independent auxiliary centers.

    Libcint differentiates its own electron coordinates. Moving a basis center
    has the opposite sign. Orbital A/B and auxiliary C owners are accumulated
    independently before any coincident-atom cancellation.
    """
    from pyscf import gto

    def molecule_and_transform(inputs):
        # PySCF groups shells by atom/angular momentum. Recover the requested
        # public AO order explicitly so auxiliary shell permutations are real
        # input changes, rather than accidental changes to the oracle's weights.
        shells = inputs["shells"]
        order = sorted(
            range(len(shells)),
            key=lambda i: (shells[i]["atom_index"], shells[i]["angular_momentum"]),
        )
        mol, scale, _ = pyscf_molecule({**inputs, "shells": [shells[i] for i in order]})
        transform = np.diag(scale) if mol.cart else mol.cart2sph_coeff() * scale
        offsets = mol.ao_loc_nr()
        permutation = np.concatenate(
            [
                np.arange(offsets[order.index(i)], offsets[order.index(i) + 1])
                for i in range(len(shells))
            ]
        )
        return mol, transform[:, permutation]

    o, co = molecule_and_transform(orbital_inputs)
    x, cx = molecule_and_transform(auxiliary_inputs)
    combined = gto.conc_mol(o, x)
    combined.cart = o.cart
    shells = (0, o.nbas, 0, o.nbas, o.nbas, o.nbas + x.nbas)
    a = combined.intor("int3c2e_cart", shls_slice=shells)
    ip1 = combined.intor("int3c2e_ip1_cart", comp=3, shls_slice=shells)
    ip2 = combined.intor("int3c2e_ip2_cart", comp=3, shls_slice=shells)
    m = x.intor("int2c2e_cart")
    metric_ip = x.intor("int2c2e_ip1_cart", comp=3)
    atoms = len(orbital_inputs["coordinates"])
    da = np.zeros((atoms, 3, *a.shape))
    dm = np.zeros((atoms, 3, *m.shape))
    for atom, (_, _, begin, end) in enumerate(
        o.aoslice_by_atom(ao_loc=o.ao_loc_nr(cart=True))
    ):
        da[atom, :, begin:end, :, :] -= ip1[:, begin:end, :, :]
        da[atom, :, :, begin:end, :] -= ip1[:, begin:end, :, :].transpose(0, 2, 1, 3)
    for atom, (_, _, begin, end) in enumerate(
        x.aoslice_by_atom(ao_loc=x.ao_loc_nr(cart=True))
    ):
        da[atom, :, :, :, begin:end] -= ip2[:, :, :, begin:end]
        dm[atom, :, begin:end, :] -= metric_ip[:, begin:end, :]
        dm[atom, :, :, begin:end] -= metric_ip[:, begin:end, :].transpose(0, 2, 1)
    a = np.einsum("ijk,ip,jq,kr->pqr", a, co, co, cx, optimize=True)
    da = np.einsum("axijk,ip,jq,kr->axpqr", da, co, co, cx, optimize=True)
    m = cx.T @ m @ cx
    dm = np.einsum("axij,ip,jq->axpq", dm, cx, cx, optimize=True)
    return a, m, da, dm
