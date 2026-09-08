"""Owned native normalized bases and explicitly ordered spatial derivative jets."""

from __future__ import annotations

import ctypes as ct
import threading
from dataclasses import asdict
from hashlib import sha256

import numpy as np
from vibeqc import Calculator, Primitive, Shell, _native
from vibeqc.profiles import canonical_hash

from tools.vibeqc_posthf.reference import immutable

from .grid import checked_int, owned_atoms

DOUBLE = ct.POINTER(ct.c_double)
SIZE = ct.POINTER(ct.c_size_t)


def pointer(array):
    return array.ctypes.data_as(DOUBLE)


def jet_indices(order):
    """Multi-indices in libcint/CCA order; ordinary derivatives, no factorials.

    A mixed xy slot occurs once. Contracting a full symmetric Hessian therefore
    gives that off-diagonal slot multiplicity two; the stored AO jet does not.
    Spatial differentiation at fixed centers has basis-center derivative -d/dr.
    Moving a physical atom can also move grid points and partition weights;
    that complete nuclear chain rule is a distinct, unimplemented operation.
    """
    checked_int(order, "spatial derivative order", low=0, high=3)
    return tuple(
        (x, degree - x - z, z)
        for degree in range(order + 1)
        for x in range(degree, -1, -1)
        for z in range(degree - x + 1)
    )


class NativeAO:
    """Copy normalized native shell state; input system handles may be released.

    Normalization and real-spherical transformations come directly from
    molecule/basis.cpp. Only public through-f bases are accepted. No SCF run,
    density assumption or integral evaluation is involved in AO preparation.
    """

    _fixed = frozenset(
        (
            "atoms",
            "shells",
            "charge",
            "multiplicity",
            "representation",
            "packed",
            "natom",
            "nprimitive",
            "nao",
            "identity",
            "numeric_bytes",
        )
    )

    def __setattr__(self, name, value):
        if name in self._fixed and name in self.__dict__:
            raise AttributeError(
                "AO scientific state is immutable; prepare a new basis"
            )
        super().__setattr__(name, value)

    def __init__(
        self,
        atoms,
        basis="sto-3g",
        *,
        representation="cartesian",
        charge=0,
        multiplicity=1,
    ):
        self._lock = threading.RLock()
        self._handle = ct.c_void_p()
        self.atoms = owned_atoms(atoms)
        self.charge = checked_int(charge, "charge", low=-(2**31))
        self.multiplicity = checked_int(multiplicity, "multiplicity")
        self.representation = (
            "real_spherical" if representation == "spherical" else representation
        )
        if self.representation not in ("cartesian", "real_spherical"):
            raise ValueError("unsupported AO representation")
        calculator = Calculator(
            basis=basis,
            basis_representation=(
                "spherical" if self.representation == "real_spherical" else "cartesian"
            ),
        )
        shells = calculator._shells_for_atoms(self.atoms)
        if any(s.angular_momentum > 3 for s in shells):
            raise ValueError("AO grids support through f")
        owned = []
        for shell in shells:
            checked_int(shell.atom_index, "shell atom", low=0, high=len(self.atoms) - 1)
            checked_int(shell.angular_momentum, "shell angular momentum", low=0, high=3)
            primitives = []
            for p in shell.primitives:
                if (
                    not np.isfinite(p.exponent)
                    or p.exponent <= 0
                    or not np.isfinite(p.coefficient)
                ):
                    raise ValueError(
                        "AO exponents must be positive and all primitives finite"
                    )
                primitives.append(Primitive(float(p.exponent), float(p.coefficient)))
            owned.append(
                Shell(shell.atom_index, shell.angular_momentum, tuple(primitives))
            )
        self.shells = tuple(owned)
        self._library = lib = calculator._library
        lib.vibeqc_grid_basis_create_v1.argtypes = [
            ct.c_void_p,
            ct.POINTER(ct.c_void_p),
            SIZE,
            ct.c_char_p,
            ct.c_size_t,
        ]
        lib.vibeqc_grid_basis_destroy_v1.argtypes = [ct.c_void_p]
        lib.vibeqc_grid_basis_destroy_v1.restype = None
        lib.vibeqc_grid_basis_pack_v1.argtypes = [
            ct.c_void_p,
            DOUBLE,
            ct.c_size_t,
            ct.c_char_p,
            ct.c_size_t,
        ]
        lib.vibeqc_grid_ao_v1.argtypes = [
            ct.c_void_p,
            DOUBLE,
            ct.c_size_t,
            ct.c_uint,
            ct.c_size_t,
            ct.c_size_t,
            DOUBLE,
            ct.c_size_t,
            ct.c_char_p,
            ct.c_size_t,
        ]
        context, system = ct.c_void_p(), ct.c_void_p()
        dimensions = (ct.c_size_t * 3)()
        _native.check(
            lib,
            lib.vibeqc_context_create(
                ct.byref(calculator._context_descriptor()), ct.byref(context)
            ),
        )
        try:
            system = calculator._create_native_system(
                context, self.atoms, charge, multiplicity, self.shells
            )
            self._call(
                "vibeqc_grid_basis_create_v1",
                system,
                ct.byref(self._handle),
                dimensions,
            )
            self.natom, self.nprimitive, self.nao = map(int, dimensions)
            packed = np.empty(3 * self.natom + 2 * self.nprimitive + 16 * self.nao)
            self._call(
                "vibeqc_grid_basis_pack_v1", self._handle, pointer(packed), packed.size
            )
            self.packed = immutable(packed)
        except Exception:
            self.close()
            raise
        finally:
            if system:
                lib.vibeqc_system_destroy(system)
            lib.vibeqc_context_destroy(context)
        # Native packed state + owned Python copy + original atom/shell records.
        self.numeric_bytes = (
            2 * self.packed.nbytes
            + 32 * self.natom
            + 32 * len(self.shells)
            + 16 * self.nprimitive
        )
        self.identity = canonical_hash(
            {
                "schema": "vibeqc.ao-grid-basis-v1",
                "atoms": [asdict(a) for a in self.atoms],
                "shells": [asdict(s) for s in self.shells],
                "representation": self.representation,
                "charge": self.charge,
                "multiplicity": self.multiplicity,
                "packed_sha256": sha256(self.packed.tobytes()).hexdigest(),
                "screening": "none",
            }
        )

    def _call(self, name, *args):
        error = ct.create_string_buffer(2048)
        if getattr(self._library, name)(*args, error, len(error)):
            raise RuntimeError(error.value.decode())

    def evaluate(
        self, points, order=1, *, ao_begin=0, ao_count=None, budget_bytes=64 << 20
    ):
        """Return owned [jet,point,AO] data for a checked, unscreened tile."""
        jets = jet_indices(order)
        checked_int(ao_begin, "AO begin", low=0, high=self.nao)
        ao_count = self.nao - ao_begin if ao_count is None else ao_count
        checked_int(ao_count, "AO count", low=0, high=self.nao - ao_begin)
        checked_int(budget_bytes, "AO tile budget", high=2**63 - 1)
        raw = np.asarray(points)
        if raw.ndim != 2 or raw.shape[1] != 3 or np.iscomplexobj(raw):
            raise ValueError("real grid points require shape (n,3)")
        elements = len(jets) * len(raw) * ao_count
        capacity = self.numeric_bytes + 16 * elements + 64 * len(raw) + 4096
        if capacity > budget_bytes:
            raise ValueError(
                f"AO tile needs {capacity} numeric bytes; budget is {budget_bytes}"
            )
        points = immutable(raw)
        with self._lock:
            if not self._handle:
                raise RuntimeError("AO basis is closed")
            result = np.empty((len(jets), len(points), ao_count))
            self._call(
                "vibeqc_grid_ao_v1",
                self._handle,
                pointer(points),
                len(points),
                order,
                ao_begin,
                ao_count,
                pointer(result),
                result.size,
            )
            return immutable(result)

    def close(self):
        """Release only this basis; previously returned detached jets survive."""
        with self._lock:
            if self._handle:
                self._library.vibeqc_grid_basis_destroy_v1(self._handle)
                self._handle = ct.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        if hasattr(self, "_lock"):
            self.close()
