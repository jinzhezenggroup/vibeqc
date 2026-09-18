"""Private, live-owner proof for the native #162 stationary-state handoff.

The opaque native token is never reconstructed from Python identity labels.
The batch must outlive validation; replay, replacement and closure revoke old
snapshots. Export is explicit and may transfer the final CUDA matrices.
"""

import ctypes as ct
from hashlib import sha256
from types import MappingProxyType

import numpy as np
from vibeqc_compiler.common.arrays import immutable
from vibeqc_compiler.common.provenance import canonical_hash

from . import _native
from .batch import PreparedBatch
from .ks import SCF_DOMAIN, resolve_ks_method


class NativeKsSnapshot:
    """Own one native snapshot and check its current batch before consumption."""

    __slots__ = (
        "_arrays",
        "_batch",
        "_handle",
        "_identity",
        "_library",
        "_residual",
        "grid",
        "metadata",
        "values",
    )
    _fixed = frozenset(__slots__)

    def __setattr__(self, name, value):
        if name in self._fixed and hasattr(self, name):
            raise AttributeError("native KS snapshot provenance is immutable")
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in self._fixed:
            raise AttributeError("native KS snapshot provenance is immutable")
        super().__delattr__(name)

    def __init__(self, batch, index):
        if not isinstance(batch, PreparedBatch):
            raise TypeError("stationary snapshot requires a native PreparedBatch")
        if type(index) is not int or not 0 <= index < batch.system_count:
            raise ValueError("stationary snapshot requires an in-range batch index")
        batch._ensure_open()
        self._batch = batch
        self._library = lib = batch._library
        # Keep the pointer as an immutable integer. Exposing a c_void_p here
        # would permit callers to mutate .value even if assignment is blocked.
        self._handle = 0
        try:
            create = lib.vibeqc_ks_snapshot_create_v1
        except AttributeError as error:
            raise NotImplementedError(
                "native library lacks the #162 snapshot bridge"
            ) from error
        create.argtypes = [
            ct.c_void_p,
            ct.c_size_t,
            ct.POINTER(ct.c_void_p),
            ct.POINTER(ct.c_uint64),
            ct.c_size_t,
        ]
        lib.vibeqc_ks_snapshot_check_v1.argtypes = [ct.c_void_p, ct.c_void_p]
        lib.vibeqc_ks_snapshot_copy_v1.argtypes = [
            ct.c_void_p,
            ct.c_void_p,
            ct.POINTER(ct.c_double),
            ct.c_size_t,
        ]
        lib.vibeqc_ks_snapshot_destroy_v1.argtypes = [ct.c_void_p]
        lib.vibeqc_ks_snapshot_destroy_v1.restype = None
        metadata = (ct.c_uint64 * 16)()
        handle = ct.c_void_p()
        try:
            _native.check(
                lib, create(batch._batch, index, ct.byref(handle), metadata, 16)
            )
            object.__setattr__(self, "_handle", handle.value)
            self.metadata = tuple(metadata)
            if metadata[0] != 1 or metadata[7] != 1:
                raise NotImplementedError(
                    "unsupported native KS snapshot/domain version"
                )
            values = np.empty(metadata[15], dtype=np.float64)
            _native.check(
                lib,
                lib.vibeqc_ks_snapshot_copy_v1(
                    batch._batch,
                    self._handle,
                    values.ctypes.data_as(ct.POINTER(ct.c_double)),
                    values.size,
                ),
            )
            self.values = immutable(values)
        except Exception:
            self.close()
            raise

    def check_current(self):
        """Host-only exact token check; numerical equality cannot renew a lease."""
        self._batch._ensure_open()
        if not self._handle or self._library.vibeqc_ks_snapshot_check_v1(
            self._batch._batch, self._handle
        ):
            raise ValueError(
                "stationary KS snapshot is stale or has no current native owner"
            )

    def decode(self, basis, grid):
        """Verify actual AO/grid sources before deriving any Python identities."""
        from vibeqc_compiler.dft.grid import ExplicitGrid

        from ._dft_gradient import (
            StationaryKsIdentity,
            native_ao_geometry_identity,
            xc_geometry_topology_identity,
        )

        self.check_current()
        (
            _,
            n,
            spins,
            natom,
            packed_count,
            npoint,
            pbe,
            _,
            owner,
            epoch,
            density_generation,
            orbital_generation,
            device,
            representation,
            multiplicity,
            _,
        ) = self.metadata
        offset = 0

        def take(shape):
            nonlocal offset
            count = int(np.prod(shape))
            value = self.values[offset : offset + count].reshape(shape)
            offset += count
            return value

        residual, charge = take((2,))
        atoms = take((natom, 4))
        arrays = {}
        for name in (
            "density",
            "fock",
            "coefficients",
            "orbital_energies",
            "occupations",
            "weighted_density",
        ):
            shape = (
                (spins, n)
                if name in ("orbital_energies", "occupations")
                else (spins, n, n)
            )
            arrays[name] = take(shape)
        arrays["overlap"] = take((n, n))
        packed, points, weights, owners = (
            take((packed_count,)),
            take((npoint, 3)),
            take((npoint,)),
            take((npoint,)),
        )
        actual_atoms = np.asarray([[a.atomic_number, *a.position] for a in basis.atoms])
        if (
            basis.nao != n
            or not np.array_equal(basis.packed, packed)
            or not np.array_equal(actual_atoms, atoms)
            or basis.charge != charge
            or basis.multiplicity != multiplicity
            or (basis.representation == "real_spherical") != bool(representation)
        ):
            raise ValueError("native stationary basis/overlap source mismatch")
        if grid is None:
            grid = ExplicitGrid(
                points,
                weights,
                tuple(map(int, owners)),
                {"source": "native-ks-snapshot-v1", "owner": owner},
            )
        if not all(
            np.array_equal(a, b)
            for a, b in (
                (grid.points, points),
                (grid.weights, weights),
                (grid.owners, owners),
            )
        ):
            raise ValueError("native stationary grid source mismatch")
        self.grid = grid
        method = ("pbe" if pbe else "lda") + ("-rks" if spins == 1 else "-uks")
        _, spec = resolve_ks_method(method)
        basis_identity = basis.identity
        identity = StationaryKsIdentity(
            method=method,
            model_identity=canonical_hash(
                {
                    "native_owner": owner,
                    "functional": spec.identity,
                    "scf_domain": SCF_DOMAIN,
                    "grid": grid.identity,
                    "basis": basis_identity,
                }
            ),
            geometry_identity=native_ao_geometry_identity(basis),
            basis_identity=basis_identity,
            overlap_identity=canonical_hash(
                {
                    "basis": basis_identity,
                    "overlap": sha256(arrays["overlap"].tobytes()).hexdigest(),
                }
            ),
            grid_identity=grid.identity,
            topology_identity=xc_geometry_topology_identity(basis, grid),
            functional_identity=spec.identity,
            # Native SCF's scaled tail/spin extension is a distinct energy
            # model from generated interior-v1. Never relabel it to authorize
            # unsupported generated molecular derivatives.
            regularization_identity=canonical_hash({"scf_domain": SCF_DOMAIN}),
            provider_identity=canonical_hash(
                {
                    "provider": "native-cuda-exact-j-fp64",
                    "owner": owner,
                    "device": device,
                }
            ),
            owner=owner,
            solve_epoch=epoch,
            density_generation=density_generation,
            fock_generation=density_generation,
            orbital_generation=orbital_generation,
        )
        self._identity, self._arrays, self._residual = (
            identity,
            MappingProxyType(arrays),
            residual,
        )
        return dict(
            identity=identity,
            **arrays,
            physical_residual=float(residual),
            successful=True,
            converged=True,
            physical=True,
            _source=self,
        )

    def validate(self, state):
        """Reject copied labels and even self-consistent replacement matrices."""
        self.check_current()
        if state.identity != self._identity:
            raise ValueError("native stationary state identity mismatch")
        if state.physical_residual != self._residual or not all(
            np.array_equal(getattr(state, name), value)
            for name, value in self._arrays.items()
        ):
            raise ValueError("native stationary snapshot content mismatch")

    def close(self):
        if self._handle:
            handle = self._handle
            # Revocation may clear the binding internally; public assignment
            # must never attach a fresh lease to this snapshot's old contents.
            object.__setattr__(self, "_handle", 0)
            self._library.vibeqc_ks_snapshot_destroy_v1(handle)

    def __del__(self):
        if hasattr(self, "_handle"):
            self.close()
