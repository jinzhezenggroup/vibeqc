"""Private, live-owner proof for the native #162 stationary-state handoff.

The opaque native token is never reconstructed from Python identity labels.
The batch must outlive validation; replay, replacement and closure revoke old
snapshots. Export is explicit and may transfer the final CUDA matrices.
"""

import ctypes as ct
import typing
from hashlib import sha256
from types import MappingProxyType

import numpy as np
from vibeqc_compiler.common.arrays import immutable
from vibeqc_compiler.common.provenance import canonical_hash

from . import _native
from .batch import PreparedBatch
from .ks import SCF_DOMAIN, resolve_ks_method


def _scf_xc_points(
    library: typing.Any,
    functional: typing.Any,
    rho: typing.Any,
    gradient: typing.Any,
    tau: typing.Any = None,
) -> typing.Any:
    """Evaluate the exact native semilocal SCF point model."""
    if type(functional) is bool:
        functional = int(functional)
    if type(functional) is not int or functional not in (0, 1, 2):
        raise TypeError("SCF point evaluator requires functional code 0, 1, or 2")
    raw_rho, raw_gradient = np.asarray(rho), np.asarray(gradient)
    if (
        np.iscomplexobj(raw_rho)
        or np.iscomplexobj(raw_gradient)
        or raw_rho.ndim != 2
        or raw_rho.shape[0] != 2
        or raw_gradient.shape != (2, raw_rho.shape[1], 3)
        or raw_rho.shape[1] == 0
    ):
        raise ValueError("SCF point evaluator requires rho[2,n] and gradient[2,n,3]")
    rho = np.ascontiguousarray(raw_rho, dtype=np.float64)
    gradient = np.ascontiguousarray(raw_gradient, dtype=np.float64)
    if tau is None:
        if functional == 2:
            raise ValueError("r2SCAN point evaluation requires tau[2,n]")
        tau = np.zeros_like(rho)
    raw_tau = np.asarray(tau)
    if np.iscomplexobj(raw_tau) or raw_tau.shape != rho.shape:
        raise ValueError("SCF point evaluator requires real tau[2,n]")
    tau = np.ascontiguousarray(raw_tau, dtype=np.float64)
    output = np.empty((rho.shape[1], 11), dtype=np.float64)
    try:
        evaluate = library.vibeqc_xc_point_batch_v2
    except AttributeError as error:
        raise NotImplementedError(
            "native library lacks the semilocal XC point bridge v2"
        ) from error
    evaluate.argtypes = [
        ct.c_uint32,
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_double),
        ct.c_size_t,
        ct.POINTER(ct.c_double),
        ct.c_size_t,
    ]
    evaluate.restype = ct.c_int
    _native.check(
        library,
        evaluate(
            functional,
            rho.ctypes.data_as(ct.POINTER(ct.c_double)),
            gradient.ctypes.data_as(ct.POINTER(ct.c_double)),
            tau.ctypes.data_as(ct.POINTER(ct.c_double)),
            rho.shape[1],
            output.ctypes.data_as(ct.POINTER(ct.c_double)),
            output.size,
        ),
    )
    return {
        "energy": immutable(output[:, 0]),
        "rho": immutable(output[:, 1:3].T),
        "gradient": immutable(output[:, 3:9].reshape(-1, 2, 3).transpose(1, 0, 2)),
        "tau": immutable(output[:, 9:11].T),
    }


class NativeKsSnapshot:
    """Own one native snapshot and check its current batch before consumption."""

    __slots__ = (
        "_arrays",
        "_batch",
        "_handle",
        "_identity",
        "_library",
        "_residual",
        "atomic_weights",
        "backend",
        "ecp_cores",
        "ecp_terms",
        "export_work",
        "grid",
        "grid_spec",
        "hamiltonian",
        "metadata",
        "values",
    )
    _fixed = frozenset(__slots__)

    def __setattr__(self, name: typing.Any, value: typing.Any) -> None:
        if name in self._fixed and hasattr(self, name):
            raise AttributeError("native KS snapshot provenance is immutable")
        super().__setattr__(name, value)

    def __delattr__(self, name: typing.Any) -> None:
        if name in self._fixed:
            raise AttributeError("native KS snapshot provenance is immutable")
        super().__delattr__(name)

    def __init__(self, batch: typing.Any, index: typing.Any) -> None:
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
            if metadata[0] not in (1, 2, 3, 4, 5) or metadata[7] != 1:
                raise NotImplementedError(
                    "unsupported native KS snapshot/domain version"
                )
            cpu = metadata[0] in (2, 4)
            if (metadata[12] == 2**64 - 1) != cpu:
                raise ValueError("native KS snapshot backend/device mismatch")
            self.backend = "cpu" if cpu else "cuda"
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

    def check_current(self) -> None:
        """Host-only exact token check; numerical equality cannot renew a lease."""
        self._batch._ensure_open()
        if not self._handle or self._library.vibeqc_ks_snapshot_check_v1(
            self._batch._batch, self._handle
        ):
            raise ValueError(
                "stationary KS snapshot is stale or has no current native owner"
            )

    def decode(self, basis: typing.Any, grid: typing.Any) -> typing.Any:
        """Verify actual AO/grid sources before deriving any Python identities."""
        from vibeqc_compiler.dft.grid import ExplicitGrid

        from ._dft_gradient import (
            StationaryKsIdentity,
            native_ao_geometry_identity,
            scf_regularization_identity,
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
            functional,
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

        def take(shape: typing.Any) -> typing.Any:
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
        if self.metadata[0] in (2, 3, 4, 5):
            from vibeqc_compiler.dft.grid import GridSpec

            version, radial, polar, azimuth, iterations, tolerance = take((6,))
            radii = take((119,))
            self.grid_spec = GridSpec(
                version=int(version),
                radial_points=int(radial),
                angular_polar=int(polar),
                angular_azimuth=int(azimuth),
                partition_iterations=int(iterations),
                coincident_tolerance=float(tolerance),
                element_radii=tuple(
                    (z, float(r)) for z, r in enumerate(radii) if z and r
                ),
            )
            self.atomic_weights = take((npoint,))
        else:
            self.grid_spec = None  # CUDA v1 has no prescription suffix.
            self.atomic_weights = None
        self.export_work = MappingProxyType(
            dict(zip(("d2h_bytes", "reads", "synchronizations"), map(int, take((3,)))))
            if self.metadata[0] in (3, 5)
            else {}
        )
        if self.metadata[0] in (4, 5):
            cores = take((natom,))
            count = float(take((1,))[0])
            if not np.isfinite(count) or count < 1 or not count.is_integer():
                raise ValueError("invalid native ECP term count")
            terms = take((int(count), 5))
            if (
                not np.isfinite(cores).all()
                or np.any(cores != np.floor(cores))
                or np.any(cores < 0)
                or np.any(cores >= atoms[:, 0])
            ):
                raise ValueError("invalid native ECP core counts")
            if not np.isfinite(terms).all():
                raise ValueError("nonfinite native ECP terms")
            self.ecp_cores = tuple(map(int, cores))
            self.ecp_terms = tuple(tuple(row) for row in terms)
            self.hamiltonian = "scalar-semilocal-ecp"
        else:
            self.ecp_cores = (0,) * natom
            self.ecp_terms = ()
            # CUDA v1/v3 do not export ECP Hamiltonian records. Their existing
            # gradient consumer independently rejects core-adjusted occupations;
            # this CPU extension must not label such snapshots all-electron.
            self.hamiltonian = "all-electron" if self.backend == "cpu" else "unbound"
        if offset != len(self.values):
            raise ValueError("native KS snapshot wire length mismatch")
        if self.hamiltonian != "unbound" and not np.isclose(
            arrays["occupations"].sum(),
            atoms[:, 0].sum() - sum(self.ecp_cores) - charge,
            atol=1e-10,
            rtol=0,
        ):
            raise ValueError("native stationary effective-charge occupation mismatch")
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
        functional_names = {0: "lda", 1: "pbe", 2: "r2scan"}
        try:
            family = functional_names[functional]
        except KeyError as error:
            raise NotImplementedError("unsupported native KS functional id") from error
        method = family + ("-rks" if spins == 1 else "-uks")
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
                    **(
                        {
                            "hamiltonian": self.hamiltonian,
                            "ecp_cores": self.ecp_cores,
                            "ecp_terms": self.ecp_terms,
                        }
                        if self.metadata[0] in (4, 5)
                        else {}
                    ),
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
            # The derivative bridge consumes this exact SCF point model;
            # interior-v1 remains a separate diagnostic contract.
            regularization_identity=scf_regularization_identity(),
            provider_identity=canonical_hash(
                {
                    "provider": f"native-{self.backend}-exact-j-fp64",
                    "owner": owner,
                    "device": -1 if self.backend == "cpu" else device,
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

    def evaluate_xc_points(
        self,
        functional: typing.Any,
        rho: typing.Any,
        gradient: typing.Any,
        tau: typing.Any = None,
    ) -> typing.Any:
        """Return SCF-domain point energy and Cartesian first derivatives."""
        self.check_current()
        values = _scf_xc_points(self._library, functional, rho, gradient, tau)
        self.check_current()
        return values

    def ecp_derivatives(self) -> typing.Any:
        """Backend-specific provider bound to this live owner's exact ECP model.

        Materializes two atom/xyz/AO-pair arrays. CUDA uses only generated CUDA
        ECP derivatives; CPU explicitly uses checked native two-grid ECP.
        Public wrappers admit and reserve this dense export before execution.
        """
        self.check_current()
        if self.hamiltonian != "scalar-semilocal-ecp":
            raise NotImplementedError(
                "ECP derivative snapshot requires a bound ECP state"
            )
        evaluate = self._library.vibeqc_ks_snapshot_ecp_derivatives_v1
        evaluate.argtypes = [
            ct.c_void_p,
            ct.c_void_p,
            ct.POINTER(ct.c_double),
            ct.c_size_t,
        ]
        evaluate.restype = ct.c_int
        n, natom = self.metadata[1], self.metadata[3]
        output = np.empty((2, natom, 3, n, n), dtype=np.float64)
        _native.check(
            self._library,
            evaluate(
                self._batch._batch,
                self._handle,
                output.ctypes.data_as(ct.POINTER(ct.c_double)),
                output.size,
            ),
        )
        self.check_current()
        if not np.isfinite(output).all():
            raise ArithmeticError("nonfinite native ECP derivatives")
        return immutable(output)

    def validate(self, state: typing.Any) -> None:
        """Reject copied labels and even self-consistent replacement matrices."""
        self.check_current()
        if state.identity != self._identity:
            raise ValueError("native stationary state identity mismatch")
        if state.physical_residual != self._residual or not all(
            np.array_equal(getattr(state, name), value)
            for name, value in self._arrays.items()
        ):
            raise ValueError("native stationary snapshot content mismatch")

    def close(self) -> None:
        if self._handle:
            handle = self._handle
            # Revocation may clear the binding internally; public assignment
            # must never attach a fresh lease to this snapshot's old contents.
            object.__setattr__(self, "_handle", 0)
            self._library.vibeqc_ks_snapshot_destroy_v1(handle)

    def __del__(self) -> None:
        if hasattr(self, "_handle"):
            self.close()
