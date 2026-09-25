"""Private, live-owner proof for the native #162 stationary-state handoff.

The opaque native token is never reconstructed from Python identity labels.
The batch must outlive validation; replay, replacement and closure revoke old
snapshots. Export is explicit and may transfer the final CUDA matrices.
"""

import ctypes as ct
import threading
import typing
from dataclasses import replace
from hashlib import sha256
from types import MappingProxyType

import numpy as np
from vibeqc_compiler.common.arrays import immutable
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.xc.spec import FunctionalSpec

from . import _native
from .batch import PreparedBatch
from .ks import native_xc_functional_code, scf_domain_for_method


def _scf_xc_points(
    library: typing.Any,
    functional: typing.Any,
    rho: typing.Any,
    gradient: typing.Any,
    tau: typing.Any = None,
    *,
    scales: typing.Any = (1.0, 1.0),
) -> typing.Any:
    """Evaluate the exact native semilocal SCF point model."""
    if type(functional) is bool:
        functional = int(functional)
    if type(functional) is not int or functional not in (0, 1, 2, 3, 4):
        raise TypeError("SCF point evaluator requires functional code 0, 1, 2, 3, or 4")
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
        if functional in (2, 4):
            raise ValueError("meta-GGA point evaluation requires tau[2,n]")
        tau = np.zeros_like(rho)
    raw_tau = np.asarray(tau)
    if np.iscomplexobj(raw_tau) or raw_tau.shape != rho.shape:
        raise ValueError("SCF point evaluator requires real tau[2,n]")
    tau = np.ascontiguousarray(raw_tau, dtype=np.float64)
    output = np.empty((rho.shape[1], 11), dtype=np.float64)
    try:
        evaluate = (
            library.vibeqc_xc_point_batch_v2
            if scales == (1.0, 1.0)
            else library.vibeqc_xc_point_batch_v3
        )
    except AttributeError as error:
        raise NotImplementedError(
            "native library lacks the required semilocal XC point bridge"
        ) from error
    prefix_types = (
        [ct.c_uint32]
        if scales == (1.0, 1.0)
        else [ct.c_uint32, ct.c_double, ct.c_double]
    )
    prefix_values = [functional] if scales == (1.0, 1.0) else [functional, *scales]
    evaluate.argtypes = [
        *prefix_types,
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
            *prefix_values,
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
        # Native r2SCAN publishes the AO kinetic coefficient vtau/2, not
        # the raw feature derivative dE/dtau.
        "kinetic": immutable(output[:, 9:11].T),
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
        "coefficients",
        "ecp_cores",
        "ecp_terms",
        "export_work",
        "functional",
        "grid",
        "grid_provenance",
        "grid_spec",
        "hamiltonian",
        "metadata",
        "method_ir",
        "model_terms",
        "nonlocal_density_policy",
        "values",
    )
    _fixed = frozenset(__slots__)

    def __setattr__(self, name: typing.Any, value: typing.Any) -> None:
        if name in self._fixed and hasattr(self, name):
            raise AttributeError("native KS snapshot provenance is immutable")
        if name == "grid_provenance" and value is not None:
            # Own the mapping as well as the attribute: write-once storage alone
            # does not prevent a caller from mutating model-defining provenance.
            value = MappingProxyType(dict(value))
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
            method_name = self._batch._calculator._method_name
            expected_domain_version = {3: 2, 4: 3}.get(
                native_xc_functional_code(method_name), 1
            )
            if (
                metadata[0] not in (1, 2, 3, 4, 5, 6, 7)
                or metadata[7] != expected_domain_version
            ):
                raise NotImplementedError(
                    "unsupported native KS snapshot/domain version"
                )
            cpu = metadata[0] in (2, 4, 6, 7)
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
        if self.metadata[0] in (2, 3, 4, 5, 6, 7):
            from vibeqc_compiler.dft.grid import GridSpec, grid_policy_provenance

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
            self.grid_provenance = grid_policy_provenance(self.grid_spec)
            self.atomic_weights = take((npoint,))
        else:
            self.grid_spec = None  # CUDA v1 has no prescription suffix.
            self.grid_provenance = None
            self.atomic_weights = None
        self.export_work = MappingProxyType(
            dict(zip(("d2h_bytes", "reads", "synchronizations"), map(int, take((3,)))))
            if self.metadata[0] in (3, 5)
            else {}
        )
        if self.metadata[0] in (4, 5, 7):
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
            # Legacy CUDA v1/v3 do not carry Hamiltonian records. Only the
            # live native proof may promote them to all-electron; absence of
            # an ECP suffix alone is insufficient provenance for CPKS.
            hamiltonian = "all-electron" if self.backend == "cpu" else "unbound"
            proof = getattr(self._library, "vibeqc_ks_snapshot_hamiltonian_v1", None)
            if self.backend == "cuda" and proof is not None:
                proof.argtypes = [ct.c_void_p, ct.c_void_p, ct.POINTER(ct.c_uint32)]
                proof.restype = ct.c_int
                kind = ct.c_uint32()
                _native.check(
                    self._library,
                    proof(self._batch._batch, self._handle, ct.byref(kind)),
                )
                if kind.value == 0:
                    hamiltonian = "all-electron"
            self.hamiltonian = hamiltonian
        self.coefficients = (
            tuple(take((3,))) if self.metadata[0] in (6, 7) else (1.0, 1.0, 0.0)
        )
        options = self._batch._calculator.ks_options
        if (
            options is None
            or options.coefficients != self.coefficients
            or functional
            != native_xc_functional_code(self._batch._calculator._method_name)
            or (options.method_ir.spin == "polarized") != (spins == 2)
        ):
            raise ValueError("native stationary composition mismatch")
        full_method_ir = options.method_ir
        method = self._batch._calculator._method_name
        if method == "pbe-d4-rks":
            from vibeqc_compiler.method import DispersionCorrectionPrimitive

            electronic_primitives = tuple(
                primitive
                for primitive in full_method_ir.primitives
                if not isinstance(primitive, DispersionCorrectionPrimitive)
            )
            if len(electronic_primitives) != 1:
                raise ValueError(
                    "PBE-D4 stationary projection requires one electronic primitive"
                )
            self.method_ir = replace(
                full_method_ir,
                identifier=f"{full_method_ir.identifier}/electronic",
                primitives=electronic_primitives,
            )
            method = "pbe-rks"
        else:
            self.method_ir = full_method_ir
        self.functional = options.functional
        self.model_terms = ()
        self.nonlocal_density_policy = None
        if functional == 4:
            from vibeqc_compiler.dft.nonlocal_policy import (
                MOLECULAR_VV10_DENSITY_POLICY,
            )

            from .ks import ks_range_exchange_parameters

            try:
                read_model = self._library.vibeqc_ks_snapshot_wb97mv_model_v1
            except AttributeError as error:
                raise NotImplementedError(
                    "native library lacks complete WB97M-V snapshot provenance"
                ) from error
            read_model.argtypes = [
                ct.c_void_p,
                ct.c_void_p,
                ct.POINTER(ct.c_double),
                ct.c_size_t,
            ]
            read_model.restype = ct.c_int
            proof = (ct.c_double * 9)()
            _native.check(
                self._library, read_model(self._batch._batch, self._handle, proof, 9)
            )
            nlc = options.execution_plan.nonlocal_correlation
            if nlc is None:
                raise ValueError("WB97M-V snapshot lost its nonlocal primitive")
            expected = (
                *ks_range_exchange_parameters(self.method_ir),
                1.0,
                float(nlc.spec.b),
                float(nlc.spec.c),
                float(nlc.coefficient),
                1.0,
                self._batch._calculator._screening_tolerance,
            )
            if tuple(proof) != expected:
                raise ValueError(
                    "WB97M-V snapshot complete native model disagrees with MethodIR"
                )
            object.__setattr__(self, "model_terms", tuple(proof))
            object.__setattr__(
                self, "nonlocal_density_policy", MOLECULAR_VV10_DENSITY_POLICY
            )
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
        spec = self.functional
        composition_identity = (
            {
                "method_ir": self.method_ir.identity,
                "coefficients": self.coefficients,
                **(
                    {
                        "native_model_terms": self.model_terms,
                        "nonlocal_density_policy": self.nonlocal_density_policy,
                    }
                    if self.model_terms
                    else {}
                ),
            }
            if self.coefficients != (1.0, 1.0, 0.0)
            else {}
        )
        basis_identity = basis.identity
        identity = StationaryKsIdentity(
            method=method,
            model_identity=canonical_hash(
                {
                    "native_owner": owner,
                    "functional": spec.identity,
                    "scf_domain": scf_domain_for_method(method),
                    "grid": grid.identity,
                    **(
                        {"grid_provenance": dict(self.grid_provenance)}
                        if self.grid_provenance is not None
                        else {}
                    ),
                    "basis": basis_identity,
                    **composition_identity,
                    **(
                        {
                            "hamiltonian": self.hamiltonian,
                            "ecp_cores": self.ecp_cores,
                            "ecp_terms": self.ecp_terms,
                        }
                        if self.metadata[0] in (4, 5, 7)
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
            regularization_identity=scf_regularization_identity(method),
            provider_identity=canonical_hash(
                {
                    "provider": f"native-{self.backend}-exact-{'jk' if self.coefficients[2] else 'j'}-fp64",
                    "owner": owner,
                    "device": -1 if self.backend == "cpu" else device,
                    **composition_identity,
                }
            ),
            owner=owner,
            solve_epoch=epoch,
            density_generation=density_generation,
            fock_generation=density_generation,
            orbital_generation=orbital_generation,
            spin=self.method_ir.spin,
            ingredients=spec.ingredients,
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
        if not isinstance(functional, FunctionalSpec):
            raise TypeError("XC point evaluation requires a typed functional")
        if functional.identity != self.functional.identity:
            raise ValueError("XC point functional disagrees with native composition")
        code = native_xc_functional_code(self._batch._calculator._method_name)
        values = _scf_xc_points(
            self._library, code, rho, gradient, tau, scales=self.coefficients[:2]
        )
        self.check_current()
        return values

    def energy(self) -> float:
        """Read the verified energy under this snapshot's current-owner lease."""
        self.check_current()
        read = self._library.vibeqc_ks_snapshot_energy_v1
        read.argtypes = [ct.c_void_p, ct.c_void_p, ct.POINTER(ct.c_double)]
        read.restype = ct.c_int
        value = ct.c_double()
        _native.check(
            self._library, read(self._batch._batch, self._handle, ct.byref(value))
        )
        self.check_current()
        return value.value

    def evaluate_rks_response_points(
        self,
        pbe: bool,
        rho: typing.Any,
        gradient: typing.Any,
        delta_rho: typing.Any,
        delta_gradient: typing.Any,
    ) -> typing.Any:
        """Differentiate the exact SCF point potential in a restricted direction.

        Inputs use total density and Cartesian gradient, with no sigma division
        or low-density clipping. This CPU bridge does not qualify UKS or CUDA.
        """
        return self._evaluate_response_points(
            pbe, rho, gradient, delta_rho, delta_gradient, spins=1
        )

    def prepare_cuda_response(
        self, *, tile_points: int, budget_bytes: int
    ) -> typing.Any:
        """Copy this live state's exact sources into a bounded CUDA XC owner."""
        return _NativeCudaXCPlan(
            self, tile_points=tile_points, budget_bytes=budget_bytes
        )

    def evaluate_uks_response_points(
        self,
        pbe: bool,
        rho: typing.Any,
        gradient: typing.Any,
        delta_rho: typing.Any,
        delta_gradient: typing.Any,
    ) -> typing.Any:
        """Return both spin potentials for a physical UKS density direction.

        Spin-major inputs preserve cross-spin correlation. Empty spins require
        zero directions; the singular exchange Hessian normal to that boundary
        is never silently regularized. This bridge executes on CPU only.
        """
        return self._evaluate_response_points(
            pbe, rho, gradient, delta_rho, delta_gradient, spins=2
        )

    def _evaluate_response_points(
        self,
        pbe: bool,
        rho: typing.Any,
        gradient: typing.Any,
        delta_rho: typing.Any,
        delta_gradient: typing.Any,
        *,
        spins: int,
    ) -> typing.Any:
        """Common checked CPU wire protocol for restricted and spin directions."""
        self.check_current()
        if self.backend != "cpu" or self.metadata[2] != spins:
            raise NotImplementedError(
                "native point response requires matching CPU spin state"
            )
        # The snapshot's functional wire code is not a boolean: newer SCF
        # methods (for example r2SCAN=2) must never be interpreted as PBE.
        if self.metadata[6] not in (0, 1) or self.coefficients != (1.0, 1.0, 0.0):
            raise NotImplementedError(
                "native point response requires unscaled LDA/PBE only"
            )
        if type(pbe) is not bool or pbe != bool(self.metadata[6]):
            raise ValueError("native response functional mismatch")
        values = [np.asarray(x) for x in (rho, gradient, delta_rho, delta_gradient)]
        n = values[0].size // spins
        rho_shape = (n,) if spins == 1 else (2, n)
        gradient_shape = (*rho_shape, 3)
        if n == 0 or any(
            x.shape != shape or np.iscomplexobj(x) or not np.isfinite(x).all()
            for x, shape in zip(
                values,
                (rho_shape, gradient_shape, rho_shape, gradient_shape),
                strict=True,
            )
        ):
            raise ValueError(
                "point response requires finite density and Cartesian gradient spin arrays"
            )
        values = [np.ascontiguousarray(x, dtype=np.float64) for x in values]
        output = np.empty((n, 4 * spins), dtype=np.float64)
        evaluate = (
            self._library.vibeqc_xc_rks_response_batch_v1
            if spins == 1
            else self._library.vibeqc_xc_uks_response_batch_v1
        )
        pointer = ct.POINTER(ct.c_double)
        evaluate.argtypes = [
            ct.c_uint32,
            pointer,
            pointer,
            pointer,
            pointer,
            ct.c_size_t,
            pointer,
            ct.c_size_t,
        ]
        evaluate.restype = ct.c_int
        _native.check(
            self._library,
            evaluate(
                int(pbe),
                *(x.ctypes.data_as(pointer) for x in values),
                n,
                output.ctypes.data_as(pointer),
                output.size,
            ),
        )
        self.check_current()
        return {
            "rho": immutable(output[:, :spins].T),
            "gradient": immutable(
                output[:, spins:].reshape(n, spins, 3).transpose(1, 0, 2)
            ),
        }

    def ecp_derivatives(self) -> typing.Any:
        """Backend-specific provider bound to this live owner's exact ECP model.

        Materializes two atom/xyz/AO-pair arrays. CPU and CUDA execute shared
        generated ECP mathematics with checked two-grid admission.
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


class _NativeCudaXCPlan:
    """Owned XC arena/stream with native token checks before publication.

    Only density directions and final AO matrices cross the host/device seam.
    AO values, base/directional features, point derivatives and assembly execute
    on device. Preparation retains the native reference density once.
    """

    def __init__(
        self, snapshot: NativeKsSnapshot, *, tile_points: int, budget_bytes: int
    ) -> None:
        self._lock = threading.RLock()
        self._handle = ct.c_void_p()
        self.snapshot = snapshot
        self._library = lib = snapshot._library
        snapshot.check_current()
        if snapshot.backend != "cuda" or snapshot.hamiltonian != "all-electron":
            raise NotImplementedError(
                "CUDA response requires a proven all-electron CUDA state"
            )
        if type(tile_points) is not int or not 0 < tile_points < 2**31:
            raise ValueError("CUDA response tile_points must be a positive int32")
        if type(budget_bytes) is not int or not 0 < budget_bytes < 2**64:
            raise ValueError("CUDA response budget must be a positive uint64")
        pointer = ct.POINTER(ct.c_double)
        signatures = {
            "create": [
                ct.c_void_p,
                ct.c_void_p,
                ct.c_size_t,
                ct.c_size_t,
                ct.POINTER(ct.c_void_p),
            ],
            "apply": [
                ct.c_void_p,
                ct.c_void_p,
                pointer,
                ct.c_size_t,
                pointer,
                ct.c_size_t,
            ],
            "diagnostic": [ct.c_void_p, ct.POINTER(ct.c_uint64), ct.c_size_t],
            "destroy": [ct.c_void_p],
        }
        for name, signature in signatures.items():
            function = getattr(lib, f"vibeqc_ks_xc_response_{name}_v1")
            function.argtypes = signature
            function.restype = None if name == "destroy" else ct.c_int
        try:
            self._check(
                lib.vibeqc_ks_xc_response_create_v1(
                    snapshot._batch._batch,
                    snapshot._handle,
                    tile_points,
                    budget_bytes,
                    ct.byref(self._handle),
                )
            )
            self.shape = (
                snapshot.metadata[2],
                snapshot.metadata[1],
                snapshot.metadata[1],
            )
            self.identity = canonical_hash(
                {
                    "owner": "native-cuda-xc-response/v1",
                    "state": snapshot._identity.to_payload(),
                    "tile_points": tile_points,
                    "device": snapshot.metadata[12],
                    "device_bytes": self.diagnostics["device_bytes"],
                }
            )
        except BaseException:
            self.close()
            raise

    def _check(self, status: int) -> None:
        if status == 7:
            raise MemoryError("native CUDA XC response device budget exhausted")
        _native.check(self._library, status, context=self.snapshot._batch._context)

    def _ensure_open(self) -> None:
        if not self._handle:
            raise RuntimeError("native CUDA XC response owner is closed")
        self.snapshot.check_current()

    @property
    def diagnostics(self) -> dict:
        """Actual native arena/transfer counters; no inferred PCIe byte counts."""
        with self._lock:
            self._ensure_open()
            values = (ct.c_uint64 * 12)()
            self._check(
                self._library.vibeqc_ks_xc_response_diagnostic_v1(
                    self._handle, values, 12
                )
            )
            return dict(
                zip(
                    (
                        "device_bytes",
                        "setup_h2d_bytes",
                        "action_h2d_bytes",
                        "d2h_bytes",
                        "synchronizations",
                        "enqueues",
                        "spins",
                        "nbf",
                        "grid_points",
                        "preparation_export_d2h_bytes",
                        "preparation_export_reads",
                        "preparation_export_synchronizations",
                    ),
                    map(int, values),
                    strict=True,
                )
            )

    def apply(self, direction: typing.Any) -> np.ndarray:
        """Publish only a complete finite AO response for the still-live state."""
        raw = np.asarray(direction)
        if (
            raw.shape != self.shape
            or np.iscomplexobj(raw)
            or not np.isfinite(raw).all()
        ):
            raise ValueError("CUDA XC response requires finite real spin AO directions")
        values = np.array(raw, dtype=np.float64, order="C", copy=True)
        output = np.empty_like(values)
        pointer = ct.POINTER(ct.c_double)
        with self._lock:
            self._ensure_open()
            self._check(
                self._library.vibeqc_ks_xc_response_apply_v1(
                    self.snapshot._batch._batch,
                    self._handle,
                    values.ctypes.data_as(pointer),
                    values.size,
                    output.ctypes.data_as(pointer),
                    output.size,
                )
            )
            self._ensure_open()
            if not np.isfinite(output).all():
                raise ArithmeticError("nonfinite CUDA XC response")
        return immutable(output)

    def close(self) -> None:
        """Destroy this arena/stream while preserving the borrowed native state."""
        with self._lock:
            if self._handle:
                self._library.vibeqc_ks_xc_response_destroy_v1(self._handle)
                self._handle = ct.c_void_p()

    def __del__(self) -> None:
        if hasattr(self, "_lock"):
            self.close()
