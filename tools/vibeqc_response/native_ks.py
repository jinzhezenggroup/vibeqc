"""Live native CPU/CUDA RKS/UKS handoff into the shared CPKS operator and solver.

The batch and AO basis are borrowed; the integral source and native snapshot
lease are owned. SCF replay, failed replay, or closure revokes this response.
No SCF rerun, recanonicalization, or relabeling of an HF reference occurs here.
"""

from __future__ import annotations

import typing
from time import perf_counter

import numpy as np
from vibeqc._dft_gradient import StationaryDerivativeContract, StationaryKsState
from vibeqc.fock import FockBuildSpec, FockTerm
from vibeqc.ks import resolve_ks_method
from vibeqc.profiles import canonical_hash
from vibeqc_compiler.dft.features import density_features, spin_densities
from vibeqc_compiler.xc.potential import assemble_coefficients

from tools.vibeqc_posthf.reference import ReferenceSnapshot
from tools.vibeqc_posthf.sources import NativeSource

from .backends import NativeJKBackend, _checked_density
from .operators import CPKSResponseOperator, cpks_operator_identity
from .problem import ResponseUnsupported
from .spin_cuda import CudaSpinJKBackend
from .uhf import UHFReferenceSnapshot, UKSResponseOperator, uks_operator_identity
from .xc import FixedDensityXCDerivativeKernel


class _NativeCudaJBackend(CudaSpinJKBackend):
    """Exact native KS Coulomb using the same bounded CUDA provider owner.

    Semilocal response requests no exchange. The unrestricted input adapter
    supplies total density in the first channel, so J is independent of the
    reference's RKS/UKS packing and no unused K contraction executes.
    """

    def _build_spec(self, approximation: str) -> FockBuildSpec:
        if approximation != "exact":
            raise ResponseUnsupported("native KS response requires exact Coulomb")
        return FockBuildSpec(
            spin="unrestricted",
            derivative_order=0,
            exchange=FockTerm(present=False, coefficient=0.0),
        )

    def validate_reference(self, reference: typing.Any) -> typing.Any:
        with self._lock:
            self._ensure_open()
            restricted = isinstance(reference, ReferenceSnapshot)
            if not restricted and not isinstance(reference, UHFReferenceSnapshot):
                raise TypeError("native CUDA J requires a KS response snapshot")
            for name, expected in (
                ("geometry_hash", self.source.geometry_hash),
                ("basis_hash", self.source.basis_hash),
                ("representation", self.source.representation),
                ("hamiltonian_id", self.hamiltonian_id),
                ("algorithm", "KS" if restricted else "UKS"),
                ("nmo" if restricted else "nbf", self.nbf),
            ):
                if getattr(reference, name, None) != expected:
                    raise ValueError(f"native CUDA J/reference {name} mismatch")
            occupied = (
                (reference.electron_count // 2,) * 2
                if restricted
                else (reference.nocc("alpha"), reference.nocc("beta"))
            )
            if occupied != (self.nalpha, self.nbeta):
                raise ValueError("native CUDA J/reference spin occupations mismatch")
        return self

    def coulomb_exchange(self, density: typing.Any) -> typing.Any:
        """J-only implementation of the shared semilocal operator seam."""
        with self._lock:
            self._ensure_open()
            d = _checked_density(density, self.nbf)
            started = perf_counter()
            result = self._plan.evaluate(np.stack([d, np.zeros_like(d)]))
            if (
                result.diagnostics != self._native_diagnostics
                or result.exchange is not None
            ):
                raise RuntimeError("native CUDA J execution identity changed")
            j = result.coulomb
            if j is None or j.shape != d.shape or not np.isfinite(j).all():
                raise RuntimeError("native CUDA J returned invalid Coulomb")
            self.statistics["actions"] += 1
            self.statistics["seconds"] += perf_counter() - started
            self.statistics["host_input_bytes"] += 2 * d.nbytes
            self.statistics["host_jk_result_bytes"] += j.nbytes
            return j, np.zeros_like(j)


class _NativeKSXCKernel(FixedDensityXCDerivativeKernel):
    """Bind the common tile/assembly path to the SCF-domain point differential."""

    def __init__(
        self,
        state: typing.Any,
        spec: typing.Any,
        basis: typing.Any,
        *,
        tile_points: int,
    ) -> None:
        self.state = state
        super().__init__(
            spec,
            basis,
            state.grid,
            state.density[0] if spec.spin == "unpolarized" else state.density,
            tile_points=tile_points,
        )
        self.identity = canonical_hash(
            {
                "kernel": self.identity,
                "point_model": state.identity.regularization_identity,
                "native_state": state.identity.to_payload(),
                "derivative": f"{spec.spin}-cartesian-directional-v1",
            }
        )

    def _response_tile(
        self,
        jets: typing.Any,
        density: typing.Any,
        direction: typing.Any,
        weights: typing.Any,
    ) -> typing.Any:
        ingredients = (
            ("rho", "gradient") if "sigma" in self.spec.ingredients else ("rho",)
        )
        features = density_features(jets, density, ingredients=ingredients)
        delta = density_features(jets, direction, ingredients=ingredients)
        zero = np.zeros((2, jets.shape[1], 3))
        if self.spec.spin == "polarized":
            coefficients = self.state._source.evaluate_uks_response_points(
                "sigma" in self.spec.ingredients,
                features["rho"],
                features.get("gradient", zero),
                delta["rho"],
                delta.get("gradient", zero),
            )
        else:
            coefficients = self.state._source.evaluate_rks_response_points(
                "sigma" in self.spec.ingredients,
                features["rho"].sum(axis=0),
                features.get("gradient", zero).sum(axis=0),
                delta["rho"].sum(axis=0),
                delta.get("gradient", zero).sum(axis=0),
            )
        if "sigma" not in self.spec.ingredients:
            coefficients.pop("gradient")
        return assemble_coefficients(jets, coefficients, weights)


class _NativeCudaXCKernel(_NativeKSXCKernel):
    """The same KS kernel contract with complete XC action execution on CUDA."""

    def __init__(
        self,
        state: typing.Any,
        spec: typing.Any,
        basis: typing.Any,
        *,
        tile_points: int,
        device_budget_bytes: int,
    ) -> None:
        super().__init__(state, spec, basis, tile_points=tile_points)
        self._cuda = state._source.prepare_cuda_response(
            tile_points=tile_points,
            budget_bytes=device_budget_bytes,
        )
        self.identity = canonical_hash(
            {"kernel": self.identity, "cuda_owner": self._cuda.identity}
        )

    def apply_spin(self, delta_density: typing.Any) -> typing.Any:
        """Upload a direction; AO, features, point derivative and assembly stay on GPU."""
        self.state._source.check_current()
        if not self.basis._handle:
            raise ValueError("native CPKS AO basis is closed")
        direction = spin_densities(delta_density, self.basis.nao)
        if self.spec.spin == "unpolarized":
            if not np.array_equal(direction[0], direction[1]):
                raise ResponseUnsupported(
                    "unpolarized response requires equal spin directions"
                )
            direction = direction.sum(axis=0, keepdims=True)
        started = perf_counter()
        result = self._cuda.apply(direction)
        self.statistics["actions"] += 1
        self.statistics["tiles"] += (
            len(self.grid.points) + self.tile_points - 1
        ) // self.tile_points
        self.statistics["seconds"] += perf_counter() - started
        self.statistics["peak_bytes"] = self._cuda.diagnostics["device_bytes"]
        return result

    def close(self) -> None:
        """Release the owned device plan, preserving the borrowed snapshot/basis."""
        self._cuda.close()

    def validate_reference(self, reference: typing.Any) -> typing.Any:
        """Keep device-owner lifetime in the common zero-RHS validation path."""
        self._cuda._ensure_open()
        return super().validate_reference(reference)


class _NativeKSLease:
    """Shared live-state binding and owned/borrowed lifetime for KS response."""

    _spin_blocks: int
    _response_identity: typing.ClassVar[typing.Any]

    @classmethod
    def from_native(
        cls,
        batch: typing.Any,
        basis: typing.Any,
        grid: typing.Any = None,
        *,
        index: int = 0,
        functional: typing.Any = None,
        tile_points: int = 256,
        axis_tile: int = 2,
        device_budget_bytes: int = 128 << 20,
        perturbation_labels: tuple[str, ...] = (),
    ) -> typing.Self:
        """Bind actual converged all-electron KS orbitals without rerunning SCF.

        CUDA owns bounded XC and Coulomb plans under one response budget; the
        borrowed SCF owner, host transforms and Krylov are separate resources.
        """
        state = StationaryKsState.from_native(batch, basis, grid, index=index)
        source = None
        backend = kernel = None
        try:
            spin_method = "rks" if cls._spin_blocks == 1 else "uks"
            if (
                state.identity.method
                not in (f"lda-{spin_method}", f"pbe-{spin_method}")
                or state._source.hamiltonian != "all-electron"
            ):
                raise ResponseUnsupported(
                    f"native CPKS requires all-electron {spin_method.upper()} LDA/PBE"
                )
            if state._source.coefficients != (1.0, 1.0, 0.0):
                raise ResponseUnsupported(
                    f"native CPKS requires unscaled LDA/PBE {spin_method.upper()}"
                )
            _, expected = resolve_ks_method(state.identity.method)
            spec = expected if functional is None else functional
            if spec.identity != state.identity.functional_identity:
                raise ValueError("native CPKS functional identity mismatch")
            source = NativeSource(
                basis.atoms,
                basis=basis.shells,
                charge=basis.charge,
                multiplicity=basis.multiplicity,
                representation=basis.representation,
            )
            overlap, hcore = source.one_electron()
            if not np.allclose(overlap, state.overlap, atol=1e-12, rtol=0):
                raise ValueError("native CPKS integral source overlap mismatch")
            common = {
                "overlap": state.overlap,
                "hcore": hcore,
                "reference_energy": state._source.energy(),
                "scf_residual": state.physical_residual,
                "geometry_hash": source.geometry_hash,
                "basis_hash": source.basis_hash,
                "generation_id": canonical_hash(state.identity.to_payload()),
                "representation": basis.representation,
                "hf_backend": f"native-{state._source.backend}-{spin_method}",
                "functional_identity": spec.identity,
                "grid_identity": state.grid.identity,
            }
            if cls._spin_blocks == 1:
                reference = ReferenceSnapshot(
                    **common,
                    fock=state.fock[0],
                    coefficients=state.coefficients[0],
                    orbital_energies=state.orbital_energies[0],
                    occupations=state.occupations[0],
                    electron_count=source.electron_count,
                    algorithm="KS",
                )
            else:
                # Preserve the independently canonical alpha and beta frames;
                # the shared spin reference never reinterprets them as RHF.
                reference = UHFReferenceSnapshot(
                    **common,
                    algorithm="UKS",
                    **{
                        f"{name}_{spin}": getattr(state, name)[index]
                        for name in (
                            "fock",
                            "coefficients",
                            "orbital_energies",
                            "occupations",
                        )
                        for index, spin in enumerate(("alpha", "beta"))
                    },
                )
            if state._source.backend == "cuda":
                kernel = _NativeCudaXCKernel(
                    state,
                    spec,
                    basis,
                    tile_points=tile_points,
                    device_budget_bytes=device_budget_bytes,
                )
                remaining = (
                    device_budget_bytes - kernel._cuda.diagnostics["device_bytes"]
                )
                if remaining <= 0:
                    raise MemoryError("native CPKS budget cannot hold Coulomb after XC")
                backend = _NativeCudaJBackend(
                    source,
                    device_id=state._source.metadata[12],
                    device_budget_bytes=remaining,
                )
            else:
                backend = NativeJKBackend(source, axis_tile=axis_tile)
                kernel = _NativeKSXCKernel(state, spec, basis, tile_points=tile_points)
            problem = cls.build_problem(
                reference, backend, kernel, perturbation_labels=perturbation_labels
            )
            result = cls(problem, backend, kernel)
            result.state = state
            result._source = source
            result._reference_identity = reference.identity
            result._backend_identity = backend.identity
            result._contract = StationaryDerivativeContract(state.identity)
            result.validate_current()
            return result
        except Exception:
            if kernel is not None and hasattr(kernel, "close"):
                kernel.close()
            if backend is not None and hasattr(backend, "close"):
                backend.close()
            state._source.close()
            if source is not None:
                source.close()
            raise

    def validate_current(self) -> None:
        """Reject revoked state even for a zero RHS or an already-converged guess."""
        self._contract.validate(self.state)
        self._source._check_open()
        if not self.xc_kernel.basis._handle:
            raise ValueError("native CPKS AO basis is closed")
        if (
            self.problem.reference.identity != self._reference_identity
            or self.backend.identity != self._backend_identity
            or self.backend.source is not self._source
            or self.xc_kernel.state is not self.state
            or self.xc_kernel.spec.identity != self.state.identity.functional_identity
            or self.xc_kernel.grid.identity != self.state.identity.grid_identity
            or self.xc_kernel.basis.identity != self.state.identity.basis_identity
            or self.problem.operator_identity
            != self._response_identity(self.backend, self.xc_kernel)
        ):
            raise ValueError("native CPKS state/provider/kernel identity mismatch")
        self.backend.validate_reference(self.problem.reference)
        self.xc_kernel.validate_reference(self.problem.reference)

    @property
    def diagnostics(self) -> dict:
        """Execution boundaries and owned response storage, separate from SCF."""
        self.validate_current()
        cuda = self.state._source.backend == "cuda"
        xc = self.xc_kernel._cuda.diagnostics if cuda else None
        return {
            "execution": "cuda" if cuda else "cpu",
            "coulomb_execution": "cuda" if cuda else "cpu",
            "xc_ao_features_points_assembly": "cuda" if cuda else "cpu",
            "orbital_transforms": "host",
            "krylov_execution": "host",
            "owned_device_bytes": (
                self.backend.device_resident_bytes + xc["device_bytes"]
            )
            if cuda
            else 0,
            "xc": xc,
            "snapshot_export": dict(self.state._source.export_work),
            "memory_scope": "retained response Coulomb/XC allocations only; excludes borrowed SCF, preparation temporaries, host arrays/Krylov, context and library-private memory",
        }

    def close(self) -> None:
        """Revoke the owned response lease; leave the borrowed batch and AO open."""
        for owner in (getattr(self, "xc_kernel", None), getattr(self, "backend", None)):
            if owner is not None and hasattr(owner, "close"):
                owner.close()
        if hasattr(self, "state"):
            self.state._source.close()
        if hasattr(self, "_source"):
            self._source.close()

    def __enter__(self) -> typing.Self:
        self.validate_current()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


class NativeRKSResponse(_NativeKSLease, CPKSResponseOperator):
    """Owned CPU/CUDA LDA/PBE RKS adapter for shared ``solve``/``solve_many``.

    Construct with ``from_native`` after successful SCF. Keep the borrowed
    batch and NativeAO open; this object owns its snapshot lease and integrals.
    The exact native state/grid/provider identity gates every action and solve.
    AO/MO transforms and Krylov remain on the host.
    """

    _spin_blocks = 1
    _response_identity = staticmethod(cpks_operator_identity)

    def induced_fock(
        self, delta_density: typing.Any, *, transpose: bool = False
    ) -> typing.Any:
        """Apply the live native KS density-response map with lease validation."""
        self.validate_current()
        result = super().induced_fock(delta_density, transpose=transpose)
        self.validate_current()
        return result

    def _base_action(
        self, vector: typing.Any, *, transpose: bool = False
    ) -> typing.Any:
        self.validate_current()
        result = super()._base_action(vector, transpose=transpose)
        self.validate_current()
        return result


class NativeUKSResponse(_NativeKSLease, UKSResponseOperator):
    """Live native CPU/CUDA LDA/PBE UKS adapter with coupled alpha/beta response.

    Uses the same lease, point model, XC assembly, spin layout and Krylov
    implementation as the existing restricted/spin consumers. No spin average,
    SCF rerun or synthetic reference substitution is performed.
    """

    _spin_blocks = 2
    _response_identity = staticmethod(uks_operator_identity)

    def apply(self, vector: typing.Any) -> typing.Any:
        self.validate_current()
        result = super().apply(vector)
        self.validate_current()
        return result
