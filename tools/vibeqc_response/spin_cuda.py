"""Spin-resolved CUDA J/K with the existing prepared Fock provider.

The provider contracts signed alpha/beta densities on device. Orbital
transforms, canonicalization, result storage and shared Krylov remain on host.
"""

from __future__ import annotations

import threading
import time
import typing
import uuid
from contextlib import ExitStack
from copy import deepcopy

import numpy as np
from vibeqc.fock import FockBuildSpec, FockPlan
from vibeqc.profiles import canonical_hash
from vibeqc_compiler.dft import NativeAO

from tools.vibeqc_posthf.export import _canonical_orbitals
from tools.vibeqc_posthf.sources import NativeSource

from .backends import _checked_density
from .direct_cuda import _source_record


class CudaSpinJKBackend:
    """Own exact or density-fitted CUDA J/K for a declared UHF source.

    One native unrestricted evaluation returns J[Da+Db], K[Da], K[Db]. Raw
    matrices avoid subtracting hcore from Fock, which loses small response
    directions through cancellation. No CPU J/K fallback is permitted.

    The NativeSource is borrowed. The copied native Fock plan is owned here.
    DF identity binds the actual plan's orbital/auxiliary basis, cutoff and
    retained metric rank; it is not interchangeable with conventional HF or
    a legacy MetricFactor identity. ``export_reference`` constructs a native
    converged reference with this exact Hamiltonian.
    """

    def __init__(
        self,
        source: typing.Any,
        *,
        approximation: str = "exact",
        metric_relative_threshold: float = 1e-10,
        device_id: int = 0,
        device_budget_bytes: int = 64 << 20,
    ) -> None:
        self._lock = threading.RLock()
        self._plan = None
        if not isinstance(source, NativeSource):
            raise TypeError("CUDA spin response requires a NativeSource")
        if approximation not in ("exact", "density_fitted"):
            raise ValueError("spin J/K approximation must be exact or density_fitted")
        if type(device_id) is not int or not 0 <= device_id < 2**31:
            raise ValueError("device_id must be a nonnegative int32")
        if type(device_budget_bytes) is not int or not 0 < device_budget_bytes < 2**64:
            raise ValueError("device_budget_bytes must be a positive uint64")
        self.source = source
        self._source_record = _source_record(source)
        self._auxiliary_record = (source.auxiliary_hash, source.naux)
        self.nbf = source.nbf
        self.device_id = device_id
        self.approximation = approximation
        self.nalpha = (source.electron_count + source.multiplicity - 1) // 2
        self.nbeta = source.electron_count - self.nalpha
        if not 0 <= self.nbeta <= self.nalpha <= self.nbf or not self.nalpha:
            raise ValueError("source spin occupations exceed the AO space")
        fitted = approximation == "density_fitted"
        if fitted and not source.naux:
            raise ValueError("DF spin response requires an explicit auxiliary basis")
        shells = (*source.shells, *(source.auxiliary_shells if fitted else ()))
        if any(shell.angular_momentum > 3 for shell in shells):
            raise NotImplementedError("CUDA spin response supports s/p/d/f shells only")
        spec = self._build_spec(approximation)
        started = time.perf_counter()
        try:
            with ExitStack() as owners:

                def ao(shell_records: typing.Any) -> typing.Any:
                    return owners.enter_context(
                        NativeAO(
                            source.atoms,
                            basis=shell_records,
                            representation=source.representation,
                            charge=source.charge,
                            multiplicity=source.multiplicity,
                        )
                    )

                basis = ao(source.shells)
                auxiliary = ao(source.auxiliary_shells) if fitted else None
                self._plan = FockPlan(
                    basis,
                    spec,
                    auxiliary=auxiliary,
                    device="cuda",
                    device_id=device_id,
                    screening_tolerance=0.0,
                    metric_relative_threshold=metric_relative_threshold,
                    device_budget_bytes=device_budget_bytes,
                )
            diag = self._plan.diagnostics
            if (
                diag["backend"] != "cuda"
                or diag["source_schedule"] != "cuda_independent"
                or diag["precision"] != "float64"
                or diag["resolved"] != spec.to_dict()
                or diag["screening_tolerance"] != 0.0
                or diag["nbf"] != self.nbf
                or bool(diag["auxiliary_rank"]) != fitted
                or (fitted and not 0 < diag["auxiliary_rank"] <= source.naux)
                or (
                    fitted
                    and diag["metric_relative_threshold"] != metric_relative_threshold
                )
            ):
                raise RuntimeError("CUDA spin response provider semantics mismatch")
            if not 0 < diag["device_bytes"] <= device_budget_bytes:
                raise RuntimeError("CUDA spin response provider device budget mismatch")
            self._native_diagnostics = deepcopy(diag)
            self.device_resident_bytes = int(diag["device_bytes"])
            self.hamiltonian_id = (
                "prepared-spin-df:"
                + canonical_hash(
                    {
                        "plan": self._plan.identity,
                        "auxiliary_rank": diag["auxiliary_rank"],
                        "metric_convention": "square-symmetric-thresholded-inverse-square-root",
                    }
                )
                if fitted
                else "conventional-unscreened"
            )
            self.identity = canonical_hash(
                {
                    "backend": "cuda-spin-fock-jk/v1",
                    "source": self._source_record,
                    "auxiliary": self._auxiliary_record,
                    "hamiltonian": self.hamiltonian_id,
                    "plan": self._plan.identity,
                    "execution": self._plan.execution_identity,
                    "device_budget_bytes": device_budget_bytes,
                }
            )
            self.statistics = {
                "actions": 0,
                "seconds": 0.0,
                "setup_seconds": time.perf_counter() - started,
                "peak_bytes": self.device_resident_bytes,
                # These are host API payloads, not measured CUDA transfer bytes.
                "host_input_bytes": 0,
                "host_jk_result_bytes": 0,
            }
        except BaseException:
            self.close()
            raise

    def _ensure_open(self) -> None:
        if self._plan is None:
            raise RuntimeError("CUDA spin response backend is closed")
        if (
            _source_record(self.source) != self._source_record
            or (self.source.auxiliary_hash, self.source.naux) != self._auxiliary_record
        ):
            raise ValueError("CUDA spin response source identity changed")
        self._plan._ensure_open()

    def _build_spec(self, approximation: str) -> FockBuildSpec:
        """HF owns both terms; a native KS subclass requests only Coulomb."""
        return FockBuildSpec.hf(
            "unrestricted",
            coulomb=approximation,
            exchange=approximation,
            derivative_order=0,
        )

    @property
    def diagnostics(self) -> typing.Any:
        """Detached execution provenance, with a bounded allocation scope."""
        with self._lock:
            self._ensure_open()
            return {
                "provider": deepcopy(self._native_diagnostics),
                "hamiltonian_id": self.hamiltonian_id,
                "jk_execution": "cuda",
                "matrix_storage": "host",
                "orbital_transforms": "host",
                "krylov_execution": "host",
                "gpu_resident_response": False,
                "reference_export_preparation": "CPU overlap/hcore and canonicalization; native CUDA J/K SCF",
                "memory_scope": "retained J/K device allocations; excludes preparation temporaries, reference-export SCF/eigensolver caches, host matrices, Krylov and CUDA context",
            }

    def validate_reference(self, reference: typing.Any) -> typing.Any:
        """Reject changed geometry, Hamiltonian or occupations, even at equal N."""
        from .uhf import UHFReferenceSnapshot

        with self._lock:
            self._ensure_open()
            if not isinstance(reference, UHFReferenceSnapshot):
                raise TypeError("CUDA spin response requires UHFReferenceSnapshot")
            for name, expected in (
                ("geometry_hash", self._source_record[1]),
                ("basis_hash", self._source_record[2]),
                ("representation", self._source_record[3]),
                ("nbf", self.nbf),
                ("hamiltonian_id", self.hamiltonian_id),
                ("algorithm", "UHF"),
            ):
                if getattr(reference, name, None) != expected:
                    raise ValueError(f"CUDA spin backend/reference {name} mismatch")
            if (reference.nocc("alpha"), reference.nocc("beta")) != (
                self.nalpha,
                self.nbeta,
            ):
                raise ValueError("CUDA spin backend/reference occupations mismatch")
        return self

    def spin_coulomb_exchange(
        self, alpha_density: typing.Any, beta_density: typing.Any
    ) -> typing.Any:
        """Return J[Da+Db], K[Da], K[Db] for symmetric signed directions."""
        with self._lock:
            self._ensure_open()
            density = np.stack(
                [
                    _checked_density(alpha_density, self.nbf),
                    _checked_density(beta_density, self.nbf),
                ]
            )
            started = time.perf_counter()
            result = self._plan.evaluate(density, derivative=False)
            if result.diagnostics != self._native_diagnostics:
                raise RuntimeError("CUDA spin response execution identity changed")
            for matrix, shape in (
                (result.coulomb, (self.nbf, self.nbf)),
                (result.exchange, (2, self.nbf, self.nbf)),
            ):
                if (
                    matrix is None
                    or matrix.shape != shape
                    or matrix.dtype != np.dtype(np.float64)
                    or not np.isfinite(matrix).all()
                ):
                    raise RuntimeError("CUDA spin response returned invalid J/K")
            self.statistics["actions"] += 1
            self.statistics["seconds"] += time.perf_counter() - started
            self.statistics["host_input_bytes"] += density.nbytes
            self.statistics["host_jk_result_bytes"] += (
                result.coulomb.nbytes + result.exchange.nbytes
            )
            return result.coulomb, result.exchange[0], result.exchange[1]

    def resident_response(
        self,
        problem: typing.Any,
        *,
        vector_slots: int = 128,
        device_budget_bytes: int = 128 << 20,
    ) -> typing.Any:
        """Create an exact unrestricted resident response owner.

        The owner shares this backend's direct CUDA stream and plan. Density
        fitted plans remain on the existing host-orchestrated path until a
        separate resident DF ABI is qualified.
        """
        with self._lock:
            self._ensure_open()
            from .resident_uhf_cuda import CudaResidentUHFResponse

            return CudaResidentUHFResponse(
                self,
                problem,
                vector_slots=vector_slots,
                device_budget_bytes=device_budget_bytes,
            )

    def export_reference(
        self,
        *,
        max_iterations: int = 100,
        energy_tolerance: float = 1e-12,
        density_tolerance: float = 1e-11,
    ) -> typing.Any:
        """Run the shared native SCF and verify its canonical spin snapshot.

        This explicit convenience solve never runs during response actions.
        Export uses CPU overlap/hcore metadata and canonicalization; both SCF
        and verification Fock contractions use this same prepared CUDA plan.
        The immutable snapshot is not a borrowed mutable calculator epoch.
        """
        from .uhf import UHFReferenceSnapshot

        with self._lock:
            self._ensure_open()
            started = time.perf_counter()
            result = self._plan.solve(
                compute_forces=False,
                max_iterations=max_iterations,
                energy_tolerance=energy_tolerance,
                density_tolerance=density_tolerance,
            )
            overlap, hcore = self.source.one_electron()
            evaluation = self._plan.evaluate(result.density)
            state = {}
            canonical = []
            residual = 0.0
            for spin, count, density, fock in zip(
                ("alpha", "beta"),
                (self.nalpha, self.nbeta),
                result.density,
                evaluation.fock,
                strict=True,
            ):
                energies, coefficients = _canonical_orbitals(overlap, fock)
                occupations = (np.arange(self.nbf) < count).astype(np.float64)
                canonical.append((coefficients * occupations) @ coefficients.T)
                residual = max(
                    residual,
                    float(
                        np.max(
                            np.abs(fock @ density @ overlap - overlap @ density @ fock)
                        )
                    ),
                )
                state.update(
                    {
                        f"coefficients_{spin}": coefficients,
                        f"orbital_energies_{spin}": energies,
                        f"occupations_{spin}": occupations,
                        f"fock_{spin}": fock,
                    }
                )
            density_drift = float(np.max(np.abs(np.stack(canonical) - result.density)))
            rebuilt = self._plan.evaluate(np.stack(canonical))
            fock_drift = float(np.max(np.abs(rebuilt.fock - evaluation.fock)))
            snapshot = UHFReferenceSnapshot(
                **state,
                overlap=overlap,
                hcore=hcore,
                reference_energy=result.energy,
                scf_residual=max(residual, density_drift, fock_drift),
                geometry_hash=self.source.geometry_hash,
                basis_hash=self.source.basis_hash,
                representation=self.source.representation,
                generation_id=str(uuid.uuid4()),
                hamiltonian_id=self.hamiltonian_id,
                hf_backend="native-cuda-prepared-fock",
            )
            self.validate_reference(snapshot)
            return snapshot, {
                "physical_residual": residual,
                "canonical_density_drift": density_drift,
                "physical_fock_drift": fock_drift,
                "iterations": result.iterations,
                "fock_builds": result.fock_builds,
                "export_seconds": time.perf_counter() - started,
                "canonicalization_backend": "cpu-numpy",
                "provider": result.diagnostics,
            }

    def close(self) -> None:
        """Release only the owned Fock plan; source lifetime stays with caller."""
        with self._lock:
            if self._plan is not None:
                self._plan.close()
                self._plan = None

    def __enter__(self) -> typing.Any:
        with self._lock:
            self._ensure_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        if hasattr(self, "_lock"):
            self.close()
