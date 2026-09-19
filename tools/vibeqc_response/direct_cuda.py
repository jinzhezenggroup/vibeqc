"""Direct CUDA J/K adapter for the shared, host-orchestrated RHF response layer.

This is not a device-resident CPHF solver or a molecular Hessian endpoint.
Scientific J/K arithmetic belongs to the existing prepared Fock provider.
"""

from __future__ import annotations

import threading
import time
from copy import deepcopy

import numpy as np
from vibeqc.fock import FockBuildSpec, FockPlan
from vibeqc.profiles import canonical_hash
from vibeqc_compiler.dft import NativeAO

from tools.vibeqc_posthf.sources import NativeSource

from .backends import _checked_density


def _source_record(source):
    """Bind NativeSource's immutable scientific fields, not just its dimensions."""
    source._check_open()
    return (
        source.identity,
        source.geometry_hash,
        source.basis_hash,
        source.representation,
        source.nbf,
        source.charge,
        source.multiplicity,
        source.electron_count,
    )


class CudaDirectJKBackend:
    """Exact, unscreened CUDA J/K actions on real symmetric signed AO inputs.

    Reuses :class:`vibeqc.fock.FockPlan` with explicit exact J and K and zero
    screening. Response densities need not be positive, normalized, or ground-
    state densities. The returned matrices are raw J/K, not a total Fock matrix
    containing a core Hamiltonian. #179 applies the existing RHF coefficients.

    Preparation and result arrays are on the host. The shared RHF AO/MO
    transforms and GMRES/orthogonalization also remain on the host; only the
    prepared J/K integral contractions execute on CUDA. No CPU or DF substitute
    is allowed. The caller must keep the borrowed NativeSource open.
    """

    hamiltonian_id = "conventional-unscreened"

    def __init__(self, source, *, device_id=0, device_budget_bytes=64 << 20):
        self._lock = threading.RLock()
        self._plan = None
        if not isinstance(source, NativeSource):
            raise TypeError("CUDA direct response requires a NativeSource")
        if type(device_id) is not int or not 0 <= device_id < 2**31:
            raise ValueError("device_id must be a nonnegative int32")
        if type(device_budget_bytes) is not int or not 0 < device_budget_bytes < 2**64:
            raise ValueError("device_budget_bytes must be a positive uint64")
        self.source = source
        self._source_record = _source_record(source)
        if (
            source.multiplicity != 1
            or source.electron_count <= 0
            or source.electron_count % 2
        ):
            raise NotImplementedError(
                "CUDA direct response is qualified for closed-shell RHF only"
            )
        if any(shell.angular_momentum > 3 for shell in source.shells):
            raise NotImplementedError(
                "CUDA direct response supports s/p/d/f shells only"
            )
        self.nbf = source.nbf
        self.device_id = device_id
        spec = FockBuildSpec.hf(derivative_order=0)
        try:
            # Copy the actual shell records, never re-resolve a named basis.
            # FockPlan owns its native source after this temporary AO owner closes.
            with NativeAO(
                source.atoms,
                basis=source.shells,
                representation=source.representation,
                charge=source.charge,
                multiplicity=source.multiplicity,
            ) as basis:
                if basis.nao != self.nbf:
                    raise ValueError(
                        "CUDA direct response source AO-dimension mismatch"
                    )
                self._plan = FockPlan(
                    basis,
                    spec,
                    device="cuda",
                    device_id=device_id,
                    screening_tolerance=0.0,
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
                or diag["auxiliary_rank"] != 0
            ):
                raise RuntimeError("CUDA direct response provider semantics mismatch")
            if not 0 < diag["device_bytes"] <= device_budget_bytes:
                raise RuntimeError(
                    "CUDA direct response provider device budget mismatch"
                )
            self._native_diagnostics = deepcopy(diag)
            self.device_resident_bytes = int(diag["device_bytes"])
            self.identity = canonical_hash(
                {
                    "backend": "cuda-direct-fock-jk/v1",
                    "source": self._source_record,
                    "hamiltonian": self.hamiltonian_id,
                    "plan": self._plan.identity,
                    "execution": self._plan.execution_identity,
                    "device_budget_bytes": device_budget_bytes,
                }
            )
            self.statistics = {
                "actions": 0,
                "seconds": 0.0,
                # Compatibility with the shared operator's workspace estimate;
                # this is not a complete response/Hessian peak-memory claim.
                "peak_bytes": self.device_resident_bytes,
            }
        except BaseException:
            self.close()
            raise

    def _ensure_open(self):
        if self._plan is None:
            raise RuntimeError("CUDA direct response backend is closed")
        if _source_record(self.source) != self._source_record:
            raise ValueError("CUDA direct response source identity changed")

    @property
    def diagnostics(self):
        """Detached provenance with explicit execution and memory boundaries."""
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
                "molecular_hessian": False,
                "memory_scope": "retained direct J/K device allocations only; excludes preparation temporaries, host transforms/results, solver and CUDA context",
            }

    def validate_reference(self, reference):
        """Bind only a matching conventional RHF reference, never DF/KS/UHF."""
        with self._lock:
            self._ensure_open()
            for name, expected in (
                ("geometry_hash", self._source_record[1]),
                ("basis_hash", self._source_record[2]),
                ("representation", self._source_record[3]),
                ("nmo", self.nbf),
                ("electron_count", self._source_record[7]),
                ("hamiltonian_id", self.hamiltonian_id),
                ("algorithm", "RHF"),
            ):
                if getattr(reference, name, None) != expected:
                    raise ValueError(f"CUDA direct backend/reference {name} mismatch")
        return self

    def coulomb_exchange(self, density):
        """Apply the prepared device provider; publish only complete raw J/K."""
        with self._lock:
            self._ensure_open()
            value = np.asarray(density)
            if np.iscomplexobj(value):
                raise ValueError("density response must be finite real FP64")
            d = _checked_density(np.asarray(value, dtype=np.float64), self.nbf)
            started = time.perf_counter()
            result = self._plan.evaluate(d, derivative=False)
            if result.diagnostics != self._native_diagnostics:
                raise RuntimeError("CUDA direct response execution identity changed")
            matrices = (result.coulomb, result.exchange)
            for matrix in matrices:
                if (
                    matrix is None
                    or matrix.shape != (self.nbf, self.nbf)
                    or matrix.dtype != np.dtype(np.float64)
                    or not np.isfinite(matrix).all()
                ):
                    raise RuntimeError("CUDA direct response returned invalid J/K")
            # FockEvaluation already owns immutable result storage. Do not
            # symmetrize outputs to hide a provider/transpose error.
            self.statistics["actions"] += 1
            self.statistics["seconds"] += time.perf_counter() - started
            return matrices

    def close(self):
        """Release only the owned Fock plan; the caller owns source lifetime."""
        with self._lock:
            if self._plan is not None:
                self._plan.close()
                self._plan = None

    def __enter__(self):
        with self._lock:
            self._ensure_open()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        if hasattr(self, "_lock"):
            self.close()
