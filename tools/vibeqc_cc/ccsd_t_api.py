"""Composed energy-only RCCSD(T) facade for issue #150 slice C.

The facade deliberately composes the already-qualified RCCSD and perturbative
triples implementations instead of introducing another coupled-cluster equation
stack.  A successful RCCSD(T) result is published only after the underlying
RCCSD state has converged and, for the resident CUDA backend, carries the exact
``resident_solved_state_identity`` produced by the independent expanded
physical replay.

This module is an internal post-HF product boundary.  It does not activate the
reserved public C-ABI ``VIBEQC_METHOD_RCCSD_T`` entry: native method
registration remains coupled to #149 C and must not be emulated by a Python
special case in ``Calculator``.
"""

from __future__ import annotations

import json
import time
import typing
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.integral.cuda_adapter import CudaCompilerAdapter

from tools.vibeqc_posthf.reference import ReferenceSnapshot

from .api import RCCSDResult
from .api import energy as rccsd_energy
from .solver import CCSDResult, SolverOptions
from .triples import INVENTORY_HASH
from .triples_cuda import (
    CudaTriplesResult,
    CudaTriplesTiles,
    TriplesTileConfig,
    cpu_triples_tiles,
)


@dataclass(frozen=True)
class RCCSDTCapabilities:
    """Internal capabilities; batch support means the Python homogeneous helper.

    This record does not describe or activate native PreparedBatch support.
    """

    method: str = "rccsd(t)"
    family: str = "coupled_cluster"
    available: bool = True
    supports_batch: bool = True
    supported_properties: frozenset = frozenset({"energy"})
    batch_shape_policy: str = "homogeneous"

    def __post_init__(self) -> None:
        if self.method != "rccsd(t)" or self.family != "coupled_cluster":
            raise ValueError("RCCSD(T) capability identity mismatch")
        if self.supported_properties != frozenset({"energy"}):
            raise ValueError("RCCSD(T) is energy-only")
        if self.batch_shape_policy != "homogeneous":
            raise ValueError("RCCSD(T) prepared batches are homogeneous")


def rccsd_t_method_capabilities(method: str = "rccsd(t)") -> RCCSDTCapabilities:
    """Report the executable internal RCCSD(T) facade capability.

    ``ccsd(t)`` is accepted as a spelling alias because the Python Calculator
    already uses that public method string for the reserved ABI identifier.
    The canonical capability identity remains ``rccsd(t)``.
    """

    normalized = method.lower().replace(" ", "")
    if normalized not in {"rccsd(t)", "ccsd(t)"}:
        raise ValueError(f"unknown method {method!r}")
    return RCCSDTCapabilities()


def _array_sha256(array: typing.Any) -> str:
    return sha256(np.ascontiguousarray(array, dtype="<f8").tobytes()).hexdigest()


def _qualified_cc_state_identity(backend: str, state: CCSDResult) -> str:
    """Return the exact accepted CC state identity used by triples.

    Resident CUDA results must carry the solved-state identity emitted only
    after independent expanded physical replay. Both backends bind exact replay
    inputs and final amplitudes; CUDA additionally binds that resident identity.
    A non-converged state is never certified here.
    """

    if not isinstance(state, CCSDResult):
        raise TypeError("RCCSD(T) requires a CCSDResult")
    if not state.converged:
        raise RuntimeError("RCCSD(T) triples require a converged RCCSD state")
    resident_identity = state.provenance.get("resident_solved_state_identity")
    if backend == "cuda-resident" and (
        not isinstance(resident_identity, str) or not resident_identity
    ):
        raise RuntimeError("resident RCCSD(T) requires resident_solved_state_identity")
    return canonical_hash(
        {
            "reference_id": state.provenance.get("reference_id"),
            "hamiltonian_id": state.provenance.get("hamiltonian_id"),
            "integral_hash": state.provenance.get("integral_hash"),
            "equation_hash": state.provenance.get("equation_hash"),
            "independent_equation_hash": state.provenance.get(
                "independent_equation_hash"
            ),
            "t1_sha256": _array_sha256(state.t1),
            "t2_sha256": _array_sha256(state.t2),
            "replay_inputs_hash": canonical_hash(state.replay_inputs),
            "ccsd_correlation_energy": state.correlation_energy,
            "resident_solved_state_identity": resident_identity,
            "qualification": "expanded-physical-replay",
        }
    )


def _triples_arrays(snapshot: typing.Any, state: CCSDResult) -> dict[str, np.ndarray]:
    """Recover the exact mathematical inputs retained by the accepted CC solve."""

    required = ("ovvv", "ovoo", "ovov", "fov", "orbital_energies")
    missing = [name for name in required if name not in state.replay_inputs]
    if missing:
        raise RuntimeError(
            "RCCSD(T) accepted CC state is missing replay inputs: " + ", ".join(missing)
        )
    nocc, nvir = state.t1.shape
    if (nocc, nvir) != (snapshot.nocc, snapshot.nmo - snapshot.nocc):
        raise ValueError("RCCSD(T) accepted state/reference shape mismatch")
    # Orbital energies belong to the accepted replay just like F and g. Never
    # fetch fresh provider data or substitute a caller's post-solve snapshot.
    eps = np.asarray(state.replay_inputs["orbital_energies"], dtype=np.float64)
    if eps.ndim != 1 or eps.size != nocc + nvir:
        raise ValueError("RCCSD(T) requires occupied and virtual orbital energies")
    arrays = {
        name: np.ascontiguousarray(state.replay_inputs[name], dtype=np.float64)
        for name in required[:-1]
    }
    arrays.update(
        t1=np.ascontiguousarray(state.t1, dtype=np.float64),
        t2=np.ascontiguousarray(state.t2, dtype=np.float64),
        eps_o=np.ascontiguousarray(eps[:nocc]),
        eps_v=np.ascontiguousarray(eps[nocc:]),
    )
    return arrays


@dataclass(frozen=True)
class RCCSDTResult:
    """Energy decomposition and exact accepted RCCSD state for RCCSD(T)."""

    backend: str
    status: str
    reason: str
    reference_energy: float
    ccsd_correlation_energy: float | None
    triples_energy: float | None
    correlation_energy: float | None
    total_energy: float | None
    ccsd: RCCSDResult
    triples: CudaTriplesResult | None
    provenance: dict

    @property
    def converged(self) -> bool:
        return self.status == "converged"

    @property
    def state(self) -> CCSDResult:
        """Compatibility alias for callers that need final T1/T2/provenance."""

        return self.ccsd.state

    def write(self, path: typing.Any) -> typing.Any:
        """Write strict compact JSON; hash all fields except record_hash.

        The scientific identities exclude timing; the artifact hash includes
        it. Full arrays/history remain available through ``state.write(path)``.
        Serialize before opening the destination so nonfinite data cannot
        destroy a previously valid artifact.
        """

        triples = None
        if self.triples is not None:
            triples = {
                "energy": self.triples_energy,
                "per_tile": tuple(float(x) for x in self.triples.per_tile),
                "tile_count": self.triples.tile_count,
                "vir_chunk_size": self.triples.vir_chunk_size,
                "peak_device_bytes": self.triples.peak_device_bytes,
                "artifact_keys": tuple(self.triples.artifact_keys),
                "timing": dict(self.triples.timing),
            }
        record = {
            "schema": "vibeqc.rccsd-t.endpoint/1",
            "status": self.status,
            "reason": self.reason,
            "backend": self.backend,
            "reference_energy": self.reference_energy,
            "ccsd_correlation_energy": self.ccsd_correlation_energy,
            "triples_energy": self.triples_energy,
            "correlation_energy": self.correlation_energy,
            "total_energy": self.total_energy,
            "ccsd_status": self.ccsd.status,
            "provenance": self.provenance,
            "triples": triples,
        }
        record["record_hash"] = canonical_hash(record)
        Path(path).write_text(
            json.dumps(record, separators=(",", ":"), sort_keys=True, allow_nan=False)
            + "\n"
        )
        return record


def _failed_from_ccsd(result: RCCSDResult, timing: dict) -> RCCSDTResult:
    """Retain diagnostics without certifying a failed state as triples-ready."""
    state_identity = canonical_hash(
        {
            "replay_inputs_hash": canonical_hash(result.state.replay_inputs),
            "t1_sha256": _array_sha256(result.state.t1),
            "t2_sha256": _array_sha256(result.state.t2),
            "status": result.status,
        }
    )
    return RCCSDTResult(
        backend=result.backend,
        status=result.status,
        reason="RCCSD stage did not converge; perturbative triples were not evaluated: "
        + result.reason,
        reference_energy=result.reference_energy,
        ccsd_correlation_energy=result.correlation_energy,
        triples_energy=None,
        correlation_energy=None,
        total_energy=None,
        ccsd=result,
        triples=None,
        provenance={
            "schema": "vibeqc.rccsd-t.result/1",
            "ccsd_status": result.status,
            "triples_evaluated": False,
            "reference_id": result.provenance.get("reference_id"),
            "ccsd_state_identity": state_identity,
            "result_identity": canonical_hash(
                {
                    "method": "rccsd(t)",
                    "backend": result.backend,
                    "state_identity": state_identity,
                    "status": result.status,
                    "reference_energy": result.reference_energy,
                    "ccsd_correlation_energy": result.correlation_energy,
                }
            ),
            "triples_tile_count": 0,
            "virtual_triple_count": 0,
            "triples_peak_device_bytes": 0,
            "triples_artifact_keys": (),
            "timing": timing,
            "ccsd": result.provenance,
        },
    )


def _validate_execution(
    *,
    backend: typing.Any = "cpu",
    options: typing.Any = None,
    compiler: typing.Any = None,
    cache: typing.Any = None,
    device: typing.Any = 0,
    provider_peak_bytes: typing.Any = 0,
    triples_max_bytes: typing.Any = 256 << 20,
    vir_chunk_size: typing.Any = 1,
    triples_oracle: typing.Any = False,
    profile: typing.Any = False,
    compute_forces: typing.Any = False,
) -> None:
    """Validate shared execution controls before any solver work, even if empty."""
    if compute_forces:
        raise NotImplementedError(
            "RCCSD(T) exposes energy only; forces are not implemented"
        )
    if backend not in ("cpu", "cuda-resident"):
        raise ValueError("RCCSD(T) backend must be 'cpu' or 'cuda-resident'")
    TriplesTileConfig(1, 1, vir_chunk_size, triples_max_bytes, device)
    if type(provider_peak_bytes) is not int or provider_peak_bytes < 0:
        raise ValueError("provider_peak_bytes must be a nonnegative integer")
    if options is not None and not isinstance(options, SolverOptions):
        raise TypeError("options must be SolverOptions")
    if backend == "cuda-resident":
        if not isinstance(compiler, CudaCompilerAdapter):
            raise ValueError("CUDA RCCSD(T) requires a CudaCompilerAdapter")
        if not isinstance(cache, Path):
            raise ValueError("CUDA RCCSD(T) requires a pathlib.Path cache")
    elif triples_oracle:
        raise ValueError("triples_oracle is only meaningful for CUDA triples")


def rccsd_t_energy(
    snapshot: typing.Any,
    provider: typing.Any,
    *,
    backend: typing.Any = "cpu",
    options: typing.Any = None,
    t1: typing.Any = None,
    t2: typing.Any = None,
    warm_start: typing.Any = None,
    compiler: typing.Any = None,
    cache: typing.Any = None,
    device: typing.Any = 0,
    provider_peak_bytes: typing.Any = 0,
    triples_max_bytes: typing.Any = 256 << 20,
    vir_chunk_size: typing.Any = 1,
    triples_oracle: typing.Any = False,
    profile: typing.Any = False,
    compute_forces: typing.Any = False,
) -> typing.Any:
    """Execute RHF-reference RCCSD followed by standard noniterative (T).

    Production GPU composition uses ``backend="cuda-resident"``: RCCSD is the
    #555 resident solve and triples are the #448 bounded generated tiles.  The
    CPU route is retained as an auditable endpoint oracle.  The earlier
    host-staged ``backend="cuda"`` RCCSD helper is intentionally not promoted
    here.

    A failed/non-converged RCCSD stage returns a failed result with no triples
    or RCCSD(T) total energy.  Invalid triples inputs or an infeasible triples
    budget raise; :func:`rccsd_t_batch_energy` converts those exceptions into
    isolated per-item failure states. ``options`` controls RCCSD, while
    ``triples_max_bytes`` independently bounds each CUDA tile plan (not CPU
    workspace). The RCCSD convenience owner closes before triples uploads
    host replay inputs and final amplitudes; no device-pointer handoff is implied.
    """

    start = time.perf_counter()
    _validate_execution(
        backend=backend,
        options=options,
        compiler=compiler,
        cache=cache,
        device=device,
        provider_peak_bytes=provider_peak_bytes,
        triples_max_bytes=triples_max_bytes,
        vir_chunk_size=vir_chunk_size,
        triples_oracle=triples_oracle,
        profile=profile,
        compute_forces=compute_forces,
    )
    cc_start = time.perf_counter()
    ccsd = rccsd_energy(
        snapshot,
        provider,
        backend=backend,
        options=options,
        t1=t1,
        t2=t2,
        warm_start=warm_start,
        compiler=compiler,
        cache=cache,
        device=device,
        provider_peak_bytes=provider_peak_bytes,
        compute_forces=False,
    )
    timing = {"ccsd_s": time.perf_counter() - cc_start, "triples_s": 0.0}
    if not ccsd.converged:
        result = _failed_from_ccsd(ccsd, timing)
        timing["endpoint_s"] = time.perf_counter() - start
        return result

    cc_state_identity = _qualified_cc_state_identity(backend, ccsd.state)
    triples_start = time.perf_counter()
    arrays = _triples_arrays(snapshot, ccsd.state)
    nocc = snapshot.nocc
    nvir = snapshot.nmo - nocc

    if backend == "cpu":
        triples = cpu_triples_tiles(
            nocc,
            nvir,
            arrays,
            vir_chunk_size=vir_chunk_size,
        )
        triples_backend = "cpu-tiled"
    else:
        config = TriplesTileConfig(
            nocc,
            nvir,
            vir_chunk_size=vir_chunk_size,
            max_bytes=triples_max_bytes,
            device=device,
        )
        with CudaTriplesTiles(config, compiler, cache) as executor:
            triples = executor.run_tiles(arrays, oracle=triples_oracle, profile=profile)
        triples_backend = "cuda-bounded-tiles"

    timing["triples_s"] = time.perf_counter() - triples_start
    timing["triples_detail"] = dict(triples.timing)

    et = float(triples.et)
    ecc = ccsd.correlation_energy
    if ecc is None or not np.isfinite(ecc) or not np.isfinite(et):
        raise FloatingPointError("RCCSD(T) energy decomposition is nonfinite")
    correlation = float(ecc + et)
    total = float(ccsd.reference_energy + correlation)
    if not np.isfinite(correlation) or not np.isfinite(total):
        raise FloatingPointError("RCCSD(T) combined energy is nonfinite")
    result_identity = canonical_hash(
        {
            "method": "rccsd(t)",
            "ccsd_state_identity": cc_state_identity,
            "triples_energy": et,
            "vir_chunk_size": vir_chunk_size,
            "triples_backend": triples_backend,
            "triples_inventory_hash": INVENTORY_HASH,
            "reference_energy": ccsd.reference_energy,
            "ccsd_correlation_energy": ecc,
            "correlation_energy": correlation,
            "total_energy": total,
        }
    )
    provenance = {
        "schema": "vibeqc.rccsd-t.result/1",
        "method": "standard-canonical-rccsd(t)",
        "reference_id": ccsd.provenance.get("reference_id"),
        "hamiltonian_id": ccsd.provenance.get("hamiltonian_id"),
        "ccsd_state_identity": cc_state_identity,
        "result_identity": result_identity,
        "resident_solved_state_identity": ccsd.provenance.get(
            "resident_solved_state_identity"
        ),
        "triples_inventory_hash": INVENTORY_HASH,
        "ccsd_backend": backend,
        "triples_backend": triples_backend,
        "triples_evaluated": True,
        "vir_chunk_size": vir_chunk_size,
        "triples_peak_device_bytes": triples.peak_device_bytes,
        "triples_tile_count": triples.tile_count,
        "virtual_triple_count": nvir * (nvir + 1) * (nvir + 2) // 6,
        "triples_artifact_keys": tuple(triples.artifact_keys),
        # The facade closes CCSD device owners before creating any tile owner.
        # Provider reservations are caller-supplied and may outlive both phases.
        "peak_device_bytes": 0
        if backend == "cpu"
        else max(
            ccsd.provenance.get("combined_peak_bytes", 0),
            provider_peak_bytes + triples.peak_device_bytes,
        ),
        "memory": {
            "ccsd_logical_required_bytes": ccsd.provenance.get(
                "logical_required_bytes"
            ),
            "ccsd_peak_device_bytes": ccsd.provenance.get("combined_peak_bytes", 0),
            "triples_input_bytes": sum(a.nbytes for a in arrays.values()),
            "triples_peak_bytes_per_tile": triples.peak_bytes_per_tile,
            "triples_cpu_workspace_peak_bytes": None,
            "triples_max_bytes": triples_max_bytes,
            "provider_peak_bytes": provider_peak_bytes,
        },
        "runtime_device": triples.runtime_device,
        "ccsd": ccsd.provenance,
        "triples": triples.provenance,
        "timing": timing,
    }
    timing["endpoint_s"] = time.perf_counter() - start
    return RCCSDTResult(
        backend=backend,
        status="converged",
        reason="RCCSD converged and standard bounded (T) energy completed",
        reference_energy=float(ccsd.reference_energy),
        ccsd_correlation_energy=float(ecc),
        triples_energy=et,
        correlation_energy=correlation,
        total_energy=total,
        ccsd=ccsd,
        triples=triples,
        provenance=provenance,
    )


@dataclass(frozen=True)
class RCCSDTBatchItemResult:
    """One isolated prepared-batch item in input order."""

    index: int
    status: str
    reason: str
    converged: bool
    ccsd_correlation_energy: float | None
    triples_energy: float | None
    total_energy: float | None
    result: RCCSDTResult | None


@dataclass(frozen=True)
class BatchRCCSDTResult:
    """Input-ordered homogeneous RCCSD(T) batch result."""

    items: tuple[RCCSDTBatchItemResult, ...]
    shape: tuple[int, int] | None


class PreparedRCCSDTBatch:
    """Prepared homogeneous batch with independent CC/triples item state.

    The implementation is intentionally sequential at the control layer: every
    system retains separate amplitudes, DIIS, convergence and failure state,
    while homogeneous shapes reuse the compiler/cache artifacts. Preparation
    borrows providers and allocates no solver/device state. Each execute call
    starts fresh; per-item warm starts and shared amplitude arrays are not batch
    settings. No padding or ragged-shape claim is made.
    """

    def __init__(
        self,
        problems: typing.Any,
        *,
        compute_forces: typing.Any = False,
        **settings: typing.Any,
    ) -> None:
        state_keys = ("t1", "t2", "warm_start")
        if any(settings.get(name) is not None for name in state_keys):
            raise ValueError(
                "RCCSD(T) prepared batches do not accept shared t1/t2 or warm_start state"
            )
        settings = {
            key: value for key, value in settings.items() if key not in state_keys
        }
        _validate_execution(compute_forces=compute_forces, **settings)
        self.problems = tuple((snapshot, provider) for snapshot, provider in problems)
        if not self.problems:
            self.shape = None
        else:
            shapes = []
            for item in self.problems:
                snapshot, _provider = item
                if not isinstance(snapshot, ReferenceSnapshot):
                    raise TypeError(
                        "RCCSD(T) batch requires validated reference snapshots"
                    )
                shape = (snapshot.nocc, snapshot.nmo - snapshot.nocc)
                shapes.append(shape)
            if any(shape != shapes[0] for shape in shapes[1:]):
                raise ValueError(
                    "RCCSD(T) prepared batches require homogeneous (nocc, nvir) shapes"
                )
            self.shape = shapes[0]
        self.settings = dict(settings)

    def execute(self, *, compute_forces: typing.Any = False) -> BatchRCCSDTResult:
        """Return input-ordered results; any item exception leaves others runnable."""
        if compute_forces:
            raise NotImplementedError(
                "RCCSD(T) exposes energy only; forces are not implemented"
            )
        items = []
        for index, (snapshot, provider) in enumerate(self.problems):
            try:
                result = rccsd_t_energy(
                    snapshot,
                    provider,
                    compute_forces=False,
                    **self.settings,
                )
                items.append(
                    RCCSDTBatchItemResult(
                        index=index,
                        status=result.status,
                        reason=result.reason,
                        converged=result.converged,
                        ccsd_correlation_energy=result.ccsd_correlation_energy,
                        triples_energy=result.triples_energy,
                        total_energy=result.total_energy,
                        result=result,
                    )
                )
            except Exception as error:  # noqa: BLE001 - item isolation is the contract
                items.append(
                    RCCSDTBatchItemResult(
                        index=index,
                        status="error",
                        reason=f"{type(error).__name__}: {error}",
                        converged=False,
                        ccsd_correlation_energy=None,
                        triples_energy=None,
                        total_energy=None,
                        result=None,
                    )
                )
        return BatchRCCSDTResult(tuple(items), self.shape)


def rccsd_t_batch_energy(
    problems: typing.Any, *, compute_forces: typing.Any = False, **settings: typing.Any
) -> BatchRCCSDTResult:
    """Prepare and execute a homogeneous energy-only RCCSD(T) batch."""

    return PreparedRCCSDTBatch(
        problems, compute_forces=compute_forces, **settings
    ).execute()
