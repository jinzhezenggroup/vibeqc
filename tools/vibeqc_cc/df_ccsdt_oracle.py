"""Same-Hamiltonian dense DF-CCSD(T) oracle for issue #157 slice A.

This module intentionally does not provide the production factorized solver.
It freezes the first supported method definition (conventional RHF reference
plus a correlation-only DF Hamiltonian), reconstructs a complete small-system
MO g_DF once, and evaluates the already-audited RCCSD/(T) equations on exactly
that Hamiltonian.
"""

from __future__ import annotations

import typing
from copy import deepcopy
from dataclasses import dataclass, replace
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
from vibeqc.profiles import canonical_hash

from tools.vibeqc_posthf.df import DFProvider, MetricFactor
from tools.vibeqc_posthf.providers import BlockResult, ConventionalProvider
from tools.vibeqc_posthf.reference import ReferenceSnapshot, immutable

from .ccsd_t_api import RCCSDTResult, rccsd_t_energy
from .df_contract import DFCCSDTMethodContract, correlation_df_reference


def _array_hash(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value, dtype="<f8").tobytes()).hexdigest()


def _validate_three_index(
    three_index: typing.Any,
    nmo: int,
    metric_dimension: int | None = None,
) -> np.ndarray:
    value = np.asarray(three_index)
    if value.dtype != np.float64 or value.ndim != 3:
        raise ValueError("DF three-index tensor must be a rank-3 float64 array")
    if value.shape[1:] != (nmo, nmo) or value.shape[0] < 1:
        raise ValueError("DF three-index tensor has incompatible MO dimensions")
    if metric_dimension is not None and value.shape[0] != metric_dimension:
        raise ValueError("DF three-index auxiliary dimension does not match contract")
    if not np.isfinite(value).all():
        raise ValueError("DF three-index tensor must be finite")
    return value


def dense_eri_from_three_index(
    three_index: typing.Any,
    *,
    max_bytes: int = 256 << 20,
) -> np.ndarray:
    """Reconstruct g[p,q,r,s]=sum_Q B[Q,p,q]B[Q,r,s] for a small oracle."""

    value = np.asarray(three_index)
    if value.ndim != 3 or value.shape[1] != value.shape[2]:
        raise ValueError("three_index must have shape (naux,nmo,nmo)")
    value = _validate_three_index(value, value.shape[1])
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("dense oracle max_bytes must be a positive integer")
    nmo = value.shape[1]
    # Count the caller-owned B plus output and one conservative full-size
    # contraction temporary.  The production path must not use this allocation.
    required = value.nbytes + 16 * nmo**4
    if required > max_bytes:
        raise MemoryError(f"dense DF oracle requires {required} numeric bytes")
    eri = np.einsum("Qpq,Qrs->pqrs", value, value, optimize=True)
    if not np.isfinite(eri).all():
        raise FloatingPointError(
            "dense DF ERI reconstruction produced nonfinite values"
        )
    return immutable(eri)


def df_fitting_error(df_eri: typing.Any, exact_eri: typing.Any) -> dict[str, float]:
    """Report fitting error separately from implementation/oracle tolerances."""

    fitted = np.asarray(df_eri, dtype=np.float64)
    exact = np.asarray(exact_eri, dtype=np.float64)
    if fitted.shape != exact.shape or fitted.ndim != 4:
        raise ValueError("DF/exact ERIs must have the same four-index shape")
    delta = fitted - exact
    return {
        "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "frobenius": float(np.linalg.norm(delta)),
        "rms": float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0,
    }


class DenseDFOracleProvider(ConventionalProvider):
    """Small-system dense g_DF provider accepted by the trusted CC equation stack."""

    def __init__(
        self,
        snapshot: ReferenceSnapshot,
        eri_mo: typing.Any,
        contract: DFCCSDTMethodContract,
    ) -> None:
        eri = np.asarray(eri_mo)
        if (
            not isinstance(snapshot, ReferenceSnapshot)
            or not isinstance(contract, DFCCSDTMethodContract)
            or snapshot.hamiltonian_id != contract.correlation_hamiltonian_id
            or snapshot.identity != contract.correlation_snapshot_identity
            or snapshot.geometry_hash != contract.geometry_hash
            or snapshot.basis_hash != contract.orbital_basis_hash
        ):
            raise ValueError("dense DF oracle identity mismatch")
        if eri.dtype != np.float64 or eri.shape != (snapshot.nmo,) * 4:
            raise ValueError("dense DF oracle requires complete float64 MO ERIs")
        if not np.isfinite(eri).all():
            raise ValueError("dense DF oracle ERIs must be finite")
        self.snapshot = snapshot
        self.g = immutable(eri)
        self.contract = contract
        self.backend = "cpu"
        self._closed = False
        self.source = SimpleNamespace(
            identity="dense-df-oracle:" + _array_hash(self.g),
            _check_open=self._check_open,
        )

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("dense DF oracle provider is closed")

    def get(self, block: typing.Any) -> BlockResult:
        self._check_open()
        block.validate(self.snapshot)
        values = immutable(self.g[np.ix_(*block.slots)])
        return BlockResult(
            block,
            values,
            self.snapshot.identity,
            self.snapshot.hamiltonian_id,
            {
                "backend": "cpu-dense-df-oracle-fp64",
                "same_hamiltonian_oracle": True,
                "method_contract_identity": self.contract.identity,
                "cache_hit": False,
            },
        )

    def clear(self) -> None:
        self._check_open()

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> typing.Self:
        self._check_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass(frozen=True)
class PreparedDenseDFCCSDTOracle:
    """Owned dense DF Hamiltonian and provider used only for qualification."""

    snapshot: ReferenceSnapshot
    contract: DFCCSDTMethodContract
    three_index: np.ndarray
    eri_mo: np.ndarray
    provider: DenseDFOracleProvider
    diagnostics: dict[str, typing.Any]

    def close(self) -> None:
        self.provider.close()


def dense_df_oracle_from_three_index(
    snapshot: ReferenceSnapshot,
    contract: DFCCSDTMethodContract,
    three_index: typing.Any,
    *,
    max_bytes: int = 256 << 20,
) -> PreparedDenseDFCCSDTOracle:
    """Create the independent dense oracle from an already-whitened B tensor."""

    if (
        snapshot.hamiltonian_id != contract.correlation_hamiltonian_id
        or snapshot.identity != contract.correlation_snapshot_identity
    ):
        raise ValueError("dense DF oracle snapshot identity mismatch")
    source_b = _validate_three_index(
        three_index, snapshot.nmo, contract.metric_dimension
    )
    # During ownership transfer the caller-owned B and the immutable oracle copy
    # coexist.  Account for both plus the dense output/contraction workspace.
    required = 2 * source_b.nbytes + 16 * snapshot.nmo**4
    if type(max_bytes) is not int or max_bytes < required:
        raise MemoryError(f"dense DF oracle requires {required} numeric bytes")
    b = immutable(np.array(source_b, copy=True))
    eri = dense_eri_from_three_index(b, max_bytes=max_bytes - source_b.nbytes)
    provider = DenseDFOracleProvider(snapshot, eri, contract)
    diagnostics = {
        "schema": "vibeqc.df-ccsd-t.dense-oracle/1",
        "method_contract_identity": contract.identity,
        "three_index_sha256": _array_hash(b),
        "eri_sha256": _array_hash(eri),
        "three_index_bytes": b.nbytes,
        "eri_bytes": eri.nbytes,
        "dense_budget_bytes": max_bytes,
        "dense_required_bytes": required,
        "production_factorized": False,
    }
    return PreparedDenseDFCCSDTOracle(snapshot, contract, b, eri, provider, diagnostics)


def prepare_same_hamiltonian_dense_oracle(
    reference: ReferenceSnapshot,
    source: typing.Any,
    metric: MetricFactor,
    *,
    provider_budget_bytes: int = 256 << 20,
    dense_budget_bytes: int = 256 << 20,
    axis_tile: int = 2,
    auxiliary_tile: int = 3,
) -> PreparedDenseDFCCSDTOracle:
    """Build B in the conventional-RHF MO basis, then reconstruct one dense g_DF."""

    snapshot, contract = correlation_df_reference(reference, source, metric)
    nmo, naux = snapshot.nmo, source.naux
    dense_required = 16 * naux * nmo * nmo + 16 * nmo**4
    if type(dense_budget_bytes) is not int or dense_budget_bytes < dense_required:
        raise MemoryError(f"dense DF oracle requires {dense_required} numeric bytes")
    with DFProvider(
        snapshot,
        source,
        metric,
        budget_bytes=provider_budget_bytes,
        axis_tile=axis_tile,
        auxiliary_tile=auxiliary_tile,
    ) as provider:
        columns = tuple(range(nmo))
        b = provider.three_index(columns, columns)
        provider_statistics = dict(provider.statistics)
    prepared = dense_df_oracle_from_three_index(
        snapshot,
        contract,
        b,
        max_bytes=dense_budget_bytes,
    )
    diagnostics = {
        **prepared.diagnostics,
        "df_provider_statistics": provider_statistics,
        "reference_identity": reference.identity,
        "correlation_snapshot_identity": snapshot.identity,
        "fock_policy": contract.fock_policy,
    }
    return replace(prepared, diagnostics=diagnostics)


@dataclass(frozen=True)
class DFCCSDTOracleResult:
    """DF method record plus the trusted RCCSD(T) result on the dense g_DF oracle."""

    result: RCCSDTResult
    contract: DFCCSDTMethodContract
    oracle_identity: str
    provenance: dict[str, typing.Any]

    @property
    def converged(self) -> bool:
        return self.result.converged

    @property
    def ccsd_correlation_energy(self) -> float | None:
        return self.result.ccsd_correlation_energy

    @property
    def triples_energy(self) -> float | None:
        return self.result.triples_energy

    @property
    def total_energy(self) -> float | None:
        return self.result.total_energy


def run_dense_df_ccsdt_oracle(
    prepared: PreparedDenseDFCCSDTOracle,
    *,
    options: typing.Any = None,
    vir_chunk_size: int = 1,
) -> DFCCSDTOracleResult:
    """Evaluate the audited conventional equations on exactly the reconstructed g_DF."""

    prepared.provider._check_open()
    result = rccsd_t_energy(
        prepared.snapshot,
        prepared.provider,
        backend="cpu",
        options=options,
        vir_chunk_size=vir_chunk_size,
    )
    diagnostics = deepcopy(prepared.diagnostics)
    diagnostics.update(
        method_contract_identity=prepared.contract.identity,
        three_index_sha256=_array_hash(prepared.three_index),
        eri_sha256=_array_hash(prepared.eri_mo),
    )
    identity = canonical_hash(
        {
            "method_contract_identity": prepared.contract.identity,
            "three_index_sha256": diagnostics["three_index_sha256"],
            "eri_sha256": diagnostics["eri_sha256"],
            "rccsd_t_result_identity": result.provenance["result_identity"],
        }
    )
    return DFCCSDTOracleResult(
        result,
        prepared.contract,
        identity,
        {
            "schema": "vibeqc.df-ccsd-t.oracle-result/1",
            "method": "df-rccsd(t)-correlation-only",
            "method_contract": prepared.contract.record(),
            "oracle": diagnostics,
            "oracle_identity": identity,
            "rccsd_t": deepcopy(result.provenance),
        },
    )
