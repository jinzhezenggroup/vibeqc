"""Factorized perturbative triples for issue #157 slice C.

The accepted DF-RCCSD state already owns whitened B_ov/B_vv factors.  This
module evaluates the standard canonical (T) correction on that same
Hamiltonian without reconstructing the resident ovvv block or any full T3
tensor.  The W1 term is reduced over the auxiliary index directly.
"""

from __future__ import annotations

import time
import typing
from dataclasses import dataclass

import numpy as np

from .df_factorized import DFCCSDResult, PreparedDFCCSD, _run

if typing.TYPE_CHECKING:
    from .df_contract import DFCCSDTMethodContract
    from .solver import SolverOptions

from .triples import (
    _LABELS,
    INVENTORY_HASH,
    OP,
    SLOW_TABLE,
    VP,
    _check_denominators,
    _degeneracy,
    _permuted,
    r3,
)


def factorized_triples_workspace_bytes(nocc: int) -> int:
    """Conservative scratch budget for one virtual triple.

    The bound includes all six W/V/Z cubes, projection temporaries and
    bounded input-validation scratch. Previous triples are released explicitly.
    B factors, accepted amplitudes and retained smaller MO blocks are owned and
    budgeted by :class:`PreparedDFCCSD`; this function budgets only temporary
    (T) work.  No term scales with nvir**3 or nvir**4.
    """

    if type(nocc) is not int or nocc < 1:
        raise ValueError("factorized DF triples require a nonempty occupied space")
    return 8 * (32 * nocc**3 + 10 * nocc**2 + nocc)


def _validate_inputs(
    bov: typing.Any,
    bvv: typing.Any,
    ovoo: typing.Any,
    ovov: typing.Any,
    fov: typing.Any,
    t1: typing.Any,
    t2: typing.Any,
    eps_o: typing.Any,
    eps_v: typing.Any,
) -> tuple[np.ndarray, ...]:
    inputs = (bov, bvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v)
    if any(not isinstance(value, np.ndarray) for value in inputs):
        raise ValueError("factorized DF triples require FP64 numpy arrays")
    values = tuple(np.asarray(value) for value in inputs)
    if any(x.dtype != np.float64 for x in values):
        raise ValueError("factorized DF triples require FP64 inputs")
    bov, bvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v = values
    if bov.ndim != 3 or bvv.ndim != 3:
        raise ValueError("DF triples factors must have rank three")
    q, o, v = bov.shape
    if min(q, o, v) < 1:
        raise ValueError(
            "factorized DF triples require nonempty factor/orbital domains"
        )
    shapes = (
        (bvv.shape, (q, v, v)),
        (ovoo.shape, (o, v, o, o)),
        (ovov.shape, (o, v, o, v)),
        (fov.shape, (o, v)),
        (t1.shape, (o, v)),
        (t2.shape, (o, o, v, v)),
        (eps_o.shape, (o,)),
        (eps_v.shape, (v,)),
    )
    if q < 1 or any(actual != expected for actual, expected in shapes):
        raise ValueError("incompatible factorized DF triples shapes")
    return values


def _validate_values(values: tuple[np.ndarray, ...], chunk: int) -> None:
    """Bound even strided validation temporaries by occupied-space scratch."""
    for value in values:
        for start in range(0, value.size, chunk):
            if not np.isfinite(value.flat[start : start + chunk]).all():
                raise ValueError("factorized DF triples require finite FP64 inputs")
    for vv in values[1]:
        for row in range(vv.shape[0]):
            for start in range(0, row + 1, chunk):
                stop = min(row + 1, start + chunk)
                difference = vv[row, start:stop] - vv[start:stop, row]
                if np.max(np.abs(difference)) > 1e-10:
                    raise ValueError("B_vv must preserve the symmetric spatial-MO pair")


def _w_factorized(
    bov: np.ndarray,
    bvv: np.ndarray,
    vooo: np.ndarray,
    t2_t: np.ndarray,
    a: int,
    b: int,
    c: int,
) -> np.ndarray:
    """W(a,b,c) with vvov reduced directly from B_ov/B_vv."""

    o = bov.shape[1]
    w = np.zeros((o, o, o), dtype=np.float64)
    for q in range(bov.shape[0]):
        # vvov[a,b,i,f] = sum_Q B_ov[Q,i,a] B_vv[Q,f,b].
        contracted_f = np.einsum("f,fkj->kj", bvv[q, :, b], t2_t[c], optimize=True)
        w += np.einsum("i,kj->ijk", bov[q, :, a], contracted_f, optimize=False)
    w -= np.einsum("ijm,mk->ijk", vooo[a], t2_t[b, c], optimize=True)
    return w


def _v_dense(
    vvoo: np.ndarray,
    fvo: np.ndarray,
    t1_t: np.ndarray,
    t2_t: np.ndarray,
    a: int,
    b: int,
    c: int,
) -> np.ndarray:
    value = np.einsum("ij,k->ijk", vvoo[a, b], t1_t[c], optimize=False)
    value += np.einsum("ij,k->ijk", t2_t[a, b], fvo[c], optimize=False)
    return value


def factorized_triples_energy(
    bov: typing.Any,
    bvv: typing.Any,
    ovoo: typing.Any,
    ovov: typing.Any,
    fov: typing.Any,
    t1: typing.Any,
    t2: typing.Any,
    eps_o: typing.Any,
    eps_v: typing.Any,
    *,
    max_bytes: int = 256 << 20,
    denominator_threshold: float = 1e-10,
) -> float:
    """Standard closed-shell (T) energy from factorized DF virtual integrals."""

    (
        bov,
        bvv,
        ovoo,
        ovov,
        fov,
        t1,
        t2,
        eps_o,
        eps_v,
    ) = _validate_inputs(bov, bvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v)
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("factorized DF triples max_bytes must be a positive integer")
    required = factorized_triples_workspace_bytes(bov.shape[1])
    if required > max_bytes:
        raise MemoryError(
            f"factorized DF triples require {required} temporary numeric bytes"
        )
    _validate_values(
        (bov, bvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v),
        min(4096, required // (3 * np.dtype(np.float64).itemsize)),
    )
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        _check_denominators(eps_o, eps_v, denominator_threshold)

        t1_t = t1.T
        t2_t = t2.transpose(2, 3, 0, 1)
        vooo = ovoo.transpose(1, 0, 2, 3)
        vvoo = ovov.transpose(1, 3, 0, 2)
        fvo = fov.T
        eijk = eps_o[:, None, None] + eps_o[None, :, None] + eps_o[None, None, :]

        et = 0.0
        for a in range(len(eps_v)):
            for b in range(a + 1):
                for c in range(b + 1):
                    d3 = (eijk - eps_v[a] - eps_v[b] - eps_v[c]) * _degeneracy(a, b, c)
                    ws: dict[str, np.ndarray] = {}
                    vs: dict[str, np.ndarray] = {}
                    for label in _LABELS:
                        pa, pb, pc = _permuted((a, b, c), VP[label])
                        ws[label] = _w_factorized(bov, bvv, vooo, t2_t, pa, pb, pc)
                        vs[label] = _v_dense(vvoo, fvo, t1_t, t2_t, pa, pb, pc)
                    zs = {
                        label: r3(ws[label] + 0.5 * vs[label]) / d3 for label in _LABELS
                    }
                    for zlabel, row in SLOW_TABLE.items():
                        for wlabel, occupied_order in row:
                            et += np.einsum(
                                "ijk,ijk",
                                ws[wlabel].transpose(OP[occupied_order]),
                                zs[zlabel],
                            )
                    # Do not retain the prior Z dictionary while constructing the
                    # next virtual triple's intermediates.
                    del ws, vs, zs, d3
        result = float(2.0 * et)
        if not np.isfinite(result):
            raise FloatingPointError("factorized DF triples energy became nonfinite")
        return result


@dataclass(frozen=True)
class DFCCSDTResult:
    """Internal factorized DF-RCCSD(T) energy result for #157 slice C."""

    status: str
    reason: str
    reference_energy: float
    ccsd_correlation_energy: float | None
    triples_energy: float | None
    correlation_energy: float | None
    total_energy: float | None
    ccsd: DFCCSDResult
    provenance: dict[str, typing.Any]

    @property
    def converged(self) -> bool:
        return self.status == "converged"


def solve_df_ccsdt(
    snapshot: typing.Any,
    provider: typing.Any,
    contract: DFCCSDTMethodContract,
    *,
    options: SolverOptions | None = None,
    t1: np.ndarray | None = None,
    t2: np.ndarray | None = None,
    triples_max_bytes: int = 256 << 20,
    denominator_threshold: float = 1e-10,
) -> DFCCSDTResult:
    """Run factorized DF-RCCSD then standard (T) on the same retained DF state."""

    prepared = PreparedDFCCSD(snapshot, provider, contract, options, t1, t2)
    try:
        start = time.perf_counter()
        ccsd = _run(prepared)
        if not ccsd.converged:
            return DFCCSDTResult(
                ccsd.status,
                ccsd.reason,
                snapshot.reference_energy,
                ccsd.correlation_energy,
                None,
                None,
                None,
                ccsd,
                {
                    "schema": "vibeqc.df-rccsd-t.result/1",
                    "method_contract_identity": contract.identity,
                    "triples_executed": False,
                    "ccsd": ccsd.provenance,
                },
            )
        eps = np.asarray(snapshot.orbital_energies, dtype=np.float64)
        o = snapshot.nocc
        triples_start = time.perf_counter()
        et = factorized_triples_energy(
            prepared.integrals.bov,
            prepared.integrals.bvv,
            prepared.feeds["ovoo"],
            prepared.feeds["ovov"],
            prepared.feeds["fov"],
            ccsd.t1,
            ccsd.t2,
            eps[:o],
            eps[o:],
            max_bytes=triples_max_bytes,
            denominator_threshold=denominator_threshold,
        )
        triples_seconds = time.perf_counter() - triples_start
        ecc = float(ccsd.correlation_energy)
        correlation = ecc + et
        total = snapshot.reference_energy + correlation
        if not np.isfinite((et, ecc, correlation, total)).all():
            raise FloatingPointError(
                "factorized DF-RCCSD(T) total energy became nonfinite"
            )
        provenance = {
            "schema": "vibeqc.df-rccsd-t.result/1",
            "method": "df-rccsd(t)-correlation-only",
            "method_contract_identity": contract.identity,
            "reference_id": snapshot.identity,
            "hamiltonian_id": snapshot.hamiltonian_id,
            "fock_policy": contract.fock_policy,
            "ccsd": ccsd.provenance,
            "triples_inventory_hash": INVENTORY_HASH,
            "triples_factorization": "B_ov+B_vv-direct-W1-v1",
            "resident_ovvv": False,
            "resident_vvvv": False,
            "full_t3": False,
            "triples_workspace_bytes": factorized_triples_workspace_bytes(o),
            "triples_max_bytes": triples_max_bytes,
            "timing": {
                "triples_s": triples_seconds,
                "endpoint_s": time.perf_counter() - start,
            },
        }
        return DFCCSDTResult(
            "converged",
            "factorized DF-RCCSD and standard (T) passed",
            snapshot.reference_energy,
            ecc,
            et,
            correlation,
            total,
            ccsd,
            provenance,
        )
    finally:
        prepared.close()
