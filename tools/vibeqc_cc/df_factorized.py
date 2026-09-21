"""Factorized correlation-DF RCCSD energy path for issue #157 slice B.

The conventional RHF Fock is preserved by the #157 method contract.  This
module removes the memory-heavy ovvv/vvvv MO tensors from the CCSD equation
feeds.  Their exact contributions are contracted directly from B_ov/B_vv and
injected as singles/doubles residual corrections into the audited TensorIR
inventory.
"""

from __future__ import annotations

import time
import typing
from dataclasses import asdict, dataclass
from hashlib import sha256

import numpy as np
from vibeqc_compiler.tensor import Program, execute

from tools.vibeqc_posthf import MOBlock, ReferenceSnapshot
from tools.vibeqc_posthf.df import DFProvider
from tools.vibeqc_posthf.reference import immutable
from tools.vibeqc_validation.schema import canonical_hash

from .df_contract import DFCCSDTMethodContract
from .doubles import build_ccsd_program
from .equations import amplitude_layouts
from .solver import _DIIS, SolverOptions, _ccsd_solver_region

_DENSE_BLOCKS = ("ovov", "ovvo", "oovv", "ovoo", "oooo")


def _array_hash(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value, dtype="<f8").tobytes()).hexdigest()


def virtual_correction_workspace_bytes(nocc: int, nvir: int) -> int:
    """Conservative logical numeric workspace for one factorized residual call."""

    if any(type(n) is not int or n < 1 for n in (nocc, nvir)):
        raise ValueError("DF RCCSD requires nonempty occupied and virtual spaces")
    doubles = nocc * nocc * nvir * nvir
    # r2 + tau + two contraction temporaries + transpose/add staging, plus
    # conservative matrix/vector work. Inputs B/T are budgeted by their owners.
    return 8 * (
        6 * doubles + nocc * nvir + 6 * nvir * nvir + 4 * nocc * nvir + nocc * nocc
    )


def _validate_factor_inputs(
    bov: typing.Any,
    bvv: typing.Any,
    t1: typing.Any,
    t2: typing.Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bov, bvv, t1, t2 = (np.asarray(x) for x in (bov, bvv, t1, t2))
    if any(x.dtype != np.float64 for x in (bov, bvv, t1, t2)):
        raise ValueError("DF virtual corrections require finite FP64 arrays")
    if bov.ndim != 3 or bvv.ndim != 3 or t1.ndim != 2 or t2.ndim != 4:
        raise ValueError("invalid DF factor/amplitude ranks")
    q, o, v = bov.shape
    if (
        q < 1
        or bvv.shape != (q, v, v)
        or t1.shape != (o, v)
        or t2.shape != (o, o, v, v)
    ):
        raise ValueError("incompatible DF factor/amplitude shapes")
    return bov, bvv, t1, t2


def virtual_corrections(
    bov: typing.Any,
    bvv: typing.Any,
    t1: typing.Any,
    t2: typing.Any,
    *,
    max_bytes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Contract all omitted ovvv/vvvv RCCSD residual terms from DF B factors.

    The formulas are the exact #148/#149 spatial equations after substituting
    g[p,q,r,s] = sum_Q B[Q,p,q] B[Q,r,s].  The auxiliary dimension is reduced
    one Q slice at a time, so no ovvv or vvvv array is formed.
    """

    bov, bvv, t1, t2 = _validate_factor_inputs(bov, bvv, t1, t2)
    _q, o, v = bov.shape
    required = virtual_correction_workspace_bytes(o, v)
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes < required):
        raise MemoryError(
            f"DF virtual residual correction requires {required} numeric bytes"
        )

    # Validate only after admitting scratch, and slice factor checks by Q so
    # validation itself cannot allocate an unbudgeted O(naux*nvir**2) temporary.
    if not all(np.isfinite(x).all() for x in (t1, t2)):
        raise ValueError("DF virtual corrections require finite FP64 arrays")
    for ov, vv in zip(bov, bvv, strict=True):
        if not np.isfinite(ov).all() or not np.isfinite(vv).all():
            raise ValueError("DF virtual corrections require finite FP64 arrays")
        if np.max(np.abs(vv - vv.T)) > 1e-10:
            raise ValueError("B_vv must preserve the symmetric spatial-MO pair")

    r1 = np.zeros_like(t1)
    r2 = np.zeros_like(t2)
    tau = t2 + np.einsum("ia,jb->ijab", t1, t1, optimize=False)

    for q in range(bov.shape[0]):
        ov = bov[q]
        vv = bvv[q]

        # Singles S09/S10/S13/S14.
        s09 = np.einsum("ikcd,kd->ic", t2, ov, optimize=True)
        r1 += 2.0 * (s09 @ vv.T)
        s10 = np.einsum("ikcd,kc->id", t2, ov, optimize=True)
        r1 -= s10 @ vv.T
        ov_t1 = ov.T @ t1
        scalar = float(np.einsum("kd,kd->", ov, t1, optimize=False))
        r1 += 2.0 * scalar * (t1 @ vv.T)
        r1 -= t1 @ (ov_t1 @ vv.T)

        # Lvv(ovvv) -> Xfvv -> D06.
        delta_lvv = 2.0 * scalar * vv - vv @ ov_t1.T
        x = np.einsum("ac,ijcb->ijab", delta_lvv, t2, optimize=True)
        r2 += x + x.transpose(1, 0, 3, 2)

        # Wvoov(ovvv) -> ring/exchange.
        vi = vv @ t1.T  # (a,i)
        ring_jb = np.einsum("kc,kjcb->jb", ov, t2, optimize=True)
        x = 2.0 * np.einsum("ai,jb->ijab", vi, ring_jb, optimize=False)
        r2 += x + x.transpose(1, 0, 3, 2)
        exchange_jb = np.einsum("kc,kjbc->jb", ov, t2, optimize=True)
        x = np.einsum("ai,jb->ijab", vi, exchange_jb, optimize=False)
        r2 -= x + x.transpose(1, 0, 3, 2)

        # Wvovo(ovvv) -> ring/cross.
        ki = ov @ t1.T  # (k,i)
        stage = np.einsum("ki,kjcb->ijcb", ki, t2, optimize=True)
        x = -np.einsum("ac,ijcb->ijab", vv, stage, optimize=True)
        r2 += x + x.transpose(1, 0, 3, 2)
        stage = np.einsum("ki,kjac->ijac", ki, t2, optimize=True)
        x = np.einsum("bc,ijac->ijab", vv, stage, optimize=True)
        r2 -= x + x.transpose(1, 0, 3, 2)

        # Xv(ovvv) -> D02.
        jb = t1 @ vv
        x = np.einsum("ia,jb->ijab", ov, jb, optimize=False)
        r2 += x + x.transpose(1, 0, 3, 2)

        # Complete D05 virtual ladder:
        # + g[a,c,b,d] tau[ijcd]
        # - g[k,d,a,c] t1[k,b] tau[ijcd]
        # - g[k,c,b,d] t1[k,a] tau[ijcd].
        stage = np.einsum("ac,ijcd->ijad", vv, tau, optimize=True)
        r2 += np.einsum("ijad,bd->ijab", stage, vv, optimize=True)
        t1_ov = t1.T @ ov
        r2 -= np.einsum("ijad,bd->ijab", stage, t1_ov, optimize=True)
        stage = np.einsum("ac,ijcd->ijad", t1_ov, tau, optimize=True)
        r2 -= np.einsum("ijad,bd->ijab", stage, vv, optimize=True)

    if not np.isfinite(r1).all() or not np.isfinite(r2).all():
        raise FloatingPointError("DF virtual residual correction became nonfinite")
    return r1, r2


class FactorizedDFIntegralState:
    """Borrow a fresh DFProvider and retain only B_ov/B_vv plus smaller MO blocks."""

    def __init__(self, snapshot: ReferenceSnapshot, provider: DFProvider) -> None:
        if not isinstance(provider, DFProvider):
            raise TypeError("factorized DF RCCSD requires a DFProvider")
        if provider.snapshot.identity != snapshot.identity:
            raise ValueError("DF provider/reference identity mismatch")
        provider._check()
        if provider._cache:
            raise ValueError(
                "factorized DF RCCSD requires a fresh DFProvider with an empty cache"
            )
        self.provider = provider
        self.snapshot = snapshot
        self.bov: np.ndarray | None = None
        self.bvv: np.ndarray | None = None
        self.blocks: dict[str, np.ndarray] = {}
        self._reserved = 0
        try:
            occupied = tuple(range(snapshot.nocc))
            virtual = tuple(range(snapshot.nocc, snapshot.nmo))
            self.bov = provider.three_index(occupied, virtual)
            provider.reserve_external(self.bov.nbytes)
            self._reserved += self.bov.nbytes
            self.bvv = provider.three_index(virtual, virtual)
            provider.reserve_external(self.bvv.nbytes)
            self._reserved += self.bvv.nbytes
            for name in _DENSE_BLOCKS:
                result = provider.get(MOBlock.from_spaces(snapshot, name))
                if (
                    result.reference_id != snapshot.identity
                    or result.hamiltonian_id != snapshot.hamiltonian_id
                ):
                    raise ValueError("DF integral block identity mismatch")
                self.blocks[name] = result.to_host()
        except BaseException:
            self.close()
            raise
        self.bov_hash = _array_hash(self.bov)
        self.bvv_hash = _array_hash(self.bvv)

    def check(self) -> None:
        if self.bov is None or self.bvv is None:
            raise RuntimeError("factorized DF integral state is closed")
        self.provider._check()

    def close(self) -> None:
        self.blocks.clear()
        self.bov = None
        self.bvv = None
        if hasattr(self, "provider"):
            self.provider.clear()
            if self._reserved:
                self.provider.release_external(self._reserved)
                self._reserved = 0


def _df_programs(nocc: int, nvir: int) -> tuple[Program, Program]:
    full = build_ccsd_program(
        nocc,
        nvir,
        form="optimized",
        external_virtual_correction=True,
    )
    program = Program(
        {
            key: full.outputs[key]
            for key in (
                "correlation_energy",
                "singles_residual",
                "doubles_residual",
                "energy_t1",
                "energy_t2",
                "energy_t1t1",
            )
        },
        provenance={
            "full_program_hash": full.logical_hash,
            "df_virtual_factorization": "B_ov+B_vv-v1",
        },
    )
    reference = build_ccsd_program(
        nocc,
        nvir,
        form="expanded",
        diagnostics=False,
        external_virtual_correction=True,
    )
    return program, reference


class PreparedDFCCSD:
    """Prepared CPU DF-RCCSD solve with no resident ovvv/vvvv MO tensors."""

    def __init__(
        self,
        snapshot: ReferenceSnapshot,
        provider: DFProvider,
        contract: DFCCSDTMethodContract,
        options: SolverOptions | None = None,
        t1: np.ndarray | None = None,
        t2: np.ndarray | None = None,
    ) -> None:
        self.options = SolverOptions() if options is None else options
        if not isinstance(self.options, SolverOptions):
            raise TypeError("options must be SolverOptions")
        if not isinstance(snapshot, ReferenceSnapshot) or snapshot.algorithm != "RHF":
            raise TypeError("DF RCCSD requires a validated RHF ReferenceSnapshot")
        if not isinstance(contract, DFCCSDTMethodContract):
            raise TypeError("DF RCCSD requires the #157 method contract")
        if (
            snapshot.identity != contract.correlation_snapshot_identity
            or snapshot.hamiltonian_id != contract.correlation_hamiltonian_id
            or snapshot.geometry_hash != contract.geometry_hash
            or snapshot.basis_hash != contract.orbital_basis_hash
            or contract.fock_policy != "preserve-conventional-rhf"
        ):
            raise ValueError("DF RCCSD method contract/reference mismatch")
        self.snapshot = snapshot
        self.provider = provider
        self.contract = contract
        o, v = snapshot.nocc, snapshot.nmo - snapshot.nocc
        self.program, self.reference_program = _df_programs(o, v)

        def retained(program: Program) -> int:
            return sum(
                node.spec.size * node.spec.itemsize for node in program.live_nodes
            ) + sum(
                node.spec.size * node.spec.itemsize for node in program.outputs.values()
            )

        amplitudes = o * v + o * o * v * v
        self.correction_workspace_bytes = virtual_correction_workspace_bytes(o, v)
        self.logical_required_bytes = (
            max(retained(program) for program in (self.program, self.reference_program))
            + (12 + 2 * self.options.diis_size) * amplitudes * 8
            + self.correction_workspace_bytes
        )
        if self.logical_required_bytes > self.options.max_bytes:
            raise ValueError(
                "DF RCCSD logical solver/correction budget needs "
                f"{self.logical_required_bytes} bytes before integral conversion"
            )

        inputs = {
            node.attrs["name"]: node
            for node in self.program.live_nodes
            if node.op == "input"
        }
        self.amplitude_check = Program({key: inputs[key] for key in ("t1", "t2")})
        if (t1 is None) != (t2 is None):
            raise ValueError("provide both DF RCCSD amplitude arrays or neither")
        if t1 is not None:
            execute(
                self.amplitude_check,
                {"t1": t1, "t2": t2},
                max_bytes=self.options.max_bytes,
            )

        eps = snapshot.orbital_energies
        d1 = eps[:o, None] - eps[None, o:]
        if np.any(d1 >= 0) or np.min(np.abs(d1)) <= self.options.denominator_threshold:
            raise ValueError("near-zero or nonnegative physical DF RCCSD denominator")
        d2 = d1[:, None, :, None] + d1[None, :, None, :]
        if np.min(np.abs(d2)) <= self.options.denominator_threshold:
            raise ValueError("near-zero physical DF RCCSD doubles denominator")
        self.denominators = (
            d1 - self.options.level_shift,
            d2 - 2 * self.options.level_shift,
        )
        self.layouts = amplitude_layouts(o, v)

        self.integrals = FactorizedDFIntegralState(snapshot, provider)
        fock = snapshot.coefficients.T @ snapshot.fock @ snapshot.coefficients
        self.feeds = {
            "foo": fock[:o, :o],
            "fov": fock[:o, o:],
            "fvv": fock[o:, o:],
            **self.integrals.blocks,
        }
        self.integral_hash = canonical_hash(
            {
                **{
                    key: _array_hash(np.asarray(value))
                    for key, value in self.feeds.items()
                },
                "bov": self.integrals.bov_hash,
                "bvv": self.integrals.bvv_hash,
                "omitted_dense_blocks": ("ovvv", "vvvv"),
            }
        )
        self.initial = (
            (
                np.zeros((o, v)),
                self.feeds["ovov"].transpose(0, 2, 1, 3) / d2,
            )
            if t1 is None
            else (np.array(t1, copy=True), np.array(t2, copy=True))
        )
        self.solver_region = _ccsd_solver_region(self)

    def evaluate(
        self,
        t1: np.ndarray,
        t2: np.ndarray,
        *,
        independent: bool = False,
    ) -> dict[str, np.ndarray]:
        self.integrals.check()
        execute(
            self.amplitude_check,
            {"t1": t1, "t2": t2},
            max_bytes=self.options.max_bytes,
        )
        r1_virtual, r2_virtual = virtual_corrections(
            self.integrals.bov,
            self.integrals.bvv,
            t1,
            t2,
            max_bytes=self.correction_workspace_bytes,
        )
        outputs = execute(
            self.reference_program if independent else self.program,
            {
                **self.feeds,
                "df_virtual_singles": r1_virtual,
                "df_virtual_doubles": r2_virtual,
                "t1": t1,
                "t2": t2,
            },
            max_bytes=self.options.max_bytes,
        ).outputs
        return outputs

    def pack(self, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [layout.pack(value) for layout, value in zip(self.layouts, (t1, t2))]
        )

    def unpack(self, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        split = self.layouts[0].size
        return (
            self.layouts[0].unpack(vector[:split]),
            self.layouts[1].unpack(vector[split:]),
        )

    def close(self) -> None:
        self.integrals.close()


@dataclass(frozen=True)
class DFCCSDResult:
    """Converged/failed correlation-DF RCCSD state and factorization evidence."""

    status: str
    reason: str
    correlation_energy: float | None
    total_energy: float | None
    t1: np.ndarray
    t2: np.ndarray
    history: tuple
    provenance: dict

    @property
    def converged(self) -> bool:
        return self.status == "converged"


def _run(prepared: PreparedDFCCSD) -> DFCCSDResult:
    options = prepared.options
    snapshot = prepared.snapshot
    current = prepared.initial
    diis = _DIIS(options.diis_size)
    weights = np.sqrt(
        np.concatenate([np.asarray(layout.weights) for layout in prepared.layouts])
    )
    history = []
    previous_energy = None
    status, reason = "not_converged", "maximum DF RCCSD iterations reached"
    start = time.perf_counter()
    last_finite = current
    energy = None

    for iteration in range(options.max_iterations + 1):
        try:
            out = prepared.evaluate(*current)
            energy = float(out["correlation_energy"])
            r1, r2 = out["singles_residual"], out["doubles_residual"]
            residual = max(float(np.max(np.abs(value))) for value in (r1, r2))
            delta = None if previous_energy is None else abs(energy - previous_energy)
            row = {
                "iteration": iteration,
                "correlation_energy": energy,
                "energy_change": delta,
                "r1_max": float(np.max(np.abs(r1))),
                "r2_max": float(np.max(np.abs(r2))),
                "energy_components": {
                    key: float(out[key])
                    for key in ("energy_t1", "energy_t2", "energy_t1t1")
                },
                "elapsed_seconds": time.perf_counter() - start,
            }
            history.append(row)
            last_finite = current
            if (
                delta is not None
                and delta <= options.energy_tolerance
                and residual <= options.residual_tolerance
            ):
                independent = prepared.evaluate(*current, independent=True)
                row["independent_r1_max"] = float(
                    np.max(np.abs(independent["singles_residual"]))
                )
                row["independent_r2_max"] = float(
                    np.max(np.abs(independent["doubles_residual"]))
                )
                if (
                    max(row["independent_r1_max"], row["independent_r2_max"])
                    <= options.residual_tolerance
                    and abs(float(independent["correlation_energy"]) - energy)
                    <= options.energy_tolerance
                ):
                    status, reason = (
                        "converged",
                        "energy change and fresh factorized expanded R1/R2 passed",
                    )
                    break
            if iteration == options.max_iterations:
                break
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                trial = tuple(
                    amplitude + (1 - options.damping) * residual_value / denominator
                    for amplitude, residual_value, denominator in zip(
                        current, (r1, r2), prepared.denominators
                    )
                )
            trial_out = prepared.evaluate(*trial)
            error = (
                prepared.pack(
                    trial_out["singles_residual"],
                    trial_out["doubles_residual"],
                )
                * weights
            )
            current = prepared.unpack(diis.update(prepared.pack(*trial), error))
            previous_energy = energy
        except (FloatingPointError, OverflowError) as error:
            status, reason = "nonfinite", str(error)
            break
        except ValueError as error:
            if not any(word in str(error).lower() for word in ("finite", "overflow")):
                raise
            status, reason = "nonfinite", str(error)
            break

    provenance = {
        "schema": "vibeqc.df-rccsd.result/1",
        "method": "df-rccsd-correlation-only",
        "method_contract_identity": prepared.contract.identity,
        "reference_id": snapshot.identity,
        "hamiltonian_id": snapshot.hamiltonian_id,
        "fock_policy": prepared.contract.fock_policy,
        "integral_hash": prepared.integral_hash,
        "bov_sha256": prepared.integrals.bov_hash,
        "bvv_sha256": prepared.integrals.bvv_hash,
        "resident_four_index_blocks": _DENSE_BLOCKS,
        "factorized_blocks": ("ovvv", "vvvv"),
        "equation_hash": prepared.program.logical_hash,
        "independent_equation_hash": prepared.reference_program.logical_hash,
        "options": asdict(options),
        "diis_restarts": diis.restarts,
        "logical_required_bytes": prepared.logical_required_bytes,
        "virtual_correction_workspace_bytes": prepared.correction_workspace_bytes,
        "solver_region_identity": prepared.solver_region.identity,
        "reference_energy": snapshot.reference_energy,
        "provider_statistics": dict(prepared.provider.statistics),
    }
    return DFCCSDResult(
        status,
        reason,
        energy,
        None if energy is None else snapshot.reference_energy + energy,
        immutable(last_finite[0]),
        immutable(last_finite[1]),
        tuple(history),
        provenance,
    )


def solve_df_ccsd(
    snapshot: ReferenceSnapshot,
    provider: DFProvider,
    contract: DFCCSDTMethodContract,
    *,
    options: SolverOptions | None = None,
    t1: np.ndarray | None = None,
    t2: np.ndarray | None = None,
) -> DFCCSDResult:
    """Solve the #157 correlation-only DF-RCCSD variant on the CPU."""

    prepared = PreparedDFCCSD(snapshot, provider, contract, options, t1, t2)
    try:
        return _run(prepared)
    finally:
        prepared.close()
