"""Validate the native supplied-Hamiltonian DF CCSD solver at hundreds of AOs.

This adapter consumes independent states from df_ccsdt_large_oracle.py. It is
validation tooling, never a production source or a complete molecular endpoint.
Run preparation with OPENBLAS_NUM_THREADS=1 for exact recorded input bytes. Run
the compiled tests/native/df_cc_solver_probe.cpp on its binary stdin in Slurm,
then invoke this adapter with --result to compare every converged amplitude.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

COLUMNS = (
    "status",
    "energy",
    "iterations",
    "r1",
    "r2",
    "capacity",
    "device_bytes",
    "h2d",
    "scalar_d2h",
    "amplitude_d2h",
    "iterations_called",
    "replays_called",
    "q_calls",
    "q_operations",
    "accumulations",
    "seconds",
)


def sha256(path: Path) -> str:
    """Hash large local artifacts incrementally; they remain outside Git."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(state: Path, reference: dict, target: Path, max_bytes: int) -> dict:
    """Reproduce PySCF's packed AO Hamiltonian, then emit an MP2 initial state.

    The oracle passes lib.pack_tril(B_AO) to PySCF. Its separately stored full
    B_MO can lose symmetry through ill-conditioned whitening/transform roundoff.
    Reconstruct that actual lower-triangle AO source, restore pair symmetry and
    transform it. Average only final floating-point MO pair roundoff. No gate
    is relaxed and no converged reference amplitude is used as a solver guess.
    """
    o, v, q = (int(reference[key]) for key in ("nocc", "nvir", "naux"))
    with np.load(state) as saved:
        c = saved["coefficients"]
        n = c.shape[0]
        assert c.shape == (n, o + v)
        w = saved["metric_whitening"]
        raw = saved["raw_three_center"]
        bao = (w.T @ raw.reshape(n * n, q).T).reshape(q, n, n)
        ao_asymmetry = float(np.max(abs(bao - bao.transpose(0, 2, 1))))
        rows, cols = np.tril_indices(n, -1)
        bao[:, cols, rows] = bao[:, rows, cols]
        b = np.einsum("Qmn,mp,nr->Qpr", bao, c, c, optimize=True)
        b = (b + b.transpose(0, 2, 1)) / 2
        stored = saved["bmo"]
        roundoff = {
            "stored_bmo_asymmetry": float(
                np.max(abs(stored - stored.transpose(0, 2, 1)))
            ),
            "packed_ao_source_asymmetry": ao_asymmetry,
            "canonical_bmo_change": float(np.max(abs(b - stored))),
        }
        eps = saved["orbital_energies"]
        fock = c.T @ saved["fock"] @ c
        oo, ov, vv = b[:, :o, :o], b[:, :o, o:], b[:, o:, o:]
        d1 = eps[:o, None] - eps[None, o:]
        d2 = d1[:, None, :, None] + d1[None, :, None, :]
        ovov = np.einsum("Qia,Qjb->iajb", ov, ov, optimize=True)
        # Exactly the native probe's field order; empty ovvv/vvvv consume zero
        # bytes. Retained smaller blocks come from the same canonical factors.
        fields = (
            fock[:o, :o],
            fock[:o, o:],
            fock[o:, o:],
            ovov,
            ovov.transpose(0, 1, 3, 2),
            np.einsum("Qij,Qab->ijab", oo, vv, optimize=True),
            np.einsum("Qia,Qjk->iajk", ov, oo, optimize=True),
            np.einsum("Qij,Qkl->ijkl", oo, oo, optimize=True),
            d1,
            d2,
            fock[:o, o:] / d1,
            ovov.transpose(0, 2, 1, 3) / d2,
            ov,
            vv,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as stream:
            # uint64: o, v, Q, complete budget, iteration cap, DIIS size, CUDA.
            stream.write(
                np.array([o, v, q, max_bytes, 100, 6, 1], dtype="<u8").tobytes()
            )
            for array in fields:
                stream.write(np.asarray(array, dtype="<f8").tobytes())
    return {
        "input_sha256": sha256(target),
        "input_bytes": target.stat().st_size,
        "factor_roundoff": roundoff,
    }


def qualify(state: Path, reference: dict, result: Path) -> dict:
    """Compare complete amplitudes and energy; failed convergence cannot pass."""
    with result.open() as stream:
        status = dict(zip(COLUMNS, map(float, stream.readline().split()), strict=True))
        amplitudes = np.fromstring(stream.readline(), sep=" ")
        reason = stream.readline().strip()
        if stream.read().strip():
            raise ValueError("unexpected trailing native result data")
    o, v, q = (int(reference[key]) for key in ("nocc", "nvir", "naux"))
    if amplitudes.size != o * v + o * o * v * v or not np.isfinite(amplitudes).all():
        raise ValueError("invalid native amplitude result")
    if not all(np.isfinite(value) for value in status.values()):
        raise ValueError("nonfinite native solver diagnostics")
    t1 = amplitudes[: o * v].reshape(o, v)
    t2 = amplitudes[o * v :].reshape(o, o, v, v)
    with np.load(state) as saved:
        errors = {
            "energy": abs(status["energy"] - reference["correlation_energy"]),
            "t1_max": float(np.max(abs(t1 - saved["t1"]))),
            "t2_max": float(np.max(abs(t2 - saved["t2"]))),
        }
    gates = {"energy": 3e-9, "t1_max": 1e-8, "t2_max": 1e-8, "residual": 1e-10}
    if status["status"] != 0 or max(status["r1"], status["r2"]) > gates["residual"]:
        raise ValueError("native solve did not pass physical convergence replay")
    if not all(errors[key] <= gates[key] for key in errors):
        raise ValueError(f"independent oracle gates failed: {errors}")
    evaluations = status["iterations_called"] + status["replays_called"]
    if (
        status["q_calls"] != q * evaluations
        or status["accumulations"] != 2 * q * evaluations
    ):
        raise ValueError("incomplete auxiliary work accounting")
    return {
        "status": status,
        "reason": reason,
        "absolute_errors": errors,
        "gates": gates,
        "result_sha256": sha256(result),
        "result_bytes": result.stat().st_size,
        "amplitude_sha256": hashlib.sha256(amplitudes.tobytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, default=24 << 30)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    if sha256(args.state) != reference["state_sha256"]:
        raise ValueError("independent state identity mismatch")
    if not args.input and not args.result:
        parser.error(
            "provide --input for preparation and/or --result for qualification"
        )
    record = {
        "case": reference["case"],
        "reference": reference,
        "scope": "Native supplied-Hamiltonian correlation-only DF CCSD from MP2 amplitudes; no native AO source, triples or force endpoint",
    }
    if args.input:
        record.update(prepare(args.state, reference, args.input, args.max_bytes))
    if args.result:
        record.update(qualify(args.state, reference, args.result))
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(args.summary)


if __name__ == "__main__":
    main()
