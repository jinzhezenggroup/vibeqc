"""Checksummed, data-only local traces and independent snapshot counterfactuals.

The schema stores validated scientific identity/arrays rather than Python or
native handles. A portable checkpoint consumer (#204) must still rebuild the
source and apply the destination solver's safeguards. Replay timings are never
reported as complete-solve acceleration.
"""

import json
import math
import time
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
from vibeqc import Atom, Primitive, ResolvedModel, Shell
from vibeqc.profiles import canonical_hash, file_hash

from tools.vibeqc_numerics.audit import StrictHFAudit, _validate_source_model
from tools.vibeqc_posthf.sources import NativeSource

from .proposals import OccupiedProposal, RotationProposal
from .state import DensityProposal, ScfSnapshot, safeguard_policy

MATRICES = ("density", "fock", "residual", "overlap", "baseline")


def source_record(source):
    """Explicit export of molecular inputs; only called by the opt-in writer."""
    return {
        "atoms": [asdict(a) for a in source.atoms],
        "basis": [asdict(s) for s in source.shells],
        "auxiliary_basis": [asdict(s) for s in source.auxiliary_shells],
        "charge": source.charge,
        "representation": source.representation,
    }


def restore_source(record):
    """Rebuild owned runtime resources from plain, normalized input records."""

    def shells(rows):
        return tuple(
            Shell(
                s["atom_index"],
                s["angular_momentum"],
                tuple(Primitive(**p) for p in s["primitives"]),
            )
            for s in rows
        )

    return NativeSource(
        tuple(Atom(a["atomic_number"], tuple(a["position"])) for a in record["atoms"]),
        shells(record["basis"]),
        auxiliary_basis=shells(record["auxiliary_basis"]) or None,
        charge=record["charge"],
        representation="spherical"
        if record["representation"] == "real_spherical"
        else "cartesian",
    )


def export_trace(result, source, path, *, provenance=None):
    """Write a local JSON manifest plus bounded NPZ matrices, with no overwrite.

    Failed/incomplete traces are preserved with their status. Invalid model
    outputs retain hashes and rejection reasons, not unsafe object payloads.
    No molecular data is uploaded and no write occurs during ordinary solves.
    """
    path = Path(path)
    archive = path.with_suffix(".npz")
    if path.suffix != ".json" or path.exists() or archive.exists():
        raise ValueError("trace needs a new .json path and sibling .npz")
    _validate_source_model(source, result.model)
    arrays = {
        f"s{i}_{name}": np.asarray(getattr(s, name), dtype="<f8")
        for i, s in enumerate(result.snapshots)
        for name in MATRICES
    }
    for name in ("density", "forces"):
        value = getattr(result, name)
        if value is not None:
            arrays["final_" + name] = np.asarray(value, dtype="<f8")
    if sum(a.nbytes for a in arrays.values()) > 128 << 20:
        raise ValueError("trace exceeds 128 MiB numeric budget")
    path.parent.mkdir(parents=True, exist_ok=True)
    with archive.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    record = {
        "schema": "vibeqc.scf_trace",
        "schema_version": 1,
        "safeguard": safeguard_policy(),
        "model": result.model.to_dict(),
        "source": source_record(source),
        "owner": result.owner,
        "controls": asdict(result.controls),
        "result": result.metrics(),
        "decisions": result.decisions,
        "snapshots": [{**s.record(), "identity": s.identity} for s in result.snapshots],
        "archive": {
            "file": archive.name,
            "bytes": archive.stat().st_size,
            "sha256": file_hash(archive),
            "shapes": {k: list(a.shape) for k, a in arrays.items()},
        },
        "provenance": provenance or {},
        "replay_is_complete_solve": False,
    }
    record["record_hash"] = canonical_hash(record)
    with path.open("x") as stream:
        json.dump(record, stream, allow_nan=False, sort_keys=True)
        stream.write("\n")
    return path


def load_trace(path):
    """Validate sizes, names, hashes and physical invariants before replay."""
    path = Path(path)
    if path.stat().st_size > 16 << 20:
        raise ValueError("trace manifest exceeds budget")
    record = json.loads(path.read_text())
    digest = record.pop("record_hash")
    if (
        record.get("schema") != "vibeqc.scf_trace"
        or type(record.get("schema_version")) is not int
        or record["schema_version"] != 1
    ):
        raise ValueError("unsupported trace schema")
    if canonical_hash(record) != digest:
        raise ValueError("trace manifest checksum mismatch")
    if record.get("safeguard") != safeguard_policy():
        raise ValueError("unsupported trace safeguard policy")
    model = ResolvedModel.from_dict(record["model"])
    info = record["archive"]
    if Path(info["file"]).name != info["file"]:
        raise ValueError("trace archive must be a sibling")
    archive = path.parent / info["file"]
    if (
        archive.stat().st_size != info["bytes"]
        or archive.stat().st_size > 128 << 20
        or file_hash(archive) != info["sha256"]
    ):
        raise ValueError("trace archive size/checksum mismatch")
    shapes = info["shapes"]
    if len(record["snapshots"]) > 10000 or len(shapes) > 50002:
        raise ValueError("trace entry budget")
    for shape in shapes.values():
        if (
            not 1 <= len(shape) <= 3
            or any(type(n) is not int or not 0 < n <= 4096 for n in shape)
            or math.prod(shape) > 3 * 4096
        ):
            raise ValueError("invalid trace array shape")
    with zipfile.ZipFile(archive) as z:
        entries = z.infolist()
        if len(entries) != len(shapes) or {e.filename for e in entries} != {
            k + ".npy" for k in shapes
        }:
            raise ValueError("trace archive names differ")
        if (
            any(
                e.file_size > 4096 + 8 * math.prod(shapes[e.filename[:-4]])
                for e in entries
            )
            or sum(e.file_size for e in entries) > 128 << 20
        ):
            raise ValueError("trace decompression budget")
    snapshots = []
    with np.load(archive, allow_pickle=False) as arrays:
        for name, shape in shapes.items():
            value = arrays[name]
            if (
                list(value.shape) != shape
                or value.dtype != np.dtype("<f8")
                or not np.isfinite(value).all()
            ):
                raise ValueError("trace array shape/type/finiteness")
        for i, metadata in enumerate(record["snapshots"]):
            if (
                metadata["schema"] != "vibeqc.scf_snapshot"
                or type(metadata["schema_version"]) is not int
                or metadata["schema_version"] != 1
            ):
                raise ValueError("unsupported snapshot schema")
            if (
                ResolvedModel.from_dict(metadata["model"]) != model
                or metadata["owner"] != record["owner"]
            ):
                raise ValueError("cross-owner/model trace state")
            state = ScfSnapshot(
                model,
                metadata["owner"],
                metadata["generation"],
                metadata["iteration"],
                metadata["fock_builds"],
                metadata["energy"],
                *(arrays[f"s{i}_{name}"] for name in MATRICES),
            )
            if state.identity != metadata["identity"] or state.record() != {
                k: v for k, v in metadata.items() if k != "identity"
            }:
                raise ValueError("snapshot content checksum mismatch")
            if snapshots and (
                state.iteration != snapshots[-1].iteration + 1
                or state.generation != snapshots[-1].generation
            ):
                raise ValueError("snapshot ordering/generation mismatch")
            snapshots.append(state)
    return record, tuple(snapshots)


class TargetOperator:
    """Independent FP64 audit contractions over native raw integrals.

    Uses the same Hamiltonian identity as NUM01, including the DF metric cutoff.
    Dense values are explicitly limited to that audit's small-system boundary.
    """

    def __init__(self, source, model):
        self.audit = StrictHFAudit(source, model)
        self.model = model
        self.fock_builds = 0

    def evaluate(self, density):
        a = self.audit
        total = density.sum(axis=0)
        if a.eri is not None:
            j = np.einsum("pqrs,rs->pq", a.eri, total)
            k = np.einsum("prqs,xrs->xpq", a.eri, density)
        else:
            j = np.einsum("pqP,rsP,rs->pq", a.factors, a.factors, total)
            k = np.einsum("prP,qsP,xrs->xpq", a.factors, a.factors, density)
        f = a.hcore + j - (0.5 if self.model.method == "rhf" else 1.0) * k
        energy = float(0.5 * np.sum(density * (a.hcore + f)) + a.nuclear_repulsion)
        residual = f @ density @ a.overlap - a.overlap @ density @ f
        self.fock_builds += 1
        return energy, f, residual


def counterfactual(state, proposer, operator):
    """Re-evaluate an identical snapshot, including all failed trial Focks.

    The original physical Fock is independently verified before any candidate
    evaluation. These costs include Python audit contractions, not a native
    full solve or an inference-only speedup estimate.
    """
    started = time.perf_counter()
    if state.model != operator.model:
        raise ValueError("counterfactual target model mismatch")
    energy, fock, _ = operator.evaluate(state.density)
    if abs(energy - state.energy) > 1e-9 or not np.allclose(
        fock, state.fock, atol=1e-9, rtol=0
    ):
        raise ValueError("snapshot Fock differs from the target operator")
    trials = 0
    action, reason, fraction = "baseline", "no_proposal", 0.0
    selected = state.baseline
    inference_started = time.perf_counter()
    try:
        proposal = None if proposer is None else proposer(state)
        if isinstance(proposal, (OccupiedProposal, RotationProposal)):
            proposal = proposal.materialize(state)
        if proposal is not None and not isinstance(proposal, DensityProposal):
            raise TypeError("proposal type")
        if proposal is not None:
            proposal.validate(state)
            if proposal.representation == "reset":
                action, reason = "reset", "requested_reset"
            else:
                action, reason = "rejected", "target_operator_no_descent"
    except Exception as exc:  # noqa: BLE001 - retain failures from arbitrary optional proposal code.
        proposal = None
        action, reason = "rejected", str(exc)
    inference_seconds = time.perf_counter() - inference_started
    if proposal is not None and action == "rejected":
        for trial_fraction in (1.0, 0.5, 0.25, 0.125):
            trial = (
                1 - trial_fraction
            ) * state.density + trial_fraction * proposal.density
            trial_energy, _, trial_residual = operator.evaluate(trial)
            trials += 1
            norm = float(np.sqrt(np.mean(trial_residual**2)))
            if (
                np.isfinite(trial_energy)
                and np.isfinite(norm)
                and trial_energy <= state.energy + 1e-9
                and norm <= max(1e-12, state.residual_rms * (1 - 1e-4 * trial_fraction))
            ):
                selected, fraction = trial, trial_fraction
                action = "accepted" if fraction == 1 else "damped"
                reason = "target_operator_descent"
                break
    final_energy, _, residual = operator.evaluate(selected)
    return {
        "snapshot_id": state.identity,
        "action": action,
        "reason": reason,
        "fraction": fraction,
        "proposal_trials": trials,
        "fock_builds": trials + 2,
        "energy": final_energy,
        "residual_rms": float(np.sqrt(np.mean(residual**2))),
        "inference_seconds": inference_seconds,
        "replay_seconds": time.perf_counter() - started,
        "complete_solve_measurement": False,
        "intended_state": "unverified",
    }
