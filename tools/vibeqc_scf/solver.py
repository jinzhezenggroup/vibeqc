"""Synchronous native CPU solves with per-item callback ownership.

This development bridge uses actual conventional/DF RHF/UHF loops, including
final force assembly. It never invokes an optional ML package, downloads a
model, exports a trace by default, or redirects CUDA execution to the host.
"""

import ctypes as ct
import threading
import time
import uuid
from dataclasses import dataclass

import numpy as np

from tools.vibeqc_numerics.audit import HFProbe, ProbeControls, _validate_source_model
from tools.vibeqc_posthf.reference import immutable
from tools.vibeqc_posthf.sources import _DOUBLE, pointer

from .state import DensityProposal, ScfSnapshot, spin_counts, validate_density


class _SnapshotView(ct.Structure):
    _fields_ = [
        ("generation", ct.c_uint64),
        ("iteration", ct.c_uint),
        ("nbf", ct.c_size_t),
        ("spins", ct.c_size_t),
        ("fock_builds", ct.c_size_t),
        ("electrons", ct.POINTER(ct.c_uint)),
        ("weight", ct.c_double),
        ("energy", ct.c_double),
        ("residual_rms", ct.c_double),
        *(
            (name, _DOUBLE)
            for name in ("density", "fock", "residual", "overlap", "baseline")
        ),
    ]


class _DecisionView(ct.Structure):
    _fields_ = [
        ("action", ct.c_int),
        ("reason", ct.c_char_p),
        ("trials", ct.c_uint),
        *(
            (name, ct.c_double)
            for name in (
                "fraction",
                "energy",
                "residual_rms",
                "inference_seconds",
                "validation_seconds",
                "operator_seconds",
            )
        ),
    ]


_PROPOSE = ct.CFUNCTYPE(
    ct.c_int,
    ct.POINTER(_SnapshotView),
    _DOUBLE,
    ct.POINTER(ct.c_uint64),
    ct.POINTER(ct.c_uint),
)
_OBSERVE = ct.CFUNCTYPE(None, ct.POINTER(_SnapshotView), ct.POINTER(_DecisionView))
_ACTIONS = ("baseline", "accepted", "damped", "rejected", "reset")


@dataclass(frozen=True)
class ScfSolve:
    """Complete-solve result; failed iterates remain ineligible as references."""

    model: object
    controls: ProbeControls
    owner: str
    status: str
    error: str | None
    energy: float | None
    energy_change: float | None
    density_rms: float | None
    iterations: int | None
    fock_builds: int | None
    density: np.ndarray | None
    forces: np.ndarray | None
    snapshots: tuple
    decisions: tuple
    solve_seconds: float
    trace_seconds: float
    trace_complete: bool

    @property
    def converged(self):
        return self.status == "converged"

    def as_probe(self):
        """Adapt a converged solve to NUM01's independent physical audit."""
        if not self.converged:
            raise ValueError("failed solve is not a converged probe")
        return HFProbe(
            self.model,
            self.controls,
            "cpu",
            self.energy,
            self.energy_change,
            self.density_rms,
            self.iterations,
            True,
            self.density,
            self.forces,
            self.solve_seconds,
        )

    def metrics(self):
        """All work, including rejected proposals and fallback, belongs here."""
        return {
            "status": self.status,
            "error": self.error,
            "energy": self.energy,
            "energy_change": self.energy_change,
            "density_rms": self.density_rms,
            "iterations": self.iterations,
            "fock_builds": self.fock_builds,
            "solve_seconds": self.solve_seconds,
            "trace_seconds": self.trace_seconds,
            "trace_complete": self.trace_complete,
            "inference_seconds": sum(d["inference_seconds"] for d in self.decisions),
            "callback_seconds": sum(d["callback_seconds"] for d in self.decisions),
            "materialization_seconds": sum(
                d["materialization_seconds"] for d in self.decisions
            ),
            "validation_seconds": sum(d["validation_seconds"] for d in self.decisions),
            "repair_seconds": sum(d["repair_seconds"] for d in self.decisions),
            "trial_operator_seconds": sum(
                d["operator_seconds"] for d in self.decisions
            ),
            "proposal_trials": sum(d["trials"] for d in self.decisions),
            "stability": "not_evaluated",
            "intended_state": "unverified",
            "reference_snapshot_eligible": False,
        }


def solve(
    source,
    model,
    controls=None,
    *,
    proposer=None,
    initial_density=None,
    capture=False,
    owner=None,
    max_trace_iterations=10000,
):
    """Run one complete CPU solve. Capture is local, bounded and explicit.

    ``proposer(snapshot)`` returns ``None`` or a typed DensityProposal. Snapshot
    and candidate buffers are copied at the boundary. Callback failures become
    recorded rejections/resets; no exception may escape a ctypes trampoline.
    Trace storage failure marks the result as failed instead of pretending a
    partial trace is complete. External exports require a separate explicit call.
    """
    started = time.perf_counter()
    controls = ProbeControls() if controls is None else controls
    if not isinstance(controls, ProbeControls) or controls.screening_tolerance != 0:
        raise ValueError("CPU proposal bridge requires typed unscreened controls")
    if controls.max_iterations > 10000 or controls.diis_history > 100:
        raise ValueError("SCF proposal bridge control limit")
    if (
        type(capture) is not bool
        or type(max_trace_iterations) is not int
        or not 1 <= max_trace_iterations <= 10000
    ):
        raise ValueError("invalid trace controls")
    if proposer is not None and not callable(proposer):
        raise TypeError("proposer must be callable or None")
    _validate_source_model(source, model)
    if source.backend != "cpu-reference-native-shell-tiles":
        raise ValueError("SCF proposal callbacks are CPU only")
    owner = uuid.uuid4().hex if owner is None else owner
    if not isinstance(owner, str) or not owner:
        raise ValueError("owner must be a nonempty string")
    spins = len(spin_counts(model)[0])
    shape = (spins, source.nbf, source.nbf)
    initial = None
    if initial_density is not None:
        initial = validate_density(initial_density, source.one_electron()[0], model)
    density = np.full(shape, np.nan)
    forces = np.full((len(source.atoms), 3), np.nan)
    scalars = np.full(6, np.nan)
    states, decisions, callback_errors = [], [], []
    pending = {}
    trace_seconds = 0.0
    trace_complete = capture

    def snapshot(view):
        def matrix(name):
            dims = (
                (view.nbf, view.nbf)
                if name == "overlap"
                else (view.spins, view.nbf, view.nbf)
            )
            return np.ctypeslib.as_array(
                getattr(view, name), shape=(int(np.prod(dims)),)
            ).reshape(dims)

        return ScfSnapshot(
            model,
            owner,
            view.generation,
            view.iteration,
            view.fock_builds,
            view.energy,
            *(
                matrix(n)
                for n in ("density", "fock", "residual", "overlap", "baseline")
            ),
        )

    @_PROPOSE
    def propose(view_ptr, out, parent, iteration):
        nonlocal trace_seconds
        view = view_ptr.contents
        parent[0], iteration[0] = view.generation, view.iteration
        details = {
            "inference_seconds": 0.0,
            "materialization_seconds": 0.0,
            "repair_seconds": 0.0,
            "repair": "none",
            "adapter_error": None,
            "proposal_hash": None,
        }
        pending["details"] = details
        try:
            copying = time.perf_counter()
            state = snapshot(view)
            trace_seconds += time.perf_counter() - copying
            pending["state"] = state
            inference_started = time.perf_counter()
            try:
                proposal = proposer(state)
            finally:
                details["inference_seconds"] = time.perf_counter() - inference_started
            if proposal is None:
                return 0
            from .proposals import OccupiedProposal, RotationProposal

            if isinstance(proposal, (OccupiedProposal, RotationProposal)):
                materializing = time.perf_counter()
                try:
                    proposal = proposal.materialize(state)
                finally:
                    details["materialization_seconds"] = (
                        time.perf_counter() - materializing
                    )
            if not isinstance(proposal, DensityProposal):
                raise TypeError("proposer must return a DensityProposal or None")
            details.update(
                repair_seconds=proposal.repair_seconds, repair=proposal.repair
            )
            # Content metadata records failed candidates without retaining a
            # mutable model-owned buffer or serializing arbitrary Python objects.
            from hashlib import sha256

            details["proposal_hash"] = sha256(
                proposal.density.astype("<f8").tobytes()
            ).hexdigest()
            if proposal.parent_id != state.identity:
                parent[0] = 0  # Native rejection independently checks stale state.
            if proposal.representation == "reset":
                if proposal.parent_id != state.identity:
                    raise ValueError("stale_state")
                return 3
            if proposal.density.shape != shape:
                raise ValueError("shape")
            np.ctypeslib.as_array(out, shape=(density.size,))[:] = (
                proposal.density.ravel()
            )
            return 2 if proposal.representation == "determinant_density" else 1
        except BaseException as exc:  # noqa: BLE001 - no Python exception may cross the C callback ABI.
            details["adapter_error"] = f"{type(exc).__name__}: {exc}"
            # Reset leaves the already computed traditional next step intact.
            return 3

    @_OBSERVE
    def observe(view_ptr, decision_ptr):
        nonlocal trace_seconds, trace_complete
        copying = time.perf_counter()
        try:
            view, decision = view_ptr.contents, decision_ptr.contents
            details = pending.pop(
                "details",
                {
                    "inference_seconds": 0.0,
                    "materialization_seconds": 0.0,
                    "repair_seconds": 0.0,
                    "repair": "none",
                    "adapter_error": None,
                    "proposal_hash": None,
                },
            )
            row = {name: getattr(decision, name) for name, _ in _DecisionView._fields_}
            # The native callback envelope includes copying and transformations;
            # inference is timed around the user generator alone. Declared repair
            # cost may overlap inference/materialization and is not additive.
            row["callback_seconds"] = row["inference_seconds"]
            row.update(
                action=_ACTIONS[decision.action],
                reason=decision.reason.decode(),
                iteration=view.iteration,
                **details,
            )
            decisions.append(row)
            state = pending.pop("state", None)
            if capture:
                if len(states) >= max_trace_iterations:
                    trace_complete = False
                else:
                    states.append(state if state is not None else snapshot(view))
        except BaseException as exc:  # noqa: BLE001 - no Python exception may cross the C callback ABI.
            callback_errors.append(f"{type(exc).__name__}: {exc}")
            trace_complete = False
        trace_seconds += time.perf_counter() - copying

    fn = source._library.vibeqc_scf_solve_v1
    fn.argtypes = [
        ct.c_void_p,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.c_uint,
        ct.c_uint,
        ct.c_double,
        ct.c_double,
        ct.c_double,
        _DOUBLE,
        _PROPOSE,
        _OBSERVE,
        _DOUBLE,
        ct.c_size_t,
        _DOUBLE,
        ct.c_size_t,
        _DOUBLE,
        ct.c_size_t,
        ct.c_char_p,
        ct.c_size_t,
    ]
    fn.restype = ct.c_int
    error = ct.create_string_buffer(2048)
    with source._lock:
        source._check_open()
        code = fn(
            source._handle,
            1 if model.method == "rhf" else 2,
            model.multiplicity,
            int(model.approximation == "density_fitting"),
            controls.max_iterations,
            controls.diis_history,
            controls.energy_tolerance,
            controls.density_tolerance,
            model.metric_relative_threshold or 1e-10,
            pointer(initial) if initial is not None else None,
            propose if proposer is not None else _PROPOSE(),
            observe if capture or proposer is not None else _OBSERVE(),
            pointer(density),
            density.size,
            pointer(forces),
            forces.size,
            pointer(scalars),
            scalars.size,
            error,
            len(error),
        )
    status = (
        "error"
        if code or callback_errors
        else "converged"
        if scalars[4] == 1
        else "nonconverged"
    )
    message = error.value.decode() if code else "; ".join(callback_errors) or None

    def scalar(i):
        return float(scalars[i]) if np.isfinite(scalars[i]) else None

    return ScfSolve(
        model,
        controls,
        owner,
        status,
        message,
        scalar(0),
        scalar(1),
        scalar(2),
        int(scalars[3]) if np.isfinite(scalars[3]) else None,
        int(scalars[5]) if np.isfinite(scalars[5]) else None,
        immutable(density) if np.isfinite(density).all() else None,
        immutable(forces)
        if status == "converged" and np.isfinite(forces).all()
        else None,
        tuple(states),
        tuple(decisions),
        time.perf_counter() - started,
        trace_seconds,
        trace_complete and not code,
    )


class ScfItem:
    """Serial per-item ownership for a caller-owned source and immutable model.

    Rebinding changes the owner epoch and drops previous results. Every solve
    gets a fresh native generation and private DIIS history. Warm/projection
    seeds are explicit inputs, never shared automatically across items.
    """

    def __init__(self, source, model):
        self._lock = threading.RLock()
        self._busy = False
        self.rebind(source, model)

    def rebind(self, source, model):
        with self._lock:
            if self._busy:
                raise RuntimeError("cannot rebind an active SCF item")
            _validate_source_model(source, model)
            self._source, self._model = source, model
            self._owner = uuid.uuid4().hex
            self.last_result = None

    def solve(self, **kwargs):
        with self._lock:
            if self._busy:
                raise RuntimeError("cannot recursively solve an active SCF item")
            self._busy = True
            try:
                self.last_result = solve(
                    self._source, self._model, owner=self._owner, **kwargs
                )
                return self.last_result
            finally:
                self._busy = False
