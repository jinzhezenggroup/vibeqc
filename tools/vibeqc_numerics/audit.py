"""Small-system native probes and explicit fixed-density/relaxed HF audits.

Reuse #147's owned raw source and immutable-array convention. These diagnostic
records are not post-HF reference snapshots: loose or failed SCF is deliberately
retained, and no correlated consumer may treat it as a converged reference.
All dense audit tensors are capped at 12 orbital/24 auxiliary functions.
"""

from __future__ import annotations

import ctypes as ct
import math
import os
import time
from dataclasses import dataclass

import numpy as np
from vibeqc.accuracy import ResolvedModel

from tools.vibeqc_posthf.df import MetricFactor
from tools.vibeqc_posthf.reference import immutable
from tools.vibeqc_posthf.sources import _DOUBLE, pointer


@dataclass(frozen=True)
class ProbeControls:
    """Iteration and screening controls for one reproducible native solve.

    Arithmetic is FP64 for CPU experiments. CUDA arithmetic must be recorded
    by its execution instrumentation; an environment request alone is not
    evidence of executed FP32 work. A strict probe refuses a mixed override.
    """

    energy_tolerance: float = 1e-13
    density_tolerance: float = 1e-11
    screening_tolerance: float = 0.0
    max_iterations: int = 200
    diis_history: int = 8

    def __post_init__(self):
        for name in ("energy_tolerance", "density_tolerance", "screening_tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise TypeError(f"{name} requires an explicit numeric control")
            if (
                not math.isfinite(value)
                or value < 0
                or (value == 0 and name != "screening_tolerance")
            ):
                raise ValueError(f"invalid {name}")
        for name in ("max_iterations", "diis_history"):
            value = getattr(self, name)
            if (
                type(value) is not int
                or not 0 <= value < 2**32
                or (name == "max_iterations" and value == 0)
            ):
                raise ValueError(f"invalid {name}")


@dataclass(frozen=True, eq=False)
class HFProbe:
    """Detached final density/forces or a failed solve's explicit diagnostics."""

    model: ResolvedModel
    controls: ProbeControls
    backend: str
    energy: float
    energy_change: float | None
    density_rms: float
    iterations: int
    converged: bool
    density: np.ndarray | None
    forces: np.ndarray | None
    solve_seconds: float
    requested_mixed_fock_threshold: float | None = None

    def __post_init__(self):
        if not isinstance(self.model, ResolvedModel) or not isinstance(
            self.controls, ProbeControls
        ):
            raise TypeError("HF probes require typed identities and controls")
        if (
            type(self.converged) is not bool
            or type(self.iterations) is not int
            or self.iterations < 1
        ):
            raise ValueError("invalid HF probe convergence record")
        if self.backend not in ("cpu", "cuda"):
            raise ValueError("unknown HF probe execution backend")
        threshold = self.requested_mixed_fock_threshold
        if threshold is not None and (
            self.backend != "cuda"
            or isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
            or threshold <= self.controls.screening_tolerance
        ):
            raise ValueError("invalid experimental mixed Fock threshold")
        for name in ("energy", "energy_change", "density_rms", "solve_seconds"):
            value = getattr(self, name)
            if value is None and name == "energy_change" and not self.converged:
                continue
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise ValueError("HF probe scalar diagnostics must be finite")
            if name != "energy" and value < 0:
                raise ValueError("HF probe diagnostic magnitude cannot be negative")
        if self.converged:
            density, forces = immutable(self.density), immutable(self.forces)
            spins = 1 if self.model.method == "rhf" else 2
            if (
                density.ndim != 3
                or density.shape[0] != spins
                or not 0 < density.shape[1] <= 12
                or density.shape[1] != density.shape[2]
            ):
                raise ValueError("invalid HF probe density/spin dimensions")
            if not np.allclose(density, density.swapaxes(-1, -2), atol=1e-12, rtol=0):
                raise ValueError("HF probe density must be Hermitian")
            if forces.ndim != 2 or forces.shape[1] != 3 or not len(forces):
                raise ValueError("invalid HF probe force dimensions")
            object.__setattr__(self, "density", density)
            object.__setattr__(self, "forces", forces)
        elif self.density is not None or self.forces is not None:
            raise ValueError("an unconverged probe cannot publish reference state")

    def scalars(self):
        """Serialize diagnostics without silently exporting molecular matrices."""
        return {
            "model_id": self.model.identity,
            "energy": self.energy,
            "energy_change": self.energy_change,
            "density_rms": self.density_rms,
            "iterations": self.iterations,
            "converged": self.converged,
            "solve_seconds": self.solve_seconds,
            "backend": self.backend,
            "storage_precision": "fp64",
            "compute_precision": "fp64" if self.backend == "cpu" else "unreported",
            "reduction_precision": "fp64" if self.backend == "cpu" else "unreported",
            "requested_mixed_fock_threshold": self.requested_mixed_fock_threshold,
            "fp32_work_count": None,
            "host_density_export_bytes": self.density.nbytes
            if self.density is not None
            else 0,
            "reference_snapshot_eligible": False,
        }


def _validate_source_model(source, model):
    """Reuse the public resolver so all audits validate the same scientific inputs."""
    from vibeqc import Calculator

    if not isinstance(model, ResolvedModel):
        raise TypeError("audit requires a typed model")
    if not 0 < source.nbf <= 12 or source.naux > 24:
        raise ValueError("audit domain is at most 12 orbital/24 auxiliary AOs")
    fitted = model.approximation == "density_fitting"
    if fitted and not source.naux:
        raise ValueError("a fitted audit requires explicit source auxiliary data")
    resolved = Calculator(
        method=model.method,
        basis=source.shells,
        basis_representation="spherical"
        if source.representation == "real_spherical"
        else "cartesian",
        density_fitting="cpu" if fitted else "none",
        auxiliary_basis=source.auxiliary_shells if fitted else None,
        density_fitting_relative_threshold=model.metric_relative_threshold or 1e-10,
    ).resolved_model(
        source.atoms, charge=source.charge, multiplicity=model.multiplicity
    )
    if resolved != model:
        raise ValueError("probe source/model identity mismatch")


def probe_hf(
    source,
    model,
    controls=None,
    *,
    backend="cpu",
    device_id=0,
    experimental_mixed_fock_threshold=None,
):
    """Execute a native HF solve and disclose the small-system host export.

    Model construction must use the same source nuclei/basis through the
    public resolver. Identity validation below prevents attaching an unrelated
    model to a density even when its matrix dimensions happen to agree.
    An arithmetic experiment must explicitly match the process environment;
    ordinary strict probes reject mixed overrides. This records a requested
    policy, without claiming that eligible FP32 work actually executed.
    """
    controls = ProbeControls() if controls is None else controls
    if not isinstance(model, ResolvedModel) or not isinstance(controls, ProbeControls):
        raise TypeError("probe requires a typed model and controls")
    if backend not in ("cpu", "cuda") or type(device_id) is not int or device_id < 0:
        raise ValueError("invalid HF probe backend/device")
    _validate_source_model(source, model)
    # This older probe ABI uses the source's minimum-spin descriptor. Generic
    # source/model validation and fixed-density audits are spin-independent;
    # SOL01's separate solve ABI supplies an explicit multiplicity.
    if model.multiplicity != (1 if source.electron_count % 2 == 0 else 2):
        raise ValueError(
            "the raw-source probe currently supports minimum-spin references"
        )
    fitted = model.approximation == "density_fitting"
    selection = os.environ.get("VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD", "0")
    if experimental_mixed_fock_threshold is None:
        if backend == "cuda" and selection not in ("", "0", "0.0", "none"):
            raise ValueError(
                "FP64 audit refuses an active experimental mixed-precision override"
            )
    else:
        threshold = experimental_mixed_fock_threshold
        if (
            backend != "cuda"
            or isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
            or threshold <= controls.screening_tolerance
        ):
            raise ValueError("invalid experimental mixed Fock threshold")
        if float(selection) != threshold:
            raise ValueError("mixed Fock experiment does not match process settings")
    spin_count = 1 if model.method == "rhf" else 2
    density = np.empty((spin_count, source.nbf, source.nbf))
    forces = np.empty((len(source.atoms), 3))
    scalars = np.empty(5)
    function = source._library.vibeqc_accuracy_hf_probe_v1
    function.argtypes = [
        ct.c_void_p,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.c_uint,
        ct.c_uint,
        ct.c_double,
        ct.c_double,
        ct.c_double,
        ct.c_int,
        ct.c_double,
        _DOUBLE,
        ct.c_size_t,
        _DOUBLE,
        ct.c_size_t,
        _DOUBLE,
        ct.c_size_t,
        ct.c_char_p,
        ct.c_size_t,
    ]
    function.restype = ct.c_int
    start = time.perf_counter()
    with source._lock:
        source._check_open()
        source._call(
            "vibeqc_accuracy_hf_probe_v1",
            source._handle,
            1 if model.method == "rhf" else 2,
            int(backend == "cuda"),
            device_id,
            controls.max_iterations,
            controls.diis_history,
            controls.energy_tolerance,
            controls.density_tolerance,
            controls.screening_tolerance,
            int(fitted),
            model.metric_relative_threshold or 1e-10,
            pointer(density),
            density.size,
            pointer(forces),
            forces.size,
            pointer(scalars),
            scalars.size,
        )
    elapsed = time.perf_counter() - start
    converged = bool(scalars[4])
    return HFProbe(
        model,
        controls,
        backend,
        float(scalars[0]),
        float(scalars[1]) if np.isfinite(scalars[1]) else None,
        float(scalars[2]),
        int(scalars[3]),
        converged,
        density if converged else None,
        forces if converged else None,
        elapsed,
        experimental_mixed_fock_threshold,
    )


class StrictHFAudit:
    """Independent FP64 operator audit in a deliberately tiny, dense domain.

    The native raw-source evaluator supplies unscreened values, independently
    of the production CUDA Fock. NumPy contractions evaluate the physical
    commutator and fixed-density energy. Relaxed target energies/forces still
    require a separate native solve; this object never declares convergence.
    """

    def __init__(self, source, model):
        start = time.perf_counter()
        _validate_source_model(source, model)
        self.model = model
        self.overlap, self.hcore = map(immutable, source.one_electron())
        n = source.nbf
        self.metric = None
        self.factors = None
        self.eri = None
        if model.approximation == "conventional":
            self.eri = immutable(
                source._read("four_center_eri", (0, 0, 0, 0), (n, n, n, n))
            )
        else:
            self.metric = MetricFactor.from_source(
                source, relative_threshold=model.metric_relative_threshold
            )
            raw = source._read("three_center_eri", (0, 0, 0), (n, n, source.naux))
            self.factors = immutable(raw @ self.metric.inverse_square_root)
        eigenvalues, vectors = np.linalg.eigh(self.overlap)
        if eigenvalues[0] <= 1e-10:
            raise ValueError("strict audit rejects a linearly dependent AO metric")
        self.orthogonalizer = immutable((vectors / np.sqrt(eigenvalues)) @ vectors.T)
        self.overlap_condition = float(eigenvalues[-1] / eigenvalues[0])
        self.nuclear_repulsion = 0.0
        for i, atom in enumerate(source.atoms):
            for other in source.atoms[:i]:
                distance = float(
                    np.linalg.norm(np.asarray(atom.position) - other.position)
                )
                if distance == 0:
                    raise ValueError(
                        "coincident nuclei do not define a finite molecular audit"
                    )
                self.nuclear_repulsion += (
                    atom.atomic_number * other.atomic_number / distance
                )
        self.setup_seconds = time.perf_counter() - start
        self.resident_array_bytes = sum(
            a.nbytes
            for a in (
                self.overlap,
                self.hcore,
                self.orthogonalizer,
                self.eri,
                self.factors,
            )
            if a is not None
        )

    def evaluate(self, probe: HFProbe) -> dict:
        """Evaluate the actual target at the saved density without relaxing it."""
        if probe.model != self.model:
            raise ValueError("strict audit model mismatch")
        if not probe.converged or probe.density is None:
            raise ValueError("failed probe has no density to audit")
        start = time.perf_counter()
        density = probe.density
        if density.shape[1:] != self.overlap.shape:
            raise ValueError("probe density does not match the audited AO space")
        total = density.sum(axis=0)
        if self.eri is not None:
            coulomb = np.einsum("pqrs,rs->pq", self.eri, total, optimize=True)
            exchange = np.einsum("prqs,xrs->xpq", self.eri, density, optimize=True)
        else:
            coulomb = np.einsum(
                "pqP,rsP,rs->pq", self.factors, self.factors, total, optimize=True
            )
            exchange = np.einsum(
                "prP,qsP,xrs->xpq", self.factors, self.factors, density, optimize=True
            )
        spin_factor = 0.5 if self.model.method == "rhf" else 1.0
        fock = self.hcore + coulomb - spin_factor * exchange
        energy = float(
            0.5 * np.sum(density * (self.hcore + fock)) + self.nuclear_repulsion
        )
        residual = fock @ density @ self.overlap - self.overlap @ density @ fock
        x = self.orthogonalizer
        physical_residual = float(np.max(np.abs(x.T @ residual @ x)))
        eps = np.linalg.eigvalsh(x.T @ fock @ x)
        counts = (
            [self.model.electron_count // 2]
            if self.model.method == "rhf"
            else [
                (self.model.electron_count + self.model.multiplicity - 1) // 2,
                (self.model.electron_count - self.model.multiplicity + 1) // 2,
            ]
        )
        gaps = [
            float(e[n] - e[n - 1])
            for e, n in zip(eps, counts, strict=True)
            if 0 < n < len(e)
        ]
        traces = np.einsum("xpq,qp->x", density, self.overlap)
        expected = [self.model.electron_count] if self.model.method == "rhf" else counts
        return {
            "scope": "fixed_density",
            "model_id": self.model.identity,
            "energy": energy,
            "energy_operator_difference": abs(energy - probe.energy),
            "orthonormal_commutator_max": physical_residual,
            "ao_commutator_max": float(np.max(np.abs(residual))),
            "density_idempotency_max": float(
                np.max(np.abs(spin_factor * density @ self.overlap @ density - density))
            ),
            "electron_trace_error_max": float(np.max(np.abs(traces - expected))),
            "overlap_condition": self.overlap_condition,
            "minimum_gap": min(gaps) if gaps else None,
            "gap_is_error_certificate": False,
            "metric_rank": self.metric.rank if self.metric else None,
            "metric_condition": self.metric.condition_number if self.metric else None,
            "audit_seconds": time.perf_counter() - start,
        }


def error_features(source, probe, audit):
    """Extract verified basis identity and physical diagnostics for calibration.

    A name such as STO-3G is assigned only after comparing actual expanded
    basis data. Custom, changed or larger bases retain their own identity and
    therefore cannot accidentally enter this initial calibration domain.
    """
    from vibeqc import Calculator
    from vibeqc.accuracy_estimator import HFErrorFeatures

    if audit["model_id"] != probe.model.identity:
        raise ValueError("error features require a matching physical audit")
    _validate_source_model(source, probe.model)
    builtin = Calculator(method=probe.model.method, basis="sto-3g")
    expected = builtin.resolved_model(
        source.atoms, charge=source.charge, multiplicity=probe.model.multiplicity
    )
    family = (
        builtin._basis.identity
        if expected.basis_hash == probe.model.basis_hash
        else "custom:" + probe.model.basis_hash
    )
    distances = [
        float(np.linalg.norm(np.asarray(a.position) - b.position))
        for i, a in enumerate(source.atoms)
        for b in source.atoms[:i]
    ]
    return HFErrorFeatures(
        probe.model.identity,
        family,
        tuple(a.atomic_number for a in source.atoms),
        source.nbf,
        audit["orthonormal_commutator_max"],
        audit["overlap_condition"],
        audit["minimum_gap"],
        audit["electron_trace_error_max"],
        audit["density_idempotency_max"],
        min(distances) if distances else None,
        backend=probe.backend,
        precision="fp64" if probe.backend == "cpu" else "unreported",
    )
