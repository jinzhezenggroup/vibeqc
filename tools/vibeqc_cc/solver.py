"""Internal conventional CPU CCSD iterations, separate from physical equations."""

import json
import time
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
from hashlib import sha256
from pathlib import Path

import numpy as np
from vibeqc_compiler.tensor import Program, execute

from tools.vibeqc_posthf import MOBlock, ReferenceSnapshot
from tools.vibeqc_posthf.providers import ConventionalProvider
from tools.vibeqc_posthf.reference import immutable
from tools.vibeqc_validation.schema import canonical_hash

from .doubles import build_ccsd_program
from .equations import amplitude_layouts


def _provider_blocks(provider, snapshot, names):
    """Dry-run the complete CPU cache transition before any AO conversion.

    This internal adapter mirrors #147's pin/LRU retention under its reentrant
    lock. It reads its private cache accounting but does not change the shared
    provider API. Saved-MO replay/test adapters have no transformation cache.
    """
    blocks = {name: MOBlock.from_spaces(snapshot, name) for name in names}
    with getattr(provider, "_lock", nullcontext()):
        if hasattr(provider, "_cache"):
            if provider._closed:
                raise RuntimeError("CCSD integral provider is closed")
            if provider.source.identity != provider._source_identity:
                raise ValueError("integral source changed before CCSD preparation")
            provider.source._check_open()
            retained = provider._retained
            cache = OrderedDict(
                (key, cost) for key, (_, cost) in provider._cache.items()
            )
            for block in blocks.values():
                plan = provider.plan(block)
                if plan.peak_bytes > provider.budget_bytes:
                    raise MemoryError(
                        "CCSD provider block exceeds budget before integral conversion"
                    )
                key = (snapshot.identity, provider.source.identity, block.slots)
                if key in cache:
                    cache.move_to_end(key)
                    continue
                while (
                    cache
                    and retained + plan.peak_bytes > provider.budget_bytes
                    and provider.cache_policy == "lru"
                ):
                    _, cost = cache.popitem(last=False)
                    retained -= cost
                if retained + plan.peak_bytes > provider.budget_bytes:
                    raise MemoryError(
                        "CCSD complete provider block set exceeds budget before integral conversion"
                    )
                cache[key] = 8 * plan.output_elements
                retained += cache[key]
        return {name: provider.get(block) for name, block in blocks.items()}


@dataclass(frozen=True)
class SolverOptions:
    """CPU iteration controls with fixed scientific acceptance limits.

    Damping, level shifts and CC DIIS affect proposals only; final acceptance
    uses the original physical equations. ``max_bytes`` bounds logical numeric
    solver/interpreter storage independently of the provider's cache budget;
    it is not a Python/BLAS process-RSS cap.
    """

    max_iterations: int = 100
    energy_tolerance: float = 1e-11
    residual_tolerance: float = 1e-9
    denominator_threshold: float = 1e-10
    damping: float = 0.0
    level_shift: float = 0.0
    diis_size: int = 6
    max_bytes: int = 256 << 20

    def __post_init__(self):
        for key in ("max_iterations", "max_bytes", "diis_size"):
            v = getattr(self, key)
            if type(v) is not int or v < (0 if key == "diis_size" else 1):
                raise ValueError(f"invalid CCSD option {key}")
        if self.diis_size not in (0, *range(2, 21)):
            raise ValueError("diis_size must be zero or between 2 and 20")
        for key in (
            "energy_tolerance",
            "residual_tolerance",
            "denominator_threshold",
            "damping",
            "level_shift",
        ):
            if not np.isfinite(getattr(self, key)):
                raise ValueError(f"nonfinite CCSD option {key}")
        if (
            not 0 < self.energy_tolerance <= 1e-8
            or not 0 < self.residual_tolerance <= 1e-9
        ):
            raise ValueError(
                "CCSD tolerances must respect energy <=1e-8 and physical residual <=1e-9"
            )
        if (
            self.denominator_threshold <= 0
            or not 0 <= self.damping < 1
            or self.level_shift < 0
        ):
            raise ValueError("invalid CCSD denominator/damping/level shift")


class PreparedCCSD:
    """Borrow a conventional provider and pin seven MO blocks under its budget.

    max_bytes bounds the equation interpreter plus conservative logical solver
    work/DIIS arrays. The provider has a separate budget. Neither bounds total
    Python/BLAS RSS. Preflight occurs before any integral read.
    """

    def __init__(self, snapshot, provider, options=None, t1=None, t2=None):
        self.options = SolverOptions() if options is None else options
        if not isinstance(self.options, SolverOptions):
            raise TypeError("options must be SolverOptions")
        if not isinstance(snapshot, ReferenceSnapshot):
            raise TypeError("CCSD requires a validated RHF ReferenceSnapshot")
        if snapshot.algorithm != "RHF":
            raise ValueError("CCSD requires an RHF reference")
        if not isinstance(provider, ConventionalProvider) or provider.backend != "cpu":
            raise ValueError("CCSD requires a conventional CPU integral provider")
        if snapshot.identity != provider.snapshot.identity:
            raise ValueError("CCSD provider/reference identity mismatch")
        self.snapshot, self.provider = snapshot, provider
        o, v = snapshot.nocc, snapshot.nmo - snapshot.nocc
        full_program = build_ccsd_program(o, v, form="optimized")
        self.program = Program(
            {
                k: full_program.outputs[k]
                for k in (
                    "correlation_energy",
                    "singles_residual",
                    "doubles_residual",
                    "energy_t1",
                    "energy_t2",
                    "energy_t1t1",
                )
            },
            provenance={"full_program_hash": full_program.logical_hash},
        )
        self.reference_program = build_ccsd_program(
            o, v, form="expanded", diagnostics=False
        )

        def retained(p):
            return sum(n.spec.size * n.spec.itemsize for n in p.live_nodes) + sum(
                n.spec.size * n.spec.itemsize for n in p.outputs.values()
            )

        amplitudes = o * v + o * o * v * v
        self.logical_required_bytes = (
            max(retained(p) for p in (self.program, self.reference_program))
            + (12 + 2 * self.options.diis_size) * amplitudes * 8
        )
        if self.logical_required_bytes > self.options.max_bytes:
            raise ValueError(
                f"CCSD logical budget needs {self.logical_required_bytes} bytes before integral conversion"
            )
        self.layouts = amplitude_layouts(o, v)
        inputs = {
            n.attrs["name"]: n for n in self.program.live_nodes if n.op == "input"
        }
        self.amplitude_check = Program({k: inputs[k] for k in ("t1", "t2")})
        if (t1 is None) != (t2 is None):
            raise ValueError("provide both CCSD amplitude arrays or neither")
        if t1 is not None:
            self.validate_amplitudes(t1, t2)
        eps = snapshot.orbital_energies
        d1 = eps[:o, None] - eps[None, o:]
        # Diagnose the physical canonical denominators before applying a shift.
        if np.any(d1 >= 0) or np.min(np.abs(d1)) <= self.options.denominator_threshold:
            raise ValueError(
                "near-zero or nonnegative physical CCSD denominator; no regularization applied"
            )
        d2 = d1[:, None, :, None] + d1[None, :, None, :]
        if np.min(np.abs(d2)) <= self.options.denominator_threshold:
            raise ValueError("near-zero physical CCSD doubles denominator")
        self.denominators = (
            d1 - self.options.level_shift,
            d2 - 2 * self.options.level_shift,
        )
        f = snapshot.coefficients.T @ snapshot.fock @ snapshot.coefficients
        self.feeds = {"foo": f[:o, :o], "fov": f[:o, o:], "fvv": f[o:, o:]}
        for name, result in _provider_blocks(
            provider, snapshot, ("ovov", "ovvo", "oovv", "ovvv", "ovoo", "oooo", "vvvv")
        ).items():
            if (
                result.reference_id != snapshot.identity
                or result.hamiltonian_id != snapshot.hamiltonian_id
            ):
                raise ValueError("CCSD integral block identity mismatch")
            self.feeds[name] = result.to_host()
        self.integral_hash = canonical_hash(
            {
                k: sha256(np.ascontiguousarray(a, dtype="<f8").tobytes()).hexdigest()
                for k, a in self.feeds.items()
            }
        )
        self.initial = (
            (np.zeros((o, v)), self.feeds["ovov"].transpose(0, 2, 1, 3) / d2)
            if t1 is None
            else (np.array(t1, copy=True), np.array(t2, copy=True))
        )

    def validate_amplitudes(self, t1, t2):
        execute(
            self.amplitude_check, {"t1": t1, "t2": t2}, max_bytes=self.options.max_bytes
        )

    def evaluate(self, t1, t2, *, independent=False):
        if getattr(self.provider, "_closed", False):
            raise RuntimeError("CCSD integral provider is closed")
        if self.provider.snapshot.identity != self.snapshot.identity:
            raise ValueError("provider reference changed during CCSD")
        self.provider.source._check_open()
        return execute(
            self.reference_program if independent else self.program,
            {**self.feeds, "t1": t1, "t2": t2},
            max_bytes=self.options.max_bytes,
        ).outputs

    def pack(self, t1, t2):
        return np.concatenate(
            [layout.pack(a) for layout, a in zip(self.layouts, (t1, t2))]
        )

    def unpack(self, vector):
        split = self.layouts[0].size
        return self.layouts[0].unpack(vector[:split]), self.layouts[1].unpack(
            vector[split:]
        )


class _DIIS:
    """Independent CC history of actual (amplitude, physical residual) pairs."""

    def __init__(self, size):
        self.size = size
        self.vectors, self.errors = [], []
        self.restarts = 0

    def update(self, vector, error):
        if not self.size:
            return vector
        self.vectors.append(vector.copy())
        self.errors.append(error.copy())
        self.vectors = self.vectors[-self.size :]
        self.errors = self.errors[-self.size :]
        while len(self.vectors) > 1:
            errors = np.array(self.errors)
            gram = errors @ errors.T
            scale = float(np.max(np.abs(gram)))
            if scale == 0:
                return vector
            n = len(gram)
            B = np.empty((n + 1, n + 1))
            B[:n, :n] = gram / scale
            B[n, :] = B[:, n] = -1
            B[n, n] = 0
            rhs = np.zeros(n + 1)
            rhs[n] = -1
            try:
                coefficients = np.linalg.solve(B, rhs)[:n]
                if (
                    not np.isfinite(coefficients).all()
                    or np.max(np.abs(coefficients)) > 1e6
                ):
                    raise np.linalg.LinAlgError("ill-conditioned CC DIIS weights")
                return coefficients @ np.array(self.vectors)
            except np.linalg.LinAlgError:
                self.vectors.pop(0)
                self.errors.pop(0)
                self.restarts += 1
        return vector


@dataclass(frozen=True)
class CCSDResult:
    """Owned last finite state, convergence evidence and reproducible inputs.

    A finite energy alone does not imply convergence: inspect ``converged`` or
    ``status``. Failure results retain their reason and iteration history, and
    can be written/replayed without turning exhaustion into successful CCSD.
    """

    status: str
    reason: str
    correlation_energy: float | None
    total_energy: float | None
    t1: np.ndarray
    t2: np.ndarray
    history: tuple
    provenance: dict
    replay_inputs: dict

    @property
    def converged(self):
        return self.status == "converged"

    def write(self, path):
        """Save successful or failed finite state and enough input to replay it."""
        record = {
            "schema": "vibeqc.ccsd.result",
            "version": 1,
            "status": self.status,
            "reason": self.reason,
            "correlation_energy": self.correlation_energy,
            "total_energy": self.total_energy,
            "t1": self.t1.tolist(),
            "t2": self.t2.tolist(),
            "history": self.history,
            "provenance": self.provenance,
            "inputs": self.replay_inputs,
            "inputs_hash": canonical_hash(self.replay_inputs),
        }
        record["record_hash"] = canonical_hash(record)
        Path(path).write_text(
            json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )


def solve(snapshot, provider, *, options=None, t1=None, t2=None):
    """Solve conventional RCCSD from an owned RHF reference and CPU provider.

    Supply both real FP64 T1/T2 arrays or neither (the default is an MP2-like
    guess). The provider is borrowed and must remain open during the call.
    Invalid references, amplitudes, denominators or budgets raise before MO
    conversion. Numerical failure/exhaustion returns a nonconverged result.
    Only fresh expanded physical R1/R2 and delta-E can accept the final root;
    the implementation never calls an external coupled-cluster solver.
    """
    prepared = PreparedCCSD(snapshot, provider, options, t1, t2)
    options = prepared.options
    current = prepared.initial
    diis = _DIIS(options.diis_size)
    weights = np.sqrt(np.concatenate([np.asarray(p.weights) for p in prepared.layouts]))
    history = []
    previous_energy = None
    status, reason = "not_converged", "maximum CCSD iterations reached"
    start = time.perf_counter()
    last_finite = current
    energy = None
    for iteration in range(options.max_iterations + 1):
        try:
            out = prepared.evaluate(*current)
            energy = float(out["correlation_energy"])
            r1, r2 = out["singles_residual"], out["doubles_residual"]
            residual = max(float(np.max(np.abs(r))) for r in (r1, r2))
            delta = None if previous_energy is None else abs(energy - previous_energy)
            row = {
                "iteration": iteration,
                "correlation_energy": energy,
                "energy_change": delta,
                "r1_max": float(np.max(np.abs(r1))),
                "r2_max": float(np.max(np.abs(r2))),
                "energy_components": {
                    k: float(out[k]) for k in ("energy_t1", "energy_t2", "energy_t1t1")
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
                        "energy change and freshly expanded physical R1/R2 passed",
                    )
                    break
            if iteration == options.max_iterations:
                break
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                trial = tuple(
                    t + (1 - options.damping) * r / d
                    for t, r, d in zip(current, (r1, r2), prepared.denominators)
                )
            # DIIS errors belong to exactly the trial vector being stored.
            trial_out = prepared.evaluate(*trial)
            error = (
                prepared.pack(
                    trial_out["singles_residual"], trial_out["doubles_residual"]
                )
                * weights
            )
            current = prepared.unpack(diis.update(prepared.pack(*trial), error))
            previous_energy = energy
        except (FloatingPointError, OverflowError) as error:
            status, reason = "nonfinite", str(error)
            break
        except ValueError as error:
            if not any(s in str(error).lower() for s in ("finite", "overflow")):
                raise
            status, reason = "nonfinite", str(error)
            break
    provenance = {
        "reference_id": snapshot.identity,
        "hamiltonian_id": snapshot.hamiltonian_id,
        "integral_hash": prepared.integral_hash,
        "equation_hash": prepared.program.logical_hash,
        "independent_equation_hash": prepared.reference_program.logical_hash,
        "options": asdict(options),
        "diis_restarts": diis.restarts,
        "logical_required_bytes": prepared.logical_required_bytes,
        "reference_energy": snapshot.reference_energy,
        "solver_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    replay = {k: v.tolist() for k, v in prepared.feeds.items()}
    replay.update(
        initial_t1=prepared.initial[0].tolist(),
        initial_t2=prepared.initial[1].tolist(),
        orbital_energies=snapshot.orbital_energies.tolist(),
        snapshot={
            field.name: (
                getattr(snapshot, field.name).tolist()
                if isinstance(getattr(snapshot, field.name), np.ndarray)
                else getattr(snapshot, field.name)
            )
            for field in fields(snapshot)
            if field.init
        },
    )
    return CCSDResult(
        status,
        reason,
        energy,
        None if energy is None else snapshot.reference_energy + energy,
        immutable(last_finite[0]),
        immutable(last_finite[1]),
        tuple(history),
        provenance,
        replay,
    )
