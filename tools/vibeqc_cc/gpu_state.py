"""CC-specific state contracts and generated Jacobi step for the GPU solver.

No CUDA execution or global budget API lives here. #146 owns allocation and
lowering, #147 owns reference compatibility; this module keeps those boundaries
while defining CC iteration controls separately from the physical equations.
"""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from tools.vibeqc_posthf.reference import ReferenceSnapshot, immutable
from tools.vibeqc_tensor import Program, add, divide, execute, input_tensor
from tools.vibeqc_tensor.cuda_plan import INT_MAX, Reservations, aligned, plan_cuda
from tools.vibeqc_tensor.types import checked_size

from .doubles import build_ccsd_program
from .equations import amplitude_specs
from .solver import SolverOptions


def iteration_program(nocc, nvir, *, damping=0.0):
    """Original physical R plus T + (1-damping)*R/D, as typed TensorIR.

    D is a resident preconditioner input, not part of R. The final acceptance
    DAG is separately built with form='expanded' and has no D dependency.
    """
    SolverOptions(damping=damping)
    physical = build_ccsd_program(nocc, nvir, form="shared", diagnostics=False)
    inputs = {n.attrs["name"]: n for n in physical.live_nodes if n.op == "input"}
    outputs = dict(physical.outputs)
    for index, residual in enumerate(("singles_residual", "doubles_residual"), 1):
        t = inputs[f"t{index}"]
        d = input_tensor(f"d{index}", t.spec)
        outputs[f"next_t{index}"] = add(
            t,
            divide(physical.outputs[residual], d),
            coefficients=(1, Fraction(1) - Fraction(float(damping))),
        )
    return Program(
        outputs,
        provenance={
            "physical_equation": physical.logical_hash,
            "iteration": "damped Jacobi; shifted denominators are controls only",
        },
    )


def denominators(snapshot, options):
    """Preflight physical gaps once, before any integral or device allocation."""
    if not isinstance(snapshot, ReferenceSnapshot):
        raise TypeError("GPU RCCSD requires a validated ReferenceSnapshot")
    # The shared snapshot also represents KS states for CPKS; RCCSD still
    # requires the RHF reference used by the audited physical equations.
    if snapshot.algorithm != "RHF":
        raise ValueError("GPU RCCSD requires an RHF reference")
    if not isinstance(options, SolverOptions):
        raise TypeError("GPU RCCSD options must be SolverOptions")
    o = snapshot.nocc
    eps = snapshot.orbital_energies
    d1 = eps[:o, None] - eps[None, o:]
    d2 = d1[:, None, :, None] + d1[None, :, None, :]
    if any(
        np.any(d >= 0) or np.min(np.abs(d)) <= options.denominator_threshold
        for d in (d1, d2)
    ):
        raise ValueError("near-zero or nonnegative physical CCSD denominator")
    return tuple(
        immutable(d - i * options.level_shift) for i, d in enumerate((d1, d2), 1)
    )


def state_reservation(nocc, nvir, diis_size):
    """CC offsets inside a #146 reservation, with no allocation or global policy.

    Main-plan inputs/outputs already own current T/R, denominators and next T.
    Reserve last-finite T, dense DIIS histories, bounded solve scratch and
    physical residual reductions. The final replay plan is charged separately.
    """
    SolverOptions(diis_size=diis_size)
    specs = amplitude_specs(nocc, nvir)
    n1, n2 = (s.size for s in specs)
    elements = checked_size(n1 + n2, "CC amplitude elements")
    if elements > INT_MAX:
        raise ValueError("CC DIIS dimension exceeds cuBLAS integer range")
    history = diis_size
    counts = {
        "last_t1": n1,
        "last_t2": n2,
        "diis_vectors": history * elements,
        "diis_errors": history * elements,
        "gram": history * history,
        "system": (history + 1) ** 2 if history else 0,
        "coefficients": history + 1 if history else 0,
        "r1_partials": min((n1 + 255) // 256, 65535),
        "r2_partials": min((n2 + 255) // 256, 65535),
        "scalars": 4,  # R1/R2 maxima, DIIS status and arithmetic status storage.
    }
    segments, offset = {}, 0
    for name, count in counts.items():
        size = checked_size(count * 8, name + " bytes")
        segments[name] = {"offset": offset, "bytes": size}
        offset = checked_size(offset + aligned(size), "CC state reservation")
    return Reservations(diis=offset), segments


def solver_plans(nocc, nvir, target, options, *, provider_peak_bytes=0):
    """Compose CC plans under #146's numeric-buffer budget, before AO work.

    The independent expanded replay is always retained and charged, even
    when this makes a budget infeasible. The caller supplies the shared #147
    provider estimate; no transformation or hidden host fallback runs here.
    Returned plans are preparation data, not evidence of GPU convergence.
    """
    if not isinstance(options, SolverOptions):
        raise TypeError("GPU RCCSD options must be SolverOptions")
    checked_size(provider_peak_bytes, "provider peak")
    available = options.max_bytes - provider_peak_bytes
    if available <= 0:
        raise ValueError("CC budget exhausted by integral provider")
    replay = plan_cuda(
        build_ccsd_program(nocc, nvir, form="expanded", diagnostics=False),
        target,
        max_bytes=available,
    )
    reservation, segments = state_reservation(nocc, nvir, options.diis_size)
    remainder = available - replay.peak_bytes
    if remainder <= 0:
        raise ValueError("CC budget cannot retain independent physical replay")
    primary = plan_cuda(
        iteration_program(nocc, nvir, damping=options.damping),
        target,
        max_bytes=remainder,
        reservations=reservation,
    )
    return (
        primary,
        replay,
        {
            "provider_peak_bytes": provider_peak_bytes,
            "primary_peak_bytes": primary.peak_bytes,
            "independent_replay_peak_bytes": replay.peak_bytes,
            "combined_peak_bytes": provider_peak_bytes
            + primary.peak_bytes
            + replay.peak_bytes,
            "state_segments": segments,
            "diis_capacity": options.diis_size,
            "scope": "#146 numeric buffers plus caller-supplied #147 provider peak",
        },
    )


@dataclass(frozen=True)
class AmplitudeSnapshot:
    """Owned amplitudes tied to the exact physical reference, never shape alone.

    This is an explicit warm-start value. It does not assert convergence and
    never transports orbitals across geometries. All incoming amplitudes are
    revalidated before device upload, including simultaneous pair symmetry.
    """

    reference_id: str
    t1: np.ndarray
    t2: np.ndarray

    def __post_init__(self):
        if not isinstance(self.reference_id, str) or not self.reference_id:
            raise ValueError("warm start requires a reference identity")
        for name in ("t1", "t2"):
            a = getattr(self, name)
            if not isinstance(a, np.ndarray) or a.dtype != np.float64:
                raise ValueError("warm-start amplitudes must be FP64 ndarrays")
        if self.t1.ndim != 2:
            raise ValueError("warm-start singles must be rank two")
        specs = amplitude_specs(*self.t1.shape)
        execute(
            Program({k: input_tensor(k, s) for k, s in zip(("t1", "t2"), specs)}),
            {"t1": self.t1, "t2": self.t2},
        )
        object.__setattr__(self, "t1", immutable(self.t1))
        object.__setattr__(self, "t2", immutable(self.t2))

    def for_reference(self, snapshot):
        if not isinstance(snapshot, ReferenceSnapshot):
            raise TypeError("warm start requires a validated ReferenceSnapshot")
        if snapshot.algorithm != "RHF":
            raise ValueError("GPU RCCSD warm start requires an RHF reference")
        if self.reference_id != snapshot.identity:
            raise ValueError(
                "warm start invalidated by reference/geometry/orbital change"
            )
        if self.t1.shape != (snapshot.nocc, snapshot.nmo - snapshot.nocc):
            raise ValueError("warm-start shape does not match reference")
        return self.t1, self.t2
