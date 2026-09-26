"""Energy-only public RCCSD facade over the audited #148/#149 solver tiers.

This is the issue #149 C-slice API. It registers an energy-only capability and
single-point entry point in the internal CC facade, the same tier as the #148
``solve`` entry point. The native C-ABI ``VIBEQC_METHOD`` registry entry is
deliberately *not* added here: a public ``VIBEQC_METHOD_RCCSD`` needs the #193
device-resident provider interface as its prerequisite, and this module does not
invent one. Force requests are rejected here directly; frozen-core, ECP,
open-shell and non-RHF references are rejected by the #147 reference/provider
validation the solver reuses (never silently reinterpreted as HF or as a
different correlated method).
"""

from __future__ import annotations

import typing
from dataclasses import dataclass
from pathlib import Path

from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter

from .solver import CCSDResult, solve


@dataclass(frozen=True)
class Capabilities:
    """Energy-only capability record for the internal RCCSD facade."""

    method: str
    family: str
    available: bool
    supports_batch: bool
    supported_properties: frozenset

    def __post_init__(self) -> None:
        if self.method != "rccsd" or self.family != "coupled_cluster":
            raise ValueError("RCCSD capability identity mismatch")
        if "energy" not in self.supported_properties:
            raise ValueError("RCCSD must report the energy property")


def method_capabilities(method: str = "rccsd") -> Capabilities:
    """Report RCCSD support without constructing a reference or provider.

    ``supports_batch`` describes native prepared batching, which this
    experimental helper does not implement. The Python
    :func:`batch_energy` helper executes independent per-item states instead.
    """
    if method.lower() != "rccsd":
        raise ValueError(f"unknown method {method!r}")
    return Capabilities(
        method="rccsd",
        family="coupled_cluster",
        available=True,
        supports_batch=False,
        supported_properties=frozenset({"energy"}),
    )


@dataclass(frozen=True)
class RCCSDResult:
    """Energy-only single-point result plus the complete replayable solver state.

    ``state`` is the #148 ``CCSDResult`` and carries amplitude, history,
    provenance and replay inputs. Convergence is judged only by that state's
    independently re-evaluated physical residuals and energy.
    """

    backend: str
    reference_energy: float
    correlation_energy: float | None
    total_energy: float | None
    state: CCSDResult

    @property
    def status(self) -> typing.Any:
        return self.state.status

    @property
    def reason(self) -> typing.Any:
        return self.state.reason

    @property
    def converged(self) -> typing.Any:
        return self.state.converged

    @property
    def iterations(self) -> typing.Any:
        return len(self.state.history)

    @property
    def final_r1_max(self) -> typing.Any:
        return (
            self.state.history[-1].get("independent_r1_max")
            if self.state.history
            else None
        )

    @property
    def final_r2_max(self) -> typing.Any:
        return (
            self.state.history[-1].get("independent_r2_max")
            if self.state.history
            else None
        )

    @property
    def t1(self) -> typing.Any:
        return self.state.t1

    @property
    def t2(self) -> typing.Any:
        return self.state.t2

    @property
    def history(self) -> typing.Any:
        return self.state.history

    @property
    def provenance(self) -> typing.Any:
        return self.state.provenance

    def write(self, path: typing.Any) -> typing.Any:
        """Export the replayable state (replayed by ``tools.replay_ccsd``)."""
        return self.state.write(path)


def energy(
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
    compute_forces: typing.Any = False,
) -> typing.Any:
    """Return the energy-only RCCSD single point from an owned RHF reference.

    ``backend`` selects the physical-equation evaluator: ``"cpu"`` is the #148
    interpreter; ``"cuda"`` compiles the same TensorIR through #146 and
    requires an explicit ``CudaCompilerAdapter`` and cache. ``"cuda-resident"``
    keeps T/R/integrals/DIIS resident across iterations and reads only scalar
    convergence control until final acceptance. Forces,
    frozen cores, ECP, open shells and non-RHF references raise before AO work;
    a CUDA provider failure never silently falls back to CPU CCSD.
    """
    if compute_forces:
        raise NotImplementedError(
            "RCCSD exposes energy only; forces are not implemented"
        )
    if warm_start is not None:
        from .gpu_state import AmplitudeSnapshot

        if not isinstance(warm_start, AmplitudeSnapshot):
            raise TypeError("RCCSD warm_start must be AmplitudeSnapshot")
        if t1 is not None or t2 is not None:
            raise ValueError("RCCSD warm_start cannot be combined with raw t1/t2")
        if backend != "cuda-resident":
            t1, t2 = warm_start.for_reference(snapshot)
    if backend == "cpu":
        state = solve(snapshot, provider, options=options, t1=t1, t2=t2)
    elif backend in ("cuda", "cuda-resident"):
        if not isinstance(compiler, CudaCompilerAdapter):
            raise ValueError("CUDA RCCSD requires a CudaCompilerAdapter")
        if not isinstance(cache, Path):
            raise ValueError("CUDA RCCSD requires a pathlib.Path cache")
        if backend == "cuda":
            from .gpu_solver import solve_gpu as selected_solver
        else:
            from .resident_solver import solve_gpu_resident as selected_solver

        kwargs = {
            "compiler": compiler,
            "cache": cache,
            "options": options,
            "t1": t1,
            "t2": t2,
            "device": device,
            "provider_peak_bytes": provider_peak_bytes,
        }
        if backend == "cuda-resident":
            kwargs["warm_start"] = warm_start
        state = selected_solver(snapshot, provider, **kwargs)
    else:
        raise ValueError("RCCSD backend must be 'cpu', 'cuda' or 'cuda-resident'")
    reference = snapshot.reference_energy
    return RCCSDResult(
        backend=backend,
        reference_energy=reference,
        correlation_energy=state.correlation_energy,
        total_energy=state.total_energy,
        state=state,
    )


@dataclass(frozen=True)
class BatchItemResult:
    """Isolated per-system energy outcome; a failure never corrupts neighbors."""

    index: int
    status: str
    reason: str
    backend: str
    converged: bool
    correlation_energy: float | None
    total_energy: float | None
    state: CCSDResult | None


@dataclass(frozen=True)
class BatchRCCSDResult:
    """Input-ordered energy batch; ragged AO shapes are run independently."""

    items: tuple


def batch_energy(
    problems: typing.Any,
    *,
    backend: typing.Any = "cpu",
    options: typing.Any = None,
    compiler: typing.Any = None,
    cache: typing.Any = None,
    device: typing.Any = 0,
    provider_peak_bytes: typing.Any = 0,
) -> typing.Any:
    """Execute independent energy-only systems, isolating every item's state.

    ``problems`` is an iterable of ``(snapshot, provider)`` pairs. Each item
    owns its amplitudes, denominators, DIIS and status; an exception on one
    item is recorded and the remaining items continue. Homogeneous-shape
    GPU compilation is deliberately *not* grouped or padded here: grouping
    into a native prepared fleet remains required production work under #149.
    """
    items = []
    for index, (snapshot, provider) in enumerate(problems):
        try:
            result = energy(
                snapshot,
                provider,
                backend=backend,
                options=options,
                compiler=compiler,
                cache=cache,
                device=device,
                provider_peak_bytes=provider_peak_bytes,
            )
            items.append(
                BatchItemResult(
                    index=index,
                    status=result.status,
                    reason=result.reason,
                    backend=result.backend,
                    converged=result.converged,
                    correlation_energy=result.correlation_energy,
                    total_energy=result.total_energy,
                    state=result.state,
                )
            )
        except Exception as error:  # noqa: BLE001 - item isolation is the contract
            items.append(
                BatchItemResult(
                    index=index,
                    status="error",
                    reason=f"{type(error).__name__}: {error}",
                    backend=backend,
                    converged=False,
                    correlation_energy=None,
                    total_energy=None,
                    state=None,
                )
            )
    return BatchRCCSDResult(tuple(items))
