"""Experimental host-staged RCCSD validation over the #146 CUDA executor.

This helper does not satisfy #149 B/C: resident T/R iteration and native
registry/prepared-batch integration remain required production work.

The audited physical equations and final acceptance are exactly #148's: this
module swaps only the residual-evaluation backend. Inputs, Fock/integral
blocks, denominators and the MP2-like initial guess are prepared by the same
:class:`tools.vibeqc_cc.solver.PreparedCCSD` used by the CPU solver, so the GPU
solver cannot drift from the audited CPU reference, provider and control law.

Each iteration executes the whole damped-Jacobi TensorIR on one ordinary
stream, so amplitudes and residuals are staged through host buffers on every
iteration; those transfers are counted and reported explicitly. A
device-resident T/R loop where the host reads only small residual scalars is
required by #149 step 2 and is *not* implemented here. This solver is
therefore an honestly reported ordinary-stream host-controlled iteration, not
a hidden resident acceleration, and it must not be advertised as such.
"""

from __future__ import annotations

import time
import typing
from contextlib import ExitStack
from dataclasses import asdict, fields
from hashlib import sha256
from pathlib import Path

import numpy as np
from vibeqc_compiler.integral.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.tensor.cuda_execute import PreparedCuda, compile_cuda

from tools.vibeqc_posthf.reference import ReferenceSnapshot, immutable

from .gpu_state import solver_plans
from .solver import _DIIS, CCSDResult, PreparedCCSD, SolverOptions


class PreparedGPUSolver:
    """Own a compiled GPU residual/energy evaluator and independent replay.

    ``cpu`` carries the complete validated provider/integral/denominator
    preparation and the amplitude layout metric. The two compiled plans share
    nothing mutable: the primary evaluates the shared-form physical DAG (and
    stages the damped-Jacobi proposal tensors on device), and the replay
    re-evaluates the expanded physical DAG for final acceptance. The host,
    not the device, forms the Jacobi trial from the residual outputs. Budgets
    follow :func:`tools.vibeqc_cc.gpu_state.solver_plans` exactly, so an
    infeasible combined peak fails before CUDA allocation.
    """

    def __init__(
        self,
        snapshot: typing.Any,
        provider: typing.Any,
        compiler: typing.Any,
        cache: typing.Any,
        *,
        options: typing.Any = None,
        t1: typing.Any = None,
        t2: typing.Any = None,
        device: typing.Any = 0,
        provider_peak_bytes: typing.Any = 0,
    ) -> None:
        if not isinstance(compiler, CudaCompilerAdapter):
            raise TypeError("GPU RCCSD requires a CudaCompilerAdapter")
        if not isinstance(cache, Path):
            raise TypeError("GPU RCCSD cache must be a pathlib.Path")
        if not isinstance(snapshot, ReferenceSnapshot):
            raise TypeError("CCSD requires a validated RHF ReferenceSnapshot")
        if type(device) is not int or device < 0:
            raise ValueError("device must be a nonnegative visible CUDA ordinal")
        self.options = SolverOptions() if options is None else options
        self.snapshot = snapshot
        o, v = snapshot.nocc, snapshot.nmo - snapshot.nocc
        self.shape = (o, v)
        self.compiler = compiler
        self.cache = cache
        self.device = device
        target = compiler.target
        primary, replay, diagnostic = solver_plans(
            o, v, target, self.options, provider_peak_bytes=provider_peak_bytes
        )
        # Reject the combined CUDA budget before PreparedCCSD reads any MO
        # blocks. Its independent host budget and scientific preflight still
        # apply before compilation or allocation.
        self.cpu = PreparedCCSD(snapshot, provider, self.options, t1, t2)
        self.diagnostic = diagnostic
        # A failed replay compile/allocation can retain this constructor's
        # traceback. Release the primary immediately so batch neighbors can
        # reuse its budget without waiting for garbage collection.
        with ExitStack() as cleanup:
            self.primary = cleanup.enter_context(
                PreparedCuda(
                    primary, compile_cuda(primary, compiler, cache), device=device
                )
            )
            self.replay = cleanup.enter_context(
                PreparedCuda(
                    replay, compile_cuda(replay, compiler, cache), device=device
                )
            )
            self._transfer = self._transfer_accounting(primary, replay)
            cleanup.pop_all()
        # Host-controlled DIIS storage follows the CPU solver's bounded
        # allowance; the plan reservation charges the future resident path.

    @staticmethod
    def _transfer_accounting(primary: typing.Any, replay: typing.Any) -> typing.Any:
        """Per-evaluation transfer sizes and successful execution counters.

        The host reads only ``correlation_energy`` and the two residual
        tensors to drive the iteration; the `next_t1/next_t2` proposal outputs
        are staged by the plan yet not consumed on the host (the control law
        forms its trial in numpy). The download figures below charge the
        complete plan outputs: the residual-driven iteration only reads a
        subset, so this is a conservative upper bound, not an undercount.
        """

        def tensor_bytes(node: typing.Any) -> typing.Any:
            return node.spec.size * node.spec.itemsize

        inputs = sum(
            tensor_bytes(primary.program.live_nodes[s]) for s in primary.inputs
        )
        outputs = sum(
            tensor_bytes(primary.program.live_nodes[s]) for _, s in primary.outputs
        )
        residual_only = sum(
            tensor_bytes(primary.program.live_nodes[s])
            for name, s in primary.outputs
            if name in ("singles_residual", "doubles_residual", "correlation_energy")
        )
        replay_inputs = sum(
            tensor_bytes(replay.program.live_nodes[s]) for s in replay.inputs
        )
        return {
            "primary_upload_bytes_per_evaluation": inputs,
            "primary_download_bytes_per_evaluation": outputs,
            "residual_scalar_download_bytes_per_evaluation": residual_only,
            # Ordinary iterations evaluate both current and trial amplitudes;
            # each call uploads the complete integral inputs again.
            "primary_evaluations": 0,
            "independent_replay_evaluations": 0,
            "independent_replay_input_bytes": replay_inputs,
        }

    # Delegate the shared CPU preparation surface so the control law is one code
    # path regardless of backend.
    @property
    def options_(self) -> typing.Any:
        return self.cpu.options

    @property
    def initial(self) -> typing.Any:
        return self.cpu.initial

    @property
    def layouts(self) -> typing.Any:
        return self.cpu.layouts

    @property
    def denominators(self) -> typing.Any:
        return self.cpu.denominators

    @property
    def integral_hash(self) -> typing.Any:
        return self.cpu.integral_hash

    @property
    def logical_required_bytes(self) -> typing.Any:
        return self.cpu.logical_required_bytes

    def pack(self, t1: typing.Any, t2: typing.Any) -> typing.Any:
        return self.cpu.pack(t1, t2)

    def unpack(self, vector: typing.Any) -> typing.Any:
        return self.cpu.unpack(vector)

    def evaluate(
        self, t1: typing.Any, t2: typing.Any, *, independent: typing.Any = False
    ) -> typing.Any:
        """Physical energy/R1/R2 (+ Jacobi proposal) on the primary or replay DAG."""
        provider = self.cpu.provider
        provider.source._check_open()
        if getattr(provider, "_closed", False):
            raise RuntimeError("CCSD integral provider is closed")
        if provider.snapshot.identity != self.snapshot.identity:
            raise ValueError("provider reference changed during CCSD")
        executor = self.replay if independent else self.primary
        d1, d2 = self.cpu.denominators
        feeds = {
            **self.cpu.feeds,
            "t1": t1,
            "t2": t2,
            "d1": d1,
            "d2": d2,
        }
        try:
            result = executor.execute(feeds)
            count = (
                "independent_replay_evaluations"
                if independent
                else "primary_evaluations"
            )
            self._transfer[count] += 1
            return result.outputs
        except RuntimeError as error:
            # TensorIR transports arithmetic failures through its native error
            # string. Match only those explicit boundaries; driver, compiler
            # and allocation failures must keep their original exception.
            if str(error).startswith(
                ("non-finite tensor at step ", "tensor division by zero at step ")
            ):
                raise FloatingPointError(str(error)) from error
            raise

    def close(self) -> None:
        self.primary.close()
        self.replay.close()

    def __enter__(self) -> typing.Any:
        return self

    def __exit__(self, *unused: object) -> None:
        self.close()


def solve_gpu(
    snapshot: typing.Any,
    provider: typing.Any,
    *,
    compiler: typing.Any,
    cache: typing.Any,
    options: typing.Any = None,
    t1: typing.Any = None,
    t2: typing.Any = None,
    device: typing.Any = 0,
    provider_peak_bytes: typing.Any = 0,
) -> typing.Any:
    """Solve conventional RCCSD with the physical equations evaluated on CUDA.

    The control law (MP2-like guess, damped shifted-denominator Jacobi, host
    CC DIIS, and expanded-DAG final acceptance) is identical to
    :func:`tools.vibeqc_cc.solve`. Only the residual/energy evaluation runs on
    the GPU; amplitudes and residuals cross the host boundary every iteration,
    which the returned provenance reports explicitly. This is an experimental
    validation helper, not the resident production solver required by #149. ``compiler`` and
    ``cache`` are required so compilation is never implicit. Provider failures
    never silently switch to CPU CCSD: they raise, exactly as on the CPU path.
    """
    with PreparedGPUSolver(
        snapshot,
        provider,
        compiler,
        cache,
        options=options,
        t1=t1,
        t2=t2,
        device=device,
        provider_peak_bytes=provider_peak_bytes,
    ) as prepared:
        options = prepared.options
        current = prepared.initial
        diis = _DIIS(options.diis_size)
        weights = np.sqrt(
            np.concatenate([np.asarray(p.weights) for p in prepared.layouts])
        )
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
                delta = (
                    None if previous_energy is None else abs(energy - previous_energy)
                )
                row = {
                    "iteration": iteration,
                    "correlation_energy": energy,
                    "energy_change": delta,
                    "r1_max": float(np.max(np.abs(r1))),
                    "r2_max": float(np.max(np.abs(r2))),
                    "elapsed_seconds": time.perf_counter() - start,
                    "backend": "cuda-fp64-ordinary-stream",
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
                            "energy change and freshly expanded physical R1/R2 passed on GPU",
                        )
                        break
                if iteration == options.max_iterations:
                    break
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    trial = tuple(
                        t + (1 - options.damping) * r / d
                        for t, r, d in zip(current, (r1, r2), prepared.denominators)
                    )
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
            "equation_hash": prepared.primary.plan.program.provenance.get(
                "physical_equation"
            ),
            "iteration_equation_hash": prepared.primary.plan.program.logical_hash,
            "independent_equation_hash": prepared.replay.plan.program.logical_hash,
            "options": asdict(options),
            "diis_restarts": diis.restarts,
            "logical_required_bytes": prepared.logical_required_bytes,
            "reference_energy": snapshot.reference_energy,
            "backend": "cuda-fp64-ordinary-stream",
            "device": prepared.primary.device,
            "graph_status": prepared.primary.graph_status,
            "primary_peak_bytes": prepared.primary.plan.peak_bytes,
            "independent_replay_peak_bytes": prepared.replay.plan.peak_bytes,
            "combined_peak_bytes": prepared.diagnostic["combined_peak_bytes"],
            "transfer": prepared._transfer,
            "residency": (
                "ordinary-stream host-controlled iteration; T/R cross host "
                "each iteration, resident T/R requires the #193 interface"
            ),
            "replay": "tools.replay_ccsd reproduces GPU results on the CPU solver",
            "solver_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        replay = {k: v.tolist() for k, v in prepared.cpu.feeds.items()}
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
