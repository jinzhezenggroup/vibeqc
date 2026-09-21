"""Generated CUDA execution candidate for ragged two-body D3(BJ).

This module deliberately lives in the compiler package.  It exercises the same
GeometryIR/PairIR/TensorIR equation and generated VJP as the CPU reference, then
uses the shared TensorIR CUDA planner/compiler/runtime.  The native D3 owner
remains the public production fallback until this candidate passes numerical,
resource, replay, failure-isolation and endpoint-performance gates.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vibeqc_compiler.common.prepared_execution import (
    PreparedArtifactBinding,
    PreparedExecutionLease,
    PreparedExecutionRequest,
)
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.tensor import Program
from vibeqc_compiler.tensor.cuda_execute import PreparedCuda, compile_cuda
from vibeqc_compiler.tensor.cuda_plan import TensorPlan, plan_cuda

from .d3 import (
    D3CompilerSpec,
    D3GeometryBatchProgram,
    D3SpecLike,
    compile_d3_bj_batch,
)

if typing.TYPE_CHECKING:
    from typing_extensions import Self


@dataclass(frozen=True)
class D3GeneratedCudaExecution:
    """Detached generated D3 batch outputs and execution evidence."""

    energies: np.ndarray
    gradients: tuple[np.ndarray, ...] | None
    rebuilt: bool
    program_identity: str
    plan_identity: str
    metrics: dict[str, typing.Any]


@dataclass(frozen=True)
class D3GeneratedCudaDiagnostic:
    """Prepared generated-owner resource and identity summary."""

    system_count: int
    total_atoms: int
    maximum_atoms: int
    peak_bytes: int
    program_identity: str
    plan_identity: str
    prepared_identity: str
    prepared_executions: int
    prepared_refreshes: int
    rebuild_count: int


def _normalize_systems(
    systems: typing.Iterable[typing.Any],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[np.ndarray, ...], np.ndarray]:
    elements: list[int] = []
    offsets = [0]
    coordinates: list[np.ndarray] = []
    for value in systems:
        try:
            atomic_numbers, raw_coordinates = value
        except (TypeError, ValueError) as error:
            raise TypeError(
                "each generated D3 system must be (atomic_numbers, coordinates)"
            ) from error
        z = np.asarray(atomic_numbers)
        if z.ndim != 1 or z.size == 0:
            raise ValueError(
                "generated D3 atomic numbers must be a non-empty one-dimensional array"
            )
        if not np.issubdtype(z.dtype, np.integer):
            raise TypeError("generated D3 atomic numbers must be integers")
        if np.any(z < 1) or np.any(z > 86):
            raise NotImplementedError(
                "generated D3(BJ) supports atomic numbers 1 through 86"
            )
        if np.iscomplexobj(raw_coordinates):
            raise TypeError("generated D3 coordinates must be real")
        xyz = np.ascontiguousarray(raw_coordinates, dtype=np.float64)
        if xyz.shape != (z.size, 3):
            raise ValueError("generated D3 coordinates must have shape (natoms, 3)")
        if not np.isfinite(xyz).all():
            raise ValueError("generated D3 coordinates must be finite")
        elements.extend(int(value) for value in z)
        offsets.append(offsets[-1] + int(z.size))
        coordinates.append(xyz.copy())
    if not coordinates:
        raise ValueError("generated D3 batch requires at least one system")
    return (
        tuple(elements),
        tuple(offsets),
        tuple(coordinates),
        np.concatenate(coordinates, axis=0),
    )


def _energy_gradient_program(
    compiled: D3GeometryBatchProgram,
) -> tuple[Program, str]:
    reverse = compiled.coordinate_vjp()
    seed_name = next(
        name for name, source in reverse.input_map.items() if source == "energy"
    )
    gradient_name = next(
        name
        for name, source in reverse.output_map.items()
        if source == compiled.geometry.coordinate_name
    )
    program = Program(
        {
            "energy": compiled.program.outputs["energy"],
            "gradient": reverse.program.outputs[gradient_name],
        },
        provenance={
            "kind": "d3-bj-generated-ragged-cuda-candidate",
            "compiler_program": compiled.identity,
            "primal_equation": compiled.program.logical_hash,
            "vjp_equation": reverse.program.logical_hash,
        },
    )
    return program, seed_name


class PreparedD3CudaBatch:
    """Prepared generated D3(BJ) ragged CUDA retirement candidate.

    One TensorIR program owns the complete heterogeneous batch.  Pair/CN/switch
    topology remains explicit compiler state.  Coordinate replay that preserves
    this state reuses the prepared CUDA program; a topology change builds and
    prepares a new generated program before the previous owner is released.
    """

    def __init__(
        self,
        spec: D3SpecLike | D3CompilerSpec,
        systems: typing.Iterable[typing.Any],
        compiler: typing.Any,
        cache: Path,
        *,
        max_bytes: int = 256 * 1024**2,
        device: int = 0,
    ) -> None:
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        (
            self._elements,
            self._offsets,
            self._defaults,
            default_flat,
        ) = _normalize_systems(systems)
        self.spec = D3CompilerSpec.from_spec(spec)
        self.compiler = compiler
        self.cache = Path(cache)
        self.max_bytes = max_bytes
        self.device = device
        self._prepared: PreparedCuda | None = None
        self._compiled: D3GeometryBatchProgram | None = None
        self._program: Program | None = None
        self._seed_name = ""
        self._plan: TensorPlan | None = None
        self._prepared_request: PreparedExecutionRequest | None = None
        self._lease = PreparedExecutionLease()
        self._rebuild_count = 0
        try:
            self._install(default_flat)
        except BaseException:
            self.close()
            raise

    @property
    def system_count(self) -> int:
        return len(self._offsets) - 1

    def _install(self, coordinates: np.ndarray) -> None:
        compiled = compile_d3_bj_batch(
            self.spec,
            self._elements,
            self._offsets,
            coordinates,
        )
        program, seed_name = _energy_gradient_program(compiled)
        plan = plan_cuda(
            program,
            self.compiler.target,
            max_bytes=self.max_bytes,
        )
        artifact = compile_cuda(plan, self.compiler, self.cache)
        request = PreparedExecutionRequest(
            "d3-generated-cuda",
            compiled.identity,
            canonical_hash(self.compiler.target.to_payload()),
            plan.identity,
            canonical_hash(
                {
                    "host_bytes": plan.host_bytes,
                    "device_bytes": plan.device_bytes,
                    "max_bytes": self.max_bytes,
                    "systems": self.system_count,
                }
            ),
            device=self.device,
        )
        lease = PreparedExecutionLease()
        # Validate the metadata-only contract before creating device resources.
        lease.install(
            request,
            (PreparedArtifactBinding.from_artifact(artifact),),
            host_bytes=plan.host_bytes,
            device_bytes=plan.device_bytes,
        )
        prepared = PreparedCuda(plan, artifact, device=self.device)

        old, old_lease = self._prepared, self._lease
        self._compiled = compiled
        self._program = program
        self._seed_name = seed_name
        self._plan = plan
        self._prepared = prepared
        self._prepared_request = request
        self._lease = lease
        if old is not None:
            old.close()
            old_lease.invalidate()
            self._rebuild_count += 1

    def _require_open(
        self,
    ) -> tuple[D3GeometryBatchProgram, PreparedCuda, TensorPlan]:
        if self._compiled is None or self._prepared is None or self._plan is None:
            raise RuntimeError("generated D3 CUDA batch is closed")
        return self._compiled, self._prepared, self._plan

    def _coordinates(
        self, geometries: typing.Iterable[np.ndarray | None] | None
    ) -> np.ndarray:
        if geometries is None:
            return np.concatenate(self._defaults, axis=0)
        geometries = tuple(geometries)
        if len(geometries) != self.system_count:
            raise ValueError(
                "changed geometry list must match the prepared generated D3 system count"
            )
        packed = []
        for default, geometry in zip(self._defaults, geometries, strict=True):
            if geometry is None:
                packed.append(default)
                continue
            if np.iscomplexobj(geometry):
                raise TypeError("changed generated D3 coordinates must be real")
            xyz = np.ascontiguousarray(geometry, dtype=np.float64)
            if xyz.shape != default.shape:
                raise ValueError("changed generated D3 geometry has the wrong shape")
            if not np.isfinite(xyz).all():
                raise ValueError("changed generated D3 coordinates must be finite")
            packed.append(xyz)
        return np.concatenate(packed, axis=0)

    def execute(
        self,
        geometries: typing.Iterable[np.ndarray | None] | None = None,
        *,
        gradients: bool = True,
        profile: bool = False,
    ) -> D3GeneratedCudaExecution:
        if type(gradients) is not bool:
            raise TypeError("gradients must be a Boolean")
        coordinates = self._coordinates(geometries)
        compiled, prepared, plan = self._require_open()
        rebuilt = False
        if self._lease.needs_refresh:
            # Rebuild method-owned runtime state before republishing after failure.
            self._install(coordinates)
            compiled, prepared, plan = self._require_open()
            rebuilt = True
        try:
            compiled.validate_coordinates(coordinates)
        except ValueError as error:
            if "stale D3 batch pair topology/switch state" not in str(error):
                raise
            self._install(coordinates)
            compiled, prepared, plan = self._require_open()
            rebuilt = True

        request = self._prepared_request
        if request is None:
            raise RuntimeError("generated D3 CUDA prepared identity is missing")
        self._lease.require(
            request,
            max_host_bytes=self.max_bytes,
            max_device_bytes=self.max_bytes,
        )
        try:
            result = prepared.execute(
                {
                    compiled.geometry.coordinate_name: coordinates,
                    self._seed_name: np.ones(self.system_count, dtype=np.float64),
                },
                profile=profile,
            )
            energies = np.asarray(result.outputs["energy"], dtype=np.float64).copy()
            gradient = np.asarray(result.outputs["gradient"], dtype=np.float64)
            split = None
            if gradients:
                split = tuple(
                    gradient[begin:end].copy()
                    for begin, end in zip(
                        self._offsets[:-1], self._offsets[1:], strict=True
                    )
                )
            execution = D3GeneratedCudaExecution(
                energies=energies,
                gradients=split,
                rebuilt=rebuilt,
                program_identity=compiled.identity,
                plan_identity=plan.identity,
                metrics=dict(result.metrics),
            )
        except BaseException:
            self._lease.mark_failure()
            raise
        self._lease.mark_success()
        return execution

    def diagnostic(self) -> D3GeneratedCudaDiagnostic:
        compiled, _, plan = self._require_open()
        counts = tuple(
            end - begin
            for begin, end in zip(self._offsets[:-1], self._offsets[1:], strict=True)
        )
        return D3GeneratedCudaDiagnostic(
            system_count=self.system_count,
            total_atoms=self._offsets[-1],
            maximum_atoms=max(counts),
            peak_bytes=plan.peak_bytes,
            program_identity=compiled.identity,
            plan_identity=plan.identity,
            prepared_identity=self._lease.identity or "",
            prepared_executions=self._lease.executions,
            prepared_refreshes=self._lease.refreshes,
            rebuild_count=self._rebuild_count,
        )

    def close(self) -> None:
        if self._prepared is not None:
            self._prepared.close()
        self._prepared = None
        self._compiled = None
        self._program = None
        self._plan = None
        self._prepared_request = None
        self._lease.invalidate()

    def __enter__(self) -> Self:
        self._require_open()
        return self

    def __exit__(self, *unused: object) -> None:
        self.close()
