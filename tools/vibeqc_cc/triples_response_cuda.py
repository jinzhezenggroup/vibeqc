"""Bounded CUDA execution for generated standard-(T) response sources.

The mathematical derivative remains owned by :mod:`triples_response`: this
module only schedules its generated per-tile VJPs through the resident TensorIR
CUDA owner and accumulates the returned cotangents into the full input blocks.

Corrected Lambda reuses :class:`PreparedCUDALambda` so the repeated shared
CCSD J^T action remains resident on device. GMRES stays host-controlled, exactly
as for the existing CUDA Lambda owner. No nuclear derivative or force capability
is registered here.
"""

from __future__ import annotations

import time
import typing
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
from vibeqc_compiler.common.provenance import canonical_hash

from tools.vibeqc_response.implicit import (
    ImplicitSolveError,
    _immutable,
    checked_transpose_solve,
)
from tools.vibeqc_response.krylov import _vector_norm
from tools.vibeqc_response.problem import ResponseCompatibilityError

from .lambda_response import BoundCCSDResponse
from .lambda_solver import CCSDLambdaResult, _feed_hash
from .triples import _check_denominators, _validate
from .triples_cuda import TriplesTileConfig
from .triples_lambda_response import (
    CorrectedLambdaResult,
    _source_identity,
)
from .triples_response import (
    _arrays,
    _scatter_prefix,
    _selected_inputs,
    build_tile_triples_vjp,
)
from .triples_tiles import TriplesTileEnumerator, _tile_input_feeds


@dataclass(frozen=True)
class CudaTriplesResponseResult:
    """Accumulated generated (T) VJP blocks and bounded-CUDA evidence."""

    sources: typing.Mapping[str, np.ndarray]
    inputs: tuple[str, ...]
    nocc: int
    nvir: int
    vir_chunk_size: int
    peak_device_bytes: int
    input_identity: str
    source_identity: str
    peak_bytes_per_tile: tuple[int, ...] = ()
    artifact_keys: tuple[str, ...] = ()
    runtime_device: typing.Mapping[str, typing.Any] | None = None
    timing: typing.Mapping[str, typing.Any] = field(default_factory=dict)
    provenance: typing.Mapping[str, typing.Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sources",
            MappingProxyType(
                {
                    name: _immutable(np.asarray(value))
                    for name, value in self.sources.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "runtime_device",
            None
            if self.runtime_device is None
            else MappingProxyType(dict(self.runtime_device)),
        )
        object.__setattr__(self, "timing", MappingProxyType(dict(self.timing)))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


class CudaTriplesResponseTiles:
    """Run generated (T) tile VJPs through bounded resident CUDA owners.

    Each tile owns one exact-shape reverse program, resident allocation and
    upload/run/download lifetime. Only requested cotangent blocks are generated
    and downloaded. The host accumulates prefix-overlapping cotangents after
    each tile closes, so neither a full T3 tensor nor a full reverse tape is
    retained on device.
    """

    backend = "cuda-fp64-resident-triples-vjp"

    def __init__(
        self,
        config: TriplesTileConfig,
        compiler: typing.Any,
        cache: typing.Any,
    ) -> None:
        from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
        from vibeqc_compiler.tensor.cuda_plan import plan_cuda
        from vibeqc_compiler.tensor.cuda_resident import (
            PreparedResident,
            compile_resident,
        )

        if not isinstance(config, TriplesTileConfig):
            raise TypeError("CUDA triples response requires TriplesTileConfig")
        if not isinstance(compiler, CudaCompilerAdapter):
            raise TypeError("CUDA triples response requires CudaCompilerAdapter")
        self.config = config
        self.compiler = compiler
        self.cache = cache
        self._plan_cuda = plan_cuda
        self._compile_resident = compile_resident
        self._PreparedResident = PreparedResident

    def run_tiles(
        self,
        arrays: typing.Mapping[str, np.ndarray],
        *,
        inputs: typing.Iterable[str] | None = None,
        profile: bool = False,
    ) -> CudaTriplesResponseResult:
        """Evaluate and accumulate requested generated VJP blocks on CUDA."""

        nocc = self.config.nocc
        nvir = self.config.nvir
        selected = _selected_inputs(inputs)
        values = _arrays(
            arrays["ovvv"],
            arrays["ovoo"],
            arrays["ovov"],
            arrays["fov"],
            arrays["t1"],
            arrays["t2"],
            arrays["eps_o"],
            arrays["eps_v"],
        )
        _validate(
            nocc,
            nvir,
            values["ovvv"],
            values["ovoo"],
            values["ovov"],
            values["fov"],
            values["t1"],
            values["t2"],
            values["eps_o"],
            values["eps_v"],
        )
        _check_denominators(values["eps_o"], values["eps_v"], 1e-10)

        totals = {
            name: np.zeros_like(np.asarray(values[name]), dtype=np.float64)
            for name in selected
        }
        tiles = list(
            TriplesTileEnumerator(nocc, nvir, vir_chunk_size=self.config.vir_chunk_size)
        )
        timing = {
            "extract_s": 0.0,
            "compile_s": 0.0,
            "upload_s": 0.0,
            "run_s": 0.0,
            "download_s": 0.0,
            "tile_count": len(tiles),
        }
        peak_bytes_per_tile: list[int] = []
        artifact_keys: list[str] = []
        reverse_hashes: list[str] = []
        primal_hashes: list[str] = []
        runtime_device = None
        total_start = time.perf_counter()

        for tile in tiles:
            started = time.perf_counter()
            feeds = _tile_input_feeds(values, tile.a_end)
            feeds = {
                **feeds,
                "bar_triples_energy": np.asarray(1.0, dtype=np.float64),
            }
            timing["extract_s"] += time.perf_counter() - started

            started = time.perf_counter()
            derivative = build_tile_triples_vjp(
                nocc,
                nvir,
                vir_chunk=(tile.a_start, tile.a_end),
                inputs=selected,
            )
            plan = self._plan_cuda(
                derivative.program,
                self.compiler.target,
                max_bytes=self.config.max_bytes,
            )
            artifact = self._compile_resident(plan, self.compiler, self.cache)
            timing["compile_s"] += time.perf_counter() - started
            peak_bytes_per_tile.append(int(plan.peak_bytes))
            artifact_keys.append(str(artifact.metadata.get("key", "")))
            reverse_hashes.append(derivative.program.logical_hash)
            primal_hashes.append(derivative.primal_logical_hash)

            with self._PreparedResident(
                plan, artifact, device=self.config.device
            ) as resident:
                started = time.perf_counter()
                # Selected VJPs can eliminate primal inputs. The resident ABI
                # accepts only output-reachable inputs, not dead definitions.
                resident.upload(
                    {
                        node.attrs["name"]: feeds[node.attrs["name"]]
                        for node in derivative.program.live_nodes
                        if node.op == "input"
                    }
                )
                timing["upload_s"] += time.perf_counter() - started

                started = time.perf_counter()
                leases, _metrics = resident.run(profile=profile)
                timing["run_s"] += time.perf_counter() - started

                started = time.perf_counter()
                for name in selected:
                    local = resident.download(leases[f"bar_{name}"])
                    _scatter_prefix(totals[name], local, name, tile.a_end)
                timing["download_s"] += time.perf_counter() - started
                if runtime_device is None:
                    runtime_device = dict(resident.device)

        timing["total_s"] = time.perf_counter() - total_start
        input_identity = _feed_hash(values)
        source_identity = canonical_hash(
            {
                "backend": self.backend,
                "nocc": nocc,
                "nvir": nvir,
                "vir_chunk_size": self.config.vir_chunk_size,
                "inputs": selected,
                "input_identity": input_identity,
                "primal_hashes": primal_hashes,
                "reverse_hashes": reverse_hashes,
                "sources": _feed_hash(totals),
            }
        )
        return CudaTriplesResponseResult(
            sources=totals,
            inputs=selected,
            nocc=nocc,
            nvir=nvir,
            vir_chunk_size=self.config.vir_chunk_size,
            peak_device_bytes=max(peak_bytes_per_tile, default=0),
            input_identity=input_identity,
            peak_bytes_per_tile=tuple(peak_bytes_per_tile),
            artifact_keys=tuple(artifact_keys),
            runtime_device=runtime_device,
            timing=timing,
            source_identity=source_identity,
            provenance={
                "schema": "vibeqc.ccsd-t.cuda-response-tiles/1",
                "backend": self.backend,
                "bounded_by": "TriplesTileConfig.max_bytes per reverse tile",
                "full_t3_resident": False,
                "full_reverse_tape_resident": False,
                "cpu_fallback": False,
            },
        )


def solve_corrected_lambda_cuda(
    prepared: typing.Any,
    baseline: CCSDLambdaResult,
    triples_response: CudaTriplesResponseResult,
    *,
    reference_identity: str,
) -> CorrectedLambdaResult:
    """Solve corrected standard-(T) Lambda using GPU triples VJP and J^T actions."""

    from .lambda_cuda import PreparedCUDALambda

    if not isinstance(prepared, PreparedCUDALambda):
        raise TypeError("corrected CUDA Lambda requires PreparedCUDALambda")
    if not isinstance(baseline, CCSDLambdaResult):
        raise TypeError("corrected CUDA Lambda requires a baseline Lambda result")
    if not isinstance(triples_response, CudaTriplesResponseResult):
        raise TypeError("corrected CUDA Lambda requires CUDA triples response")
    if not {"t1", "t2"}.issubset(triples_response.sources):
        raise ValueError("CUDA triples response must include t1 and t2 sources")

    bound = prepared.bound
    prepared._assert_current(reference_identity)
    baseline_response = BoundCCSDResponse(bound, baseline)
    nocc = bound.reference.nocc
    nvir = bound.reference.nmo - nocc
    if triples_response.nocc != nocc or triples_response.nvir != nvir:
        raise ResponseCompatibilityError(
            "CUDA triples response shape belongs to another CC state"
        )
    eps = bound.reference.orbital_energies
    expected_inputs = {
        "ovvv": bound.feeds["ovvv"],
        "ovoo": bound.feeds["ovoo"],
        "ovov": bound.feeds["ovov"],
        "fov": bound.feeds["fov"],
        "t1": bound.feeds["t1"],
        "t2": bound.feeds["t2"],
        "eps_o": eps[:nocc],
        "eps_v": eps[nocc:],
    }
    if triples_response.input_identity != _feed_hash(expected_inputs):
        raise ResponseCompatibilityError(
            "CUDA triples response inputs belong to another CC state"
        )

    dense_sources = {
        "t1": np.asarray(triples_response.sources["t1"]),
        "t2": np.asarray(triples_response.sources["t2"]),
    }
    t2_layout = bound.layouts[1]
    projected_t2 = t2_layout.unpack(t2_layout.unpack_transpose(dense_sources["t2"]))
    projected_sources = {"t1": dense_sources["t1"], "t2": projected_t2}
    source = bound.sqrt_weights * bound._pack(
        (projected_sources["t1"], projected_sources["t2"])
    )
    shared_rhs = prepared._rhs("shared") - source
    independent_rhs = prepared._rhs("independent") - source
    owner = prepared

    class Operator:
        dimension = len(bound.sqrt_weights)

        def apply(self, vector: typing.Any) -> np.ndarray:
            return owner._transpose("shared", np.asarray(vector))

    solved = checked_transpose_solve(
        Operator(),
        shared_rhs,
        solver=bound.solver,
        assert_current=lambda: prepared._assert_current(reference_identity),
    )
    independent = prepared._transpose("independent", solved.solution) - independent_rhs
    independent_norm = _vector_norm(independent)
    independent_max = float(np.max(np.abs(independent / bound.sqrt_weights)))
    if (
        max(solved.residual_norm, independent_norm, independent_max)
        > bound.options.lambda_tolerance
    ):
        raise ImplicitSolveError("corrected CUDA RCCSD(T) Lambda residual gate failed")

    total1, total2 = bound._unpack(solved.solution / bound.sqrt_weights)
    delta1 = total1 - baseline.lambda1
    delta2 = total2 - baseline.lambda2
    mathematical_source_identity = _source_identity(
        bound,
        dense_sources,
        projected_sources,
        triples_response.vir_chunk_size,
    )
    prepared._assert_current(reference_identity)
    return CorrectedLambdaResult(
        total1,
        total2,
        delta1,
        delta2,
        bound.reference_identity,
        bound.cc_state_identity,
        mathematical_source_identity,
        solved.residual_norm,
        independent_norm,
        independent_max,
        solved.iterations,
        solved.operator_actions,
        {
            "lagrangian": "E_CCSD + E_(T) + <lambda_total, R_CCSD>",
            "jacobian": "RCCSD residual Jacobian; no iterative T3 equations",
            "triples_source_identity": mathematical_source_identity,
            "cuda_triples_response_identity": triples_response.source_identity,
            "baseline_lambda_identity": baseline_response.lambda_identity,
            "equation_identity": bound.equation_identity,
            "vir_chunk_size": triples_response.vir_chunk_size,
            "tensor_backend": (
                "cuda-fp64-resident-triples-vjp+cuda-fp64-resident-lambda-actions"
            ),
            "lambda_execution_owner_identity": prepared.identity,
            "orbital_response": "excluded",
            "cpu_fallback": False,
        },
    )
