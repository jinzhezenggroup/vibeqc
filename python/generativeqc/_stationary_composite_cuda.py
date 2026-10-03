"""Complete compiler-planned composite CUDA stationary forces from a live KS state.

The shared stationary CUDA arena owns semilocal sources. This module composes
the additional providers selected by StationaryGradientPlan, currently the
qualified range-separated-exchange plus nonlocal-correlation source inventory.
No method-name dispatch or method-specific scientific kernel lives here.

The native owner evaluates integral derivatives; generated CUDA contracts AO
jets, semilocal/nonlocal feature adjoints and partition motion. Components that
are already published on the host are summed there exactly once, avoiding a
redundant host->device->host final-add round trip. Nonlocal features and force
seeds remain on the device throughout composition.
"""

from __future__ import annotations

import ctypes as ct
import typing
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter

import numpy as np
from generativeqc_compiler.dft.cuda import CudaGrid
from generativeqc_compiler.dft.nonlocal_policy import (
    MOLECULAR_VV10_DENSITY_POLICY,
    MOLECULAR_VV10_DENSITY_THRESHOLD,
)
from generativeqc_compiler.dft.plan import plan_tiles
from generativeqc_compiler.integral.first_derivative_native import (
    emit_first_derivative_cuda,
)
from generativeqc_compiler.method.nonlocal_correlation import (
    NonlocalCorrelationPrimitive,
)
from generativeqc_compiler.method.stationary_composite_resources import (
    plan_composite_stationary_cuda_resources,
)
from generativeqc_compiler.method.stationary_cuda import (
    compile_stationary_cuda,
    stationary_external_provider_sources,
    stationary_runtime_sources,
)
from generativeqc_compiler.method.stationary_feature_lease import (
    plan_stationary_feature_leases,
)
from generativeqc_compiler.method.stationary_gradient import (
    SCF_POINT_MODEL,
    StationaryGradientPlan,
    StationaryMeanField,
)
from generativeqc_compiler.method.stationary_prepared import (
    compile_stationary_prepared_plan,
)

from . import _native
from ._dft_gradient import StationaryDerivativeContract, native_ao_geometry_identity
from ._resident_ao_maps import ResidentAoMapCache, ResidentAoMapDomain
from ._stationary_cuda import _DOUBLE, _CudaSources, _native_grid_artifact, _ptr
from ._stationary_nonlocal_cuda import resident_nonlocal_geometry
from .nonlocal_runtime import _ResidentNonlocalForceOwner

_COMPOSITE_EXTERNAL_SOURCES = (
    "exchange_short_range",
    "exchange_long_range",
    "nonlocal_ao",
    "nonlocal_grid",
    "nonlocal_weight",
)


def _plan_for_state(state: typing.Any) -> StationaryGradientPlan:
    source = state._source
    point_model = (
        source._batch._calculator._ks_options.scf_domain
        if getattr(source, "nonlocal_density_policy", None)
        == MOLECULAR_VV10_DENSITY_POLICY
        else SCF_POINT_MODEL
    )
    return StationaryGradientPlan(
        source.method_ir,
        StationaryMeanField(point_model, hamiltonian=source.hamiltonian),
    )


def requires_composite_stationary_cuda(state: typing.Any) -> bool:
    """Select the qualified composite owner or fail closed on unowned sources."""
    plan = _plan_for_state(state)
    external_sources = stationary_external_provider_sources(plan)
    if not external_sources:
        return False
    if external_sources != _COMPOSITE_EXTERNAL_SOURCES:
        raise NotImplementedError(
            "stationary CUDA external-provider source inventory is not qualified: "
            + ", ".join(external_sources)
        )
    return True


def _canonical_gradient_sum(
    plan: StationaryGradientPlan,
    components: typing.Mapping[str, typing.Any],
    natom: int,
) -> np.ndarray:
    """Validate complete source coverage and sum already-host-resident gradients."""
    plan.reduction_program(atoms=natom, sources=components)
    gradient = np.zeros((natom, 3), dtype=np.float64)
    for name in plan.source_names:
        component = np.asarray(components[name])
        if component.dtype != np.float64 or component.shape != (natom, 3):
            raise ValueError(
                "composite stationary gradient component shape/dtype mismatch"
            )
        if not np.all(np.isfinite(component)):
            raise ValueError("composite stationary gradient component is nonfinite")
        np.add(gradient, component, out=gradient)
    if not np.all(np.isfinite(gradient)):
        raise ValueError("composite stationary final gradient is nonfinite")
    return gradient


class PreparedCompositeStationaryCudaGradient:
    """Retain geometry-bound CUDA owners, rebuilding explicitly on geometry change.

    Numeric capacity is checked before constructing any derivative owner.
    Driver modules/compiler objects and the pre-existing SCF snapshot are
    reported separately from this additional host/device allowance.
    """

    def __init__(self) -> None:
        self._stack = ExitStack()
        self._identity = None
        self._nonlocal = None
        self._ao_maps = None
        self.executions = 0
        self.last_work = None

    def close(self) -> None:
        if self._nonlocal is not None:
            # The native nonlocal owner drains its bound stream before freeing
            # seeds, including a failed downstream geometry enqueue.
            self._nonlocal.close()
            self._nonlocal = None
        self._ao_maps = None
        self._stack.close()
        self._identity = None
        self.last_work = None

    def execute(
        self,
        state: typing.Any,
        basis: typing.Any,
        *,
        compiler: typing.Any,
        cache: Path,
        library: Path,
        tile_points: int | None = None,
        max_device_bytes: int = 1 << 30,
        max_host_bytes: int = 2 << 30,
        active_ao_cutoff: float | None = None,
        active_ao_cache_bytes: int = 64 << 20,
    ) -> tuple[np.ndarray, dict[str, typing.Any]]:
        """Publish only a complete result; failed executions discard retained scratch."""
        try:
            return self._execute(
                state,
                basis,
                compiler=compiler,
                cache=cache,
                library=library,
                tile_points=tile_points,
                max_device_bytes=max_device_bytes,
                max_host_bytes=max_host_bytes,
                active_ao_cutoff=active_ao_cutoff,
                active_ao_cache_bytes=active_ao_cache_bytes,
            )
        except BaseException:
            self.close()
            raise

    def _execute(
        self,
        state: typing.Any,
        basis: typing.Any,
        *,
        compiler: typing.Any,
        cache: Path,
        library: Path,
        tile_points: int | None,
        max_device_bytes: int,
        max_host_bytes: int,
        active_ao_cutoff: float | None,
        active_ao_cache_bytes: int,
    ) -> tuple[np.ndarray, dict[str, typing.Any]]:
        """Contract all twelve gradients under the live SCF token and publish forces."""
        started = perf_counter()
        if active_ao_cutoff is not None and (
            type(active_ao_cutoff) not in (int, float)
            or not np.isfinite(active_ao_cutoff)
            or active_ao_cutoff <= 0
        ):
            raise ValueError("active AO cutoff must be finite and positive")
        if type(active_ao_cache_bytes) is not int or active_ao_cache_bytes < 0:
            raise ValueError("active AO cache budget must be nonnegative")
        if not callable(getattr(_CudaSources, "geometry_external_device", None)):
            raise NotImplementedError(
                "resident nonlocal force composition requires the stationary seed consumer"
            )
        contract = StationaryDerivativeContract(state.identity)
        contract.validate(state)
        source = state._source
        if (
            source.backend != "cuda"
            or source.metadata[0] != 8
            or source.hamiltonian != "all-electron"
            or source.nonlocal_density_policy != MOLECULAR_VV10_DENSITY_POLICY
        ):
            raise NotImplementedError(
                "composite stationary forces require its complete FP64 CUDA owner"
            )
        if (
            basis.identity != state.identity.basis_identity
            or native_ao_geometry_identity(basis) != state.identity.geometry_identity
        ):
            raise ValueError("composite stationary stationary basis/geometry mismatch")
        functional = int(source.functional_code)
        n, na, npnt = basis.nao, basis.natom, len(state.grid.points)
        if not (1 <= n <= 1024 and 1 <= na <= 128 and 1 <= npnt <= 4_000_000):
            raise ValueError(
                "composite stationary CUDA stationary shape exceeds its bounded domain"
            )
        if any(shell.angular_momentum > 3 for shell in basis.shells):
            raise NotImplementedError(
                "composite stationary CUDA forces currently qualify through-f AOs"
            )
        device = int(source.metadata[12])
        plan = _plan_for_state(state)
        external_sources = stationary_external_provider_sources(plan)
        if external_sources != _COMPOSITE_EXTERNAL_SOURCES:
            raise NotImplementedError(
                "composite stationary CUDA owner does not cover this source inventory"
            )
        prepared_plan = compile_stationary_prepared_plan(plan)
        feature_plan = plan_stationary_feature_leases(prepared_plan)
        grid_features = feature_plan.features
        if "rho" not in grid_features or "gradient" not in grid_features:
            raise RuntimeError(
                "stationary execution planner omitted nonlocal density features"
            )
        nlc = next(
            p
            for p in source.method_ir.primitives
            if isinstance(p, NonlocalCorrelationPrimitive)
        )
        # Integral work belongs to the native source, never an AO^4 host loop.
        capacity = 1
        # The resident pair/seed arena scales with the complete grid, while the
        # AO and geometry owners scale with one tile. Reserve its exact native
        # capacity, then admit all simultaneously live owners under both totals.
        # An explicit user cap still fails before any force allocation or JIT.
        nlc_budget = _ResidentNonlocalForceOwner.required_device_bytes(
            source._library, npnt, 256 if tile_points is None else tile_points
        )
        if (
            nlc_budget
            > source._batch._calculator.ks_options.nonlocal_memory_budget_bytes
        ):
            raise ValueError(
                "resident nonlocal force exceeds nonlocal_memory_budget_bytes"
            )
        layout = plan_composite_stationary_cuda_resources(
            basis,
            grid_plan=lambda points: plan_tiles(
                basis,
                backend="cuda",
                order=2,
                tile_points=points,
                active_ao_capacity=n,
                budget_bytes=max_device_bytes,
            ),
            grid_points=npnt,
            spins=plan.spin_blocks,
            source_count=len(stationary_runtime_sources(plan)),
            nonlocal_bytes=nlc_budget,
            target=compiler.target,
            max_device_bytes=max_device_bytes,
            max_host_bytes=max_host_bytes,
            tile_points=tile_points,
        )
        gp = layout.grid
        tile_points = gp.tile_points
        source_bytes = layout.sources.allocation_bytes
        native_budget = layout.native_bytes
        device_bound, host_bound = layout.device_bound, layout.host_bound
        # Admit dense scratch first. Optional maps use only remaining host
        # capacity, with a zero-budget dense fallback when no room remains.
        ao_cache_allowance = (
            min(active_ao_cache_bytes, max(0, max_host_bytes - host_bound))
            if active_ao_cutoff is not None
            else 0
        )
        host_bound += ao_cache_allowance
        cache = Path(cache)
        identity = (
            basis.identity,
            state.identity.geometry_identity,
            prepared_plan.identity,
            feature_plan.identity,
            source.grid_spec,
            device,
            tile_points,
            str(library),
            str(compiler.target),
            max_device_bytes,
            max_host_bytes,
            nlc_budget,
            active_ao_cutoff,
            ao_cache_allowance,
        )
        reused = identity == self._identity
        if not reused:
            self.close()
            self._stack = ExitStack()
            try:
                primitive = emit_first_derivative_cuda((("nuclear", ()),))
                artifact = compile_stationary_cuda(
                    primitive,
                    functional=functional,
                    plan=plan,
                    iterations=source.grid_spec.partition_iterations,
                    compiler=compiler,
                    cache=cache,
                )
                self.sources = self._stack.enter_context(
                    _CudaSources(
                        basis,
                        artifact,
                        compiler,
                        device,
                        tile_points,
                        capacity,
                        source_bytes,
                        spin_blocks=plan.spin_blocks,
                        integral_derivatives=False,
                        cooperative_becke=True,
                        # The batched nuclear call owns every unordered atom
                        # pair in one deterministic native page.
                        page_work_budget=max(1, na * (na - 1) // 2),
                    )
                )
                self.sources.kinds[("nuclear", ())] = 0
                # Keep semilocal and nonlocal source accounting distinct while
                # both consumers borrow the same GridTask lease. This second
                # bounded accumulator replaces the former second AO/grid pass.
                self.nonlocal_sources = self._stack.enter_context(
                    _CudaSources(
                        basis,
                        artifact,
                        compiler,
                        device,
                        tile_points,
                        capacity,
                        source_bytes,
                        spin_blocks=plan.spin_blocks,
                        integral_derivatives=False,
                        cooperative_becke=True,
                        page_work_budget=1,
                    )
                )
                self.grid = self._stack.enter_context(
                    CudaGrid(
                        basis,
                        _native_grid_artifact(library, compiler.target.architecture),
                        order=2,
                        tile_points=tile_points,
                        active_ao_capacity=n,
                        budget_bytes=gp.peak_bytes,
                        device_id=device,
                        ingredients=grid_features,
                    )
                )
                self._nonlocal = _ResidentNonlocalForceOwner(
                    nlc.spec,
                    state.grid.points,
                    state.grid.weights,
                    coefficient=nlc.coefficient,
                    tile_points=tile_points,
                    maximum_bytes=nlc_budget,
                    density_threshold=float(MOLECULAR_VV10_DENSITY_THRESHOLD),
                    device_id=device,
                    context=source._batch._context,
                    library=source._library,
                )
                self._identity = identity
            except BaseException:
                self.close()
                raise
        component_seconds = {"prepare": perf_counter() - started}
        component_start = perf_counter()
        evaluate = source._library.generativeqc_ks_snapshot_cuda_integral_gradient_v1
        evaluate.argtypes = [
            ct.c_void_p,
            ct.c_void_p,
            _DOUBLE,
            ct.c_size_t,
            ct.c_size_t,
            ct.POINTER(ct.c_uint64),
            ct.c_size_t,
        ]
        evaluate.restype = ct.c_int
        range_sources = tuple(source.name for source in plan.range_exchange_sources)
        integral_names = (
            "one_electron",
            "overlap_pulay",
            "coulomb",
            *range_sources,
        )
        integral = np.empty((len(integral_names), na, 3))
        native_usage = np.zeros(9, dtype=np.uint64)
        _native.check(
            source._library,
            evaluate(
                source._batch._batch,
                source._handle,
                _ptr(integral),
                integral.size,
                native_budget,
                native_usage.ctypes.data_as(ct.POINTER(ct.c_uint64)),
                native_usage.size,
            ),
            context=source._batch._context,
        )
        components = dict(zip(integral_names, integral, strict=True))
        component_seconds["integral_derivatives"] = perf_counter() - component_start
        component_start = perf_counter()
        resident_density = source.cuda_resident_density()
        if resident_density is None:
            raise NotImplementedError(
                "composite stationary CUDA force requires the resident final-density bridge"
            )
        self.grid.set_density_device(
            device_id=resident_density.device,
            alpha=resident_density.alpha,
            beta=resident_density.beta,
            matrix_elements=resident_density.matrix_elements,
            spins=resident_density.spins,
            source_stream=resident_density.source_stream,
        )
        # A concurrent replacement can only occur before the producer-stream
        # copy is enqueued; reject it before any stationary source publication.
        source.check_current()
        # Integral derivatives already come from the token-bound native prepared
        # owner above. These retained CUDA accumulators execute only nuclear and
        # grid-geometry sources, so uploading detached host D/W here is redundant.
        self.sources.reset_geometry(source.grid_spec.coincident_tolerance)
        self.nonlocal_sources.reset_geometry(source.grid_spec.coincident_tolerance)
        charges = np.array([a.atomic_number for a in basis.atoms], dtype=float)
        self.sources.nuclear_all(charges)
        component_seconds["density_and_nuclear_setup"] = (
            perf_counter() - component_start
        )
        ao_domain = None
        if active_ao_cutoff is not None:
            lease = source.cuda_resident_grid()
            if lease is None:
                raise NotImplementedError(
                    "active AO maps require the resident molecular grid"
                )
            ao_domain = ResidentAoMapDomain(
                self.grid.basis_identity,
                state.identity.geometry_identity,
                state.grid.identity,
                lease.device,
                lease.points,
                lease.point_count,
                tile_points,
                self.grid.plan.order,
            )
            if self._ao_maps is None or self._ao_maps.domain != ao_domain:
                self._ao_maps = ResidentAoMapCache(
                    self.grid,
                    ao_domain,
                    cutoff=active_ao_cutoff,
                    budget_bytes=ao_cache_allowance,
                )
        resident_parts, resident_seconds, resident_work = resident_nonlocal_geometry(
            grid=self.grid,
            sources=self.sources,
            nonlocal_sources=self.nonlocal_sources,
            nonlocal_owner=self._nonlocal,
            state=state,
            raw_weights=source.atomic_weights,
            tile_points=tile_points,
            ao_count=n,
            functional=functional,
            ingredients=grid_features,
            ao_maps=self._ao_maps,
            ao_domain=ao_domain,
        )
        components.update(resident_parts)
        component_seconds.update(resident_seconds)
        component_start = perf_counter()
        # Every component is already host-resident at this boundary. Keep the
        # compiler coverage gate, but do not upload them solely to add and
        # download the same 3*Natom result again.
        gradient = _canonical_gradient_sum(plan, components, na)
        contract.validate(state)
        component_seconds["reduction_and_validation"] = perf_counter() - component_start
        self.executions += 1
        work = {
            "execution": "cuda-complete-composite",
            "plan_identity": plan.identity,
            "prepared_plan_identity": prepared_plan.identity,
            "execution_graph_identity": prepared_plan.graph.identity,
            "lifetime_plan_identity": prepared_plan.lifetimes.identity,
            "feature_lease_identity": feature_plan.identity,
            "grid_features": list(feature_plan.features),
            "retained_grid_features": list(feature_plan.retained_features),
            "source_names": list(plan.source_names),
            "grid_points": npnt,
            "grid_tile_points": tile_points,
            "grid_tile_count": (npnt + tile_points - 1) // tile_points,
            "geometry_planned_lanes": layout.sources.geometry_lanes,
            "becke_planned_threads_per_point": layout.sources.becke_threads_per_point,
            "grid_density_source": "exact-final-scf-device-binding",
            "grid_density_h2d_bytes": 0,
            "final_reduction": "host-canonical-source-sum",
            "final_reduction_h2d_bytes": 0,
            "final_reduction_d2h_bytes": 0,
            **resident_work,
            "partition_pair_visits": 2 * npnt * na * (na - 1),
            # The native v1 result does not identify whether optional shell
            # capability or its public-AO fallback executed. Do not infer work
            # from the method name or turn unavailable evidence into zero.
            "symmetry_unique_quartets_per_integral_source": None,
            "two_electron_quartet_traversals": None,
            "maximum_center_dual3_evaluations_total": None,
            "two_electron_shell_traversals": None,
            "two_electron_radial_operators": [
                "full-range",
                *(primitive.operator for primitive in plan.range_exchange_primitives),
            ],
            "range_recurrences_per_participating_center": None,
            "two_electron_work_scope": (
                "unavailable: native execution route is not exported; screened shell "
                "scheduler and public-AO capability fallback are both supported"
            ),
            "additional_device_peak_bound": device_bound,
            "additional_host_numeric_bound": host_bound,
            "active_ao_cache_allowance_bytes": ao_cache_allowance,
            "active_ao_cutoff": active_ao_cutoff,
            "native_integral_resources": dict(
                zip(
                    (
                        "retained_device_bytes",
                        "source_host_preparation_bytes",
                        "one_electron_device_peak_bytes",
                        "one_electron_host_peak_bytes",
                        "one_electron_h2d_bytes",
                        "one_electron_d2h_bytes",
                        "final_state_export_d2h_bytes",
                        "final_state_export_reads",
                        "final_state_export_synchronizations",
                    ),
                    map(int, native_usage),
                    strict=True,
                )
            ),
            "prepared_execution_reused": reused,
            "execution_index": self.executions,
            "snapshot_export_work": dict(source.export_work),
            "host_scope": "snapshot validation, bounded tile scheduling, and canonical host source sum",
            "endpoint_seconds": perf_counter() - started,
            "component_seconds": component_seconds,
        }
        self.last_work = work
        return -gradient.copy(), work
