"""Complete WB97M-V CUDA stationary composition from a live native KS state.

The native owner evaluates integral derivatives; generated CUDA contracts AO
jets, semilocal/nonlocal feature adjoints, partition motion and the final sum.
Host work is explicit snapshot validation, tiling and VV10 active-set packing.
No CPU derivative evaluator, reference SCF, or finite difference is used here.
"""

from __future__ import annotations

import ctypes as ct
import typing
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter

import numpy as np
from vibeqc_compiler.dft.cuda import CudaGrid, GridTaskView
from vibeqc_compiler.dft.nonlocal_policy import (
    MOLECULAR_VV10_DENSITY_POLICY,
    MOLECULAR_VV10_DENSITY_THRESHOLD,
)
from vibeqc_compiler.dft.plan import plan_tiles
from vibeqc_compiler.integral.first_derivative_native import emit_first_derivative_cuda
from vibeqc_compiler.method.nonlocal_correlation import NonlocalCorrelationPrimitive
from vibeqc_compiler.method.stationary_cuda import compile_stationary_cuda
from vibeqc_compiler.method.stationary_gradient import (
    StationaryGradientPlan,
    StationaryMeanField,
)
from vibeqc_compiler.tensor.cuda_execute import PreparedCuda, compile_cuda
from vibeqc_compiler.tensor.cuda_plan import plan_cuda

from . import _native
from ._dft_gradient import StationaryDerivativeContract, native_ao_geometry_identity
from ._stationary_cuda import _DOUBLE, _INT, _CudaSources, _native_grid_artifact, _ptr
from .nonlocal_runtime import NonlocalFixedGridPlan


class PreparedWb97mvCudaGradient:
    """Retain geometry-bound CUDA owners, rebuilding explicitly on geometry change.

    Numeric capacity is checked before constructing any derivative owner.
    Driver modules/compiler objects and the pre-existing SCF snapshot are
    reported separately from this additional host/device allowance.
    """

    def __init__(self) -> None:
        self._stack = ExitStack()
        self._identity = None
        self._nonlocal = None
        self.executions = 0
        self.last_work = None

    def close(self) -> None:
        if self._nonlocal is not None:
            self._nonlocal.close()
            self._nonlocal = None
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
        tile_points: int = 256,
        max_device_bytes: int = 1 << 30,
        max_host_bytes: int = 2 << 30,
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
        tile_points: int,
        max_device_bytes: int,
        max_host_bytes: int,
    ) -> tuple[np.ndarray, dict[str, typing.Any]]:
        """Contract all twelve gradients under the live SCF token and publish forces."""
        started = perf_counter()
        contract = StationaryDerivativeContract(state.identity)
        contract.validate(state)
        source = state._source
        if (
            source.backend != "cuda"
            or source.metadata[0] != 8
            or source.metadata[6] != 4
            or source.hamiltonian != "all-electron"
            or source.nonlocal_density_policy != MOLECULAR_VV10_DENSITY_POLICY
        ):
            raise NotImplementedError(
                "WB97M-V forces require its complete FP64 CUDA owner"
            )
        if (
            basis.identity != state.identity.basis_identity
            or native_ao_geometry_identity(basis) != state.identity.geometry_identity
        ):
            raise ValueError("WB97M-V stationary basis/geometry mismatch")
        n, na, npnt = basis.nao, basis.natom, len(state.grid.points)
        if not (1 <= n <= 1024 and 1 <= na <= 128 and 1 <= npnt <= 4_000_000):
            raise ValueError("WB97M-V CUDA stationary shape exceeds its bounded domain")
        if any(shell.angular_momentum > 2 for shell in basis.shells):
            raise NotImplementedError("WB97M-V CUDA forces currently qualify s/p/d AOs")
        device = int(source.metadata[12])
        plan = StationaryGradientPlan(
            source.method_ir,
            StationaryMeanField("libxc-7.0/work-mgga-v1/smooth-lr-a1.35-order16"),
        )
        nlc = next(
            p
            for p in source.method_ir.primitives
            if isinstance(p, NonlocalCorrelationPrimitive)
        )
        gp = plan_tiles(
            basis,
            backend="cuda",
            order=2,
            tile_points=tile_points,
            active_ao_capacity=n,
            budget_bytes=max_device_bytes,
        )
        # The stationary primitive arena is used only for nuclear repulsion;
        # integral work belongs to the native source, never an AO^4 host loop.
        capacity = 1
        source_bytes = (
            8
            * (
                22 * capacity
                + 2 * basis.nprimitive
                + 4 * n
                + 600 * na
                + 3 * tile_points
                + 2 * plan.spin_blocks * n * n
            )
            + 256
        )
        nlc_budget = min(
            source._batch._calculator.ks_options.nonlocal_memory_budget_bytes,
            max_host_bytes // 4,
            max_device_bytes // 4,
        )
        # Separate upper allowances for the retained native direct source and
        # its transient one-electron bridge. Both enforce this cap natively;
        # the matrix term also covers final-state revalidation/export on host.
        native_budget = 256 * n * n + 1024 * (
            na + n + basis.nprimitive + len(basis.shells)
        )
        device_bound = (
            gp.peak_bytes
            + source_bytes
            + 48 * tile_points
            + nlc_budget
            + 2 * native_budget
        )
        host_bound = (
            gp.host_bytes
            + 8 * (64 * npnt + 8 * n * n + 128 * na)
            + nlc_budget
            + 2 * native_budget
        )
        if device_bound > max_device_bytes or host_bound > max_host_bytes:
            raise ValueError("WB97M-V stationary numeric capacity budget exceeded")
        cache = Path(cache)
        identity = (
            basis.identity,
            state.identity.geometry_identity,
            plan.identity,
            source.grid_spec,
            device,
            tile_points,
            str(library),
            str(compiler.target),
            max_device_bytes,
            max_host_bytes,
            nlc_budget,
        )
        reused = identity == self._identity
        if not reused:
            self.close()
            self._stack = ExitStack()
            try:
                primitive = emit_first_derivative_cuda((("nuclear", ()),))
                artifact = compile_stationary_cuda(
                    primitive,
                    functional=4,
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
                        work_budget=max(1, na * (na - 1) // 2),
                    )
                )
                self.sources.kinds[("nuclear", ())] = 0
                lib = self.sources.library
                lib.stationary_geometry_external.argtypes = [
                    ct.c_void_p,
                    ct.POINTER(GridTaskView),
                    _DOUBLE,
                    _INT,
                    _DOUBLE,
                    _DOUBLE,
                    _DOUBLE,
                    ct.c_char_p,
                    ct.c_size_t,
                ]
                self.grid = self._stack.enter_context(
                    CudaGrid(
                        basis,
                        _native_grid_artifact(library, compiler.target.architecture),
                        order=2,
                        tile_points=tile_points,
                        active_ao_capacity=n,
                        budget_bytes=gp.peak_bytes,
                        device_id=device,
                        ingredients=("rho", "gradient", "tau"),
                    )
                )
                rp = plan_cuda(
                    plan.reduction_program(atoms=na),
                    compiler.target,
                    max_bytes=max_device_bytes - device_bound,
                )
                self.reduction = self._stack.enter_context(
                    PreparedCuda(rp, compile_cuda(rp, compiler, cache), device=device)
                )
                self._reduction_device_bytes = rp.peak_bytes
                self._reduction_host_bytes = rp.host_bytes
                if host_bound + rp.host_bytes > max_host_bytes:
                    raise ValueError(
                        "WB97M-V stationary reduction exceeds host capacity"
                    )
                self._identity = identity
            except BaseException:
                self.close()
                raise

        device_bound += self._reduction_device_bytes
        host_bound += self._reduction_host_bytes

        # The native owner keeps the derivative topology across warm replays.
        component_seconds = {"prepare": perf_counter() - started}
        component_start = perf_counter()
        evaluate = source._library.vibeqc_ks_snapshot_cuda_integral_gradient_v1
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
        integral = np.empty((5, na, 3))
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
        names = (
            "one_electron",
            "overlap_pulay",
            "coulomb",
            "exchange_short_range",
            "exchange_long_range",
        )
        components = dict(zip(names, integral, strict=True))
        component_seconds["integral_derivatives"] = perf_counter() - component_start
        component_start = perf_counter()
        density = state.density if plan.spin_blocks == 2 else state.density[0]
        self.grid.set_density(density)
        self.sources.reset(
            source.grid_spec.coincident_tolerance, state.density, state.weighted_density
        )
        charges = np.array([a.atomic_number for a in basis.atoms], dtype=float)
        for atom in range(na):
            for other in range(atom):
                # This module contains one raw nuclear primitive (kind zero).
                # Spherical d AOs select component encoding in the generic
                # integral consumer, but no AO component tags belong to this
                # nuclear-only dispatch; integral derivatives are native here.
                self.sources._call(
                    "stationary_nuclear",
                    self.sources.handle,
                    0,
                    atom,
                    other,
                    float(charges[atom]),
                    float(charges[other]),
                )
        ids = np.arange(n, dtype=np.uintp)
        rho, gradient = np.empty(npnt), np.empty((npnt, 3))
        for begin in range(0, npnt, tile_points):
            end = min(begin + tile_points, npnt)
            points = state.grid.points[begin:end]
            features = self.grid.evaluate(points, ao_ids=ids)
            rho[begin:end] = features["rho"].sum(axis=0)
            gradient[begin:end] = features["gradient"].sum(axis=0)
            with self.grid.xc_task(points, ids, "WB97M-V") as task:
                self.sources.geometry(
                    task,
                    np.asarray(state.grid.owners[begin:end], dtype=np.int64),
                    state.grid.weights[begin:end],
                    source.atomic_weights[begin:end],
                    functional=4,
                )
        local = self.sources.finish()
        for name in ("xc_ao", "xc_grid", "xc_weight", "nuclear"):
            components[name] = local[name]
        component_seconds["semilocal_geometry_and_features"] = (
            perf_counter() - component_start
        )
        component_start = perf_counter()

        # Compact BOTH VV10 domains with the same density policy as SCF.
        # The derivative is on this fixed active branch; inactive seeds are zero.
        active = rho >= float(MOLECULAR_VV10_DENSITY_THRESHOLD)
        count = int(active.sum())
        seeds = np.zeros((6, npnt))
        if count:
            if self._nonlocal is None or self._nonlocal.point_count != count:
                if self._nonlocal is not None:
                    self._nonlocal.close()
                self._nonlocal = NonlocalFixedGridPlan(
                    nlc.spec,
                    count,
                    coefficient=nlc.coefficient,
                    tile_points=tile_points,
                    maximum_bytes=nlc_budget,
                    device="cuda",
                    device_id=device,
                    library=source._library,
                )
            result = self._nonlocal.execute(
                state.grid.points[active],
                state.grid.weights[active],
                rho[active],
                gradient[active],
                geometry=True,
            )
            seeds[0, active], seeds[1, active] = result.vrho, result.vsigma
            seeds[2:5, active] = result.point_derivative.T
            seeds[5, active] = result.weight_derivative
        component_seconds["vv10_pairs"] = perf_counter() - component_start
        component_start = perf_counter()
        self.sources.reset(
            source.grid_spec.coincident_tolerance, state.density, state.weighted_density
        )
        for begin in range(0, npnt, tile_points):
            end = min(begin + tile_points, npnt)
            with self.grid.xc_task(
                state.grid.points[begin:end], ids, "WB97M-V"
            ) as task:
                owners = np.ascontiguousarray(
                    state.grid.owners[begin:end], dtype=np.int64
                )
                weights = np.ascontiguousarray(
                    state.grid.weights[begin:end] * active[begin:end]
                )
                raw = np.ascontiguousarray(source.atomic_weights[begin:end])
                external = np.ascontiguousarray(seeds[:, begin:end])
                self.sources._call(
                    "stationary_geometry_external",
                    self.sources.handle,
                    ct.byref(task.view),
                    task.density_jets(4),
                    _ptr(owners),
                    _ptr(weights),
                    _ptr(raw),
                    _ptr(external),
                )
        nonlocal_parts = self.sources.finish()
        for suffix in ("ao", "grid", "weight"):
            components["nonlocal_" + suffix] = nonlocal_parts["xc_" + suffix]
        component_seconds["nonlocal_geometry"] = perf_counter() - component_start
        component_start = perf_counter()
        plan.reduction_program(atoms=na, sources=components)
        result = self.reduction.execute(components)
        contract.validate(state)
        component_seconds["reduction_and_validation"] = perf_counter() - component_start
        self.executions += 1
        work = {
            "execution": "cuda-complete-wb97mv",
            "plan_identity": plan.identity,
            "source_names": list(plan.source_names),
            "grid_points": npnt,
            "ao_collocation_point_visits": 3 * npnt,
            "geometry_point_visits": 2 * npnt,
            "partition_pair_visits": 2 * npnt * na * (na - 1),
            "nonlocal_active_points": count,
            "nonlocal_pair_evaluations": count**2,
            "ordered_quartets_per_integral_source": n**4,
            "two_electron_quartet_traversals": 1,
            "ordered_quartet_visits_total": n**4,
            "maximum_center_dual3_evaluations_total": 6 * n**4,
            "range_recurrences_per_participating_center": 2,
            "additional_device_peak_bound": device_bound,
            "additional_host_numeric_bound": host_bound,
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
            "host_scope": "snapshot validation, tiling, total-feature/active-set packing",
            "endpoint_seconds": perf_counter() - started,
            "component_seconds": component_seconds,
        }
        self.last_work = work
        return -np.asarray(result.outputs["gradient"]).copy(), work
