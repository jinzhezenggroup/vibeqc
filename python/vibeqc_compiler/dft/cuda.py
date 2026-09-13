"""Explicit CUDA AO/feature execution with a persistent private plan/arena."""

from __future__ import annotations

import ctypes as ct
import threading
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import numpy as np

from vibeqc_compiler.common.arrays import immutable
from vibeqc_compiler.common.cuda_runtime import _PREPARATION_LOCK, _Metrics
from vibeqc_compiler.common.native_runtime import compile_runtime
from vibeqc_compiler.common.provenance import file_hash
from vibeqc_compiler.common.resources import MAX_BYTES, ResourceBudget, plan_resources

from .ao import DOUBLE, SIZE, jet_indices, pointer
from .ao_cuda import emit_grid_source
from .density_source import DensitySource
from .features import requested_ingredients, spin_densities
from .grid import checked_int
from .plan import plan_tiles


class GridTaskView(ct.Structure):
    """Layout mirror of grid_task_view.cuh, borrowed only inside a task lease."""

    _fields_ = [
        ("version", ct.c_uint64),
        ("generation", ct.c_uint64),
        ("npoint", ct.c_size_t),
        ("nao", ct.c_size_t),
        ("nactive", ct.c_size_t),
        ("jets", ct.c_size_t),
        ("ao_ids", SIZE),
        ("points", DOUBLE),
        ("ao", DOUBLE),
        ("features", DOUBLE),
        ("local_potential", DOUBLE),
        ("potential", DOUBLE),
        ("stream", ct.c_void_p),
        ("error", ct.POINTER(ct.c_int)),
    ]


class DeviceGridTask:
    """Lease of a native task view; later consumers can operate on its stream.

    The owner serializes preparation and consumption. Only explicit diagnostic
    scatter inputs/outputs cross the host boundary; ordinary consumers write
    the view's local potential and call scatter without host arrays.
    """

    def __init__(self, owner, view):
        self._owner, self._view, self._active = owner, view, True

    @property
    def view(self):
        if not self._active:
            raise RuntimeError("expired device grid task lease")
        return self._view

    def scatter(self, local=None, *, reset=False, download=False):
        """Accumulate symmetric spin-local matrices through the explicit AO map.

        A device failure may partially update the global matrix. Retry with
        ``reset=True`` (or start a new density execution) to discard it; error
        status is fresh for every scatter attempt.
        """
        view = self.view
        if type(reset) is not bool or type(download) is not bool:
            raise ValueError("scatter flags must be boolean")
        if local is not None:
            local = immutable(local, shape=(2, view.nactive, view.nactive))
            if not np.allclose(local, local.swapaxes(1, 2), atol=1e-12, rtol=1e-10):
                raise ValueError("local potential must be symmetric")
        result = np.empty((2, view.nao, view.nao)) if download else None
        self._owner._call(
            "grid_cuda_scatter_v1",
            self._owner._handle,
            view.generation,
            None if local is None else pointer(local),
            int(reset),
            None if result is None else pointer(result),
        )
        return None if result is None else immutable(result)


def compile_cuda(compiler, cache):
    """Compile the device runtime without running a GPU or importing PySCF."""
    source, identity, headers = emit_grid_source()
    folder = Path(cache).resolve() / "source" / identity
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "grid.cu"
    if path.exists() and path.read_text() != source:
        raise ValueError("generated grid source identity mismatch")
    path.write_text(source)
    return compile_runtime(
        compiler,
        cache,
        path,
        headers=headers,
        libraries=("cublas",),
        options=("--fmad=false", f"-I{headers[0].parent}"),
    )


class CudaGrid:
    """Own basis/D/B/tiles and execute AO jets, cuBLAS contractions and invariants.

    Grid generation and partition weights remain on the CPU. Points upload
    explicitly, features download explicitly, and AO jets download only when
    requested by validation or a CPU consumer. No molecular AO grid is retained. Calls on
    one plan are serialized; different plans own independent arenas/streams.
    """

    _fixed = frozenset(
        (
            "plan",
            "ingredients",
            "basis_identity",
            "basis_generation",
            "artifact",
            "resource_plan",
            "device_id",
        )
    )

    def __setattr__(self, name, value):
        if name in self._fixed and name in self.__dict__:
            raise AttributeError(
                "CUDA scientific topology is immutable; prepare a new owner"
            )
        super().__setattr__(name, value)

    @property
    def source_stamp(self):
        """Read-only identity of the successfully uploaded current source."""
        return self._source_stamp

    @property
    def source_kind(self):
        """Selected route; reset to the empty D state when an upload fails."""
        return self._source_kind

    @property
    def fallback_reason(self):
        """Explicit availability decision; no matrix approximation is implied."""
        return self._fallback_reason

    @property
    def source_statistics(self):
        """Detached identity, packing and upload diagnostics."""
        return dict(self._source_statistics)

    def __init__(
        self,
        basis,
        artifact,
        *,
        order=1,
        tile_points=256,
        budget_bytes=None,
        device_id=0,
        grid=None,
        active_ao_capacity=None,
        orbital_capacity=None,
        orbital_tile=32,
        ingredients=None,
        resource_budget=None,
        basis_generation=0,
    ):
        self._lock = threading.RLock()
        self._handle = ct.c_void_p()
        self._density_ready = False
        self._borrowed = False
        self._source_stamp = None
        self._source_kind = "density_matrix"
        self._fallback_reason = "missing_orbitals"
        self._source_statistics = {}
        self.ingredients = requested_ingredients(ingredients)
        self.basis_generation = checked_int(basis_generation, "basis generation", low=0)
        checked_int(device_id, "visible device ordinal", low=0)
        if resource_budget is not None and budget_bytes is not None:
            raise ValueError("use the shared resource budget or the legacy grid budget")
        self.plan = plan_tiles(
            basis,
            backend="cuda",
            order=order,
            tile_points=tile_points,
            budget_bytes=(
                MAX_BYTES
                if resource_budget is not None
                else (256 << 20 if budget_bytes is None else budget_bytes)
            ),
            grid=grid,
            active_ao_capacity=active_ao_capacity,
            orbital_capacity=orbital_capacity,
            orbital_tile=orbital_tile,
        )
        self.resource_plan = plan_resources(
            (self.plan.resource_request(self.ingredients, device_id),),
            resource_budget or ResourceBudget(),
        ).require_feasible()
        if file_hash(artifact.library) != artifact.metadata["binary_sha256"]:
            raise ValueError("CUDA grid binary hash mismatch")
        self.basis_identity = basis.identity
        self.artifact = artifact
        self.device_id = device_id
        lib = self._library = ct.CDLL(str(artifact.library))
        lib.grid_cuda_create_v1.argtypes = [
            ct.c_int,
            ct.c_int,
            ct.c_int,
            SIZE,
            DOUBLE,
            ct.c_size_t,
            ct.c_uint,
            ct.c_size_t,
            ct.POINTER(ct.c_void_p),
            ct.c_char_p,
            ct.c_size_t,
        ]
        lib.grid_cuda_create_v2.argtypes = [
            *lib.grid_cuda_create_v1.argtypes[:8],
            ct.c_size_t,
            *lib.grid_cuda_create_v1.argtypes[8:],
        ]
        lib.grid_cuda_create_v3.argtypes = [
            *lib.grid_cuda_create_v2.argtypes[:-3],
            SIZE,
            ct.c_size_t,
            ct.c_uint,
            *lib.grid_cuda_create_v2.argtypes[-3:],
        ]
        lib.grid_cuda_destroy_v1.argtypes = [ct.c_void_p]
        lib.grid_cuda_destroy_v1.restype = None
        lib.grid_cuda_density_v1.argtypes = [
            ct.c_void_p,
            DOUBLE,
            ct.c_size_t,
            ct.c_char_p,
            ct.c_size_t,
        ]
        lib.grid_cuda_source_v1.argtypes = [
            ct.c_void_p,
            DOUBLE,
            ct.c_size_t,
            DOUBLE,
            DOUBLE,
            SIZE,
            ct.c_int,
            ct.c_char_p,
            ct.c_size_t,
        ]
        lib.grid_cuda_run_v1.argtypes = [
            ct.c_void_p,
            DOUBLE,
            ct.c_size_t,
            ct.c_int,
            DOUBLE,
            DOUBLE,
            ct.c_char_p,
            ct.c_size_t,
        ]
        lib.grid_cuda_run_selected_v1.argtypes = [
            ct.c_void_p,
            DOUBLE,
            ct.c_size_t,
            ct.c_int,
            SIZE,
            ct.c_size_t,
            DOUBLE,
            DOUBLE,
            ct.c_char_p,
            ct.c_size_t,
        ]
        lib.grid_cuda_view_v1.argtypes = [
            ct.c_void_p,
            ct.POINTER(GridTaskView),
            ct.c_char_p,
            ct.c_size_t,
        ]
        lib.grid_cuda_scatter_v1.argtypes = [
            ct.c_void_p,
            ct.c_uint64,
            DOUBLE,
            ct.c_int,
            DOUBLE,
            ct.c_char_p,
            ct.c_size_t,
        ]
        lib.grid_cuda_metrics_v1.argtypes = [
            ct.c_void_p,
            ct.POINTER(_Metrics),
            ct.POINTER(ct.c_int),
            ct.c_char_p,
            ct.c_size_t,
        ]
        architecture = artifact.metadata["identity"]["target"]["architecture"]
        number = int(architecture.removeprefix("sm_"))
        with _PREPARATION_LOCK:
            self._call(
                "grid_cuda_create_v3",
                device_id,
                number // 10,
                number % 10,
                (ct.c_size_t * 3)(basis.natom, basis.nprimitive, basis.nao),
                pointer(basis.packed),
                tile_points,
                order,
                self.plan.allocation_bytes,
                0 if active_ao_capacity is None else active_ao_capacity,
                None
                if self.plan.orbital_capacity is None
                else (ct.c_size_t * 2)(*self.plan.orbital_capacity),
                self.plan.orbital_tile,
                sum(
                    1 << ("rho", "gradient", "sigma", "tau").index(k)
                    for k in self.ingredients
                ),
                ct.byref(self._handle),
            )

    def _call(self, name, *args):
        error = ct.create_string_buffer(2048)
        if getattr(self._library, name)(*args, error, len(error)):
            raise RuntimeError(error.value.decode())

    def _check_open(self):
        if not self._handle:
            raise RuntimeError("CUDA grid plan is closed")
        if self._borrowed:
            raise RuntimeError("CUDA grid buffers are leased to a task consumer")

    def set_density(self, density):
        """Validate and replace both spin matrices; no old-density reuse is implicit."""
        with self._lock:
            self._check_open()
            d = spin_densities(density, self.plan.nao)
            self._density_ready = False
            self._source_stamp = None
            self._source_kind = "density_matrix"
            self._fallback_reason = "missing_orbitals"
            self._source_statistics = {}
            self._call("grid_cuda_density_v1", self._handle, pointer(d), d.size)
            self._density_ready = True

    def set_source(self, source, *, stamp, route="auto"):
        """Upload one checked current D/B pair; never reconstruct D on tile replay.

        DensitySource performs external-factor validation before this boundary.
        The caller carries its current stamp; geometry/basis generations must
        match this owner. Auto selects an available validated route, with an
        explicit D fallback when factors or prepared capacity are unavailable.
        This is an availability policy, not the #168 performance selector.
        """
        with self._lock:
            self._check_open()
            if not isinstance(source, DensitySource):
                raise TypeError("expected a validated DensitySource")
            if stamp != source.stamp or (
                stamp.basis_identity != self.basis_identity
                or stamp.basis_generation != self.basis_generation
                or source.density.shape != (2, self.plan.nao, self.plan.nao)
            ):
                raise ValueError("stale density source or CUDA basis generation")
            if route not in ("auto", "density_matrix", "orbitals"):
                raise ValueError("unsupported density feature route")
            counts = (
                (0, 0)
                if source.occupations is None
                else tuple(map(len, source.occupations))
            )
            reason = source.fallback_reason
            available = source.source_kind == "orbitals"
            if available and (
                self.plan.orbital_capacity is None
                or any(
                    n > cap
                    for n, cap in zip(counts, self.plan.orbital_capacity, strict=True)
                )
            ):
                available, reason = False, "orbital_capacity_exceeded"
            if route == "orbitals" and not available:
                raise ValueError(f"orbital route unavailable: {reason}")
            use_orbitals = available and route != "density_matrix"
            before = perf_counter()
            factors = (
                tuple(
                    immutable(c * np.sqrt(f))
                    for c, f in zip(
                        source.coefficients, source.occupations, strict=True
                    )
                )
                if use_orbitals
                else ()
            )
            packing_seconds = perf_counter() - before
            counts = counts if use_orbitals else (0, 0)
            self._density_ready = False
            self._source_stamp = None
            self._source_kind = "density_matrix"
            self._fallback_reason = "missing_orbitals"
            self._source_statistics = {}
            before = perf_counter()
            self._call(
                "grid_cuda_source_v1",
                self._handle,
                pointer(source.density),
                source.density.size,
                pointer(factors[0]) if factors else None,
                pointer(factors[1]) if factors else None,
                (ct.c_size_t * 2)(*counts),
                int(use_orbitals),
            )
            self._source_stamp = stamp
            self._source_kind = "orbitals" if use_orbitals else "density_matrix"
            self._fallback_reason = (
                None
                if use_orbitals
                else (
                    "requested_density_matrix" if route == "density_matrix" else reason
                )
            )
            self._source_statistics = {
                "source_identity": stamp.identity,
                "factor_identity": source.factor_identity if use_orbitals else None,
                "source_kind": self.source_kind,
                "fallback_reason": self.fallback_reason,
                "occupied_counts": counts,
                "factor_packing_seconds": packing_seconds,
                "source_upload_seconds": perf_counter() - before,
                "source_upload_bytes": source.density.nbytes
                + sum(f.nbytes for f in factors),
            }
            self._density_ready = True

    def evaluate(
        self,
        points,
        *,
        features=True,
        download_jets=False,
        ao_ids=None,
        download_features=True,
        stamp=None,
    ):
        """Return one detached result tile; no downstream CPU arithmetic fallback."""
        raw = np.asarray(points)
        if raw.ndim != 2 or raw.shape[1] != 3 or len(raw) > self.plan.tile_points:
            raise ValueError("grid points exceed the prepared tile shape")
        if (
            type(features) is not bool
            or type(download_jets) is not bool
            or type(download_features) is not bool
            or not (features or download_jets)
        ):
            raise ValueError("request features and/or AO jets")
        with self._lock:
            self._check_open()
            need_first = any(k != "rho" for k in self.ingredients)
            if features and (
                (need_first and self.plan.order < 1) or not self._density_ready
            ):
                raise ValueError(
                    "features require the requested derivatives and supplied density"
                )
            if (
                features
                and self.source_stamp is not None
                and stamp != self.source_stamp
            ):
                raise ValueError("stale or missing current CUDA density source stamp")
            points = immutable(raw)
            active = self.plan.nao
            selected = None
            if self.plan.active_ao_capacity is None:
                if ao_ids is not None:
                    raise ValueError("active AO maps require a local CUDA plan")
            else:
                raw_ids = np.asarray(ao_ids)
                if raw_ids.ndim != 1 or (
                    raw_ids.size
                    and (
                        raw_ids.dtype.kind not in "iu"
                        or np.any(raw_ids < 0)
                        or np.any(raw_ids >= self.plan.nao)
                        or np.any(raw_ids[1:] <= raw_ids[:-1])
                    )
                ):
                    raise ValueError(
                        "active AO IDs must be sorted unique in-range integers"
                    )
                active = len(raw_ids)
                if active > self.plan.active_ao_capacity:
                    raise ValueError("active AO map exceeds the prepared capacity")
                selected = np.array(raw_ids, dtype=np.uintp, copy=True)
            values = (
                np.empty((13, len(points))) if features and download_features else None
            )
            jets = (
                np.empty((len(jet_indices(self.plan.order)), len(points), active))
                if download_jets
                else None
            )
            self._call(
                "grid_cuda_run_selected_v1",
                self._handle,
                pointer(points),
                len(points),
                int(features),
                None if selected is None else selected.ctypes.data_as(SIZE),
                active,
                pointer(values) if values is not None else None,
                pointer(jets) if jets is not None else None,
            )
            result = {}
            if values is not None:
                # The transfer ABI has thirteen slots, but publication copies
                # belong only to the features actually requested on the GPU.
                for key in self.ingredients:
                    if key == "gradient":
                        value = (
                            values[[1, 2, 3, 6, 7, 8]]
                            .reshape(2, 3, len(points))
                            .transpose(0, 2, 1)
                        )
                    else:
                        value = values[
                            {"rho": [0, 5], "tau": [4, 9], "sigma": slice(10, 13)}[key]
                        ]
                    result[key] = immutable(value)
            if download_jets:
                result["ao_jets"] = immutable(jets)
            return result

    @contextmanager
    def task(self, points, ao_ids, *, stamp=None):
        """Evaluate local features and lend device buffers with no array D2H.

        Consumers enqueue on ``lease.view.stream`` and finish while the lease
        is held. Reconfiguration, density changes and nested tasks are rejected.
        Only the scalar device error status is downloaded automatically.
        """
        with self._lock:
            self._check_open()
            if self.plan.active_ao_capacity is None:
                raise ValueError("device task views require a local CUDA plan")
            if set(self.ingredients) != {"rho", "gradient", "sigma", "tau"}:
                raise ValueError("device task ABI v1 requires the full feature layout")
            self.evaluate(points, ao_ids=ao_ids, download_features=False, stamp=stamp)
            view = GridTaskView()
            self._call("grid_cuda_view_v1", self._handle, ct.byref(view))
            self._borrowed = True
            lease = DeviceGridTask(self, view)
            try:
                yield lease
            finally:
                lease._active = False
                self._borrowed = False

    def metrics(self):
        """Synchronized cumulative timings, owned allocations and loaded versions."""
        with self._lock:
            self._check_open()
            metrics = _Metrics()
            versions = (ct.c_int * 3)()
            self._call(
                "grid_cuda_metrics_v1", self._handle, ct.byref(metrics), versions
            )
            return {
                **{name: getattr(metrics, name) for name, _ in metrics._fields_},
                "runtime_version": versions[0],
                "driver_version": versions[1],
                "cublas_version": versions[2],
            }

    def close(self):
        with self._lock, _PREPARATION_LOCK:
            if self._borrowed:
                raise RuntimeError("CUDA grid buffers are leased to a task consumer")
            if self._handle:
                self._library.grid_cuda_destroy_v1(self._handle)
                self._handle = ct.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        if hasattr(self, "_lock"):
            self.close()
