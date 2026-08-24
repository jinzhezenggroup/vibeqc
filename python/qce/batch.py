"""Persistent ragged-batch interface backed by the native fleet plan."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from . import _native
from .calculator import Atom, Calculator


@dataclass(frozen=True)
class BatchItemResult:
    index: int
    status: int
    status_message: str
    energy: float
    forces: np.ndarray | None
    converged: bool
    iterations: int
    energy_change: float
    density_rms: float
    executed_backend: str
    bucket_id: int
    warm_start_used: bool
    warm_start_fallback: bool

    @property
    def succeeded(self) -> bool:
        return self.status == _native.STATUS_SUCCESS


@dataclass(frozen=True)
class BatchResult:
    """Input-ordered results for a ragged batch; forces are never padded."""

    items: tuple[BatchItemResult, ...]

    @property
    def energies(self) -> np.ndarray:
        return np.asarray(
            [item.energy if item.succeeded else np.nan for item in self.items],
            dtype=np.float64,
        )

    @property
    def failure_indices(self) -> tuple[int, ...]:
        return tuple(item.index for item in self.items if not item.succeeded)

    @property
    def succeeded(self) -> bool:
        return not self.failure_indices

    def raise_for_failures(self) -> None:
        failures = [
            f"{item.index}: {item.status_message}"
            for item in self.items
            if not item.succeeded
        ]
        if failures:
            raise RuntimeError("batched HF item failures: " + "; ".join(failures))


class PreparedBatch:
    """Persistent topology-aware native fleet plan.

    The object is not concurrently re-entrant because successful executions
    update per-system warm-start densities. Use separate plans for concurrent
    callers.
    """

    def __init__(
        self,
        calculator: Calculator,
        systems: Sequence[Iterable[Atom | tuple[str | int, Sequence[float]]]],
        *,
        charges: Sequence[int] | None = None,
        multiplicities: Sequence[int] | None = None,
        warm_start: bool = True,
    ) -> None:
        if not systems:
            raise ValueError("a batch requires at least one system")
        self._calculator = calculator
        self._library = calculator._library
        self._systems = tuple(tuple(Atom.from_value(atom) for atom in system) for system in systems)
        if any(not system for system in self._systems):
            raise ValueError("every batch item requires at least one atom")
        count = len(self._systems)
        self._charges = tuple(0 for _ in range(count)) if charges is None else tuple(charges)
        self._multiplicities = (
            tuple(1 for _ in range(count))
            if multiplicities is None
            else tuple(multiplicities)
        )
        if len(self._charges) != count or len(self._multiplicities) != count:
            raise ValueError("charges and multiplicities must match the batch size")
        self._atom_counts = tuple(len(system) for system in self._systems)
        self._atomic_numbers = tuple(
            tuple(atom.atomic_number for atom in system) for system in self._systems
        )
        self._context = ctypes.c_void_p()
        self._batch = ctypes.c_void_p()

        _native.check(
            self._library,
            self._library.qce_context_create(
                ctypes.byref(calculator._context_descriptor()), ctypes.byref(self._context)
            ),
        )
        system_handles: list[ctypes.c_void_p] = []
        try:
            for atoms, charge, multiplicity in zip(
                self._systems, self._charges, self._multiplicities, strict=True
            ):
                system_handles.append(
                    calculator._create_native_system(
                        self._context, atoms, charge, multiplicity
                    )
                )
            handle_array = (ctypes.c_void_p * count)(
                *(handle.value for handle in system_handles)
            )
            method = calculator._method_descriptor()
            flags = _native.BATCH_ENABLE_WARM_STARTS if warm_start else 0
            _native.check(
                self._library,
                self._library.qce_batch_prepare(
                    self._context,
                    handle_array,
                    count,
                    ctypes.byref(method),
                    flags,
                    ctypes.byref(self._batch),
                ),
            )
        except Exception:
            self.close()
            raise
        finally:
            for handle in system_handles:
                self._library.qce_system_destroy(handle)

    @property
    def system_count(self) -> int:
        self._ensure_open()
        return int(self._library.qce_batch_get_system_count(self._batch))

    @property
    def atomic_numbers(self) -> tuple[tuple[int, ...], ...]:
        return self._atomic_numbers

    @property
    def charges(self) -> tuple[int, ...]:
        """Charges retained by this fixed-topology native plan."""

        return self._charges

    @property
    def multiplicities(self) -> tuple[int, ...]:
        """Spin multiplicities retained by this fixed-topology native plan."""

        return self._multiplicities

    def _ensure_open(self) -> None:
        if not self._batch.value:
            raise RuntimeError("prepared batch is closed")

    def execute(
        self,
        coordinates: Sequence[Sequence[Sequence[float]] | np.ndarray | None] | None = None,
        *,
        strict: bool = False,
    ) -> BatchResult:
        self._ensure_open()
        count = len(self._systems)
        coordinate_storage: list[np.ndarray] = []
        inputs_pointer = None
        input_count = 0
        if coordinates is not None:
            if len(coordinates) != count:
                raise ValueError("coordinate list must match the prepared batch size")
            input_descriptors: list[_native.BatchInputDescriptor] = []
            for item in coordinates:
                if item is None:
                    input_descriptors.append(
                        _native.BatchInputDescriptor(
                            ctypes.sizeof(_native.BatchInputDescriptor),
                            _native.ABI_VERSION,
                            None,
                            0,
                        )
                    )
                    continue
                array = np.ascontiguousarray(item, dtype=np.float64).reshape(-1)
                coordinate_storage.append(array)
                input_descriptors.append(
                    _native.BatchInputDescriptor(
                        ctypes.sizeof(_native.BatchInputDescriptor),
                        _native.ABI_VERSION,
                        array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        array.size,
                    )
                )
            input_array = (_native.BatchInputDescriptor * count)(*input_descriptors)
            inputs_pointer = input_array
            input_count = count

        force_storage = [
            (ctypes.c_double * (3 * atom_count))() for atom_count in self._atom_counts
        ]
        output_array = (_native.BatchItemResultDescriptor * count)(
            *(
                _native.BatchItemResultDescriptor(
                    ctypes.sizeof(_native.BatchItemResultDescriptor),
                    _native.ABI_VERSION,
                    _native.STATUS_INVALID_ARGUMENT,
                    0.0,
                    force_storage[index],
                    len(force_storage[index]),
                    0,
                    0.0,
                    0.0,
                    0,
                    _native.BACKEND_CPU_REFERENCE,
                    0,
                    0,
                    0,
                )
                for index in range(count)
            )
        )
        _native.check(
            self._library,
            self._library.qce_batch_execute(
                self._batch,
                inputs_pointer,
                input_count,
                output_array,
                count,
            ),
        )

        items: list[BatchItemResult] = []
        for index, output in enumerate(output_array):
            succeeded = output.status == _native.STATUS_SUCCESS
            forces = (
                np.ctypeslib.as_array(force_storage[index]).copy().reshape(-1, 3)
                if succeeded
                else None
            )
            message = self._library.qce_status_message(output.status).decode("utf-8")
            items.append(
                BatchItemResult(
                    index=index,
                    status=output.status,
                    status_message=message,
                    energy=output.energy,
                    forces=forces,
                    converged=bool(output.converged),
                    iterations=output.iterations,
                    energy_change=output.energy_change,
                    density_rms=output.density_rms,
                    executed_backend={
                        _native.BACKEND_CPU_REFERENCE: "cpu_reference",
                        _native.BACKEND_CUDA: "cuda",
                        _native.BACKEND_HYBRID_CUDA: "hybrid_cuda",
                    }.get(output.executed_backend, "unknown"),
                    bucket_id=output.bucket_id,
                    warm_start_used=bool(output.warm_start_used),
                    warm_start_fallback=bool(output.warm_start_fallback),
                )
            )
        result = BatchResult(tuple(items))
        if strict:
            result.raise_for_failures()
        return result

    def clear_warm_starts(self) -> None:
        self._ensure_open()
        _native.check(
            self._library,
            self._library.qce_batch_clear_warm_starts(self._batch),
        )

    def close(self) -> None:
        if self._batch.value:
            self._library.qce_batch_destroy(self._batch)
            self._batch = ctypes.c_void_p()
        if self._context.value:
            self._library.qce_context_destroy(self._context)
            self._context = ctypes.c_void_p()

    def __enter__(self) -> "PreparedBatch":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
