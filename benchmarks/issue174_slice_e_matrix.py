"""Measure the #174 slice-E acceptance matrix inside one Slurm allocation.

Slice E asks for an end-to-end error/cost ablation, not an isolated kernel
speedup. This harness therefore measures *complete solves* under identical
numerical controls, at several convergence tolerances, for the two public
precision policies, and reports the actual observable error against a
separately computed, tighter FP64 reference.

Design constraints taken from the issue:

* Every tolerance is measured for both ``fp64`` and ``auto`` at the *same*
  density/screening settings, so the precision effect is isolated.
* Actual error is measured against a stricter FP64 reference, never against a
  same-tolerance FP64 solve.
* Five interleaved samples per configuration, with the accuracy-requesting
  order alternated so one policy does not systematically see a colder state.
* Singlepoint calls are always cold native solves. Persistent prepared batches
  measure cold, same-geometry warm and changed-geometry warm states separately;
  failures and unconverged runs are retained.
* Per-item provenance comes from the public getters, so a run that fell back to
  FP64 is reported as such instead of being assumed mixed.
* The large topology case is bounded-streaming and needs an explicit flag: it
  has no exact device tile arena, which is exactly the domain boundary this
  slice has to expose rather than hide.

No Python GPU package is required. When ``cupy`` is importable it is used, and
otherwise a small ctypes facade over ``libcudart``/``libcuda`` supplies the same
synchronization and device-property surface, so the recorded accelerator fields
stay real values rather than placeholders.

The harness only *measures*. It selects no policy, promotes nothing, and
records its own limitations.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import sys
import time
import types
import typing
from contextlib import ExitStack
from pathlib import Path
from statistics import median
from typing import Any

try:
    from benchmarks._retention import raw_output_path
except ModuleNotFoundError:
    from _retention import raw_output_path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    # Direct ``python benchmarks/...`` execution otherwise exposes only the
    # benchmarks directory, not its namespace-package parent.
    sys.path.insert(0, str(_REPOSITORY_ROOT))

DEFAULT_TOLERANCES = (1.0e-9, 1.0e-7, 1.0e-6, 1.0e-5)
DEFAULT_CASES = (
    "water-tetramer-def2-svp-spherical",
    "water-octamer-s4-def2-svp-spherical",
)
LARGE_CASES = ("water-hexadecamer-2s4-def2-svp-spherical",)
BOUNDED_STREAMING_CASES = ("water-32mer-4s4-def2-svp-spherical",)
MODES = ("fp64", "auto")

# CUDA driver attributes used to describe the device without cupy. The numeric
# values are the stable CUDA_DEVICE_ATTRIBUTE_* enumerators from cuda.h.
_CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13
_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76


class _CudaRuntimeFacade:
    """The ``cupy.cuda.runtime`` subset this harness and its helpers use."""

    def __init__(self) -> None:
        self._cudart = ctypes.CDLL("libcudart.so")
        self._driver = ctypes.CDLL("libcuda.so.1")
        self._bind()
        self._check(self._driver.cuInit(0), "cuInit")

    def _bind(self) -> None:
        """Declare pointer and size types before any call is possible."""

        integer = ctypes.POINTER(ctypes.c_int)
        driver = self._driver
        driver.cuInit.argtypes = [ctypes.c_uint]
        driver.cuInit.restype = ctypes.c_int
        driver.cuDeviceGet.argtypes = [integer, ctypes.c_int]
        driver.cuDeviceGet.restype = ctypes.c_int
        driver.cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        driver.cuDeviceGetName.restype = ctypes.c_int
        driver.cuDeviceGetAttribute.argtypes = [integer, ctypes.c_int, ctypes.c_int]
        driver.cuDeviceGetAttribute.restype = ctypes.c_int
        driver.cuDeviceTotalMem_v2.argtypes = [
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_int,
        ]
        driver.cuDeviceTotalMem_v2.restype = ctypes.c_int
        driver.cuDriverGetVersion.argtypes = [integer]
        driver.cuDriverGetVersion.restype = ctypes.c_int
        cudart = self._cudart
        cudart.cudaGetDevice.argtypes = [integer]
        cudart.cudaGetDevice.restype = ctypes.c_int
        cudart.cudaRuntimeGetVersion.argtypes = [integer]
        cudart.cudaRuntimeGetVersion.restype = ctypes.c_int
        cudart.cudaDeviceSynchronize.argtypes = []
        cudart.cudaDeviceSynchronize.restype = ctypes.c_int

    @staticmethod
    def _check(status: int, name: str) -> None:
        """Fail loudly instead of reporting placeholder device fields."""

        if status != 0:
            raise RuntimeError(f"{name} failed with CUDA error {status}")

    def _device_handle(self, device_id: int) -> int:
        """Resolve one visible device ordinal into a driver handle."""

        handle = ctypes.c_int()
        self._check(
            self._driver.cuDeviceGet(ctypes.byref(handle), device_id), "cuDeviceGet"
        )
        return handle.value

    def _attribute(self, handle: int, attribute: int) -> int:
        value = ctypes.c_int()
        self._check(
            self._driver.cuDeviceGetAttribute(ctypes.byref(value), attribute, handle),
            f"cuDeviceGetAttribute({attribute})",
        )
        return value.value

    def current_device(self) -> int:
        """Return the ordinal of the device this process is bound to."""

        device = ctypes.c_int()
        self._check(self._cudart.cudaGetDevice(ctypes.byref(device)), "cudaGetDevice")
        return device.value

    def getDeviceProperties(self, device_id: int) -> dict[str, Any]:
        """Return the fields ``benchmarks._support`` reads from cupy."""

        handle = self._device_handle(device_id)
        name = ctypes.create_string_buffer(256)
        self._check(
            self._driver.cuDeviceGetName(name, len(name), handle), "cuDeviceGetName"
        )
        total_memory = ctypes.c_size_t()
        self._check(
            self._driver.cuDeviceTotalMem_v2(ctypes.byref(total_memory), handle),
            "cuDeviceTotalMem",
        )
        return {
            "name": name.value,
            "major": self._attribute(
                handle, _CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
            ),
            "minor": self._attribute(
                handle, _CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
            ),
            "totalGlobalMem": total_memory.value,
            "multiProcessorCount": self._attribute(
                handle, _CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
            ),
            "clockRate": self._attribute(handle, _CU_DEVICE_ATTRIBUTE_CLOCK_RATE),
        }

    def driverGetVersion(self) -> int:
        version = ctypes.c_int()
        self._check(
            self._driver.cuDriverGetVersion(ctypes.byref(version)), "cuDriverGetVersion"
        )
        return version.value

    def runtimeGetVersion(self) -> int:
        version = ctypes.c_int()
        self._check(
            self._cudart.cudaRuntimeGetVersion(ctypes.byref(version)),
            "cudaRuntimeGetVersion",
        )
        return version.value

    def deviceSynchronize(self) -> None:
        """Block until all outstanding device work is complete."""

        self._check(self._cudart.cudaDeviceSynchronize(), "cudaDeviceSynchronize")


class _CudaFacade:
    """A ``cupy``-shaped facade so the repository helpers stay reusable."""

    def __init__(self) -> None:
        runtime = _CudaRuntimeFacade()

        def device() -> types.SimpleNamespace:
            return types.SimpleNamespace(id=runtime.current_device())

        def synchronize() -> None:
            runtime.deviceSynchronize()

        self.cuda = types.SimpleNamespace(
            Device=device,
            runtime=runtime,
            Stream=types.SimpleNamespace(
                null=types.SimpleNamespace(synchronize=synchronize)
            ),
        )


def _device_backend() -> typing.Any:
    """Return ``(module, name)`` for device metadata and synchronization."""

    try:
        import cupy

        return cupy, "cupy"
    except ModuleNotFoundError:
        return _CudaFacade(), "ctypes-cuda-runtime"


def _synchronize(backend: typing.Any) -> None:
    """Block on the default stream of the active device backend."""

    backend.cuda.Stream.null.synchronize()


def _displaced_atoms(atoms: typing.Any, displacement_bohr: float) -> typing.Any:
    """Move the last nucleus along +x to build a changed-geometry warm start.

    A single deterministic displacement keeps the model identity difference
    explicit: the geometry hash changes while basis, charge and multiplicity do
    not, so a warm density carried across this pair exercises the geometry
    revalidation path instead of a same-geometry replay.
    """
    moved = [list(entry) for entry in atoms]
    element, position = moved[-1]
    moved[-1] = [element, (position[0] + displacement_bohr, position[1], position[2])]
    return tuple((entry[0], tuple(entry[1])) for entry in moved)


def _calculator(
    case: typing.Any,
    arguments: typing.Any,
    *,
    precision: str,
    energy_tolerance: float,
    density_tolerance: float | None = None,
    screening_tolerance: float | None = None,
    max_iterations: int | None = None,
) -> typing.Any:
    """Construct one calculator with the controls fixed for this matrix."""

    from vibeqc import Calculator

    return Calculator(
        method=case.method,
        basis=case.vibeqc_basis,
        basis_representation=case.basis_representation,
        device="cuda",
        max_iterations=(
            arguments.max_iterations if max_iterations is None else max_iterations
        ),
        energy_tolerance=energy_tolerance,
        density_tolerance=(
            arguments.density_tolerance
            if density_tolerance is None
            else density_tolerance
        ),
        screening_tolerance=(
            arguments.screening_tolerance
            if screening_tolerance is None
            else screening_tolerance
        ),
        precision=precision,
    )


def _force_max_abs(result: typing.Any) -> float | None:
    """Return the maximum absolute Cartesian force component, if evaluated."""

    if result.forces is None:
        return None
    return float(abs(result.forces).max()) if result.forces.size else None


def _strict_reference(
    case: typing.Any,
    arguments: typing.Any,
    atoms: typing.Any,
    properties: typing.Any,
    backend: typing.Any,
) -> dict[str, Any]:
    """Compute the tighter FP64 reference the relaxed runs are compared to."""

    calculator = _calculator(
        case,
        arguments,
        precision="fp64",
        energy_tolerance=arguments.reference_energy_tolerance,
        density_tolerance=arguments.reference_density_tolerance,
        screening_tolerance=arguments.reference_screening_tolerance,
        max_iterations=arguments.reference_max_iterations,
    )
    model = calculator.resolved_model(
        atoms, charge=case.charge, multiplicity=case.multiplicity
    )
    record = {
        "controls": {
            "energy_tolerance": arguments.reference_energy_tolerance,
            "density_tolerance": arguments.reference_density_tolerance,
            "screening_tolerance": arguments.reference_screening_tolerance,
            "max_iterations": arguments.reference_max_iterations,
        }
    }
    result = None
    try:
        _synchronize(backend)
        started = time.perf_counter()
        result = calculator.singlepoint(
            atoms,
            charge=case.charge,
            multiplicity=case.multiplicity,
            properties=properties,
        )
        record.update(
            {
                "wall_seconds": _synchronized_seconds(started, backend),
                "converged": result.converged,
                "iterations": result.iterations,
                "energy": result.energy,
                "force_max_abs": _force_max_abs(result),
                "precision": result.precision,
            }
        )
    except Exception as error:  # noqa: BLE001 - preserve failed reference evidence
        record.update(converged=False, failure=f"{type(error).__name__}: {error}")
    return {"result": result, "model": model, "record": record}


def _accuracy_target(arguments: typing.Any, tolerance: float) -> typing.Any:
    """Build the observable requirements for the requested properties."""

    from vibeqc import ObservableTarget, TargetAccuracy

    observables = [ObservableTarget("energy", "absolute", "Eh", absolute=tolerance)]
    if "forces" in arguments.properties:
        observables.append(
            ObservableTarget(
                "forces", "max_abs", "Eh/bohr", absolute=arguments.force_target
            )
        )
    return TargetAccuracy(tuple(observables))


def _evidence(
    model: typing.Any,
    values: typing.Any,
    reference_values: typing.Any,
    target: typing.Any,
    converged: typing.Any,
) -> typing.Any:
    """Build the repository's own observable-error verdict for one run.

    The full assessment document repeats the model identity, its hashes and the
    provenance strings in every record, which pushes the artifact past the
    repository's 1 MiB review guard without adding information: the raw error
    columns carry the magnitudes and the request is recorded once per tolerance.
    Only the verdict and its per-observable outcomes are kept here.
    """

    if target is None or not converged:
        return None
    from vibeqc import compare_observables

    assessment = compare_observables(
        model,
        model,
        model,
        target,
        values,
        reference_values,
        scope="relaxed_target",
        provenance=(("reference", "independent-strict-fp64-solve"),),
        converged=converged,
    )
    return {"status": assessment.status, "outcomes": list(assessment.outcomes)}


def _error_columns(result: typing.Any, reference: typing.Any) -> dict[str, Any]:
    """Report raw observable differences next to the typed evidence record."""

    columns: dict[str, Any] = {
        "energy_absolute_error": None,
        "force_max_abs_error": None,
    }
    if (
        reference is None
        or reference["result"] is None
        or not reference["result"].converged
    ):
        return columns
    reference_result = reference["result"]
    columns["energy_absolute_error"] = abs(result.energy - reference_result.energy)
    if result.forces is not None and reference_result.forces is not None:
        columns["force_max_abs_error"] = float(
            abs(result.forces - reference_result.forces).max()
        )
    return columns


def _tolerance_matrix(
    case_name: typing.Any,
    case: typing.Any,
    atoms_base: typing.Any,
    atoms_moved: typing.Any,
    arguments: typing.Any,
    backend: typing.Any,
) -> typing.Any:
    """Run the tolerance sweep for both policies over one case."""

    references = {
        label: _strict_reference(
            case, arguments, atoms, properties=arguments.properties, backend=backend
        )
        for label, atoms in (("base", atoms_base), ("moved", atoms_moved))
    }
    records: list[dict[str, Any]] = []
    # singlepoint creates and destroys its native calculation on every call;
    # retaining a Python Calculator does not retain an SCF seed or native plan.
    plan = (("base", atoms_base), ("moved", atoms_moved))
    for tolerance in arguments.tolerances:
        target = _accuracy_target(arguments, tolerance)
        calculators = {
            mode: _calculator(
                case, arguments, precision=mode, energy_tolerance=tolerance
            )
            for mode in arguments.modes
        }
        for cycle in range(arguments.repeats):
            for step, (label, atoms) in enumerate(plan):
                ordered = (
                    arguments.modes
                    if (cycle + step) % 2 == 0
                    else tuple(reversed(arguments.modes))
                )
                for order, mode in enumerate(ordered):
                    reference = references[label]
                    record: dict[str, Any] = {
                        "case": case_name,
                        "tolerance": tolerance,
                        "mode": mode,
                        "geometry": label,
                        "kind": "cold",
                        "cycle": cycle,
                        "step": step,
                        "order": order,
                    }
                    try:
                        _synchronize(backend)
                        started = time.perf_counter()
                        result = calculators[mode].singlepoint(
                            atoms,
                            charge=case.charge,
                            multiplicity=case.multiplicity,
                            properties=arguments.properties,
                        )
                        _synchronize(backend)
                        record["wall_seconds"] = time.perf_counter() - started
                        record.update(
                            {
                                "converged": result.converged,
                                "iterations": result.iterations,
                                "energy": result.energy,
                                "energy_change": result.energy_change,
                                "density_rms": result.density_rms,
                                "force_max_abs": _force_max_abs(result),
                                "precision": result.precision,
                                "executed_backend": result.executed_backend,
                            }
                        )
                        record.update(_error_columns(result, reference))
                        if (
                            result.converged
                            and reference["result"] is not None
                            and reference["result"].converged
                        ):
                            values = {"energy": result.energy}
                            reference_values = {"energy": reference["result"].energy}
                            if result.forces is not None:
                                values["forces"] = result.forces
                                reference_values["forces"] = reference["result"].forces
                            record["accuracy"] = _evidence(
                                reference["model"],
                                values,
                                reference_values,
                                target,
                                True,
                            )
                    except Exception as error:  # noqa: BLE001 - retained in report
                        record["failure"] = f"{type(error).__name__}: {error}"
                    records.append(record)
    return records, references


def _batch_matrix(
    case_name: typing.Any,
    case: typing.Any,
    atoms_base: typing.Any,
    atoms_moved: typing.Any,
    references: typing.Any,
    arguments: typing.Any,
    backend: typing.Any,
) -> typing.Any:
    """Measure complete ragged batch solves, including per-item provenance."""

    records: list[dict[str, Any]] = []
    for tolerance in arguments.tolerances:
        target = _accuracy_target(arguments, tolerance)
        for batch_size in arguments.batch_sizes:
            labels = tuple(
                "base" if index % 2 == 0 else "moved" for index in range(batch_size)
            )
            changed_labels = tuple(
                "moved" if label == "base" else "base" for label in labels
            )
            geometries = {"base": atoms_base, "moved": atoms_moved}
            systems = [geometries[label] for label in labels]
            changed_coordinates = [
                [position for _, position in geometries[label]]
                for label in changed_labels
            ]
            for sample in range(arguments.repeats):
                # Each sample owns a fresh plan, so all three states receive
                # exactly repeats observations. Keep both policies alive and
                # alternate their order separately at every measured state.
                with ExitStack() as stack:
                    batches = {}
                    sample_records = {}
                    ordered = (
                        arguments.modes
                        if sample % 2 == 0
                        else tuple(reversed(arguments.modes))
                    )
                    for order, mode in enumerate(ordered):
                        record = {
                            "case": case_name,
                            "tolerance": tolerance,
                            "mode": mode,
                            "sample": sample,
                            "batch_size": batch_size,
                            "prepare_order": order,
                            "properties": ["energy", "forces"],
                        }
                        records.append(record)
                        sample_records[mode] = record
                        try:
                            calculator = _calculator(
                                case,
                                arguments,
                                precision=mode,
                                energy_tolerance=tolerance,
                            )
                            _synchronize(backend)
                            started = time.perf_counter()
                            batches[mode] = stack.enter_context(
                                calculator.prepare_batch(
                                    systems,
                                    charges=[case.charge] * batch_size,
                                    multiplicities=[case.multiplicity] * batch_size,
                                    warm_start=True,
                                )
                            )
                            record["prepare_wall_seconds"] = _synchronized_seconds(
                                started, backend
                            )
                        except Exception as error:  # noqa: BLE001 - retain failures
                            record["failure"] = f"{type(error).__name__}: {error}"
                    for step, (state, coordinates, expected_labels) in enumerate(
                        (
                            ("cold", None, labels),
                            ("warm", None, labels),
                            ("changed", changed_coordinates, changed_labels),
                        )
                    ):
                        ordered = (
                            arguments.modes
                            if (sample + step) % 2 == 0
                            else tuple(reversed(arguments.modes))
                        )
                        expected = [references[label] for label in expected_labels]
                        for order, mode in enumerate(ordered):
                            record = sample_records[mode]
                            if "failure" in record:
                                continue
                            record[f"{state}_order"] = order
                            try:
                                _synchronize(backend)
                                started = time.perf_counter()
                                result = batches[mode].execute(
                                    coordinates, strict=False
                                )
                                record[f"{state}_execute_wall_seconds"] = (
                                    _synchronized_seconds(started, backend)
                                )
                                # Save each result before the next state runs;
                                # a later failure must not erase earlier samples.
                                record[f"{state}_items"] = _batch_items(
                                    result, expected, target
                                )
                            except Exception as error:  # noqa: BLE001 - retain failures
                                record["failure"] = (
                                    f"{state}: {type(error).__name__}: {error}"
                                )
    return records


def _synchronized_seconds(started: float, backend: typing.Any) -> float:
    """Finish all outstanding device work before reporting a wall time."""

    _synchronize(backend)
    return time.perf_counter() - started


def _batch_items(
    result: typing.Any, expected: typing.Any, target: typing.Any
) -> list[dict[str, Any]]:
    """Describe each input-ordered item, keeping failures in their slot."""

    items = []
    for item in result.items:
        record: dict[str, Any] = {
            "index": item.index,
            "status": item.status_message,
            "converged": item.converged,
            "iterations": item.iterations,
            "density_rms": item.density_rms,
            "warm_start_used": item.warm_start_used,
            "warm_start_fallback": item.warm_start_fallback,
            "restart_origin": item.restart_origin,
            "fock_builds": item.fock_builds,
            "precision": item.precision,
            "energy": item.energy,
            "energy_absolute_error": None,
            "force_max_abs": (
                None if item.forces is None else float(abs(item.forces).max())
            ),
            "force_max_abs_error": None,
        }
        if 0 <= item.index < len(expected):
            reference = expected[item.index]
            reference_result = reference["result"]
            if (
                item.succeeded
                and reference_result is not None
                and reference_result.converged
            ):
                record.update(_error_columns(item, reference))
                values = {"energy": item.energy}
                reference_values = {"energy": reference_result.energy}
                if item.forces is not None:
                    values["forces"] = item.forces
                    reference_values["forces"] = reference_result.forces
                record["accuracy"] = _evidence(
                    reference["model"],
                    values,
                    reference_values,
                    target,
                    item.converged,
                )
        items.append(record)
    return items


def _summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate the per-configuration cost, error and policy activation."""

    groups: dict[tuple, list[dict[str, Any]]] = {}
    for record in records:
        key = (
            record.get("case"),
            record.get("tolerance"),
            record.get("mode"),
            record.get("kind"),
            record.get("geometry"),
        )
        groups.setdefault(key, []).append(record)
    summary = []
    for (case, tolerance, mode, kind, geometry), items in sorted(
        groups.items(), key=lambda entry: str(entry[0])
    ):
        walls = [item["wall_seconds"] for item in items if "wall_seconds" in item]
        energy_errors = [
            item["energy_absolute_error"]
            for item in items
            if item.get("energy_absolute_error") is not None
        ]
        force_errors = [
            item["force_max_abs_error"]
            for item in items
            if item.get("force_max_abs_error") is not None
        ]
        precision = [item.get("precision") or {} for item in items]
        summary.append(
            {
                "case": case,
                "tolerance": tolerance,
                "mode": mode,
                "kind": kind,
                "geometry": geometry,
                "samples": len(items),
                "failures": sum(1 for item in items if "failure" in item),
                "unconverged": sum(
                    1 for item in items if item.get("converged") is False
                ),
                "median_wall_seconds": median(walls) if walls else None,
                "maximum_energy_absolute_error": (
                    max(energy_errors) if energy_errors else None
                ),
                "maximum_force_max_abs_error": (
                    max(force_errors) if force_errors else None
                ),
                "mixed_activated_samples": sum(
                    1 for item in precision if item.get("effective_bits") == 32
                ),
                "fp64_samples": sum(
                    1 for item in precision if item.get("effective_bits") == 64
                ),
                "strict_refinement_samples": sum(
                    1 for item in precision if item.get("strict_refinement_applied")
                ),
                "maximum_refinement_iterations": max(
                    (item.get("refinement_iterations") or 0 for item in precision),
                    default=0,
                ),
            }
        )
    return summary


def _summarize_batch(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate batch cost, per-item error and per-item policy activation."""

    summary = []
    for record in records:
        if "failure" in record:
            summary.append(
                {
                    "case": record["case"],
                    "tolerance": record["tolerance"],
                    "mode": record["mode"],
                    "batch_size": record["batch_size"],
                    "failure": record["failure"],
                }
            )
            continue
        items = [
            item
            for state in ("cold", "warm", "changed")
            for item in record[f"{state}_items"]
        ]
        energy_errors = [
            item["energy_absolute_error"]
            for item in items
            if item["energy_absolute_error"] is not None
        ]
        summary.append(
            {
                "case": record["case"],
                "tolerance": record["tolerance"],
                "mode": record["mode"],
                "batch_size": record["batch_size"],
                "sample": record["sample"],
                "prepare_wall_seconds": record["prepare_wall_seconds"],
                "cold_execute_wall_seconds": record["cold_execute_wall_seconds"],
                "warm_execute_wall_seconds": record["warm_execute_wall_seconds"],
                "changed_execute_wall_seconds": record["changed_execute_wall_seconds"],
                "mixed_items": sum(
                    1
                    for item in items
                    if (item["precision"] or {}).get("effective_bits") == 32
                ),
                "item_count": len(items),
                "unconverged_items": sum(1 for item in items if not item["converged"]),
                "maximum_energy_absolute_error": (
                    max(energy_errors) if energy_errors else None
                ),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES))
    parser.add_argument("--include-large", action="store_true")
    parser.add_argument("--include-bounded-streaming", action="store_true")
    parser.add_argument(
        "--tolerances",
        default=",".join(f"{value:.0e}" for value in DEFAULT_TOLERANCES),
    )
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--reference-max-iterations", type=int, default=200)
    parser.add_argument("--density-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--screening-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--reference-energy-tolerance", type=float, default=1.0e-13)
    parser.add_argument("--reference-density-tolerance", type=float, default=1.0e-11)
    parser.add_argument("--reference-screening-tolerance", type=float, default=1.0e-14)
    parser.add_argument("--force-target", type=float, default=1.0e-6)
    parser.add_argument("--displacement-bohr", type=float, default=0.05)
    parser.add_argument("--properties", default="energy,forces")
    parser.add_argument("--skip-batch", action="store_true")
    parser.add_argument("--batch-sizes", default="1,4")
    parser.add_argument("--output", type=raw_output_path)
    arguments = parser.parse_args()
    if (
        arguments.repeats < 1
        or arguments.max_iterations < 1
        or arguments.reference_max_iterations < 1
    ):
        parser.error("repeat and iteration counts must be positive")
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("this real-GPU benchmark must run inside a Slurm allocation")

    requested = [name.strip() for name in arguments.cases.split(",") if name.strip()]
    if arguments.include_large:
        requested.extend(LARGE_CASES)
    if arguments.include_bounded_streaming:
        requested.extend(BOUNDED_STREAMING_CASES)
    try:
        arguments.tolerances = [
            float(value) for value in arguments.tolerances.split(",")
        ]
        arguments.batch_sizes = tuple(
            int(value) for value in arguments.batch_sizes.split(",")
        )
    except ValueError:
        parser.error("tolerances must be numbers and batch sizes must be integers")
    if any(size < 1 for size in arguments.batch_sizes):
        parser.error("batch sizes must be positive")
    controls = [
        *arguments.tolerances,
        arguments.density_tolerance,
        arguments.screening_tolerance,
        arguments.reference_energy_tolerance,
        arguments.reference_density_tolerance,
        arguments.reference_screening_tolerance,
        arguments.force_target,
    ]
    if any(not math.isfinite(value) or value <= 0 for value in controls):
        parser.error("tolerances and force targets must be finite and positive")
    if (
        arguments.reference_density_tolerance >= arguments.density_tolerance
        or arguments.reference_screening_tolerance >= arguments.screening_tolerance
    ):
        parser.error("the reference density and screening controls must be stricter")
    if (
        not math.isfinite(arguments.displacement_bohr)
        or arguments.displacement_bohr == 0
    ):
        parser.error("the changed-geometry displacement must be finite and nonzero")
    arguments.modes = tuple(
        mode.strip() for mode in arguments.modes.split(",") if mode.strip()
    )
    unsupported = sorted(set(arguments.modes) - set(MODES))
    if not arguments.modes or not requested:
        parser.error("at least one case and precision mode are required")
    if unsupported:
        parser.error(f"unsupported precision modes: {unsupported}")
    arguments.properties = tuple(
        name.strip() for name in arguments.properties.split(",") if name.strip()
    )
    if not arguments.skip_batch and set(arguments.properties) != {"energy", "forces"}:
        parser.error(
            "prepared batches evaluate energy and forces; use --skip-batch for energy-only"
        )
    if "energy" not in arguments.properties:
        parser.error("--properties must include energy")
    if arguments.reference_energy_tolerance >= min(arguments.tolerances):
        parser.error(
            "the FP64 reference must be tighter than every swept energy tolerance"
        )

    # Device handles are opened only after the Slurm guard above, so a login-node
    # invocation never initializes the driver outside the scheduler.
    backend, backend_name = _device_backend()

    from benchmarks._cases import benchmark_cases
    from benchmarks._support import (
        cuda_accelerator_metadata,
        environment_metadata,
        write_result,
    )

    available = benchmark_cases()
    unknown = [name for name in requested if name not in available]
    if unknown:
        parser.error(f"unknown benchmark cases: {unknown}")

    matrix: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    references: dict[str, Any] = {}
    for name in requested:
        case = available[name]
        atoms_base = case.atoms
        atoms_moved = _displaced_atoms(atoms_base, arguments.displacement_bohr)
        records, case_references = _tolerance_matrix(
            name, case, atoms_base, atoms_moved, arguments, backend
        )
        matrix.extend(records)
        references[name] = {
            label: {
                "model": payload["model"].to_dict(),
                "record": payload["record"],
            }
            for label, payload in case_references.items()
        }
        if not arguments.skip_batch:
            batches.extend(
                _batch_matrix(
                    name,
                    case,
                    atoms_base,
                    atoms_moved,
                    case_references,
                    arguments,
                    backend,
                )
            )

    payload = {
        "schema_version": 2,
        "benchmark": "issue174_slice_e_matrix",
        "controls": {
            "cases": requested,
            "tolerances": arguments.tolerances,
            "modes": list(arguments.modes),
            "repeats": arguments.repeats,
            "batch_sizes": list(arguments.batch_sizes),
            "max_iterations": arguments.max_iterations,
            "density_tolerance": arguments.density_tolerance,
            "screening_tolerance": arguments.screening_tolerance,
            "force_target": arguments.force_target,
            "properties": list(arguments.properties),
            "displacement_bohr": arguments.displacement_bohr,
        },
        "targets": {
            f"{tolerance:.0e}": _accuracy_target(arguments, tolerance).to_dict()
            for tolerance in arguments.tolerances
        },
        "references": references,
        "matrix": matrix,
        "summary": _summarize(matrix),
        "batch": {"records": batches, "summary": _summarize_batch(batches)},
        "environment": {
            "device_api": backend_name,
            **environment_metadata(
                distributions={"numpy": ("numpy",), "cupy": ("cupy-cuda12x", "cupy")},
                accelerator=cuda_accelerator_metadata(backend),
            ),
        },
        "limitations": [
            (
                "The matrix only measures the existing policy: a cold item and "
                "any item without a validated warm state stays FP64, so a zero "
                "mixed activation count is a policy result, not a kernel result."
            ),
            (
                "Bounded-streaming topologies have no exact device tile census "
                "and are refused by the budget-aware policy; the 768-AO case is "
                "included only to expose that boundary."
            ),
            (
                "density_rms and energy_change are the recorded convergence "
                "residuals. The independent physical-residual and fixed-density "
                "operator audits live in tools.vibeqc_numerics.audit and are not "
                "re-run here."
            ),
            (
                "Every singlepoint call is a cold native solve. Prepared batches "
                "own warm states and carry warm_start_used/warm_start_fallback "
                "per item; changed rows include the geometry update in timing."
            ),
            (
                "Reported wall time is synchronized host time around the public "
                "call, including preparation, finalization and any strict "
                "refinement."
            ),
        ],
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(encoded, end="")
    else:
        destination = write_result(arguments.output, payload)
        print(f"JSON result: {destination}")


if __name__ == "__main__":
    main()
