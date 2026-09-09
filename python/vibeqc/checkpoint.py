"""Versioned HF scientific checkpoints, independent of device/runtime objects.

The file is a fixed header, checksummed UTF-8 JSON manifest, and contiguous
uncompressed little-endian IEEE-754 binary64 arrays in C order. No executable
objects, external filenames, native structs or solver histories are accepted.
A checkpoint only proposes a seed: compatibility and physical validation are
repeated against a separately requested, freshly prepared calculation.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import struct
import tempfile
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import _native
from .accuracy import ResolvedModel, TargetAccuracy
from .calculator import Atom
from .profiles import canonical_hash
from .resources_hf import _CUDA_SCHEDULE_VARIABLES

_MAGIC = b"VQHFCP01"
_HEADER = struct.Struct("<8sQ32s")
_FORMAT = "ieee754-binary64-little-endian-c-order"
_MAX_MANIFEST = 4 << 20
_MAX_ITEMS = 10000


class CheckpointError(ValueError):
    """A checkpoint is malformed or incompatible; no seeds have been applied."""


@dataclass(frozen=True)
class CheckpointManifest:
    """Versioned metadata; ``to_dict`` returns detached portable records.

    ``items`` preserve input order, source provenance and failed/no-state slots.
    Only density and its source coordinates are reusable fields in schema 1.
    CC amplitudes, DIIS and response vectors require future explicit schemas.
    """

    items: tuple[dict, ...]
    blobs: tuple[dict, ...]
    schema_version: int = 1

    def to_dict(self):
        return deepcopy(
            {
                "schema": "vibeqc.hf_checkpoint",
                "schema_version": self.schema_version,
                "array_format": _FORMAT,
                "items": list(self.items),
                "blobs": list(self.blobs),
            }
        )


def _json(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _integer(value, name, maximum, minimum=0):
    if type(value) is not int or not minimum <= value <= maximum:
        raise CheckpointError(f"invalid {name}")
    return value


def _keys(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys.split()):
        raise CheckpointError(f"unknown or missing required {name} fields")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointError(f"duplicate manifest key: {key}")
        result[key] = value
    return result


def _hash(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _controls(calc):
    return {
        "max_iterations": calc._max_iterations,
        "energy_tolerance": calc._energy_tolerance,
        "density_tolerance": calc._density_tolerance,
        "diis_history": calc._diis_history,
        "screening_tolerance": calc._screening_tolerance,
        "target_accuracy": calc._target_accuracy.to_dict()
        if calc._target_accuracy
        else None,
        "precision_policy": os.environ.get("VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD"),
        # Share the resource planner's arithmetic/schedule inventory rather
        # than overlooking a force-screening or final-Fock override. These
        # remain source controls, separate from backend-independent physics.
        "runtime_policy": {
            name: os.environ.get(name) for name in _CUDA_SCHEDULE_VARIABLES
        },
    }


def _provider(model):
    return (
        "vibeqc.hf.ri-pseudoinverse/v1"
        if model.approximation == "density_fitting"
        else "vibeqc.hf.direct/v1"
    )


def _validate_controls(controls):
    """Keep source provenance strict without changing target solver controls."""
    _keys(
        controls,
        "max_iterations energy_tolerance density_tolerance diis_history "
        "screening_tolerance target_accuracy precision_policy runtime_policy",
        "solver controls",
    )
    for name in ("max_iterations", "diis_history"):
        _integer(controls[name], name, 2**31 - 1)
    for name in ("energy_tolerance", "density_tolerance", "screening_tolerance"):
        value = controls[name]
        if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
            raise CheckpointError(f"invalid source {name}")
    if controls["target_accuracy"] is not None:
        TargetAccuracy.from_dict(controls["target_accuracy"])
    if controls["precision_policy"] is not None and not isinstance(
        controls["precision_policy"], str
    ):
        raise CheckpointError("invalid source precision policy")
    policy = controls["runtime_policy"]
    _keys(policy, " ".join(_CUDA_SCHEDULE_VARIABLES), "runtime policy")
    if any(
        value is not None and not isinstance(value, str) for value in policy.values()
    ):
        raise CheckpointError("invalid source runtime policy")
    if policy["VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD"] != controls["precision_policy"]:
        raise CheckpointError("inconsistent source precision policy")


def _check_batch(batch):
    batch._ensure_open()
    if batch._calculator._model_signature() != batch._model_signature:
        raise CheckpointError("prepared model identity changed; prepare a new batch")


def _descriptor():
    return _native.HfWarmState(
        struct_size=ctypes.sizeof(_native.HfWarmState), abi_version=_native.ABI_VERSION
    )


def _pointer(array):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _resource_check(batch, numeric_bytes, largest_density):
    """Charge staging and source-metric validation above the current HF plan.

    The conservative host bound includes owned arrays, native candidate copies,
    endian conversion, serialization and the shared guard's dense eigensolver
    scratch. No saved ResourcePlan can override current-run limits/schedules.
    """
    extra = 4 * numeric_bytes + 24 * largest_density + _MAX_MANIFEST
    plan = batch.resource_plan
    if plan is not None:
        limit = plan.budget.limits().get("host")
        if limit is not None and plan.peak_bytes.get("host", 0) + extra > limit:
            raise MemoryError(
                "checkpoint staging/validation exceeds current ResourcePlan host headroom"
            )
    return extra


def _parse_manifest(raw, payload_bytes, max_bytes):
    try:
        doc = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=lambda v: (_ for _ in ()).throw(
                CheckpointError(f"nonfinite JSON: {v}")
            ),
        )
        _keys(doc, "schema schema_version array_format items blobs", "manifest")
        if (
            doc["schema"] != "vibeqc.hf_checkpoint"
            or type(doc["schema_version"]) is not int
            or doc["schema_version"] != 1
        ):
            raise CheckpointError("unsupported required checkpoint schema")
        if doc["array_format"] != _FORMAT:
            raise CheckpointError("unsupported dtype/endian/array format")
        if (
            not isinstance(doc["items"], list)
            or not 1 <= len(doc["items"]) <= _MAX_ITEMS
        ):
            raise CheckpointError("invalid checkpoint item count")
        if not isinstance(doc["blobs"], list) or len(doc["blobs"]) > 2 * len(
            doc["items"]
        ):
            raise CheckpointError("invalid checkpoint blob count")
        expected = {}
        for index, item in enumerate(doc["items"]):
            _keys(
                item,
                "index atomic_numbers request_identity basis_metadata model provider controls backend status seed",
                "item",
            )
            if type(item["index"]) is not int or item["index"] != index:
                raise CheckpointError("checkpoint item ordering mismatch")
            numbers = item["atomic_numbers"]
            if (
                not isinstance(numbers, list)
                or not numbers
                or len(numbers) > max_bytes // 24
            ):
                raise CheckpointError("invalid ordered nuclei")
            for number in numbers:
                _integer(number, "atomic number", 118, 1)
            if not _hash(item["request_identity"]) or not isinstance(
                item["basis_metadata"], dict
            ):
                raise CheckpointError("missing prepared identity/provenance")
            _integer(item["status"], "item status", 2**31 - 1, -1)
            if not isinstance(item["controls"], dict) or item["backend"] not in (
                "cpu",
                "cuda",
            ):
                raise CheckpointError("invalid solver provenance")
            _validate_controls(item["controls"])
            if item["basis_metadata"].get("model_identity") != item["request_identity"]:
                raise CheckpointError("prepared identity/provenance mismatch")
            seed = item["seed"]
            if seed is None:
                if item["model"] is not None or item["provider"] is not None:
                    raise CheckpointError("no-state item carries reusable identity")
                continue
            model = ResolvedModel.from_dict(item["model"])
            metadata = item["basis_metadata"]
            if metadata["orbital"]["mathematical_identity"] != model.basis_hash:
                raise CheckpointError("orbital basis provenance/model mismatch")
            if (
                model.approximation == "density_fitting"
                and metadata.get("auxiliary", metadata["orbital"])[
                    "mathematical_identity"
                ]
                != model.auxiliary_basis_hash
            ):
                raise CheckpointError("auxiliary basis provenance/model mismatch")
            if sum(numbers) - model.charge != model.electron_count:
                raise CheckpointError("impossible electron count for ordered nuclei")
            if item["provider"] != _provider(model):
                raise CheckpointError("unsupported scientific provider identity")
            _keys(
                seed,
                "density coordinates nbf energy energy_change density_rms iterations converged",
                "seed",
            )
            n = _integer(seed["nbf"], "AO dimension", math.isqrt(max_bytes // 8), 1)
            spins = 2 if model.method == "uhf" else 1
            if (model.electron_count + model.multiplicity - 1) // 2 > n:
                raise CheckpointError("spin population exceeds AO dimension")
            _integer(seed["iterations"], "source iterations", 2**31 - 1)
            if seed["converged"] is not True:
                raise CheckpointError("schema 1 requires a converged source seed")
            for name in ("energy", "energy_change", "density_rms"):
                value = seed[name]
                if type(value) not in (float, int) or not math.isfinite(value):
                    raise CheckpointError("nonfinite source diagnostics")
            if seed["density_rms"] < 0:
                raise CheckpointError("negative source residual")
            for name, shape in (
                ("density", [spins, n, n]),
                ("coordinates", [len(numbers), 3]),
            ):
                if seed[name] != f"{name}_{index}":
                    raise CheckpointError("unexpected blob name")
                expected[seed[name]] = shape
        offset = 0
        seen = set()
        for blob in doc["blobs"]:
            _keys(blob, "name shape offset bytes sha256", "blob")
            name = blob["name"]
            if not isinstance(blob["shape"], list) or any(
                type(dimension) is not int for dimension in blob["shape"]
            ):
                raise CheckpointError("noninteger blob dimensions")
            if (
                not isinstance(name, str)
                or name in seen
                or name not in expected
                or blob["shape"] != expected[name]
            ):
                raise CheckpointError("inconsistent or duplicate blob dimensions")
            size = math.prod(expected[name]) * 8
            if (
                size > max_bytes
                or type(blob["offset"]) is not int
                or blob["offset"] != offset
                or type(blob["bytes"]) is not int
                or blob["bytes"] != size
            ):
                raise CheckpointError("inconsistent blob byte count/offset")
            if not _hash(blob["sha256"]):
                raise CheckpointError("invalid blob checksum")
            seen.add(name)
            offset += size
        if seen != set(expected) or offset != payload_bytes:
            raise CheckpointError("truncated, trailing or missing checkpoint blobs")
        return CheckpointManifest(tuple(doc["items"]), tuple(doc["blobs"]))
    except (
        KeyError,
        TypeError,
        OverflowError,
        UnicodeError,
        RecursionError,
        ValueError,
    ) as error:
        if isinstance(error, CheckpointError):
            raise
        raise CheckpointError(f"invalid checkpoint manifest: {error}") from error


def _read(path, max_bytes, *, retain=True, batch=None):
    _integer(max_bytes, "checkpoint byte limit", 2**63 - 1, _HEADER.size)
    arrays = {}
    with open(path, "rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        if size > max_bytes:
            raise MemoryError("checkpoint file exceeds max_bytes")
        header = stream.read(_HEADER.size)
        if len(header) != _HEADER.size:
            raise CheckpointError("truncated checkpoint header")
        magic, length, checksum = _HEADER.unpack(header)
        if magic != _MAGIC or length > min(_MAX_MANIFEST, size - _HEADER.size):
            raise CheckpointError("invalid checkpoint header/schema or manifest length")
        raw = stream.read(length)
        if hashlib.sha256(raw).digest() != checksum:
            raise CheckpointError("manifest checksum mismatch")
        manifest = _parse_manifest(raw, size - _HEADER.size - length, max_bytes)
        total = sum(b["bytes"] for b in manifest.blobs)
        largest = max(
            (b["bytes"] for b in manifest.blobs if b["name"].startswith("density_")),
            default=0,
        )
        if batch is not None:
            _resource_check(batch, total, largest)
        for blob in manifest.blobs:
            digest = hashlib.sha256()
            if retain:
                data = bytearray(blob["bytes"])
                view = memoryview(data)
            remaining = blob["bytes"]
            cursor = 0
            while remaining:
                chunk = stream.read(min(remaining, 1 << 20))
                if not chunk:
                    raise CheckpointError("truncated checkpoint blob")
                digest.update(chunk)
                if retain:
                    view[cursor : cursor + len(chunk)] = chunk
                cursor += len(chunk)
                remaining -= len(chunk)
            if digest.hexdigest() != blob["sha256"]:
                raise CheckpointError(f"blob checksum mismatch: {blob['name']}")
            if retain:
                array = np.frombuffer(data, dtype="<f8").reshape(blob["shape"])
                if not np.isfinite(array).all():
                    raise CheckpointError("nonfinite checkpoint array")
                arrays[blob["name"]] = array
        if stream.read(1):
            raise CheckpointError("checkpoint changed while reading")
    return manifest, arrays, size


def inspect_checkpoint(path, *, max_bytes=256 << 20):
    """Verify schema, lengths and checksums without initializing any runtime.

    Physical/target validation happens only on load into a prepared object.
    Numeric blobs are streamed without retaining their complete contents.
    """
    return _read(path, max_bytes, retain=False)[0]


def save_checkpoint(batch, path, *, max_bytes=256 << 20):
    """Write, fsync, verify and atomically replace a single portable HF file."""
    start = time.perf_counter()
    _check_batch(batch)
    _integer(max_bytes, "checkpoint byte limit", 2**63 - 1, _HEADER.size)
    if batch.system_count > _MAX_ITEMS:
        raise CheckpointError("checkpoint item count exceeds schema limit")
    descriptors = []
    total = largest = 0
    for index in range(batch.system_count):
        state = _descriptor()
        _native.check(
            batch._library,
            batch._library.vibeqc_batch_get_hf_warm_state(
                batch._batch, index, ctypes.byref(state)
            ),
        )
        descriptors.append(state)
        total += 8 * (state.density_count + state.coordinate_count)
        largest = max(largest, 8 * state.density_count)
    if total > max_bytes:
        raise MemoryError("checkpoint arrays exceed max_bytes")
    extra = _resource_check(batch, total, largest)
    calc = batch._calculator
    items, arrays, blobs = [], [], []
    offset = 0
    for index, state in enumerate(descriptors):
        item = {
            "index": index,
            "atomic_numbers": list(batch.atomic_numbers[index]),
            "request_identity": batch._basis_metadata[index]["model_identity"],
            "basis_metadata": deepcopy(batch._basis_metadata[index]),
            "model": None,
            "provider": None,
            "controls": _controls(calc),
            "backend": calc._device_name,
            "status": batch._last_statuses[index] if batch._last_statuses else -1,
            "seed": None,
        }
        if state.present:
            provenance = batch._warm_metadata[index]
            if provenance is None:
                raise CheckpointError("retained seed is missing its source provenance")
            item.update(deepcopy(provenance))
            density = np.empty(state.density_count, dtype=np.float64)
            coordinates = np.empty(state.coordinate_count, dtype=np.float64)
            state.density, state.coordinates = _pointer(density), _pointer(coordinates)
            _native.check(
                batch._library,
                batch._library.vibeqc_batch_get_hf_warm_state(
                    batch._batch, index, ctypes.byref(state)
                ),
            )
            coordinates = coordinates.reshape(-1, 3)
            atoms = tuple(
                Atom(z, tuple(x))
                for z, x in zip(item["atomic_numbers"], coordinates, strict=True)
            )
            model = calc.resolved_model(
                atoms,
                charge=batch.charges[index],
                multiplicity=batch.multiplicities[index],
            )
            spins = 2 if model.method == "uhf" else 1
            n = math.isqrt(density.size // spins)
            density = density.reshape(spins, n, n)
            item.update(model=model.to_dict(), provider=_provider(model))
            item["seed"] = {
                "density": f"density_{index}",
                "coordinates": f"coordinates_{index}",
                "nbf": n,
                "energy": state.energy,
                "energy_change": state.energy_change,
                "density_rms": state.density_rms,
                "iterations": state.iterations,
                "converged": True,
            }
            for name, array in (("density", density), ("coordinates", coordinates)):
                array = np.ascontiguousarray(array, dtype="<f8")
                if not np.isfinite(array).all():
                    raise CheckpointError("nonfinite retained state")
                data = memoryview(array).cast("B")
                blobs.append(
                    {
                        "name": f"{name}_{index}",
                        "shape": list(array.shape),
                        "offset": offset,
                        "bytes": data.nbytes,
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                )
                arrays.append(array)
                offset += data.nbytes
        items.append(item)
    manifest = CheckpointManifest(tuple(items), tuple(blobs))
    raw = _json(manifest.to_dict())
    size = _HEADER.size + len(raw) + offset
    if len(raw) > _MAX_MANIFEST or size > max_bytes:
        raise MemoryError("checkpoint manifest/file exceeds byte limit")
    _parse_manifest(raw, offset, max_bytes)
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(_HEADER.pack(_MAGIC, len(raw), hashlib.sha256(raw).digest()))
            stream.write(raw)
            for array in arrays:
                stream.write(memoryview(array).cast("B"))
            stream.flush()
            os.fsync(stream.fileno())
        inspect_checkpoint(temporary, max_bytes=max_bytes)
        os.replace(temporary, path)
        temporary = None
        if hasattr(os, "O_DIRECTORY"):
            descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    report = {
        "schema_version": 1,
        "bytes": size,
        "write_seconds": time.perf_counter() - start,
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "staging_host_bound_bytes": extra,
        "saved_indices": [i["index"] for i in items if i["seed"] is not None],
    }
    batch.checkpoint_diagnostics = report
    return report


def load_checkpoint(batch, path, *, allow_warm=False, strict=True, max_bytes=256 << 20):
    """Validate a complete file, classify each slot, then atomically import seeds.

    Cross-basis/orbital transport is deliberately rejected. Changed-geometry
    seeds use the fleet's existing symmetry/metric-trace normalization and
    cold retry on numerical failure. The actual target Fock/residual is always
    recomputed during ``execute``; source convergence never bypasses SCF.
    """
    start = time.perf_counter()
    _check_batch(batch)
    manifest, arrays, size = _read(path, max_bytes, batch=batch)
    if len(manifest.items) != batch.system_count:
        raise CheckpointError("checkpoint item count/order differs from target fleet")
    calc = batch._calculator
    candidates = (_native.HfWarmState * batch.system_count)()
    reports, storage = [], []
    for index, item in enumerate(manifest.items):
        candidates[index] = _descriptor()
        level, reason = "incompatible", "missing reusable state"
        seed = item["seed"]
        if seed is not None:
            source = ResolvedModel.from_dict(item["model"])
            coords = arrays[seed["coordinates"]]
            source_atoms = tuple(
                Atom(z, tuple(x))
                for z, x in zip(item["atomic_numbers"], coords, strict=True)
            )
            geometry = canonical_hash(
                [
                    (
                        a.atomic_number,
                        tuple(float(x).hex() if x else "0x0.0p+0" for x in a.position),
                    )
                    for a in source_atoms
                ]
            )
            if geometry != source.geometry_hash:
                raise CheckpointError("source coordinate/model identity mismatch")
            try:
                target = calc.resolved_model(
                    batch._systems[index],
                    charge=batch.charges[index],
                    multiplicity=batch.multiplicities[index],
                )
                source_id, target_id = source.to_dict(), target.to_dict()
                for value in (source_id, target_id):
                    value.pop("identity")
                    value.pop("geometry_hash")
                if list(batch.atomic_numbers[index]) != item["atomic_numbers"]:
                    reason = "ordered nuclei differ"
                elif source_id != target_id:
                    basis_differs = (
                        source.basis_hash != target.basis_hash
                        or source.representation != target.representation
                    )
                    level = "transport_required" if basis_differs else "incompatible"
                    reason = (
                        "basis/AO transport required"
                        if basis_differs
                        else "method/core/spin/provider identity differs"
                    )
                elif item["provider"] != _provider(target):
                    reason = "scientific provider identity differs"
                elif source.geometry_hash != target.geometry_hash or item[
                    "controls"
                ] != _controls(calc):
                    level, reason = (
                        "warm_start_compatible",
                        "geometry or numerical controls differ",
                    )
                else:
                    level, reason = (
                        "exact_restart",
                        "same scientific identity, geometry and controls",
                    )
            except ValueError as error:
                reason = f"invalid target: {error}"
        accepted = level == "exact_restart" or (
            allow_warm and level == "warm_start_compatible"
        )
        reports.append(
            {
                "index": index,
                "compatibility": level,
                "reason": reason,
                "restored_fields": ["density"] if accepted else [],
                "rejected_fields": [] if accepted or seed is None else ["density"],
                "source_status": item["status"],
                "source_model_identity": item["model"]["identity"] if seed else None,
            }
        )
        if accepted:
            density = np.ascontiguousarray(arrays[seed["density"]], dtype=np.float64)
            coordinates = np.ascontiguousarray(
                arrays[seed["coordinates"]], dtype=np.float64
            )
            storage.extend((density, coordinates))
            state = candidates[index]
            state.density, state.coordinates = _pointer(density), _pointer(coordinates)
            state.density_count, state.coordinate_count = density.size, coordinates.size
            for name in ("energy", "energy_change", "density_rms", "iterations"):
                setattr(state, name, seed[name])
            state.present = 1
    rejected = [r for r in reports if not r["restored_fields"]]
    if strict and rejected:
        raise CheckpointError(
            "checkpoint restore rejected: "
            + "; ".join(f"{r['index']}: {r['reason']}" for r in rejected)
        )
    accepted_indices = {r["index"] for r in reports if r["restored_fields"]}
    # Allocate Python ownership/provenance before the native no-throw commit,
    # so a failed staging allocation cannot strand an imported seed without
    # its source controls or restart origin.
    origins = batch._restart_indices | accepted_indices
    metadata = list(batch._warm_metadata)
    for index in accepted_indices:
        source = manifest.items[index]
        metadata[index] = {
            "controls": deepcopy(source["controls"]),
            "backend": source["backend"],
        }
    report = {
        "schema_version": 1,
        "bytes": size,
        "manifest_identity": canonical_hash(manifest.to_dict()),
        "read_seconds": 0.0,
        "items": reports,
        "target_verification": "pending_execute",
        "validation_backend": "cpu_source_overlap",
    }
    if accepted_indices:
        status = batch._library.vibeqc_batch_restore_hf_warm_states(
            batch._batch, candidates, len(candidates)
        )
        if status != _native.STATUS_SUCCESS:
            detail = batch._library.vibeqc_context_get_last_detail(
                batch._context
            ).decode("utf-8")
            raise CheckpointError(
                f"native checkpoint validation failed: {detail or status}"
            )
    batch._restart_indices = origins
    batch._warm_metadata = metadata
    report["read_seconds"] = time.perf_counter() - start
    batch.checkpoint_diagnostics = report
    return report
