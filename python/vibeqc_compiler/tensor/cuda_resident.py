"""Resident execution over the #146 owner with pinned planned spans.

This module owns the host-side contract of the resident ABI emitted by
:mod:`vibeqc_compiler.tensor.cuda_resident_emit`. Storage, compilation and
device lifetime remain #146's; this layer only pins named spans, uploads
each input once, evaluates the *same* generated program repeatedly without
per-call staging, and downloads a *named* output when the caller asks for it.
No numerical CPU fallback or per-contraction Python execution is provided.

The old host-input/host-output ABI stays compiled and unchanged: a resident
artifact is still a valid ordinary artifact, so the host-staged path and the
resident path can be compared directly on the same binary.
"""

from __future__ import annotations

import ctypes
import json
import os
import tempfile
import time
import typing
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vibeqc_compiler.common.cuda_runtime import _Metrics
from vibeqc_compiler.common.paths import asset_path
from vibeqc_compiler.common.provenance import (
    atomic_json,
    canonical_hash,
    file_hash,
)
from vibeqc_compiler.tensor.cuda_dtype import compile_options
from vibeqc_compiler.tensor.cuda_execute import CudaArtifact, PreparedCuda, compile_cuda
from vibeqc_compiler.tensor.cuda_resident_emit import resident_source

RESIDENT_SCHEMA = "vibeqc.tensor.resident/1"


def compile_resident(
    plan: typing.Any,
    compiler: typing.Any,
    cache: typing.Any,
    *,
    extension: typing.Any = "",
    dependencies: typing.Any = (),
) -> typing.Any:
    """Compile/cache the verified ordinary TU plus the resident ABI.

    The wrapped generated source is re-hashed from the ordinary cache entry,
    so a resident artifact can never be built from an edited program. Pass
    ``extension`` (a C++ snippet) to add a plan-specific post-run action; its
    text is part of the artifact identity together with the extra headers in
    ``dependencies``.
    """
    base = compile_cuda(plan, compiler, cache)
    generated = base.library.parent / "program.cu"
    if (
        not generated.is_file()
        or canonical_hash(generated.read_text())
        != base.metadata["identity"]["generated"]
    ):
        raise ValueError("ordinary generated source identity mismatch")
    import vibeqc_compiler.tensor.cuda_resident_emit as emit_module

    source = resident_source(plan, extension=extension)
    # Resolve the resident header and generated-TU include directory through
    # the same asset helper the ordinary compile uses: a source checkout and
    # an installed wheel must both find them.
    header = asset_path("src/tensor/cuda_resident.cuh")
    paths = [*[Path(p) for p in dependencies], Path(emit_module.__file__), header]
    # Every dependency gets a stable logical name so the identity is the same
    # across different source-checkout prefixes, between source and installed-
    # wheel layouts, and across platforms.  We use the same pattern as
    # ``source_hashes(assets=...)``: files under ``PACKAGE`` are named
    # ``python/vibeqc_compiler/<rest>``; non-package assets use their
    # repository-relative name (``src/tensor/cuda_resident.cuh``).
    from vibeqc_compiler.common.paths import PACKAGE as _PKG

    def _logical_name(path: typing.Any) -> typing.Any:
        # Is it under the installed package tree?
        try:
            rel = Path(os.path.relpath(path, _PKG)).as_posix()
        except ValueError:
            rel = None
        if rel is not None and not rel.startswith(".."):
            # Package assets carry an ``assets/`` prefix we strip so the
            # installed key matches the checkout key for the same header.
            if rel.startswith("assets/"):
                rel = rel[len("assets/") :]
            else:
                rel = f"python/vibeqc_compiler/{rel}"
            return rel
        # Not under the package tree — must be under the checkout root.
        from vibeqc_compiler.common.paths import source_root

        root = source_root()
        rel = Path(os.path.relpath(path, root)).as_posix()
        if rel.startswith(".."):
            raise ValueError(
                f"resident dependency {path} is outside both the source"
                " checkout and the installed package"
            )
        return rel

    identity = {
        "schema": RESIDENT_SCHEMA,
        "base": base.metadata["key"],
        "base_binary": base.metadata["binary_sha256"],
        "generated": canonical_hash(source),
        "extension": canonical_hash(extension),
        "sources": {
            _logical_name(path): file_hash(path) for path in paths if path.is_file()
        },
    }
    key = canonical_hash(identity)
    cache = Path(cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / key
    if not destination.exists():
        with tempfile.TemporaryDirectory(
            prefix=".resident-build-", dir=cache
        ) as temporary:
            directory = Path(temporary)
            cu, library = directory / "resident.cu", directory / "program.so"
            cu.write_text(source)
            # The generated TU includes "cuda_runtime.cuh" and the resident
            # ABI includes "cuda_resident.cuh"; both live in the asset
            # directory so the include search path works source and wheel.
            # Extra ``dependencies`` headers must also be resolvable.
            includes = (base.library.parent, header.parent) + tuple(
                Path(d).parent.resolve() for d in dependencies
            )
            result = compiler.compile_shared(
                cu,
                library,
                includes=includes,
                libraries=("cublas",),
                options=compile_options(plan),
            )
            (directory / "compiler.log").write_text(result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError(
                    "resident NVCC compilation failed:\n"
                    + result.stdout
                    + result.stderr
                )
            atomic_json(
                directory / "artifact.json",
                {
                    "key": key,
                    "identity": identity,
                    "binary_sha256": file_hash(library),
                    "base_artifact": base.metadata,
                    "compile_seconds": result.duration_seconds,
                },
            )
            try:
                os.rename(directory, destination)
            except OSError:
                if not destination.is_dir():
                    raise
    metadata = json.loads((destination / "artifact.json").read_text())
    library = destination / "program.so"
    if (
        metadata.get("identity") != json.loads(json.dumps(identity))
        or metadata.get("key") != key
        or file_hash(library) != metadata.get("binary_sha256")
    ):
        raise ValueError("resident artifact identity or binary hash mismatch")
    return CudaArtifact(library, metadata)


@dataclass(frozen=True)
class DeviceTensor:
    """Output lease retaining its owner; invalidated on mutation/run/close.

    Only a named output of this exact plan can be leased: matching a shape
    alone never confers physical compatibility.
    """

    owner: object
    name: str
    generation: int

    def _step(self) -> typing.Any:
        owner = self.owner
        if (
            not owner._pointer
            or not owner._ready
            or self.generation != owner._generation
        ):
            raise RuntimeError("resident output is closed, invalidated or stale")
        try:
            return owner.plan.steps[dict(owner.plan.outputs)[self.name]]
        except KeyError as error:
            raise ValueError("unknown resident output") from error

    @property
    def shape(self) -> typing.Any:
        return self._step().node.spec.shape

    @property
    def dtype(self) -> typing.Any:
        return np.dtype(self._step().node.spec.dtype)

    def to_host(self) -> typing.Any:
        return self.owner.download(self)


def _check_lease(owner: typing.Any, value: typing.Any) -> None:
    """Validate a DeviceTensor lease: owner + readiness + generation.

    Raises RuntimeError when the lease is stale (generation mismatch or the
    plan is not ready), or ValueError when it belongs to another owner.
    ``value`` must be a non-None ``DeviceTensor`` — the name-only download
    path checks readiness separately.
    """
    if value is None:
        raise TypeError(
            "download requires a DeviceTensor lease; pass name=... only with a valid lease"
        )
    if value.owner is not owner:
        raise ValueError("resident output owner mismatch")
    value._step()  # checks _ready and generation


class PreparedResident(PreparedCuda):
    """One #146 context owns all storage; residents never create another arena.

    Attributes are fixed at preparation: the ABI version, the per-slot spans
    and the input/output names. ``upload`` pins inputs once, ``run`` evaluates
    the whole program without host staging, and ``download`` copies exactly
    one named output requested by the caller.
    """

    def __init__(
        self,
        plan: typing.Any,
        artifact: typing.Any,
        *,
        device: typing.Any = 0,
        resource_plan: typing.Any = None,
        resource_owner: typing.Any = None,
    ) -> None:
        super().__init__(
            plan,
            artifact,
            device=device,
            resource_plan=resource_plan,
            resource_owner=resource_owner,
        )
        lib = self._library
        lib.resident_abi.restype = ctypes.c_int
        if lib.resident_abi() != 1:
            self.close()
            raise ValueError("unsupported resident ABI")
        transfer = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        for name in ("resident_upload", "resident_download"):
            function = getattr(lib, name)
            function.argtypes, function.restype = transfer, ctypes.c_int
        lib.resident_run.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(_Metrics),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.resident_run.restype = ctypes.c_int
        self._names = {
            plan.steps[i].node.attrs["name"]: slot for slot, i in enumerate(plan.inputs)
        }
        self._output_slots = {name: slot for slot, (name, _) in enumerate(plan.outputs)}
        self._loaded, self._generation, self._ready = set(), 0, False
        self.transfers = {
            "h2d_bytes": 0,
            "d2h_bytes": 0,
            "runs": 0,
            "synchronizations": 0,
        }

    def _invalidate(self) -> None:
        self._ready = False
        self._generation += 1

    def upload(self, feeds: typing.Any) -> None:
        """Upload all or selected named inputs once; reject invalid tensors first."""
        with self._lock:
            if not self._pointer:
                raise RuntimeError("resident plan is closed")
            unknown = set(feeds) - self._names.keys()
            if unknown:
                raise ValueError(f"unknown resident inputs: {sorted(unknown)}")
            for name, value in feeds.items():
                slot = self._names[name]
                node = self.plan.steps[self.plan.inputs[slot]].node
                if (
                    not isinstance(value, np.ndarray)
                    or value.dtype != np.dtype(node.spec.dtype)
                    or value.shape != node.spec.shape
                ):
                    raise ValueError(
                        f"resident input {name} must be {node.spec.dtype} with shape "
                        f"{node.spec.shape}"
                    )
                np.copyto(self._inputs[slot], value)
                self._validate(self._inputs[slot], node)
            self._invalidate()
            for name in feeds:
                slot, error = self._names[name], ctypes.create_string_buffer(2048)
                array = self._inputs[slot]
                self._loaded.discard(name)
                if self._library.resident_upload(
                    self._pointer,
                    slot,
                    array.ctypes.data,
                    array.nbytes,
                    error,
                    len(error),
                ):
                    raise RuntimeError(error.value.decode())
                self._loaded.add(name)
                self.transfers["h2d_bytes"] += array.nbytes
                self.transfers["synchronizations"] += 1

    def run(self, *, profile: typing.Any = False) -> typing.Any:
        """Evaluate the whole program; download only a 4-byte error status."""
        with self._lock:
            if not self._pointer:
                raise RuntimeError("resident plan is closed")
            if self._loaded != self._names.keys():
                raise ValueError("resident inputs are not all initialized")
            self._invalidate()
            native, error = _Metrics(), ctypes.create_string_buffer(2048)
            start = time.perf_counter()
            status = self._library.resident_run(
                self._pointer, profile, ctypes.byref(native), error, len(error)
            )
            self.transfers["d2h_bytes"] += 4
            self.transfers["synchronizations"] += 1
            if status:
                raise RuntimeError(error.value.decode())
            self._ready = True
            self.transfers["runs"] += 1
            metrics = {name: getattr(native, name) for name, _ in native._fields_}
            metrics.update(
                precision=self.plan.precision,
                host_seconds=time.perf_counter() - start,
                h2d_bytes=0,
                d2h_bytes=4,
                profiled=bool(profile),
                synchronization_scope="one completion plus per-section events if profiled",
            )
            return {
                name: DeviceTensor(self, name, self._generation)
                for name in self._output_slots
            }, metrics

    def download(self, value: typing.Any, name: typing.Any = None) -> typing.Any:
        """Download exactly one named output the caller asked for.

        ``_check_lease`` validates ownership, readiness and the precise
        generation recorded when the lease was returned by ``run``.  A
        lease from run N is rejected after a later ``upload`` or ``run``
        invalidated it.
        """
        with self._lock:
            if value is None:
                if not self._pointer:
                    raise RuntimeError("resident plan is closed")
                if not self._ready:
                    raise RuntimeError(
                        "no resident run has completed yet; pass a valid DeviceTensor lease"
                    )
            _check_lease(self, value)
            target = name if name is not None else value.name
            if value is not None and target != value.name:
                raise ValueError("resident output name/target mismatch")
            slot = self._output_slots.get(target)
            if slot is None:
                raise ValueError("unknown resident output")
            step = self.plan.steps[dict(self.plan.outputs)[target]]
            output = np.empty(step.node.spec.shape, dtype=step.node.spec.dtype)
            error = ctypes.create_string_buffer(2048)
            if self._library.resident_download(
                self._pointer,
                slot,
                output.ctypes.data,
                output.nbytes,
                error,
                len(error),
            ):
                raise RuntimeError(error.value.decode())
            self.transfers["d2h_bytes"] += output.nbytes
            self.transfers["synchronizations"] += 1
            return output

    def copy_input(self, name: typing.Any, host_array: typing.Any) -> None:
        """Upload one named input into an already-running resident owner."""
        self.upload({name: host_array})

    def execute(self, feeds: typing.Any, *, profile: typing.Any = False) -> typing.Any:
        raise RuntimeError("use explicit upload/run/download on a resident owner")
