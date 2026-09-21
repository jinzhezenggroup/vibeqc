"""Shared compiled-runtime artifact caching; process control stays in cuda_adapter."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import typing
from dataclasses import asdict, dataclass
from pathlib import Path

from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.common.cuda_resources import parse_resources
from vibeqc_compiler.common.cuda_runtime import CudaArtifact
from vibeqc_compiler.common.provenance import (
    atomic_json,
    canonical_hash,
    file_hash,
    toolchain_identity,
)


@dataclass(frozen=True, slots=True)
class CudaObjectArtifact:
    """Hash-verified relocatable CUDA object for reuse across runtime wrappers."""

    path: Path
    metadata: dict[str, typing.Any]


def _cuda_host_identity(
    compiler: CudaCompilerAdapter,
) -> tuple[Path, dict[str, typing.Any]]:
    host_compiler = os.environ.get("NVCC_CCBIN") or shutil.which("gcc")
    if host_compiler is None:
        raise RuntimeError("CUDA host compiler not found")
    resolved = shutil.which(host_compiler)
    if resolved is None:
        raise RuntimeError(f"CUDA host compiler not found: {host_compiler}")
    path = Path(resolved).resolve()
    return path, {
        "host_compiler": file_hash(path),
        "host_version": subprocess.check_output(
            [str(path), "--version"], text=True, timeout=30
        ),
    }


def compile_cuda_object(
    compiler: typing.Any,
    cache: typing.Any,
    source: typing.Any,
    *,
    headers: typing.Any = (),
    options: typing.Any = (),
) -> CudaObjectArtifact:
    """Compile/cache one PIC relocatable CUDA object without loading a device."""

    if not isinstance(compiler, CudaCompilerAdapter):
        raise TypeError("CUDA object compilation requires a CUDA compiler adapter")
    source = Path(source).resolve()
    options = tuple(options)
    _, host = _cuda_host_identity(compiler)
    identity = {
        "schema": "vibeqc.cuda.object/1",
        "source": file_hash(source),
        "headers": {
            os.path.relpath(Path(p).resolve(), source.parent): file_hash(p)
            for p in headers
        },
        "toolchain": toolchain_identity(compiler.nvcc),
        **host,
        "target": asdict(compiler.target),
        "flags": [
            "c++17",
            "O3",
            "relocatable-device-code",
            "fPIC",
            *options,
        ],
        "environment": {
            k: os.environ.get(k, "")
            for k in (
                "NVCC_CCBIN",
                "NVCC_PREPEND_FLAGS",
                "NVCC_APPEND_FLAGS",
                "CPATH",
                "LIBRARY_PATH",
            )
        },
    }
    key = canonical_hash(identity)
    cache = Path(cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / key
    if not destination.exists():
        with tempfile.TemporaryDirectory(
            prefix=".cuda-object-", dir=cache
        ) as temporary:
            folder = Path(temporary)
            binary = folder / "object.o"
            result = compiler.compile(
                source,
                binary,
                options=(
                    "--relocatable-device-code=true",
                    "-Xcompiler=-fPIC",
                    *options,
                ),
            )
            (folder / "compiler.log").write_text(result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError(
                    f"CUDA object compilation failed: {result.stdout}{result.stderr}"
                )
            metadata = {
                "identity": identity,
                "key": key,
                "binary_sha256": file_hash(binary),
                "compile_seconds": result.duration_seconds,
                "resources": [asdict(x) for x in parse_resources(result.stderr)],
            }
            atomic_json(folder / "artifact.json", metadata)
            try:
                os.rename(folder, destination)
            except OSError:
                if not destination.is_dir():
                    raise
    metadata = json.loads((destination / "artifact.json").read_text())
    binary = destination / "object.o"
    if (
        canonical_hash(metadata.get("identity")) != key
        or metadata.get("key") != key
        or metadata.get("binary_sha256") != file_hash(binary)
    ):
        raise ValueError("CUDA object cache integrity failure")
    return CudaObjectArtifact(binary, metadata)


def link_cuda_objects(
    compiler: typing.Any,
    cache: typing.Any,
    objects: typing.Iterable[CudaObjectArtifact],
    *,
    libraries: typing.Any = (),
    options: typing.Any = (),
) -> CudaArtifact:
    """Device-link cached relocatable CUDA objects into one verified runtime."""

    if not isinstance(compiler, CudaCompilerAdapter):
        raise TypeError("CUDA object linking requires a CUDA compiler adapter")
    objects = tuple(objects)
    if not objects:
        raise ValueError("CUDA runtime link requires at least one object")
    for item in objects:
        if not item.path.is_file() or item.metadata.get("binary_sha256") != file_hash(
            item.path
        ):
            raise ValueError("CUDA object artifact integrity failure")
    libraries = tuple(libraries)
    options = tuple(options)
    _, host = _cuda_host_identity(compiler)
    identity = {
        "schema": "vibeqc.cuda.link/1",
        "objects": [
            {
                "key": item.metadata.get("key"),
                "binary_sha256": item.metadata.get("binary_sha256"),
            }
            for item in objects
        ],
        "toolchain": toolchain_identity(compiler.nvcc),
        **host,
        "target": asdict(compiler.target),
        "flags": ["c++17", "O3", "shared", "relocatable-device-code", "fPIC", *options],
        "libraries": list(libraries),
        "environment": {
            k: os.environ.get(k, "")
            for k in (
                "NVCC_CCBIN",
                "NVCC_PREPEND_FLAGS",
                "NVCC_APPEND_FLAGS",
                "CPATH",
                "LIBRARY_PATH",
            )
        },
    }
    key = canonical_hash(identity)
    cache = Path(cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / key
    if not destination.exists():
        with tempfile.TemporaryDirectory(prefix=".cuda-link-", dir=cache) as temporary:
            folder = Path(temporary)
            library = folder / "runtime.so"
            result = compiler.link_shared_objects(
                [item.path for item in objects],
                library,
                libraries=libraries,
                options=options,
            )
            (folder / "compiler.log").write_text(result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError(
                    f"CUDA object link failed: {result.stdout}{result.stderr}"
                )
            metadata = {
                "identity": identity,
                "key": key,
                "binary_sha256": file_hash(library),
                "compile_seconds": result.duration_seconds,
                "resources": [asdict(x) for x in parse_resources(result.stderr)],
            }
            atomic_json(folder / "artifact.json", metadata)
            try:
                os.rename(folder, destination)
            except OSError:
                if not destination.is_dir():
                    raise
    metadata = json.loads((destination / "artifact.json").read_text())
    library = destination / "runtime.so"
    if (
        canonical_hash(metadata.get("identity")) != key
        or metadata.get("key") != key
        or metadata.get("binary_sha256") != file_hash(library)
    ):
        raise ValueError("CUDA linked-runtime cache integrity failure")
    return CudaArtifact(library, metadata)


def compile_runtime(
    compiler: typing.Any,
    cache: typing.Any,
    source: typing.Any,
    *,
    headers: typing.Any = (),
    libraries: typing.Any = (),
    options: typing.Any = None,
) -> typing.Any:
    """Hash-verified native runtime cache using explicit CPU or CUDA adapters.

    Callers supply scientific source/header and library identities explicitly.
    Compilation does not execute GPU code or acquire a device.
    """
    source = Path(source).resolve()
    cpu = isinstance(compiler, CppCompilerAdapter)
    if options is None:
        options = ("-ffp-contract=off",) if cpu else ("--fmad=false",)
    if cpu:
        executable = Path(compiler.cxx).resolve()
        identity = {
            "schema": 2,
            "backend": "cpu",
            "source": file_hash(source),
            "headers": {
                os.path.relpath(Path(p).resolve(), source.parent): file_hash(p)
                for p in headers
            },
            "compiler": {
                "invocation": str(compiler.cxx),
                "path": str(executable),
                "sha256": file_hash(executable),
                "version": subprocess.check_output(
                    [str(executable), "--version"], text=True, timeout=30
                ),
            },
            "target": asdict(compiler.target),
            "flags": ["c++17", "O3", "shared", "fPIC", *options],
            "libraries": list(libraries),
            "environment": {
                k: os.environ.get(k, "")
                for k in (
                    "CPATH",
                    "CPLUS_INCLUDE_PATH",
                    "LIBRARY_PATH",
                    "COMPILER_PATH",
                    "GCC_EXEC_PREFIX",
                    "SOURCE_DATE_EPOCH",
                )
            },
        }
    else:
        # Preserve the existing CUDA schema, keys and defaults exactly. Adding
        # CPU compilation must not relabel any accepted CUDA artifact/profile.
        host_compiler = os.environ.get("NVCC_CCBIN") or shutil.which("gcc")
        if host_compiler is None:
            raise RuntimeError("CUDA host compiler not found")
        host_compiler_path = Path(host_compiler).resolve()
        identity = {
            "schema": 1,
            "source": file_hash(source),
            "headers": {
                os.path.relpath(Path(p).resolve(), source.parent): file_hash(p)
                for p in headers
            },
            "toolchain": toolchain_identity(compiler.nvcc),
            "host_compiler": file_hash(host_compiler_path),
            "host_version": subprocess.check_output(
                [str(host_compiler_path), "--version"],
                text=True,
                timeout=30,
            ),
            "target": asdict(compiler.target),
            "flags": ["c++17", "O3", "shared", "fPIC", *options],
            "libraries": list(libraries),
            "environment": {
                k: os.environ.get(k, "")
                for k in (
                    "NVCC_CCBIN",
                    "NVCC_PREPEND_FLAGS",
                    "NVCC_APPEND_FLAGS",
                    "CPATH",
                    "LIBRARY_PATH",
                )
            },
        }
    key = canonical_hash(identity)
    cache = Path(cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / key
    if not destination.exists():
        with tempfile.TemporaryDirectory(
            prefix=".native-runtime-", dir=cache
        ) as temporary:
            folder = Path(temporary)
            library = folder / "runtime.so"
            result = compiler.compile_shared(
                source, library, libraries=libraries, options=options
            )
            (folder / "compiler.log").write_text(result.stdout + result.stderr)
            if result.returncode:
                label = "CPU" if cpu else "CUDA"
                raise RuntimeError(
                    f"{label} runtime compilation failed: {result.stdout}{result.stderr}"
                )
            metadata = {
                "identity": identity,
                "key": key,
                "binary_sha256": file_hash(library),
                "compile_seconds": result.duration_seconds,
                "resources": []
                if cpu
                else [asdict(x) for x in parse_resources(result.stderr)],
            }
            atomic_json(folder / "artifact.json", metadata)
            try:
                os.rename(folder, destination)
            except OSError:
                if not destination.is_dir():
                    raise
    metadata = json.loads((destination / "artifact.json").read_text())
    library = destination / "runtime.so"
    if canonical_hash(metadata.get("identity")) != key or metadata.get(
        "binary_sha256"
    ) != file_hash(library):
        raise ValueError(f"{'CPU' if cpu else 'CUDA'} runtime cache integrity failure")
    return CudaArtifact(library, metadata)
