"""Shared compiled-runtime artifact caching; process control stays in cuda_adapter."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path

from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.common.cuda_resources import parse_resources
from vibeqc_compiler.common.cuda_runtime import CudaArtifact
from vibeqc_compiler.common.provenance import (
    atomic_json,
    canonical_hash,
    file_hash,
    toolchain_identity,
)


def compile_runtime(compiler, cache, source, *, headers=(), libraries=(), options=None):
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
        identity = {
            "schema": 1,
            "source": file_hash(source),
            "headers": {
                os.path.relpath(Path(p).resolve(), source.parent): file_hash(p)
                for p in headers
            },
            "toolchain": toolchain_identity(compiler.nvcc),
            "host_compiler": file_hash(
                Path(os.environ.get("NVCC_CCBIN") or shutil.which("gcc")).resolve()
            ),
            "host_version": subprocess.check_output(
                [os.environ.get("NVCC_CCBIN") or shutil.which("gcc"), "--version"],
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
