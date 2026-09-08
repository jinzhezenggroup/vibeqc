"""Shared compiled-runtime artifact caching; process control stays in cuda_adapter."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path

from vibeqc.profiles import atomic_json, canonical_hash, file_hash, toolchain_identity

from tools.vibeqc_tensor.cuda_execute import CudaArtifact
from tools.vibeqc_tensor.cuda_resources import parse_resources


def compile_runtime(
    compiler, cache, source, *, headers=(), libraries=(), options=("--fmad=false",)
):
    """Hash-verified native runtime cache using the shared finite NVCC adapter.

    Callers supply scientific source/header and library identities explicitly.
    Compilation does not execute GPU code or acquire a device.
    """
    source = Path(source).resolve()
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
                raise RuntimeError(
                    f"CUDA runtime compilation failed: {result.stdout}{result.stderr}"
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
    if canonical_hash(metadata.get("identity")) != key or metadata.get(
        "binary_sha256"
    ) != file_hash(library):
        raise ValueError("CUDA runtime cache integrity failure")
    return CudaArtifact(library, metadata)
