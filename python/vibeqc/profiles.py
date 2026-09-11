# ruff: noqa: PLC0414
"""Validated, atomic user-local CUDA profile storage and compatibility checks.

A profile contains a complete native library built with the existing AOT
manifest machinery. Merely writing a schedule JSON cannot enable uncompiled
kernels. Installation publishes a pointer only after every artifact and gate
has been checked, leaving a previous profile usable after interrupted tuning.
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from vibeqc_compiler.common.provenance import atomic_json as atomic_json
from vibeqc_compiler.common.provenance import canonical_hash as canonical_hash
from vibeqc_compiler.common.provenance import file_hash as file_hash
from vibeqc_compiler.common.provenance import find_nvcc as find_nvcc
from vibeqc_compiler.common.provenance import toolchain_identity as toolchain_identity

PROFILE_SCHEMA = 1
POLICY = {"precision": "fp64", "spin": ["rhf", "uhf"], "consumers": ["fock", "force"]}
BUNDLE_FILES = ("profile.json", "libvibeqc.so", "manifest.json", "evidence.json")


class DeviceDescriptor(ctypes.Structure):
    """C ABI record; only the native CUDA runtime interprets device visibility."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("name", ctypes.c_char * 256),
        ("official_profile", ctypes.c_char * 128),
        *[
            (name, ctypes.c_int32)
            for name in (
                "major",
                "minor",
                "warp_size",
                "maximum_threads_per_block",
                "maximum_threads_per_sm",
                "maximum_blocks_per_sm",
                "registers_per_sm",
                "maximum_registers_per_thread",
                "sm_count",
            )
        ],
        *[
            (name, ctypes.c_uint64)
            for name in (
                "shared_memory_per_block",
                "shared_memory_per_block_optin",
                "shared_memory_per_sm",
            )
        ],
        *[
            (name, ctypes.c_int32)
            for name in (
                "runtime_version",
                "driver_version",
                "toolkit_version",
                "release_build",
                "fast_compile",
                "portable",
            )
        ],
    ]


def cache_root() -> Path:
    """Honor XDG_CACHE_HOME and an explicit VIBEQC_PROFILE_CACHE override."""
    if configured := os.environ.get("VIBEQC_PROFILE_CACHE"):
        return Path(configured).expanduser()
    return (
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "vibeqc/profiles"
    )


def probe_device(library: ctypes.CDLL, device_id: int = 0) -> dict:
    """Probe the allocated GPU; this function must run inside its GPU allocation."""
    from . import _native

    probe = library.vibeqc_cuda_tuning_device
    probe.argtypes = [ctypes.c_int32, ctypes.POINTER(DeviceDescriptor)]
    probe.restype = ctypes.c_int
    descriptor = DeviceDescriptor()
    descriptor.struct_size = ctypes.sizeof(descriptor)
    descriptor.abi_version = _native.ABI_VERSION
    status = probe(device_id, ctypes.byref(descriptor))
    if status != 0:
        raise RuntimeError(f"CUDA tuning device probe failed (status {status})")
    library.vibeqc_get_source_identity.restype = ctypes.c_char_p
    device = {}
    for name, _ in DeviceDescriptor._fields_:
        if name in ("struct_size", "abi_version"):
            continue
        value = getattr(descriptor, name)
        device[name] = value.decode() if isinstance(value, bytes) else int(value)
    return {
        "source_identity": library.vibeqc_get_source_identity().decode(),
        "native_abi": _native.ABI_VERSION,
        "schema": PROFILE_SCHEMA,
        "policy": POLICY,
        "host": {
            "system": platform.system(),
            "machine": platform.machine(),
            "libc": list(platform.libc_ver()),
        },
        "device": device,
    }


def compatibility_identity(probe: dict) -> dict:
    """Exclude selected-profile diagnostics; retain every actual hardware limit."""
    return {
        **probe,
        "device": {
            k: v
            for k, v in probe["device"].items()
            if k not in ("official_profile", "portable")
        },
    }


def validate_bundle(
    directory: Path, probe: dict | None = None, toolchain: dict | None = None
) -> dict:
    """Reject incomplete, tampered, stale, or merely timed tuning output."""
    profile = json.loads((directory / "profile.json").read_text())
    if (
        profile.get("schema") != "vibeqc.local_profile"
        or profile.get("schema_version") != PROFILE_SCHEMA
    ):
        raise ValueError("unsupported local profile schema")
    identity = profile["identity"]
    if identity.get("policy") != POLICY or identity.get("schema") != PROFILE_SCHEMA:
        raise ValueError("incompatible codegen/consumer policy")
    if (
        identity["device"].get("release_build") != 1
        or identity["device"].get("fast_compile") != 0
    ):
        raise ValueError(
            "local profiles require release builds without fast compilation"
        )
    if probe is not None and identity != compatibility_identity(probe):
        raise ValueError("GPU, runtime, driver, native source, or ABI identity changed")
    if toolchain is not None and profile["toolchain"] != toolchain:
        raise ValueError("NVCC/PTXAS identity changed")
    if set(profile["toolchain"]) != {"nvcc", "ptxas"} or not all(
        profile["toolchain"].values()
    ):
        raise ValueError("missing compiler identity")
    if set(profile["artifacts"]) != set(BUNDLE_FILES) - {"profile.json"}:
        raise ValueError("incomplete local profile artifacts")
    for name, digest in profile["artifacts"].items():
        path = directory / name
        if path.is_symlink() or file_hash(path) != digest:
            raise ValueError(f"local profile artifact hash differs: {name}")
    kernels = profile["kernels"]
    if not kernels:
        raise ValueError("a local profile requires at least one validated winner")
    seen = set()
    for kernel in kernels:
        key = (kernel["shell_class"], kernel["consumer"])
        if key in seen or key[1] not in POLICY["consumers"]:
            raise ValueError("duplicate or unsupported kernel consumer")
        seen.add(key)
        if len(key[0]) != 4 or any(c not in "spdf" for c in key[0]):
            raise ValueError("unknown shell class")
        if kernel["schedule_hash"] != canonical_hash(kernel["schedule"]):
            raise ValueError("schedule hash differs")
        if len(kernel["source_hash"]) != 64 or any(
            c not in "0123456789abcdef" for c in kernel["source_hash"]
        ):
            raise ValueError("missing generated source identity")
        if kernel["gates"] != {
            k: "pass"
            for k in ("target", "resources", "numerical", "performance", "endpoint")
        }:
            raise ValueError("candidate did not pass all local promotion gates")
    evidence = json.loads((directory / "evidence.json").read_text())
    if not evidence.get("endpoint", {}).get("passed"):
        raise ValueError("complete endpoint acceptance is missing")
    accepted = {
        (row["shell_class"], row["consumer"]): row
        for row in evidence.get("candidates", [])
        if row.get("accepted")
    }
    if set(accepted) != seen:
        raise ValueError("profile winners differ from accepted candidate evidence")
    for kernel in kernels:
        row = accepted[(kernel["shell_class"], kernel["consumer"])]
        isolated = row.get("isolated", {})
        measured_device = isolated.get("device", {})
        actual_device = identity["device"]
        if any(
            measured_device.get(record) != actual_device[expected]
            for record, expected in (
                ("name", "name"),
                ("major", "major"),
                ("minor", "minor"),
                ("driver", "driver_version"),
                ("runtime", "runtime_version"),
            )
        ):
            raise ValueError(
                "numerical evidence was collected on a different CUDA target/runtime"
            )
        winner = next(
            (
                w
                for w in row.get("tuning", {}).get("winners", [])
                if w.get("trial_key") == row.get("selected_trial")
            ),
            {},
        )
        if (
            not isolated.get("passed")
            or isolated.get("source_hash") != kernel["source_hash"]
            or isolated.get("schedule_hash") != kernel["schedule_hash"]
            or winner.get("schedule") != kernel["schedule"]
            or not winner.get("production_validation", {}).get("accepted")
            or not isolated.get("errors")
            or not all(e["passed"] for e in isolated["errors"].values())
            or not row.get("endpoint", {}).get("passed")
        ):
            raise ValueError(
                "kernel lacks matching independent numerical/endpoint evidence"
            )
    return profile


def verify_library(directory: Path, probe: dict, device_id: int = 0) -> ctypes.CDLL:
    """Verify the binary itself against source/ABI and actual CUDA identity."""
    selected = ctypes.CDLL(str(directory / "libvibeqc.so"))
    actual = probe_device(selected, device_id)
    if actual["device"]["portable"]:
        raise ValueError("cached binary has no tuned profile for the allocated GPU")
    if compatibility_identity(actual) != compatibility_identity(probe):
        raise ValueError(
            "cached native library has incompatible source/ABI/toolkit identity"
        )
    return selected


def install_bundle(
    directory: Path,
    probe: dict,
    toolchain: dict | None = None,
    *,
    root: Path | None = None,
) -> Path:
    """Publish a validated immutable bundle, then atomically activate its identity."""
    import fcntl

    root = cache_root() if root is None else root
    profile = validate_bundle(directory, probe, toolchain)
    verify_library(directory, probe)
    bundle_id = canonical_hash(profile)
    root.mkdir(parents=True, exist_ok=True)
    bundles = root / "bundles"
    bundles.mkdir(exist_ok=True)
    destination = bundles / bundle_id
    # Serialize index updates so tuning for two devices cannot lose the other
    # device's accepted profile. A process exit releases the advisory lock.
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not destination.exists():
            with tempfile.TemporaryDirectory(
                prefix=".pending-", dir=bundles
            ) as staging:
                staged = Path(staging) / "bundle"
                staged.mkdir()
                for name in BUNDLE_FILES:
                    shutil.copyfile(directory / name, staged / name)
                validate_bundle(staged, probe, toolchain)
                os.replace(staged, destination)
        else:
            validate_bundle(destination, probe, toolchain)
        index_path = root / "active.json"
        index = json.loads(index_path.read_text()) if index_path.exists() else {}
        index[canonical_hash(profile["identity"])] = bundle_id
        atomic_json(index_path, index)
    return destination


def export_bundle(directory: Path, destination: Path) -> None:
    """Export binary and evidence together; schedule-only exports are not runnable."""
    validate_bundle(directory)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for name in BUNDLE_FILES:
                archive.write(directory / name, name)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def import_bundle(
    archive_path: Path,
    probe: dict,
    toolchain: dict | None = None,
    *,
    root: Path | None = None,
) -> Path:
    """Install a cluster bundle after the same checks, without archive path traversal."""
    with tempfile.TemporaryDirectory(prefix="vibeqc-profile-import-") as staging:
        directory = Path(staging)
        with zipfile.ZipFile(archive_path) as archive:
            if sorted(archive.namelist()) != sorted(BUNDLE_FILES):
                raise ValueError(
                    "profile archive must contain exactly the four bundle files"
                )
            for name in BUNDLE_FILES:
                if archive.getinfo(name).file_size > 2 * 1024**3:
                    raise ValueError("profile artifact exceeds the import size limit")
                with (
                    archive.open(name) as source,
                    (directory / name).open("wb") as target,
                ):
                    shutil.copyfileobj(source, target)
        return install_bundle(directory, probe, toolchain, root=root)


def clear_profiles(*, root: Path | None = None) -> None:
    """Deactivate profiles atomically; retain immutable binaries used by live processes."""
    import fcntl

    root = cache_root() if root is None else root
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        atomic_json(root / "active.json", {})


def select_library(base: ctypes.CDLL, device_id: int = 0) -> tuple[ctypes.CDLL, dict]:
    """Prefer an explicit/compatible local profile, then official/portable CUDA.

    CPU callers never enter this function. Missing or invalid caches are an
    optimization miss and retain the caller's usable baseline library.
    """
    diagnostics = {
        "source": "official",
        "identity": None,
        "kernels": [],
        "rejected": [],
    }
    if os.environ.get("VIBEQC_PROFILE") == "off":
        diagnostics["rejected"].append("local profiles explicitly disabled")
    try:
        probe = probe_device(base, device_id)
        diagnostics.update(
            source="portable" if probe["device"]["portable"] else "official",
            identity=probe["device"]["official_profile"],
        )
        if os.environ.get("VIBEQC_PROFILE") == "off":
            return base, diagnostics
        root = cache_root()
        explicit = os.environ.get("VIBEQC_PROFILE")
        if explicit:
            directory = Path(explicit).expanduser()
        else:
            index_path = root / "active.json"
            if not index_path.exists():
                return base, diagnostics
            index = json.loads(index_path.read_text())
            bundle_id = index.get(canonical_hash(compatibility_identity(probe)))
            if bundle_id is None:
                if index:
                    diagnostics["rejected"].append(
                        "cached profiles have incompatible hardware/source/runtime identities"
                    )
                return base, diagnostics
            if len(bundle_id) != 64 or any(
                c not in "0123456789abcdef" for c in bundle_id
            ):
                raise ValueError("invalid active bundle identifier")
            directory = root / "bundles" / bundle_id
        # Running an immutable accepted binary needs no compiler. If a toolkit
        # is available, a change to NVCC/PTXAS also invalidates its local cache.
        nvcc = find_nvcc()
        toolchain = toolchain_identity(nvcc) if nvcc else None
        profile = validate_bundle(directory, probe, toolchain)
        selected = verify_library(directory, probe, device_id)
        diagnostics.update(
            source="local",
            identity=canonical_hash(profile),
            kernels=profile["kernels"],
            directory=str(directory),
        )
        return selected, diagnostics
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        RuntimeError,
        subprocess.SubprocessError,
    ) as error:
        diagnostics["rejected"].append(str(error))
        return base, diagnostics
