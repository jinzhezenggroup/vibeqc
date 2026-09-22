"""Audit the six qualified stationary CUDA AOT artifacts in a package directory.

This tool performs no CUDA initialization and never discovers NVCC. It validates
the same plan/source/target/binary contract used by public forces, reports exact
binary/package footprint, and can compare a checkout build directory with an
installed/wheel directory byte-for-byte.
"""

from __future__ import annotations

# Source-tree CLI bootstrap; installed dependencies remain ordinary imports.
import sys as _compiler_sys
from pathlib import Path as _CompilerPath

if __package__ in (None, ""):
    _compiler_sys.path.insert(
        0, str(_CompilerPath(__file__).resolve().parents[1] / "python")
    )

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vibeqc_compiler.method.stationary_cuda import (
    _qualified_aot_plan,
    load_stationary_aot_artifact,
)

QUALIFIED_STATIONARY_AOT = (
    (0, "unpolarized", "lda_rks"),
    (0, "polarized", "lda_uks"),
    (1, "unpolarized", "pbe_rks"),
    (1, "polarized", "pbe_uks"),
    (2, "unpolarized", "r2scan_rks"),
    (2, "polarized", "r2scan_uks"),
)


@dataclass(frozen=True, slots=True)
class ArtifactAudit:
    name: str
    functional: int
    spin: str
    library: str
    binary_bytes: int
    binary_sha256: str
    source_identity: str
    contract_identity: str
    plan_identity: str
    artifact_key: str
    code_kinds: tuple[str, ...]
    driver_ptx_jit_possible: bool
    driver_ptx_jit_required: bool


@dataclass(frozen=True, slots=True)
class PackageAudit:
    directory: str
    architecture: str
    artifacts: tuple[ArtifactAudit, ...]
    aot_binary_bytes: int
    manifest_bytes: int
    aot_package_bytes: int
    native_library_bytes: int | None
    aot_to_native_ratio: float | None

    def to_payload(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "artifacts": [asdict(item) for item in self.artifacts],
        }


def audit_stationary_aot_directory(
    directory: Path,
    *,
    architecture: str,
    native_library: Path | None = None,
) -> PackageAudit:
    """Validate and size one checkout/install directory with all six artifacts."""

    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"stationary AOT directory does not exist: {directory}")
    if type(architecture) is not str or not architecture.startswith("sm_"):
        raise ValueError("architecture must be an sm_XX identity")

    records = []
    manifest_bytes = 0
    for functional, spin, name in QUALIFIED_STATIONARY_AOT:
        plan = _qualified_aot_plan(functional, spin)
        artifact = load_stationary_aot_artifact(
            directory,
            functional=functional,
            spin=spin,
            plan=plan,
            architecture=architecture,
        )
        metadata = artifact.metadata
        manifest = directory / f"vibeqc_stationary_{name}.json"
        manifest_bytes += manifest.stat().st_size
        code_kinds = tuple(metadata["identity"]["target"]["code_kinds"])
        records.append(
            ArtifactAudit(
                name=name,
                functional=functional,
                spin=spin,
                library=str(Path(artifact.library).resolve()),
                binary_bytes=Path(artifact.library).stat().st_size,
                binary_sha256=metadata["binary_sha256"],
                source_identity=metadata["source_identity"],
                contract_identity=metadata["contract_identity"],
                plan_identity=metadata["plan_identity"],
                artifact_key=metadata["key"],
                code_kinds=code_kinds,
                driver_ptx_jit_possible=metadata["driver_ptx_jit_possible"],
                driver_ptx_jit_required=metadata["driver_ptx_jit_required"],
            )
        )

    aot_binary_bytes = sum(record.binary_bytes for record in records)
    native_bytes = None
    ratio = None
    if native_library is not None:
        native_library = Path(native_library).resolve()
        if not native_library.is_file():
            raise FileNotFoundError(
                f"native VibeQC library does not exist: {native_library}"
            )
        native_bytes = native_library.stat().st_size
        ratio = aot_binary_bytes / native_bytes if native_bytes else None

    return PackageAudit(
        directory=str(directory),
        architecture=architecture,
        artifacts=tuple(records),
        aot_binary_bytes=aot_binary_bytes,
        manifest_bytes=manifest_bytes,
        aot_package_bytes=aot_binary_bytes + manifest_bytes,
        native_library_bytes=native_bytes,
        aot_to_native_ratio=ratio,
    )


def assert_native_cubin_path(package: PackageAudit) -> None:
    """Require every qualified artifact to execute without driver PTX JIT."""

    missing = [
        item.name
        for item in package.artifacts
        if "cubin" not in item.code_kinds or item.driver_ptx_jit_required
    ]
    if missing:
        raise ValueError(
            "stationary AOT package lacks a native cubin path for: "
            + ", ".join(sorted(missing))
        )


def assert_same_artifact_identity(
    checkout: PackageAudit, installed: PackageAudit
) -> None:
    """Require checkout and installed layouts to carry identical AOT artifacts."""

    if checkout.architecture != installed.architecture:
        raise ValueError("stationary AOT comparison architecture mismatch")
    left = {
        item.name: (
            item.binary_sha256,
            item.source_identity,
            item.contract_identity,
            item.plan_identity,
            item.artifact_key,
            item.code_kinds,
        )
        for item in checkout.artifacts
    }
    right = {
        item.name: (
            item.binary_sha256,
            item.source_identity,
            item.contract_identity,
            item.plan_identity,
            item.artifact_key,
            item.code_kinds,
        )
        for item in installed.artifacts
    }
    if left != right:
        changed = sorted(
            name for name in set(left) | set(right) if left.get(name) != right.get(name)
        )
        raise ValueError(
            "installed stationary AOT identity differs from checkout: "
            + ", ".join(changed)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--native-library", type=Path)
    parser.add_argument("--compare-directory", type=Path)
    parser.add_argument("--compare-native-library", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-native-cubin",
        action="store_true",
        help="fail if any qualified artifact would require driver PTX JIT",
    )
    args = parser.parse_args()

    primary = audit_stationary_aot_directory(
        args.directory,
        architecture=args.architecture,
        native_library=args.native_library,
    )
    if args.require_native_cubin:
        assert_native_cubin_path(primary)
    payload: dict[str, Any] = {"primary": primary.to_payload()}
    if args.compare_directory is not None:
        installed = audit_stationary_aot_directory(
            args.compare_directory,
            architecture=args.architecture,
            native_library=args.compare_native_library,
        )
        assert_same_artifact_identity(primary, installed)
        if args.require_native_cubin:
            assert_native_cubin_path(installed)
        payload["comparison"] = installed.to_payload()
        payload["identity_match"] = True

    text = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)


if __name__ == "__main__":
    main()
