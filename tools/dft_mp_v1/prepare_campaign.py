"""Prepare a source-matched installed-production DFT-MP-v1 campaign plan.

This tool does not run science and cannot establish a PASS.  It only emits a
runner plan after binding one official-upstream merged source revision to the
exact frozen contract, installed native library, prebuilt scientific artifact,
adapter, build record, operating conditions and RTX 5090 identity.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from .freeze_contract import REPO, canonical, digest
from .validate import (
    _git_is_ancestor,
    _official_master_oid,
    manifest,
    read_json,
    require,
)


def _file_record(path: Path) -> dict:
    path = path.expanduser().resolve()
    require(path.is_file(), f"missing campaign file: {path}")
    return {"path": str(path), "sha256": digest(path.read_bytes())}


def _check_merged_source(source_commit: str, contract: dict) -> None:
    require(
        type(source_commit) is str
        and len(source_commit) == 40
        and source_commit == source_commit.lower()
        and all(c in "0123456789abcdef" for c in source_commit),
        "source commit must be exact 40-hex",
    )
    official_master = _official_master_oid(REPO)
    require(official_master is not None, "could not resolve official upstream master")
    require(
        _git_is_ancestor(REPO, source_commit, official_master),
        "source revision is not in official upstream master",
    )
    source_contract = subprocess.run(
        ["git", "show", f"{source_commit}:tools/dft_mp_v1/manifest.json"],
        cwd=REPO,
        check=False,
        capture_output=True,
        timeout=30,
    )
    require(source_contract.returncode == 0, "merged source has no DFT-MP-v1 manifest")
    try:
        recorded = json.loads(source_contract.stdout)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("merged source has invalid DFT-MP-v1 manifest") from error
    require(
        type(recorded) is dict
        and recorded.get("contract_sha256") == contract["contract_sha256"],
        "merged source does not contain this exact contract",
    )


def build_plan(
    *,
    source_commit: str,
    library: Path,
    artifact: Path,
    adapter: Path,
    build_record: Path,
    conditions: Path,
    device_uuid: str,
    driver: str,
    toolchain: str,
    build_profile: str,
    timeout_seconds: int,
    adapter_command: list[str],
) -> dict:
    """Return a fully hashed runner plan or fail before any calculation starts."""

    contract = manifest()
    _check_merged_source(source_commit, contract)
    require(
        type(timeout_seconds) is int and 0 < timeout_seconds <= 86400,
        "bounded timeout_seconds required",
    )
    require(
        adapter_command
        and all(type(argument) is str and argument for argument in adapter_command),
        "adapter_command must be a non-empty argv list",
    )
    for value, label in (
        (device_uuid, "device UUID"),
        (driver, "driver"),
        (toolchain, "toolchain"),
        (build_profile, "build profile"),
    ):
        require(
            type(value) is str and bool(value.strip()), f"{label} identity required"
        )

    library_record = _file_record(library)
    artifact_record = _file_record(artifact)
    adapter_record = _file_record(adapter)
    build_record_file = _file_record(build_record)
    conditions_record = _file_record(conditions)
    read_json(Path(conditions_record["path"]))

    build = read_json(Path(build_record_file["path"]))
    require(
        build.get("source_commit") == source_commit
        and build.get("library_sha256") == library_record["sha256"]
        and build.get("artifact_sha256") == artifact_record["sha256"],
        "build record does not bind source/library/artifact",
    )
    require(
        build.get("scientific_cuda_artifacts_prebuilt") is True,
        "build record does not prove prebuilt scientific CUDA artifacts",
    )

    adapter_path = Path(adapter_record["path"])
    matches = []
    for index, argument in enumerate(adapter_command):
        try:
            if Path(argument).expanduser().resolve() == adapter_path:
                matches.append(index)
        except OSError:
            continue
    require(
        len(matches) == 1,
        "adapter_command must contain the exact hashed adapter path exactly once",
    )
    command = list(adapter_command)
    command[matches[0]] = str(adapter_path)

    campaign = {
        "execution_kind": "installed_production",
        "source_commit": source_commit,
        "build_source_commit": source_commit,
        "library_source_commit": source_commit,
        "library": library_record,
        "artifact": artifact_record,
        "adapter": adapter_record,
        "build_record": build_record_file,
        "conditions": conditions_record,
        "conditions_sha256": conditions_record["sha256"],
        "hardware": {
            "device": "RTX 5090",
            "sm": 120,
            "device_uuid": device_uuid,
            "driver": driver,
            "toolchain": toolchain,
            "build_profile": build_profile,
        },
    }
    return {
        "adapter_command": command,
        "adapter_command_file_index": matches[0],
        "timeout_seconds": timeout_seconds,
        "campaign": campaign,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--build-record", required=True, type=Path)
    parser.add_argument("--conditions", required=True, type=Path)
    parser.add_argument("--device-uuid", required=True)
    parser.add_argument("--driver", required=True)
    parser.add_argument("--toolchain", required=True)
    parser.add_argument("--build-profile", required=True)
    parser.add_argument("--timeout-seconds", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--adapter-command",
        nargs=argparse.REMAINDER,
        help="adapter argv; this must be final and include the hashed adapter file path exactly once",
    )
    args = parser.parse_args()
    command = list(args.adapter_command or [])
    if command[:1] == ["--"]:
        command = command[1:]
    plan = build_plan(
        source_commit=args.source_commit,
        library=args.library,
        artifact=args.artifact,
        adapter=args.adapter,
        build_record=args.build_record,
        conditions=args.conditions,
        device_uuid=args.device_uuid,
        driver=args.driver,
        toolchain=args.toolchain,
        build_profile=args.build_profile,
        timeout_seconds=args.timeout_seconds,
        adapter_command=command,
    )
    output = args.output.expanduser().resolve()
    require(output.parent.is_dir(), f"campaign plan directory missing: {output.parent}")
    require(not output.exists(), f"campaign plan already exists: {output}")
    output.write_bytes(canonical(plan))
    print(output)


if __name__ == "__main__":
    main()
