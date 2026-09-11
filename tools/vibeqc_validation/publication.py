"""Publish selected benchmark evidence without copying a working directory.

The existing validation envelope owns scientific gates and raw measurements.
This small storage manifest adds reproduction, dirty-source and archive
retention information. Publication never changes a production selector.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlparse

from .retention import RESULT_ROOT, classify, digest, safe_relative
from .schema import validate_evidence

SCHEMA = "vibeqc.benchmark-publication.v1"
MANIFEST = "publication.json"
ROLES = {"evidence", "summary", "samples", "input", "reproduction", "source-patch"}


def _passing_error_record(row: dict) -> bool:
    """Require the quantitative fields emitted by validation.block_error.

    A numerical stage flag alone must not admit an edited record which lost
    its tolerances. Finiteness is checked by the shared evidence validator.
    """
    required = {
        "passed",
        "atol",
        "rtol",
        "max_absolute_error",
        "max_scaled_error",
        "rms_error",
        "shape",
    }
    if not required <= row.keys() or row["passed"] is not True:
        return False
    scalars = [row[k] for k in required - {"passed", "shape"}]
    return (
        all(type(value) in {int, float} and value >= 0 for value in scalars)
        and row["atol"] + row["rtol"] > 0
        and row["max_scaled_error"] <= 1
        and isinstance(row["shape"], list)
        and all(type(size) is int and size > 0 for size in row["shape"])
    )


def validate_publication(manifest: dict, files: dict[str, bytes]) -> None:
    """Verify a self-contained selected bundle and its scientific decision.

    Large files require a per-file rationale in addition to the repository's
    size review. Remote archives are optional debugging material: permanent
    test or reproduction inputs must be in the selected bundle or source tree.
    """
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported publication manifest")
    source = manifest["source"]
    if (
        not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", source.get("revision", ""))
        or type(source.get("dirty")) is not bool
    ):
        raise ValueError("publication requires exact source revision and dirty state")
    command = manifest["reproduction"]["command"]
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(x, str) and x for x in command)
    ):
        raise ValueError("reproduction command must be a nonempty argv list")
    entries = manifest["files"]
    paths = [safe_relative(entry["path"]) for entry in entries]
    if len(set(paths)) != len(paths) or set(paths) != set(files) or MANIFEST in paths:
        raise ValueError("publication inventory differs from selected files")
    for entry in entries:
        path = entry["path"]
        if entry["role"] not in ROLES:
            raise ValueError("unknown publication file role")
        if classify(RESULT_ROOT + path) in {"transient", "generated-build"}:
            raise ValueError(f"publish a summary instead of run debris: {path}")
        if digest(files[path]) != entry["sha256"] or len(files[path]) != entry["bytes"]:
            raise ValueError(f"publication checksum/size mismatch: {path}")
        if len(files[path]) > 1 << 20 and not entry.get("reason"):
            raise ValueError(f"large file requires a review justification: {path}")
    evidence_paths = [e["path"] for e in entries if e["role"] == "evidence"]
    if len(evidence_paths) != 1:
        raise ValueError("publication requires exactly one validation envelope")
    evidence = json.loads(files[evidence_paths[0]])
    validate_evidence(evidence)
    for attachment in evidence["attachments"]:
        path = safe_relative(attachment["path"])
        if path not in files or digest(files[path]) != attachment["sha256"]:
            raise ValueError(
                f"validation attachment must be retained unchanged: {path}"
            )
    if evidence["revision"] != source["revision"]:
        raise ValueError("publication source differs from measured source")
    if (
        not evidence["device"]
        or not evidence["toolchain"]
        or not evidence["backend_selected"]
    ):
        raise ValueError("publication requires hardware/software/backend provenance")
    for key in ("equation", "ir", "source", "schedule"):
        if not re.fullmatch(
            r"[0-9a-f]{64}", evidence["hashes"].get(key) or ""
        ) and not evidence["hash_reasons"].get(key):
            raise ValueError(
                f"missing scientific identity or explicit inapplicability: {key}"
            )
    if source["dirty"] and not any(
        e["role"] == "source-patch" and e["bytes"] > 0 for e in entries
    ):
        raise ValueError(
            "dirty measured source requires a retained reconstruction patch"
        )
    decision = manifest["decision"]
    if decision.get("status") not in {
        "accepted",
        "rejected",
        "inconclusive",
    } or not decision.get("reason"):
        raise ValueError("publication requires a decision and reason")
    scope = decision.get("scope")
    if scope not in {"numerical", "performance"}:
        raise ValueError("decision scope must be numerical or performance")
    gate = (
        evidence["performance"]
        if scope == "performance"
        else evidence["stages"]["numerical"]
    )
    if decision["status"] == "accepted" and gate["status"] != "pass":
        raise ValueError(
            "accepted decision lacks passing evidence in its declared scope"
        )
    if (
        decision["status"] == "accepted"
        and scope == "numerical"
        and (
            not evidence["block_errors"]
            or not all(
                _passing_error_record(row) for row in evidence["block_errors"].values()
            )
        )
    ):
        raise ValueError(
            "numerical acceptance requires passing quantitative errors and tolerances"
        )
    for archive in manifest["archives"]:
        parsed = urlparse(archive["uri"])
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or not re.fullmatch(r"[0-9a-f]{64}", archive["sha256"])
        ):
            raise ValueError("archive requires HTTPS identity and checksum")
        if (
            type(archive["bytes"]) is not int
            or archive["bytes"] < 0
            or not archive["retention"]
        ):
            raise ValueError("archive requires size and explicit retention/expiry")
        if archive["required_for_reproduction"] is not False:
            raise ValueError(
                "reproduction inputs cannot depend on external archive expiry"
            )


def publish(run_directory: Path, specification: dict, destination: Path) -> Path:
    """Validate all selected bytes before creating a new review directory.

    The specification names relative files and roles, not an entire retry tree.
    Source revision/dirty state come from the measured run, never from the
    publishing checkout. Existing publications are never overwritten.
    """
    if destination.exists():
        raise FileExistsError(destination)
    manifest = json.loads(json.dumps(specification, allow_nan=False))
    manifest["schema"] = SCHEMA
    files = {}
    root = run_directory.resolve()
    for entry in manifest["files"]:
        path = safe_relative(entry["path"])
        candidate = (root / path).resolve(strict=True)
        if not candidate.is_relative_to(root):
            raise ValueError("publication input escapes run directory")
        data = candidate.read_bytes()
        files[path] = data
        entry.update(sha256=digest(data), bytes=len(data))
    validate_publication(manifest, files)
    destination.mkdir(parents=True, exist_ok=False)
    try:
        for path, data in files.items():
            target = destination / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        (destination / MANIFEST).write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    except Exception:
        # Remove only this newly created bundle on an I/O failure. A partially
        # published directory must not be mistaken for complete evidence.
        import shutil

        shutil.rmtree(destination)
        raise
    return destination
