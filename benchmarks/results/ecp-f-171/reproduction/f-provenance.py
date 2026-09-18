"""Verify every exported source file against the exact baseline plus overlay."""

import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(
    "/inspire/qb-ilm/project/chemicalreaction/diwenxi-CZXS25120072/vibeqc-workspace"
)
TASK = ROOT / "build-171/f-20260917"
label = sys.argv[1]
source = TASK / sys.argv[2]
expected = {}
archives = [ROOT / "evidence-171/f-baseline.tar.gz"]
if label != "baseline":
    archives.append(ROOT / "evidence-171/f-formatted.tar.gz")
archive_hashes = {}
for path in archives:
    archive_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    with tarfile.open(path) as archive:
        if path == archives[0]:
            commit = archive.pax_headers.get("comment")
        for member in archive:
            if member.isfile():
                expected[member.name] = {
                    "sha256": hashlib.sha256(
                        archive.extractfile(member).read()
                    ).hexdigest()
                }
            elif member.issym():
                expected[member.name] = {"symlink": member.linkname}
for name, identity in expected.items():
    path = source / name
    actual = (
        {"symlink": str(path.readlink())}
        if "symlink" in identity
        else {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    )
    assert actual == identity, (name, actual, identity)
manifest_bytes = json.dumps(expected, sort_keys=True).encode()
record = {
    "base_commit": commit,
    "source_kind": "verified git archive snapshot, not a Git checkout",
    "modified_from_base": label != "baseline",
    "overlay_files": []
    if label == "baseline"
    else [
        "python/vibeqc_compiler/integral/ecp.py",
        "python/vibeqc_compiler/integral/ecp_grid.py",
        "python/vibeqc_compiler/integral/ir.py",
        "python/vibeqc/ecp.py",
        "src/api/c_api_ecp.cpp",
        "tests/native/test_ecp_projector.cpp",
        "tests/native/test_ecp_capabilities.cpp",
        "cmake/VibeQCTests.cmake",
        "tests/python/test_ecp.py",
        "tests/python/test_ecp_f.py",
        "tests/python/test_ecp_ir.py",
        "tests/python/test_ecp_validation.py",
    ],
    "archive_sha256": archive_hashes,
    "verified_file_count": len(expected),
    "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    "source_files": expected,
}
build = TASK / ("cpu" if label == "cpu" else "cuda")
for name in ["libvibeqc.so", "generated/generated_ecp_ao.cuh", "CMakeCache.txt"]:
    path = build / name
    if path.exists():
        record.setdefault("build_sha256", {})[name] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
record["toolchain"] = subprocess.check_output(["nvcc", "--version"], text=True)
record["gpu"] = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
    text=True,
)
(TASK / "evidence" / (label + "-provenance.json")).write_text(
    json.dumps(record, indent=2) + "\n"
)
print(
    label,
    "verified",
    len(expected),
    "files at baseline",
    commit,
    "modified:",
    label != "baseline",
)
