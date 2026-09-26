"""Retain reproducible f-projector qualification without committing build debris."""

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(
    "/inspire/qb-ilm/project/chemicalreaction/diwenxi-CZXS25120072/vibeqc-workspace"
)
TASK = ROOT / "build-171/fprojector-20260918"
evidence = TASK / "evidence"
for mode in ("cpu", "baseline", "cuda"):
    assert (evidence / (mode + ".exit")).read_text().strip() == "0", mode
for name in (
    "prepare.sh",
    "run.sh",
    "run-cuda.sh",
    "run-cuda-v2.sh",
    "cuda-queue.sh",
    "provenance.py",
    "endpoints.py",
    "files.txt",
    "input-checks.sh",
    "input-lf.py",
    "collect.py",
):
    shutil.copyfile(
        ROOT / "evidence-171" / ("fprojector-" + name),
        evidence / ("fprojector-" + name),
    )
shutil.copyfile(
    ROOT / "evidence-171/fprojector-formatted.tar.gz",
    evidence / "source-overlay.tar.gz",
)
shutil.copyfile(ROOT / "evidence-171/fprojector-prepare.log", evidence / "prepare.log")
for mode in ("cpu", "cuda"):
    shutil.copyfile(
        TASK / mode / "CMakeCache.txt", evidence / (mode + "-CMakeCache.txt")
    )
validation = {}
for name in (
    "cpu-ctest",
    "cuda-ctest",
    "cpu-python",
    "cuda-python",
    "cuda-memcheck",
    "cuda-f-memcheck",
    "cpu-input-lf",
):
    lines = (evidence / (name + ".log")).read_text().splitlines()
    if "memcheck" in name:
        assert any("ERROR SUMMARY: 0 errors" in s for s in lines), name
    elif "ctest" in name:
        assert any("100% tests passed" in s for s in lines), name
    else:
        assert any("passed" in s for s in lines) and not any(
            "FAILED " in s for s in lines
        ), name
    validation[name] = lines[-24:]
summary = {
    "validation": validation,
    "baseline": json.loads((evidence / "baseline-benchmark.json").read_text()),
    "candidate": json.loads((evidence / "candidate-benchmark.json").read_text()),
    "f_projector_endpoints": {},
    "provenance": {},
}
for device in ("cpu", "cuda"):
    record = json.loads((evidence / (device + "-f-endpoints.json")).read_text())
    assert len(record["cases"]) == (2 if device == "cpu" else 5)
    summary["f_projector_endpoints"][device] = record
for label in ("baseline", "candidate", "cpu"):
    record = json.loads((evidence / (label + "-provenance.json")).read_text())
    record.pop("source_files")
    summary["provenance"][label] = record
assert (
    summary["provenance"]["candidate"]["source_manifest_sha256"]
    == summary["provenance"]["cpu"]["source_manifest_sha256"]
)
for device, label in [("cpu", "cpu"), ("cuda", "candidate")]:
    assert (
        summary["f_projector_endpoints"][device]["library_sha256"]
        == summary["provenance"][label]["build_sha256"]["libvibeqc.so"]
    )
out = ROOT / "evidence-171/fprojector-results"
out.mkdir(exist_ok=True)
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
for name in ("baseline-resources.txt", "candidate-resources.txt"):
    shutil.copyfile(evidence / name, out / name)
records = []
with zipfile.ZipFile(
    out / "raw-evidence.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9
) as archive:
    for path in sorted(evidence.iterdir()):
        if path.is_file():
            data = path.read_bytes()
            records.append(
                {
                    "path": path.name,
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
            archive.writestr(path.name, data)
raw = (out / "raw-evidence.zip").read_bytes()
(out / "raw-evidence.manifest.json").write_text(
    json.dumps(
        {
            "schema": "vibeqc.evidence-archive.v1",
            "archive_bytes": len(raw),
            "archive_sha256": hashlib.sha256(raw).hexdigest(),
            "files": records,
        },
        indent=2,
    )
    + "\n"
)
print("Archived", len(records), "files;", len(raw), "bytes")
