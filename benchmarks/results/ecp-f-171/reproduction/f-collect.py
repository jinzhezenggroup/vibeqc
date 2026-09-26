"""Archive orbital-f qualification with exact source and library identities."""

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(
    "/inspire/qb-ilm/project/chemicalreaction/diwenxi-CZXS25120072/vibeqc-workspace"
)
TASK = ROOT / "build-171/f-20260917"
evidence = TASK / "evidence"
for kind in ("baseline", "cpu", "cuda"):
    assert (evidence / (kind + ".exit")).read_text().strip() == "0", kind
for name in (
    "prepare.sh",
    "run.sh",
    "run-v2.sh",
    "provenance.py",
    "collect.py",
    "endpoints.py",
):
    shutil.copyfile(ROOT / "evidence-171" / ("f-" + name), evidence / ("f-" + name))
shutil.copyfile(
    ROOT / "evidence-171/f-formatted.tar.gz", evidence / "source-overlay.tar.gz"
)
for label, build in [("candidate", "cuda"), ("cpu", "cpu")]:
    shutil.copyfile(
        TASK / build / "CMakeCache.txt", evidence / (label + "-CMakeCache.txt")
    )
validation = {}
for name in (
    "cpu-ctest",
    "cuda-ctest",
    "cpu-python",
    "cuda-focused-python",
    "cpu-focused-python",
    "cuda-memcheck",
    "cuda-f-memcheck",
):
    lines = (evidence / (name + ".log")).read_text().splitlines()
    if "memcheck" in name:
        assert any("ERROR SUMMARY: 0 errors" in line for line in lines), name
    elif "ctest" in name:
        assert any("100% tests passed" in line for line in lines), name
    else:
        assert any("passed" in line for line in lines) and not any(
            "FAILED " in line for line in lines
        ), name
    validation[name] = lines[-4:]
summary = {
    "validation": validation,
    "baseline": json.loads((evidence / "baseline-benchmark.json").read_text()),
    "candidate": json.loads((evidence / "candidate-benchmark.json").read_text()),
    "f_endpoints": {},
    "provenance": {},
}
for device in ("cpu", "cuda"):
    record = json.loads((evidence / (device + "-f-endpoints.json")).read_text())
    assert len(record["cases"]) == (4 if device == "cpu" else 5)
    summary["f_endpoints"][device] = record
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
        summary["f_endpoints"][device]["library_sha256"]
        == summary["provenance"][label]["build_sha256"]["libvibeqc.so"]
    )
out = ROOT / "evidence-171/f-results"
out.mkdir(exist_ok=True)
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
records = []
with zipfile.ZipFile(
    out / "raw-evidence.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9
) as archive:
    for path in sorted(evidence.iterdir()):
        if not path.is_file():
            continue
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
