"""Qualify pinned input hashes in a separate copy with canonical Git line endings."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

root = Path(
    "/inspire/qb-ilm/project/chemicalreaction/diwenxi-CZXS25120072/vibeqc-workspace"
)
task = root / "build-171/fprojector-20260918"
target = task / "input-check-source"
shutil.copytree(
    task / "candidate",
    target,
    ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".ruff_cache"),
)
fixtures = target / "tests/data/external_basis"
manifest = json.loads((fixtures / "manifest.json").read_text())
paths = [fixtures / name for name in manifest["files"]]
paths.append(target / "tools/generate_external_basis_references.py")
changes = []
for path in paths:
    original = path.read_bytes()
    normalized = original.replace(b"\r\n", b"\n")
    path.write_bytes(normalized)
    changes.append(
        {
            "path": str(path.relative_to(target)),
            "archive_sha256": hashlib.sha256(original).hexdigest(),
            "lf_sha256": hashlib.sha256(normalized).hexdigest(),
        }
    )
(task / "evidence/input-line-endings.json").write_text(
    json.dumps(changes, indent=2) + "\n"
)
# The library and imported scientific Python remain the measured candidate.
os.environ["PYTHONPATH"] = str(task / "candidate/python")
os.environ["VIBEQC_LIBRARY"] = str(task / "cpu/libvibeqc.so")
with (task / "evidence/cpu-input-lf.log").open("w") as log:
    result = subprocess.run(
        ["python", "-m", "pytest", "tests/python/test_external_basis.py", "-q"],
        cwd=target,
        stdout=log,
        stderr=subprocess.STDOUT,
        check=False,
    )
(task / "evidence/cpu-input-lf.exit").write_text(str(result.returncode) + "\n")
raise SystemExit(result.returncode)
