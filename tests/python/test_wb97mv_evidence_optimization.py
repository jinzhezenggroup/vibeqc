"""Published benchmark gates must remain effective under Python optimization."""

from __future__ import annotations

import hashlib
import json
import lzma
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

EVIDENCE = (
    Path(__file__).resolve().parents[2] / "benchmarks/results/wb97mv-active-ao-20261003"
)


@pytest.mark.parametrize("optimization", ["flag", "environment"])
@pytest.mark.parametrize(
    "corruption,diagnostic",
    [
        ("none", None),
        ("stored_hash", "Stored report size or SHA-256 mismatch"),
        ("decoded_hash", "Decoded report size or SHA-256 mismatch"),
        ("library", "Native library identity mismatch"),
        ("source", "Native source identity mismatch"),
        ("energy", "Energy/force acceptance gate exceeded"),
        ("force", "Energy/force acceptance gate exceeded"),
        ("nonfinite", "matching shapes and finite values"),
        ("convergence", "Unconverged endpoint"),
        ("iterations", "exactly one SCF iteration"),
        ("xc_backend", "Reference XC must execute on GPU"),
        ("force_work", "discovery performed density contractions"),
        ("scf_work", "SCF AO selection differs"),
        ("cold_total", "Complete cold total does not include"),
    ],
)
def test_evidence_checks_survive_optimization(
    tmp_path: Path, optimization: str, corruption: str, diagnostic: str | None
) -> None:
    """Corrupt copied receipts, rebinding hashes to exercise scientific gates too."""
    shutil.copyfile(EVIDENCE / "verify.py", tmp_path / "verify.py")
    manifest = json.loads((EVIDENCE / "manifest.json").read_text())
    name = "matched3-sparse.json.xz"
    report = json.loads(lzma.decompress((EVIDENCE / name).read_bytes()))
    cold = report["native_cold"]
    if corruption == "library":
        report["native_build"]["library_sha256"] = "invalid"
    elif corruption == "source":
        report["native_build"]["probe"]["source_identity"] = "invalid"
    elif corruption == "energy":
        cold["energies_hartree"][0] += 1e-4
    elif corruption == "force":
        cold["forces_hartree_per_bohr"][0][0][0] += 1e-4
    elif corruption == "nonfinite":
        cold["energies_hartree"][0] = float("nan")
    elif corruption == "convergence":
        cold["convergence"][0]["converged"] = False
    elif corruption == "iterations":
        report["native_priming"]["convergence"][0]["iterations"] = 2
    elif corruption == "xc_backend":
        report["reference_cold"]["reference_xc_backend"][0]["on_gpu"] = False
    elif corruption == "force_work":
        cold["force_work"]["active_ao_maps"]["discovery_density_contractions"] = 1
    elif corruption == "scf_work":
        cold["native_scf_ao_work"]["selected"] = 0
    elif corruption == "cold_total":
        report["native_complete_cold"]["seconds"] += 1

    payload = json.dumps(report).encode("utf-8")
    stored = lzma.compress(payload)
    manifest["files"][name] = {
        "bytes": len(stored),
        "sha256": hashlib.sha256(stored).hexdigest(),
        "decoded_bytes": len(payload),
        "decoded_sha256": hashlib.sha256(payload).hexdigest(),
    }
    if corruption == "stored_hash":
        manifest["files"][name]["sha256"] = "invalid"
    elif corruption == "decoded_hash":
        manifest["files"][name]["decoded_sha256"] = "invalid"
    (tmp_path / name).write_bytes(stored)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    env = {
        **os.environ,
        "PYTHONOPTIMIZE": "1" if optimization == "environment" else "0",
    }
    command = [sys.executable]
    if optimization == "flag":
        command.append("-O")
    result = subprocess.run(
        [*command, str(tmp_path / "verify.py"), "3"],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if diagnostic is None:
        assert result.returncode == 0, result.stderr
        assert "passed." in result.stdout
    else:
        assert result.returncode != 0
        assert diagnostic in result.stderr
        assert "passed." not in result.stdout
