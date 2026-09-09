"""Artifact identity, fail-closed fixture loading and no external CC delegation."""

import ast
import copy
import json
from pathlib import Path

import pytest

from tools.validate_cc import equation_artifact, load_references, run

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "tests/reference_data/cc/rccsd-a.json"


def test_committed_equation_export_is_current():
    saved = json.loads((REFERENCE.parent / "equations-2o2v.json").read_text())
    assert saved == json.loads(json.dumps(equation_artifact()))


@pytest.mark.parametrize("mutation", ["input", "output", "version", "upstream"])
def test_reference_corruption_is_rejected(tmp_path, mutation):
    data = copy.deepcopy(load_references(REFERENCE))
    if mutation == "input":
        data["cases"][0]["inputs"]["t1"][0][0] += 0.1
    elif mutation == "output":
        data["cases"][0]["correlation_energy"] += 0.1
    elif mutation == "version":
        data["pyscf"] = "unverified"
    else:
        data["upstream"]["files"][0]["sha256"] = "0" * 64
    path = tmp_path / "corrupt.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="provenance|hash"):
        load_references(path)


def test_shared_evidence_schema_and_replay_exports(tmp_path):
    records = run(tmp_path, REFERENCE)
    assert len(records) == 5
    assert all(r["stages"]["numerical"]["status"] == "pass" for r in records)
    assert all(r["stages"]["production"]["status"] == "not-run" for r in records)


def test_production_facade_does_not_import_reference_or_pyscf():
    for name in ("__init__.py", "evaluate.py", "equations.py", "inventory.py"):
        tree = ast.parse((ROOT / "tools/vibeqc_cc" / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any("pyscf" in n or "oracle" in n for n in names)
