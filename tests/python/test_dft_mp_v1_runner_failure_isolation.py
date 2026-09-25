"""Malformed-result row finalization, not scientific campaign qualification."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import pytest

from tools.dft_mp_v1 import run as runner
from tools.dft_mp_v1.freeze_contract import digest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("field", "malformed", "diagnostic"),
    [
        pytest.param("checks", None, "has no attribute", id="null-checks"),
        pytest.param("checks", [], "has no attribute", id="list-checks"),
        pytest.param("checks", "invalid", "has no attribute", id="string-checks"),
        pytest.param("energy_eh", 10**1000, "too large", id="overflow-energy"),
        pytest.param(
            "physical_residual", 10**1000, "too large", id="overflow-residual"
        ),
        pytest.param("checks", {}, "missing grid_convergence", id="missing-check"),
    ],
)
def test_malformed_pass_finalizes_row_and_preserves_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    malformed: object,
    diagnostic: str,
) -> None:
    # Keep this orchestration regression independent of manifest generation and
    # hardware. The actual per-row validator and real child process still run.
    row = {
        "id": "fixture/first",
        "case": "fixture",
        "method": "fixture",
        "backend": "cuda",
        "provider": "native-dft",
        "j_k": "direct",
        "spin": "rks",
        "product": "energy+analytic_forces",
    }
    scf = {
        "energy_tolerance_eh": 1e-10,
        "density_tolerance": 1e-8,
        "max_iterations": 100,
        "screening_thresholds": {"direct_eri": 1e-12},
    }
    contract = {
        "contract_sha256": "b" * 64,
        "rows": [row, {**row, "id": "fixture/second"}],
        "cases": {
            "fixture": {
                "input": "fixture.json",
                "input_sha256": "c" * 64,
                "grid_identity": "d" * 64,
                "ao_count_spherical": 1,
                "atom_count": 1,
                "charge": 0,
                "multiplicity": 1,
            }
        },
        "basis": {"basis_pack_sha256": "e" * 64},
        "model": {
            "methods": {"fixture": {"version": "fixture-v1"}},
            "scf": {**scf, "physical_residual_max": 1e-8},
        },
        "gates": {
            "independent_total_energy_abs_eh": 1e-8,
            "independent_force_component_max_eh_per_bohr": 1e-7,
        },
    }
    monkeypatch.setattr(runner, "manifest", lambda: contract)
    monkeypatch.setattr(runner, "audit", lambda *_: None)
    monkeypatch.setattr(runner, "ROOT", tmp_path)

    raw = tmp_path / "fixture.raw"
    raw.write_bytes(b"unit-test fixture, not scientific evidence\n")
    record = {"path": str(raw), "sha256": digest(raw.read_bytes())}
    campaign = {
        "source_commit": "a" * 40,
        **{
            name: dict(record)
            for name in ("library", "artifact", "build_record", "conditions")
        },
    }
    payload = {
        **{key: row[key] for key in ("backend", "provider", "j_k", "spin")},
        "status": "pass",
        "row_id": row["id"],
        "contract_sha256": contract["contract_sha256"],
        "source_commit": campaign["source_commit"],
        "library_sha256": record["sha256"],
        "artifact_sha256": record["sha256"],
        "input_sha256": "c" * 64,
        "grid_identity": "d" * 64,
        "basis_pack_sha256": "e" * 64,
        "method_version": "fixture-v1",
        "ao_count": 1,
        "atom_count": 1,
        "charge": 0,
        "multiplicity": 1,
        "basis_representation": "real_spherical",
        "auxiliary": None,
        "ecp": None,
        "periodic": False,
        "attained_state": "fixture-state",
        "reference_attained_state": "fixture-state",
        "scf": {
            **scf,
            "converged": True,
            "physical_residual": 1e-9,
            "final_forces_state": "fixture-state",
        },
        "energy_eh": -1.0,
        "forces_eh_per_bohr": [[0.0, 0.0, 0.0]],
        "independent_oracle": {
            "provider": "fixture-oracle",
            "raw": record,
            "source_sha256": "f" * 64,
            "method_version": "fixture-v1",
            "basis_pack_sha256": "e" * 64,
            "total_energy_error_eh": 1e-9,
            "energy_gate_eh": 1e-8,
            "force_component_max_error_eh_per_bohr": 1e-8,
            "force_gate_eh_per_bohr": 1e-7,
        },
        "checks": {},
    }
    target = payload["scf"] if field == "physical_residual" else payload
    target[field] = malformed
    (tmp_path / "payload.json").write_text(json.dumps(payload), encoding="utf-8")
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "row = sys.argv[sys.argv.index('--row') + 1]\n"
        "progress = Path(sys.argv[sys.argv.index('--progress') + 1])\n"
        "progress.write_text(json.dumps({'row': row}) + '\\n', encoding='utf-8')\n"
        "print('fixture stderr', file=sys.stderr)\n"
        "if row == 'fixture/first':\n"
        "    print(Path('payload.json').read_text(encoding='utf-8'))\n"
        "else:\n"
        "    print(json.dumps({'status': 'unsupported', 'reason': 'fixture only'}))\n",
        encoding="utf-8",
    )
    campaign["adapter"] = {
        "path": str(adapter),
        "sha256": digest(adapter.read_bytes()),
    }
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "adapter_command": [sys.executable, str(adapter)],
                "adapter_command_file_index": 1,
                "timeout_seconds": 10,
                "campaign": campaign,
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    receipt_path = runner.run(plan, out)
    rows = json.loads(receipt_path.read_text(encoding="utf-8"))["rows"]
    assert [entry["status"] for entry in rows] == ["failed", "unsupported"]
    assert rows[0]["reason"].startswith("adapter pass rejected:")
    assert diagnostic in rows[0]["reason"]
    assert "evidence" not in rows[0]
    assert rows[0]["partial_progress"] == rows[0]["capture"]["progress"]
    for entry in rows:
        for capture in entry["capture"].values():
            assert digest((out / capture["path"]).read_bytes()) == capture["sha256"]
    captured_stdout = out / rows[0]["capture"]["stdout"]["path"]
    assert json.loads(captured_stdout.read_bytes()) == payload
    journal = [
        json.loads(line)
        for line in (out / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [entry["status"] for entry in journal] == [
        "running",
        "failed",
        "running",
        "unsupported",
    ]
    retained = {path.name: path.read_bytes() for path in out.iterdir()}
    assert runner.run(plan, out) == receipt_path
    assert retained == {path.name: path.read_bytes() for path in out.iterdir()}
