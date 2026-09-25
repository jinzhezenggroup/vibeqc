"""Negative controls for the frozen DFT-MP-v1 evidence contract."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import time
from typing import TYPE_CHECKING

import pytest
from vibeqc import Atom, GridSpec
from vibeqc_compiler.dft.grid import molecular_grid_identity

if TYPE_CHECKING:
    from pathlib import Path

from tools.dft_mp_v1 import validate as contract_validator
from tools.dft_mp_v1.freeze_contract import ROOT, canonical, digest, source_digest
from tools.dft_mp_v1.run import run as capture
from tools.dft_mp_v1.validate import (
    InvalidEvidence,
    _check_run,
    _performance_gate,
    audit,
    manifest,
)


@pytest.fixture(scope="module")
def contract() -> dict:
    return manifest()


def test_source_text_identity_is_crlf_independent() -> None:
    assert source_digest(b"a\nb\n") == source_digest(b"a\r\nb\r\n")


def test_frozen_geometry_reproduces_exact_bytes_and_actual_basis_counts(
    contract: dict,
) -> None:
    pytest.importorskip("rdkit")
    from tools.dft_mp_v1 import generate_inputs

    if not generate_inputs.generator_is_qualified():
        with pytest.raises(RuntimeError, match="unqualified DFT-MP-v1 generator"):
            generate_inputs.build()
        pytest.skip(
            "frozen geometry reconstruction requires the recorded generator wheel"
        )
    regenerated = generate_inputs.build()
    for key, source in regenerated.items():
        original = json.loads(
            (ROOT / "inputs" / f"{key}.json").read_text(encoding="utf-8")
        )
        assert original == {"schema_version": 1, "id": key, "units": "bohr", **source}
    assert [
        (
            key,
            contract["cases"][key]["atom_count"],
            contract["cases"][key]["ao_count_spherical"],
        )
        for key in ("water8", "water16", "water32")
    ] == [("water8", 24, 192), ("water16", 48, 384), ("water32", 96, 768)]
    assert contract["cases"]["ace_glygly_nme"]["ao_count_spherical"] == 247
    assert sum(row["required"] for row in contract["rows"]) == 115
    assert sum(not row["required"] for row in contract["rows"]) == 2
    assert (
        contract["model"]["grid_selection"]
        == "explicit_common_pbe_tight_derivative_v2_not_per_method_default"
    )
    water = json.loads((ROOT / "inputs/water.json").read_text(encoding="utf-8"))
    atoms = [Atom.from_value((element, xyz)) for element, xyz in water["atoms"]]
    grid = GridSpec(**contract["model"]["grid_spec"])
    assert (
        molecular_grid_identity(atoms, grid)
        == contract["cases"]["water"]["grid_identity"]
    )
    for key in regenerated:
        changed = json.loads(
            (ROOT / "inputs" / f"{key}-changed.json").read_text(encoding="utf-8")
        )
        assert changed["source"]["delta_bohr"] == 0.01


def test_generator_provenance_rejects_platform_or_wheel_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("rdkit")
    from tools.dft_mp_v1 import generate_inputs

    recorded = generate_inputs.recorded_generator_provenance()
    drifted = {**recorded, "system": "unqualified-system"}
    monkeypatch.setattr(
        generate_inputs, "current_generator_provenance", lambda: drifted
    )

    assert not generate_inputs.generator_is_qualified()
    with pytest.raises(RuntimeError, match="unqualified DFT-MP-v1 generator"):
        generate_inputs.build()


def test_generator_audit_preserves_frozen_tree_on_mismatch(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    from tools.dft_mp_v1 import generate_inputs

    frozen = tmp_path / "inputs"
    frozen.mkdir()
    case = frozen / "case.json"
    case.write_bytes(b'{"frozen":true}\n')
    before = {path.name: path.read_bytes() for path in frozen.iterdir()}

    with pytest.raises(RuntimeError, match="frozen input audit mismatch"):
        generate_inputs.audit_serialized({"case.json": b'{"candidate":true}\n'}, frozen)

    assert {path.name: path.read_bytes() for path in frozen.iterdir()} == before


def test_generator_candidate_output_is_separate_and_empty(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    from tools.dft_mp_v1 import generate_inputs

    frozen = tmp_path / "inputs"
    frozen.mkdir()
    (frozen / "case.json").write_bytes(b"frozen\n")
    candidate = {"case.json": b"candidate\n"}

    with pytest.raises(ValueError, match="separate output tree"):
        generate_inputs.write_candidate_files(candidate, frozen, frozen=frozen)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "prior.txt").write_bytes(b"prior\n")
    with pytest.raises(ValueError, match="empty"):
        generate_inputs.write_candidate_files(candidate, occupied, frozen=frozen)

    output = tmp_path / "candidate-v2"
    generate_inputs.write_candidate_files(candidate, output, frozen=frozen)
    assert (output / "case.json").read_bytes() == b"candidate\n"


def _file(path: Path, content: bytes = b"raw evidence\n") -> dict:
    path.write_bytes(content)
    return {"path": path.name, "sha256": digest(content)}


def _campaign(tmp_path: Path) -> dict:
    record = _file(tmp_path / "dummy.bin")
    conditions = _file(tmp_path / "conditions.json", b'{"mode":"fixture"}\n')
    build = {
        "source_commit": "a" * 40,
        "library_sha256": record["sha256"],
        "artifact_sha256": record["sha256"],
        "scientific_cuda_artifacts_prebuilt": True,
    }
    build_file = _file(tmp_path / "build.json", canonical(build))
    return {
        "execution_kind": "installed_production",
        "source_commit": "a" * 40,
        "build_source_commit": "a" * 40,
        "library_source_commit": "a" * 40,
        "library": record,
        "artifact": record,
        "adapter": record,
        "build_record": build_file,
        "conditions": conditions,
        "conditions_sha256": conditions["sha256"],
        "hardware": {
            "device": "RTX 5090",
            "sm": 120,
            "driver": "pinned",
            "toolchain": "pinned",
            "build_profile": "pinned",
            "device_uuid": "pinned",
        },
    }


def _run_record(tmp_path: Path, contract: dict, row: dict, campaign: dict) -> dict:
    case = contract["cases"][row["case"]]
    raw = _file(tmp_path / "oracle.raw")
    check = {"status": "pass", "raw": raw}
    return {
        "status": "pass",
        "row_id": row["id"],
        "contract_sha256": contract["contract_sha256"],
        "source_commit": campaign["source_commit"],
        "library_sha256": campaign["library"]["sha256"],
        "artifact_sha256": campaign["artifact"]["sha256"],
        "input_sha256": case["input_sha256"],
        "basis_pack_sha256": contract["basis"]["basis_pack_sha256"],
        "grid_identity": case["grid_identity"],
        "method_version": contract["model"]["methods"][row["method"]]["version"],
        "ao_count": case["ao_count_spherical"],
        "atom_count": case["atom_count"],
        "backend": "cuda",
        "provider": "native-dft",
        "j_k": "direct",
        "spin": row["spin"],
        "charge": case["charge"],
        "multiplicity": case["multiplicity"],
        "basis_representation": "real_spherical",
        "auxiliary": None,
        "ecp": None,
        "periodic": False,
        "attained_state": "state-1",
        "reference_attained_state": "state-1",
        "schedule_identity": "schedule-1",
        "scf": {
            "converged": True,
            "energy_tolerance_eh": 1e-10,
            "density_tolerance": 1e-8,
            "max_iterations": 100,
            "physical_residual": 1e-9,
            "screening_thresholds": {"direct_eri": 1e-12},
            "final_forces_state": "state-1",
        },
        "energy_eh": -1.0,
        "forces_eh_per_bohr": [[0.0, 0.0, 0.0] for _ in range(case["atom_count"])],
        "independent_oracle": {
            "provider": "other-implementation",
            "raw": raw,
            "source_sha256": "c" * 64,
            "method_version": contract["model"]["methods"][row["method"]]["version"],
            "basis_pack_sha256": contract["basis"]["basis_pack_sha256"],
            "total_energy_error_eh": 1e-9,
            "energy_gate_eh": 1e-8,
            "force_component_max_error_eh_per_bohr": 1e-8,
            "force_gate_eh_per_bohr": 1e-7,
        },
        "checks": {
            **{
                name: copy.deepcopy(check)
                for name in (
                    "grid_convergence",
                    "finite_difference",
                    "changed_geometry",
                    "warm_replay",
                    "batch_isolation",
                    "failure_recovery",
                )
            }
        },
        "online_scientific_cuda_compiles": 0,
        "precision": {
            "requested_mode": "auto",
            "operator_inventory_complete": True,
            "native_provenance": {
                "mixed_stage_fock_builds": 2,
                "strict_stage_fock_builds": 2,
                "post_scf_fock_builds": 1,
                "refinement_iterations": 2,
                "execution_retries": 0,
                "final_residual_audits": 1,
                "strict_refinement_applied": True,
                "operator_work_counters_valid": True,
            },
            "scf_fock_timeline": [
                {
                    "kind": kind,
                    "count": 1,
                    "sequence": index,
                    "iteration": index,
                    "state": "state-1",
                    "phase": "refinement" if kind == "strict_fock" else "scf",
                }
                for index, kind in enumerate(
                    (
                        "mixed_fock",
                        "mixed_fock",
                        "strict_fock",
                        "strict_fock",
                        "post_scf_fock",
                        "final_audit",
                    )
                )
            ],
            "conversion_count": 0,
            "fallback_count": 0,
            "operators": [
                {
                    "name": "J",
                    "count": 2,
                    "storage": "fp64",
                    "compute": "fp32",
                    "accumulation": "fp64",
                    "reduction": "fp64",
                    "arithmetic_mode": "mixed",
                }
            ],
        },
        "strict_comparator": {
            "source_commit": campaign["source_commit"],
            "state": "state-1",
            "library_sha256": campaign["library"]["sha256"],
            "schedule_identity": "schedule-1",
            "physical_residual": 1e-9,
            "energy_tolerance_eh": 1e-10,
            "density_tolerance": 1e-8,
            "max_iterations": 100,
            "screening_thresholds": {"direct_eri": 1e-12},
        },
        "mixed_vs_strict": {
            "total_energy_abs_eh": 1e-8,
            "force_component_max_eh_per_bohr": 1e-7,
            "matched_model_and_grid": True,
        },
    }


def test_mixed_numerical_oracle_and_work_negative_controls(
    tmp_path: Path, contract: dict
) -> None:
    row = next(r for r in contract["rows"] if r["id"] == "pbe/rks/water/mixed_correct")
    campaign = _campaign(tmp_path)
    value = _run_record(tmp_path, contract, row, campaign)
    value["checks"]["finite_difference"].update(
        step_bohr=[0.01, 0.005], reconverged_each_displacement=True
    )
    value["checks"]["grid_convergence"]["independent_finer_grid"] = True
    value["checks"]["changed_geometry"].update(
        input_sha256=contract["cases"]["water"]["changed_input_sha256"],
        grid_identity=contract["cases"]["water"]["changed_grid_identity"],
        complete_energy_forces=True,
    )
    _check_run(value, campaign, row, contract, tmp_path)

    def late_mixed(record: dict) -> None:
        events = record["precision"]["scf_fock_timeline"]
        events[1].update(kind="strict_fock", phase="refinement")
        events[3].update(kind="mixed_fock", phase="scf")

    mutations = [
        (
            "zero actual mixed work",
            lambda x: x["precision"]["native_provenance"].update(
                mixed_stage_fock_builds=0
            ),
        ),
        ("audit before Fock", lambda x: x["precision"]["scf_fock_timeline"].reverse()),
        ("mixed after refinement", late_mixed),
        (
            "missing refinement event",
            lambda x: x["precision"]["scf_fock_timeline"][2].update(phase="scf"),
        ),
        ("different SCF root", lambda x: x.update(reference_attained_state="state-2")),
        ("wrong spin", lambda x: x.update(spin="uks")),
        (
            "wrong screening",
            lambda x: x["scf"].update(screening_thresholds={"direct_eri": 1e-9}),
        ),
        (
            "strict screening drift",
            lambda x: x["strict_comparator"].update(
                screening_thresholds={"direct_eri": 1e-9}
            ),
        ),
        ("shared oracle", lambda x: x["independent_oracle"].update(provider="vibeqc")),
        (
            "force error",
            lambda x: x["mixed_vs_strict"].update(force_component_max_eh_per_bohr=1e-5),
        ),
        (
            "one-step FD",
            lambda x: x["checks"]["finite_difference"].update(step_bohr=[0.01]),
        ),
        (
            "shared coarse grid",
            lambda x: x["checks"]["grid_convergence"].update(
                independent_finer_grid=False
            ),
        ),
        (
            "fake raw oracle",
            lambda x: x["independent_oracle"]["raw"].update(sha256="0" * 64),
        ),
        ("different source", lambda x: x.update(source_commit="d" * 40)),
    ]
    for label, mutate in mutations:
        altered = copy.deepcopy(value)
        mutate(altered)
        with pytest.raises(InvalidEvidence, match=".+"):
            _check_run(altered, campaign, row, contract, tmp_path)


@pytest.mark.parametrize(
    "provider",
    [None, "", False, 7, [], {}, "vibeqc", "native-dft"],
)
def test_oracle_provider_is_a_nonempty_independent_string(
    tmp_path: Path, contract: dict, provider: object
) -> None:
    row = next(r for r in contract["rows"] if r["id"] == "pbe/rks/water/mixed_correct")
    campaign = _campaign(tmp_path)
    value = _run_record(tmp_path, contract, row, campaign)
    value["checks"]["finite_difference"].update(
        step_bohr=[0.01, 0.005], reconverged_each_displacement=True
    )
    value["checks"]["grid_convergence"]["independent_finer_grid"] = True
    value["checks"]["changed_geometry"].update(
        input_sha256=contract["cases"]["water"]["changed_input_sha256"],
        grid_identity=contract["cases"]["water"]["changed_grid_identity"],
        complete_energy_forces=True,
    )
    value["independent_oracle"]["provider"] = provider

    with pytest.raises(InvalidEvidence, match="oracle provider"):
        _check_run(value, campaign, row, contract, tmp_path)


def test_schemas_encode_oracle_and_review_identity_constraints() -> None:
    result_schema = json.loads(
        (ROOT / "result.schema.json").read_text(encoding="utf-8")
    )
    provider_schema = result_schema["properties"]["independent_oracle"]["properties"][
        "provider"
    ]
    assert provider_schema["type"] == "string"
    assert provider_schema["minLength"] == 1
    assert set(provider_schema["not"]["enum"]) == {"vibeqc", "native-dft"}

    receipt_schema = json.loads(
        (ROOT / "receipt.schema.json").read_text(encoding="utf-8")
    )
    assert (
        receipt_schema["properties"]["final_acceptance"]["properties"]["review_url"][
            "pattern"
        ]
        == contract_validator.FINAL_REVIEW_URL_PATTERN
    )


@pytest.mark.parametrize(
    ("residual", "accepted"),
    [
        (0.0, True),
        (1e-9, True),
        (1e-8, True),
        (-1.0, False),
        (float("-inf"), False),
        (False, False),
        (True, False),
        (None, False),
        ("0", False),
        (float("nan"), False),
        (float("inf"), False),
        (2e-8, False),
    ],
)
def test_strict_comparator_residual_uses_finite_nonnegative_norm(
    tmp_path: Path, contract: dict, residual: object, accepted: bool
) -> None:
    row = next(r for r in contract["rows"] if r["id"] == "pbe/rks/water/mixed_correct")
    campaign = _campaign(tmp_path)
    value = _run_record(tmp_path, contract, row, campaign)
    value["checks"]["finite_difference"].update(
        step_bohr=[0.01, 0.005], reconverged_each_displacement=True
    )
    value["checks"]["grid_convergence"]["independent_finer_grid"] = True
    value["checks"]["changed_geometry"].update(
        input_sha256=contract["cases"]["water"]["changed_input_sha256"],
        grid_identity=contract["cases"]["water"]["changed_grid_identity"],
        complete_energy_forces=True,
    )
    value["strict_comparator"]["physical_residual"] = residual
    if accepted:
        _check_run(value, campaign, row, contract, tmp_path)
    else:
        with pytest.raises(InvalidEvidence, match="strict comparator"):
            _check_run(value, campaign, row, contract, tmp_path)


def test_missing_timeout_and_fake_pass_cannot_complete(
    tmp_path: Path, contract: dict
) -> None:
    campaign = _campaign(tmp_path)
    rows = [
        {"id": row["id"], "status": "not-run", "reason": "no allocated GPU run"}
        for row in contract["rows"]
    ]
    receipt = {
        "schema_version": 1,
        "contract_sha256": contract["contract_sha256"],
        "campaign": campaign,
        "rows": rows,
    }
    path = tmp_path / "receipt.json"
    path.write_bytes(canonical(receipt))
    result = audit(path)
    assert result["product_status"] == "BLOCKED" and result["passed_rows"] == 0
    assert result["required_rows"] == 115 and result["optional_rows"] == 2
    assert len(result["optional_findings"]) == 2
    assert not any("b3lyp/uks/o2" in error for error in result["failures"])
    receipt["rows"][0] = {
        "id": rows[0]["id"],
        "status": "timed-out",
        "reason": "watchdog",
    }
    path.write_bytes(canonical(receipt))
    assert audit(path)["product_status"] == "BLOCKED"
    receipt["rows"][0] = {
        "id": rows[0]["id"],
        "status": "pass",
        "evidence": _file(tmp_path / "fake.json", b"{}"),
    }
    path.write_bytes(canonical(receipt))
    assert any("invalid pass" in error for error in audit(path)["failures"])
    receipt["rows"][0] = rows[0]
    optional = next(
        index for index, row in enumerate(contract["rows"]) if not row["required"]
    )
    receipt["rows"][optional] = {
        "id": rows[optional]["id"],
        "status": "pass",
        "evidence": _file(tmp_path / "fake_optional.json", b"{}"),
    }
    path.write_bytes(canonical(receipt))
    assert any(
        "b3lyp/uks/o2" in error and "invalid pass" in error
        for error in audit(path)["failures"]
    )
    receipt["rows"].pop()
    path.write_bytes(canonical(receipt))
    with pytest.raises(InvalidEvidence, match="missing or extra"):
        audit(path)


def test_runner_retains_explicit_unrun_rows_and_raw_journal(
    tmp_path: Path, contract: dict
) -> None:
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        'import json\nprint(json.dumps({"status":"unsupported","reason":"no production capability"}))\n',
        encoding="utf-8",
    )
    campaign = _campaign(tmp_path)
    campaign["adapter"] = {"path": adapter.name, "sha256": digest(adapter.read_bytes())}
    plan = {
        "adapter_command": [sys.executable, str(adapter)],
        "adapter_command_file_index": 1,
        "timeout_seconds": 10,
        "campaign": campaign,
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(canonical(plan))
    first = contract["rows"][0]["id"]
    result = capture(plan_path, tmp_path / "out", {first})
    receipt = json.loads(result.read_text(encoding="utf-8"))
    assert len(receipt["rows"]) == len(contract["rows"])
    assert receipt["rows"][0]["status"] == "unsupported"
    assert receipt["rows"][0]["capture"]["progress"]["sha256"] == digest(b"")
    assert all(row["status"] == "not-run" for row in receipt["rows"][1:])
    assert (result.parent / "progress.jsonl").read_text(encoding="utf-8").count(
        "\n"
    ) == 2
    assert capture(plan_path, tmp_path / "out", {first}) == result
    assert (result.parent / "progress.jsonl").read_text(encoding="utf-8").count(
        "\n"
    ) == 2
    interrupted = json.loads(result.read_text(encoding="utf-8"))
    interrupted["rows"][0] = {"id": first, "status": "running", "reason": "interrupted"}
    result.write_bytes(canonical(interrupted))
    partial = result.parent / f"{first.replace('/', '__')}.progress.jsonl"
    partial.write_text('{"sample":1}\n', encoding="utf-8")
    with pytest.raises(InvalidEvidence, match="running adapter"):
        capture(plan_path, tmp_path / "out", {first})
    resumed = json.loads(result.read_text(encoding="utf-8"))
    assert resumed["rows"][0]["status"] == "running"
    assert partial.read_text(encoding="utf-8").strip() == '{"sample":1}'
    assert (result.parent / "progress.jsonl").read_text(encoding="utf-8").count(
        "\n"
    ) == 2
    second = contract["rows"][1]["id"]
    interrupted = receipt
    interrupted["rows"][1] = {
        "id": second,
        "status": "running",
        "reason": "zero-progress interruption",
    }
    result.write_bytes(canonical(interrupted))
    for suffix in ("stdout", "stderr", "progress.jsonl"):
        (result.parent / f"{second.replace('/', '__')}.{suffix}").write_bytes(b"")
    with pytest.raises(InvalidEvidence, match="running adapter"):
        capture(plan_path, tmp_path / "out", {second})
    resumed = json.loads(result.read_text(encoding="utf-8"))
    assert resumed["rows"][1]["status"] == "running"


def test_runner_retains_non_object_json_failure(tmp_path: Path, contract: dict) -> None:
    adapter = tmp_path / "adapter.py"
    adapter.write_text('print("[]")\n', encoding="utf-8")
    campaign = _campaign(tmp_path)
    campaign["adapter"] = {"path": adapter.name, "sha256": digest(adapter.read_bytes())}
    plan = {
        "adapter_command": [sys.executable, str(adapter)],
        "adapter_command_file_index": 1,
        "timeout_seconds": 10,
        "campaign": campaign,
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(canonical(plan))
    first = contract["rows"][0]["id"]
    result = capture(plan_path, tmp_path / "out", {first})
    entry = json.loads(result.read_text(encoding="utf-8"))["rows"][0]
    assert entry["status"] == "failed"
    assert "JSON must be an object" in entry["reason"]
    assert entry["capture"]["stdout"]["sha256"] == digest(b"[]\n") or entry["capture"][
        "stdout"
    ]["sha256"] == digest(b"[]\r\n")
    assert (result.parent / "progress.jsonl").read_text(encoding="utf-8").count(
        "\n"
    ) == 2


def test_runner_resolves_relative_output_before_adapter_cwd_change(
    tmp_path: Path,
    contract: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_dir = tmp_path / "plan"
    caller_dir = tmp_path / "caller"
    plan_dir.mkdir()
    caller_dir.mkdir()
    adapter = plan_dir / "adapter.py"
    adapter.write_text(
        "import json, pathlib, sys\n"
        "progress = pathlib.Path(sys.argv[sys.argv.index('--progress') + 1])\n"
        "progress.write_text(json.dumps({'attempt_id': 0}) + '\\n')\n"
        "print(json.dumps({'status': 'unsupported', 'reason': 'fixture'}))\n",
        encoding="utf-8",
    )
    campaign = _campaign(plan_dir)
    campaign["adapter"] = {"path": adapter.name, "sha256": digest(adapter.read_bytes())}
    plan = {
        "adapter_command": [sys.executable, str(adapter)],
        "adapter_command_file_index": 1,
        "timeout_seconds": 10,
        "campaign": campaign,
    }
    plan_path = plan_dir / "plan.json"
    plan_path.write_bytes(canonical(plan))
    first = contract["rows"][0]["id"]
    monkeypatch.chdir(caller_dir)

    receipt_path = capture(plan_path, type(plan_path)("results"), {first})

    assert receipt_path == caller_dir / "results" / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["rows"][0]["status"] == "unsupported"
    progress = receipt_path.parent / receipt["rows"][0]["capture"]["progress"]["path"]
    assert json.loads(progress.read_text(encoding="utf-8"))["attempt_id"] == 0


def test_timeout_terminates_adapter_descendants(tmp_path: Path, contract: dict) -> None:
    marker = tmp_path / "leaked-child.txt"
    child = (
        "import pathlib,time; time.sleep(2); "
        f"pathlib.Path({str(marker)!r}).write_text('leaked')"
    )
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        "import pathlib,subprocess,sys,time\n"
        f"child=subprocess.Popen([sys.executable,'-c',{child!r}])\n"
        "progress=pathlib.Path(sys.argv[sys.argv.index('--progress')+1])\n"
        "progress.write_text(str(child.pid)+'\\n')\n"
        "time.sleep(10)\n",
        encoding="utf-8",
    )
    campaign = _campaign(tmp_path)
    campaign["adapter"] = {"path": adapter.name, "sha256": digest(adapter.read_bytes())}
    plan = {
        "adapter_command": [sys.executable, str(adapter)],
        "adapter_command_file_index": 1,
        "timeout_seconds": 1,
        "campaign": campaign,
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(canonical(plan))
    first = contract["rows"][0]["id"]
    result = capture(plan_path, tmp_path / "out", {first})
    entry = json.loads(result.read_text(encoding="utf-8"))["rows"][0]
    assert entry["status"] == "timed-out"
    progress = result.parent / entry["capture"]["progress"]["path"]
    assert progress.read_text(encoding="utf-8").strip().isdigit()
    time.sleep(2.5)
    assert not marker.exists()


def test_cold_gain_cannot_mask_changed_geometry_loss(contract: dict) -> None:
    records = {}
    for method in ("pbe", "r2scan", "pbe0"):
        for case in contract["gates"]["performance_cases"]:
            records[f"{method}/rks/{case}/promoted"] = {
                "timing": {
                    "cold": [
                        {"strict": {"total_ms": 125.0}, "mixed": {"total_ms": 100.0}}
                        for _ in range(5)
                    ],
                    "warm": [
                        {"strict": {"total_ms": 100.0}, "mixed": {"total_ms": 100.0}}
                        for _ in range(5)
                    ],
                    "changed_geometry": [
                        {"strict": {"total_ms": 90.0}, "mixed": {"total_ms": 100.0}}
                        for _ in range(5)
                    ],
                }
            }
    errors = _performance_gate(records, contract)
    assert any("changed_geometry" in error for error in errors)
    assert not any("cold" in error for error in errors)
    for record in records.values():
        for pair in record["timing"]["changed_geometry"]:
            pair["strict"]["total_ms"] = 125.0
        for pair in record["timing"]["warm"]:
            pair["strict"]["total_ms"] = 90.0
    warm_errors = _performance_gate(records, contract)
    assert any("warm" in error for error in warm_errors)
    assert not any("changed_geometry" in error for error in warm_errors)


def test_promoted_row_rejects_unreported_failed_attempt(
    tmp_path: Path, contract: dict
) -> None:
    row = next(r for r in contract["rows"] if r["id"] == "pbe/rks/water8/promoted")
    campaign = _campaign(tmp_path)
    value = _run_record(tmp_path, contract, row, campaign)
    raw = value["independent_oracle"]["raw"]
    value["checks"]["finite_difference"].update(
        step_bohr=[0.01, 0.005], reconverged_each_displacement=True
    )
    value["checks"]["grid_convergence"]["independent_finer_grid"] = True
    value["checks"]["changed_geometry"].update(
        input_sha256=contract["cases"]["water8"]["changed_input_sha256"],
        grid_identity=contract["cases"]["water8"]["changed_grid_identity"],
        complete_energy_forces=True,
    )
    value["checks"]["batch_throughput"] = {
        "status": "pass",
        "raw": raw,
        "batch_size": 4,
        "complete_energy_forces": True,
    }
    value["checks"]["strict_fallback"] = {"status": "pass", "raw": raw}
    value["profiled_runs_separate"] = True
    value["peak_simultaneous_bytes"] = 1024
    value["ablations"] = {
        name: {"raw": raw, "cold_ms": 100.0, "changed_geometry_ms": 100.0}
        for name in contract["required_ablations"]
    }
    components = {
        "prepare": 0.0,
        "scf": 80.0,
        "final_verification": 0.0,
        "forces": 20.0,
        "transfers": 0.0,
        "synchronization": 0.0,
        "selection": 0.0,
        "conversion": 0.0,
        "other": 0.0,
    }
    value["timing"] = {
        boundary: [
            {
                "order": ["strict", "mixed"] if index % 2 == 0 else ["mixed", "strict"],
                "unprofiled": True,
                "complete_endpoint": True,
                "conditions_sha256": campaign["conditions_sha256"],
                "strict": {"total_ms": 100.0, "components_ms": components},
                "mixed": {"total_ms": 100.0, "components_ms": components},
            }
            for index in range(5)
        ]
        for boundary in ("cold", "warm", "changed_geometry")
    }
    attempts = [
        {
            "attempt_id": attempt_id,
            "boundary": boundary,
            "pair_index": index,
            "route": route,
            "status": "pass",
            "conditions_sha256": campaign["conditions_sha256"],
            "total_ms": 100.0,
        }
        for attempt_id, (boundary, index, route) in enumerate(
            (boundary, index, route)
            for boundary in ("cold", "warm", "changed_geometry")
            for index in range(5)
            for route in value["timing"][boundary][index]["order"]
        )
    ]
    progress = tmp_path / "attempts.jsonl"
    progress.write_bytes(b"\n".join(canonical(attempt) for attempt in attempts) + b"\n")
    progress_record = {"path": progress.name, "sha256": digest(progress.read_bytes())}
    capture_record = {"stdout": raw, "stderr": raw, "progress": progress_record}
    value["attempt_ledger"] = progress_record
    _check_run(value, campaign, row, contract, tmp_path, capture_record)
    attempts.append(
        {
            "attempt_id": 30,
            "boundary": "cold",
            "pair_index": 5,
            "route": "mixed",
            "status": "failed",
        }
    )
    progress.write_bytes(b"\n".join(canonical(attempt) for attempt in attempts) + b"\n")
    capture_record["progress"] = {
        "path": progress.name,
        "sha256": digest(progress.read_bytes()),
    }
    value["attempt_ledger"] = capture_record["progress"]
    with pytest.raises(InvalidEvidence, match="timing attempts"):
        _check_run(value, campaign, row, contract, tmp_path, capture_record)


def test_optional_nonpass_never_replaces_required_final_row(
    tmp_path: Path, contract: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(contract_validator, "_check_run", lambda *_: None)
    monkeypatch.setattr(contract_validator, "_performance_gate", lambda *_: [])
    monkeypatch.setattr(contract_validator, "_official_master_oid", lambda *_: "e" * 40)
    monkeypatch.setattr(contract_validator, "_git_is_ancestor", lambda *_: True)

    def git_result(argv: list[str], **_: object) -> object:
        output = (
            canonical({"contract_sha256": contract["contract_sha256"]})
            if argv[1] == "show"
            else b""
        )
        return type("Done", (), {"returncode": 0, "stdout": output})()

    monkeypatch.setattr(contract_validator.subprocess, "run", git_result)
    campaign = _campaign(tmp_path)
    evidence = _file(tmp_path / "mock_only.json", b'{"status":"pass"}\n')
    mock_capture = {
        "stdout": evidence,
        "stderr": _file(tmp_path / "mock_stderr.txt", b""),
        "progress": _file(tmp_path / "mock_progress.jsonl", b""),
    }
    raw_final = _file(
        tmp_path / "mock_final.json",
        canonical(
            {
                "status": "PASS",
                "source_commit": campaign["source_commit"],
                "contract_sha256": contract["contract_sha256"],
            }
        ),
    )
    rows = [
        {
            "id": row["id"],
            "status": "pass",
            "evidence": evidence,
            "capture": mock_capture,
        }
        if row["required"]
        else {"id": row["id"], "status": "not-run", "reason": "optional stress case"}
        for row in contract["rows"]
    ]
    receipt = {
        "schema_version": 1,
        "contract_sha256": contract["contract_sha256"],
        "campaign": campaign,
        "rows": rows,
        "final_acceptance": {
            "issue": 1190,
            "status": "PASS",
            "source_commit": campaign["source_commit"],
            "review_url": "https://github.com/jinzhezenggroup/vibeqc/issues/1190#issuecomment-123",
            "raw_receipt": raw_final,
        },
    }
    path = tmp_path / "receipt.json"
    path.write_bytes(canonical(receipt))
    outcome = audit(path, final=True)
    assert outcome["product_status"] == "PASS"
    assert outcome["passed_required_rows"] == 115
    assert len(outcome["optional_findings"]) == 2
    valid_review_url = receipt["final_acceptance"]["review_url"]
    for invalid_review_url in (
        "https://github.com/jinzhezenggroup/vibeqc/README.md",
        "https://github.com/jinzhezenggroup/vibeqc/issues/1191#issuecomment-123",
        "https://github.com/jinzhezenggroup/vibeqc/issues/1190",
        "https://github.com/jinzhezenggroup/vibeqc/issues/1190#issuecomment-not-a-number",
    ):
        receipt["final_acceptance"]["review_url"] = invalid_review_url
        path.write_bytes(canonical(receipt))
        with pytest.raises(InvalidEvidence, match="#1190 review URL"):
            audit(path, final=True)
    receipt["final_acceptance"]["review_url"] = valid_review_url
    path.write_bytes(canonical(receipt))
    optional_indices = [
        index for index, row in enumerate(contract["rows"]) if not row["required"]
    ]
    for index, status in zip(optional_indices, ("unsupported", "failed"), strict=True):
        receipt["rows"][index] = {
            "id": contract["rows"][index]["id"],
            "status": status,
            "reason": "optional stress case",
        }
    path.write_bytes(canonical(receipt))
    outcome = audit(path, final=True)
    assert outcome["product_status"] == "PASS"
    assert len(outcome["optional_findings"]) == 2
    receipt["rows"][optional_indices[0]]["status"] = "running"
    path.write_bytes(canonical(receipt))
    assert audit(path, final=True)["product_status"] == "BLOCKED"
    receipt["rows"][optional_indices[0]]["status"] = "unsupported"
    path.write_bytes(canonical(receipt))
    monkeypatch.setattr(contract_validator, "_official_master_oid", lambda *_: None)
    assert audit(path, final=True)["product_status"] == "BLOCKED"
    monkeypatch.setattr(contract_validator, "_official_master_oid", lambda *_: "e" * 40)
    receipt["rows"][0] = {
        "id": contract["rows"][0]["id"],
        "status": "not-run",
        "reason": "mandatory capability absent",
    }
    path.write_bytes(canonical(receipt))
    assert audit(path, final=True)["product_status"] == "BLOCKED"


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=repository, text=True
    ).strip()


@pytest.mark.parametrize(
    ("advertisement", "expected"),
    (
        (b"a" * 40 + b"\trefs/heads/master\n", "a" * 40),
        (b"", None),
        (b"a" * 39 + b"\trefs/heads/master\n", None),
        (b"g" * 40 + b"\trefs/heads/master\n", None),
        (b"a" * 40 + b"\trefs/heads/main\n", None),
        (
            b"a" * 40 + b"\trefs/heads/master\n" + b"b" * 40 + b"\trefs/heads/master\n",
            None,
        ),
    ),
)
def test_official_master_advertisement_is_exact(
    advertisement: bytes, expected: str | None
) -> None:
    assert contract_validator._parse_official_master_oid(advertisement) == expected


def test_official_master_fetch_failure_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oid = "a" * 40

    def failed_fetch(argv: list[str], **_: object) -> object:
        if argv[1] == "ls-remote":
            return type(
                "Done",
                (),
                {
                    "returncode": 0,
                    "stdout": f"{oid}\trefs/heads/master\n".encode("ascii"),
                },
            )()
        return type("Done", (), {"returncode": 1, "stdout": b""})()

    monkeypatch.setattr(contract_validator.subprocess, "run", failed_fetch)
    assert contract_validator._official_master_oid(tmp_path) is None


def test_official_upstream_rejects_fork_only_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    official = tmp_path / "official.git"
    seed = tmp_path / "seed"
    fork = tmp_path / "fork.git"
    checkout = tmp_path / "checkout"
    subprocess.run(
        ["git", "init", "--bare", str(official)], check=True, capture_output=True
    )
    subprocess.run(
        ["git", "init", "-b", "master", str(seed)], check=True, capture_output=True
    )
    _git(seed, "config", "user.name", "fixture")
    _git(seed, "config", "user.email", "fixture@example.invalid")
    (seed / "base.txt").write_text("official\n", encoding="utf-8")
    _git(seed, "add", "base.txt")
    _git(seed, "commit", "-m", "official base")
    official_base = _git(seed, "rev-parse", "HEAD")
    _git(seed, "remote", "add", "official", str(official))
    _git(seed, "push", "official", "master")
    subprocess.run(
        ["git", "clone", "--bare", str(official), str(fork)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "clone", str(fork), str(checkout)],
        check=True,
        capture_output=True,
    )
    _git(checkout, "config", "user.name", "fixture")
    _git(checkout, "config", "user.email", "fixture@example.invalid")
    (checkout / "fork-only.txt").write_text("fork\n", encoding="utf-8")
    _git(checkout, "add", "fork-only.txt")
    _git(checkout, "commit", "-m", "fork only")
    fork_only = _git(checkout, "rev-parse", "HEAD")
    _git(checkout, "push", "origin", "master")
    assert (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", fork_only, "origin/master"],
            cwd=checkout,
            check=False,
        ).returncode
        == 0
    )
    fetch_head = checkout / ".git" / "FETCH_HEAD"
    fetch_head.write_text("sentinel\n", encoding="utf-8")
    monkeypatch.setattr(contract_validator, "OFFICIAL_UPSTREAM_URL", str(official))

    official_oid = contract_validator._official_master_oid(checkout)

    assert official_oid == official_base
    assert contract_validator._git_is_ancestor(checkout, official_base, official_oid)
    assert not contract_validator._git_is_ancestor(checkout, fork_only, official_oid)
    assert fetch_head.read_text(encoding="utf-8") == "sentinel\n"
