"""Independent fixtures and deliberate corruption of scientific evidence."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tools.vibeqc_validation.capabilities import capability_table
from tools.vibeqc_validation.fixtures import (
    REFERENCE_DIRECTORY,
    calculator_inputs,
    load_fixtures,
    mathematical_hash,
    molecular_inputs,
    small_inputs,
    validate_fixture,
)
from tools.vibeqc_validation.integrals import evaluate_quartet
from tools.vibeqc_validation.performance import assess_comparison, measure_interleaved
from tools.vibeqc_validation.schema import (
    GATES,
    attach_artifact,
    block_error,
    canonical_hash,
    finite_difference,
    new_evidence,
    outcome,
    validate_evidence,
    write_evidence,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = load_fixtures()


def test_mathematical_inputs_reproduce_and_do_not_depend_on_provenance():
    expected = {r["inputs"]["name"]: r["inputs_hash"] for r in FIXTURES}
    for inputs in small_inputs() + molecular_inputs():
        assert mathematical_hash(inputs) == expected[inputs["name"]]
        assert (
            mathematical_hash(dict(reversed(list(inputs.items()))))
            == expected[inputs["name"]]
        )
    assert small_inputs() == small_inputs()
    assert small_inputs(139) != small_inputs(138)
    renamed = {
        **small_inputs()[0],
        "name": "renamed",
        "seed": 999,
        "basis_name": "alias",
    }
    assert mathematical_hash(renamed) == mathematical_hash(small_inputs()[0])
    assert mathematical_hash({"coordinates": [0.0, 1.0]}) == mathematical_hash(
        {"coordinates": [-0.0, 1]}
    )
    modified = small_inputs()
    modified[0]["conventions"]["force_sign"] = "corrupted"
    assert small_inputs()[0]["conventions"]["force_sign"] != "corrupted"


def test_two_generations_have_measured_stability_and_distinct_provenance():
    stability = json.loads((REFERENCE_DIRECTORY / "stability.json").read_text())
    assert len(stability["generations"]) == len(FIXTURES)
    for row in stability["generations"]:
        assert (
            row["first_provenance"]["generated_at"]
            != row["second_provenance"]["generated_at"]
        )
        assert row["first_record_hash"] != row["second_record_hash"]
        assert row["first_data_hash"] == row["second_data_hash"]
        assert all(e["passed"] for e in row["errors"].values())
        fixture = next(r for r in FIXTURES if r["inputs_hash"] == row["inputs_hash"])
        assert canonical_hash(fixture) == row["second_record_hash"]
        first = copy.deepcopy(fixture)
        first["provenance"] = row["first_provenance"]
        first["provenance_hash"] = canonical_hash(first["provenance"])
        assert canonical_hash(first) == row["first_record_hash"]


@pytest.mark.parametrize(
    "reference",
    [
        r
        for r in FIXTURES
        if r["inputs"]["kind"] == "quartet"
        and r["inputs"]["basis_representation"] == "cartesian"
    ],
    ids=lambda r: r["inputs"]["name"],
)
def test_generator_integrals_and_derivatives_against_independent_libcint(reference):
    actual = evaluate_quartet(reference["inputs"])
    for key in ("eri", "gradient"):
        assert block_error(
            actual[key], reference["data"][key], **GATES["integral_fp64"]
        )["passed"]
    np.testing.assert_allclose(actual["gradient"].sum(axis=0), 0, atol=1e-13, rtol=0)


@pytest.mark.parametrize(
    "reference",
    [r for r in FIXTURES if r["inputs"]["kind"] == "molecule"],
    ids=lambda r: r["inputs"]["name"],
)
def test_small_molecular_cpu_endpoints_against_pinned_pyscf(reference):
    from vibeqc import Calculator

    inputs = reference["inputs"]
    result = Calculator(device="cpu", **calculator_inputs(inputs)).singlepoint(
        list(zip(inputs["atomic_numbers"], inputs["coordinates"], strict=True)),
        charge=inputs["charge"],
        multiplicity=inputs["multiplicity"],
    )
    assert result.executed_backend == "cpu_reference"
    assert block_error(result.energy, reference["data"]["energy"], atol=1e-10, rtol=0)[
        "passed"
    ]
    assert block_error(result.forces, reference["data"]["forces"], atol=1e-7, rtol=0)[
        "passed"
    ]


def test_triples_reference_and_occupations_are_nontrivial_and_explicit():
    row = next(r for r in FIXTURES if r["inputs"]["name"] == "nh3")
    assert abs(row["data"]["ccsd_t"]["triples_energy"]) > 1e-7
    assert row["data"]["ccsd_t"]["amplitude_update_max"] < 1e-9
    assert sum(row["data"]["mo_occ"]) == 10
    assert row["inputs"]["cc_settings"]["frozen_core"] == 0
    assert row["data"]["scf_residual_max"] < 1e-9


@pytest.mark.parametrize(
    "field,value",
    [
        ("cartesian_order", "reverse"),
        ("force_sign", "force=+dE/dR"),
        ("spherical_order", "reverse"),
    ],
)
def test_corrupted_conventions_rejected_even_with_recomputed_input_hash(field, value):
    row = copy.deepcopy(FIXTURES[0])
    row["inputs"]["conventions"][field] = value
    row["inputs_hash"] = mathematical_hash(row["inputs"])
    with pytest.raises(ValueError, match="conventions"):
        validate_fixture(row)


def test_corrupted_ao_values_and_force_signs_fail_numerical_gates():
    row = next(r for r in FIXTURES if r["inputs"]["name"] == "dpss-asymmetric")
    values = np.asarray(row["data"]["eri"])
    assert not block_error(values[::-1], values, **GATES["integral_fp64"])["passed"]
    row = copy.deepcopy(next(r for r in FIXTURES if r["inputs"]["name"] == "h2"))
    row["data"]["forces"] = (-np.asarray(row["data"]["forces"])).tolist()
    row["data_hash"] = canonical_hash(row["data"])
    with pytest.raises(ValueError, match="force sign"):
        validate_fixture(row)


@pytest.mark.parametrize("corruption", ["reference-version", "program-version", "data"])
def test_reference_version_and_hash_corruption(corruption):
    row = copy.deepcopy(FIXTURES[0])
    if corruption == "reference-version":
        row["schema_version"] += 1
    elif corruption == "program-version":
        row["provenance"]["pyscf"] = "unverified-version"
    else:
        row["data"]["eri"][0][0][0][0] += 0.1
    with pytest.raises(ValueError):
        validate_fixture(row)


def test_near_zero_errors_use_absolute_floor_and_reject_nonfinite_values():
    assert block_error([1e-12], [0], **GATES["integral_fp64"])["passed"]
    assert not block_error([1e-8], [0], **GATES["integral_fp64"])["passed"]
    for bad in (np.nan, np.inf):
        with pytest.raises(ValueError, match="non-finite"):
            block_error([bad], [0], **GATES["integral_fp64"])
    with pytest.raises(ValueError, match="shapes"):
        block_error([1, 2], [1], **GATES["integral_fp64"])


def test_finite_difference_records_every_step_and_freezes_policy():
    seen = []

    def energy(xyz, settings):
        seen.append(settings.copy())
        settings["screening"] = 42
        return float(np.sum(xyz**4))

    coordinates = np.array([[0.3, 0.7, -0.2]])
    result = finite_difference(
        energy, coordinates, 4 * coordinates**3, settings={"screening": 1e-14}
    )
    errors = [r["error"]["max_absolute_error"] for r in result["samples"]]
    assert len(errors) == 3 and errors[0] > errors[1] > errors[2]
    assert all(s == {"screening": 1e-14} for s in seen)
    with pytest.raises(ValueError, match="three"):
        finite_difference(
            energy, coordinates, coordinates, settings={}, steps=(0.1, 0.01)
        )


def _samples(candidate_seconds=0.8):
    counter = iter(range(100))
    rows = measure_interleaved(
        lambda side: {"iterations": 2},
        lambda: None,
        workload="energy-plus-force",
        inputs_hash=canonical_hash({"amplitudes": [1, 2]}),
        clock=lambda: next(counter),
    )
    for row in rows:
        row["seconds"] = 1.0 if row["selection"] == "baseline" else candidate_seconds
    return rows


def test_performance_uses_existing_abba_order_and_synchronizes_every_sample():
    calls = []
    rows = measure_interleaved(
        lambda side: calls.append(side),
        lambda: calls.append("sync"),
        prepare=lambda side: calls.append("prepare"),
        workload="cold-start",
        inputs_hash=canonical_hash({}),
    )
    assert len(rows) == 10
    assert calls[:4] == ["prepare", "sync", "baseline", "sync"]
    assert calls.count("sync") == 20
    assert assess_comparison(_samples())["status"] == "pass"
    assert assess_comparison(_samples(0.99))["status"] == "not-run"
    with pytest.raises(ValueError, match="five"):
        measure_interleaved(
            lambda side: None,
            lambda: None,
            workload="cold-start",
            inputs_hash="a",
            repeats=4,
        )


@pytest.mark.parametrize("corruption", ["inputs", "sync", "order", "count", "noise"])
def test_bad_or_noisy_performance_measurements_cannot_pass(corruption):
    rows = _samples()
    if corruption == "inputs":
        rows[0]["inputs_hash"] = canonical_hash({"different": True})
    elif corruption == "sync":
        rows[0]["synchronized"] = False
    elif corruption == "order":
        rows.sort(key=lambda r: r["selection"])
    elif corruption == "count":
        rows = rows[:8]
    else:
        for i, r in enumerate(rows):
            r["seconds"] *= (0.4, 1.0, 1.6)[i % 3]
    assert assess_comparison(rows)["status"] != "pass"


def _complete_record():
    record = new_evidence(
        tier="endpoint", subject="synthetic kernel test", inputs_hash=canonical_hash({})
    )
    record.update(
        revision="a" * 40,
        device={"name": "test GPU"},
        toolchain={"nvcc": "test"},
        backend_selected="cuda",
        settings={
            "device": "cuda",
            "comparison_kind": "kernel",
            "fast_compile": False,
            "fixed_state_hash": canonical_hash({"amplitudes": [1, 2]}),
            "promotion_limits": {"peak_bytes": 8192, "compile_seconds": 2},
        },
    )
    record["hashes"] = {k: canonical_hash(k) for k in record["hashes"]}
    record["hardware"] = outcome("pass")
    record["stages"] = {k: outcome("pass") for k in record["stages"]}
    record["memory"] = {"allocated_bytes": 4096, "peak_bytes": 8192, "reason": None}
    record["compilation"] = {"seconds": 2, "reason": None}
    record["block_errors"] = {"eri": block_error([1], [1], **GATES["integral_fp64"])}
    record["timings"] = _samples()
    record["performance"] = assess_comparison(record["timings"])
    return record


def test_complete_schema_roundtrip_and_existing_artifact_registration(tmp_path):
    record = _complete_record()
    output = tmp_path / "result.json"
    write_evidence(output, record)
    validate_evidence(json.loads(output.read_text()))
    artifact = tmp_path / "autotune.json"
    artifact.write_text(
        json.dumps({"schema_version": 1, "candidates": [], "winners": []})
    )
    attach_artifact(record, artifact, kind="autotune")
    assert record["attachments"][0]["sha256"]


@pytest.mark.parametrize(
    "field",
    [
        "revision",
        "source",
        "ir",
        "equation",
        "schedule",
        "device",
        "toolchain",
        "memory",
        "compilation",
        "timings",
        "numerical",
    ],
)
def test_performance_pass_rejects_missing_provenance_or_evidence(field):
    record = _complete_record()
    if field in record["hashes"]:
        record["hashes"][field] = None
    elif field == "memory":
        record[field]["peak_bytes"] = None
    elif field == "compilation":
        record[field]["seconds"] = None
    elif field == "numerical":
        record["stages"][field] = outcome("not-run", "no reference")
    else:
        record[field] = [] if field == "timings" else None
    with pytest.raises(ValueError):
        validate_evidence(record)


def test_missing_gpu_status_and_backend_fallback_cannot_promote():
    for corrupt in ("hardware", "backend"):
        record = _complete_record()
        if corrupt == "hardware":
            record["hardware"] = outcome("not-run", "GPU unavailable")
        else:
            record["backend_selected"] = "cpu_reference"
        with pytest.raises(ValueError, match="GPU"):
            validate_evidence(record)
    with pytest.raises(ValueError, match="reason"):
        outcome("not-run")
    record = _complete_record()
    record["schema_version"] = 999
    with pytest.raises(ValueError, match="version"):
        validate_evidence(record)


def test_solver_pass_requires_iteration_history_and_residual():
    record = _complete_record()
    record["settings"]["comparison_kind"] = "solver"
    with pytest.raises(ValueError, match="iteration"):
        validate_evidence(record)


def test_solver_history_cannot_omit_intermediate_iterations_or_final_residuals():
    record = _complete_record()
    record["settings"].update(comparison_kind="solver", residual_limit=1e-9)
    record["solver_iterations"] = [
        {"sample_index": i, "iteration": iteration, "energy": -1.0, "residual": 1e-10}
        for i in range(10)
        for iteration in (1, 2)
    ]
    record["residuals"] = {
        str(i): {"value": 1e-10, "independently_evaluated": True} for i in range(10)
    }
    validate_evidence(record)
    record["solver_iterations"].pop(0)
    with pytest.raises(ValueError, match="every iteration"):
        validate_evidence(record)


@pytest.mark.parametrize("field,value", [("peak_bytes", 8000), ("compile_seconds", 1)])
def test_runtime_win_cannot_override_memory_or_compile_budget(field, value):
    record = _complete_record()
    record["settings"]["promotion_limits"][field] = value
    with pytest.raises(ValueError, match="budget"):
        validate_evidence(record)


def test_kernel_performance_requires_frozen_state_identity():
    record = _complete_record()
    record["settings"].pop("fixed_state_hash")
    with pytest.raises(ValueError, match="fixed density"):
        validate_evidence(record)


def test_fast_compile_and_hidden_gpu_selection_cannot_promote():
    record = _complete_record()
    record["settings"]["fast_compile"] = True
    with pytest.raises(ValueError, match="fast_compile"):
        validate_evidence(record)
    record = _complete_record()
    record["settings"].pop("device")
    record["hardware"] = outcome("not-run", "GPU absent")
    with pytest.raises(ValueError, match="GPU"):
        validate_evidence(record)


def test_tier_command_timeout_is_a_failed_artifact(tmp_path):
    output = tmp_path / "timeout.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks/validation_gate.py"),
            "run",
            "--tier",
            "cuda-compile",
            "--subject",
            "timeout control",
            "--timeout",
            "0.05",
            "--output",
            str(output),
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(10)",
        ],
        check=False,
    )
    assert result.returncode == 1
    record = json.loads(output.read_text())
    assert record["stages"]["compilation"]["status"] == "fail"
    assert "timeout" in record["stages"]["compilation"]["reason"]


def test_capability_stages_preserve_the_existing_f_shell_matrix():
    table = capability_table()
    assert len(table["shell_classes"]) == 55
    assert sum(r["contains_f"] for r in table["shell_classes"]) == 34
    for row in table["shell_classes"]:
        assert set(row["stages"]) == {
            "representation",
            "source",
            "compilation",
            "numerical",
            "endpoint",
            "production",
        }
        assert row["stages"]["compilation"]["status"] == "not-run"
    fpps = next(r for r in table["shell_classes"] if r["shell_class"] == "fpps")
    assert fpps["stages"]["production"]["acceptance_status"] == "provisional"


def test_hf_protocol_control_runs_all_available_workloads_and_fd(tmp_path):
    from benchmarks.validation_gate import hf_evidence

    record = hf_evidence(check_fd=True)
    validate_evidence(record)
    assert record["stages"]["endpoint"]["status"] == "pass"
    assert len(record["timings"]) == 40
    assert record["workloads"]["energy-only"]["status"] == "not-run"
    assert record["performance"]["status"] == "not-run"
    assert len(record["finite_difference"]["samples"]) == 3
    write_evidence(tmp_path / "hf.json", record)


def test_tier_cli_records_missing_gpu_and_compilation_separately(tmp_path):
    script = ROOT / "benchmarks/validation_gate.py"
    output = tmp_path / "gpu.json"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "run",
            "--tier",
            "gpu-numerical",
            "--subject",
            "fsss",
            "--unavailable",
            "no GPU in this test",
            "--output",
            str(output),
        ],
        check=True,
    )
    record = json.loads(output.read_text())
    assert record["hardware"]["status"] == "not-run"
    assert record["stages"]["numerical"]["status"] == "not-run"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "run",
            "--tier",
            "cuda-compile",
            "--subject",
            "compiler stub",
            "--output",
            str(output),
            "--",
            sys.executable,
            "-c",
            "pass",
        ],
        check=True,
    )
    record = json.loads(output.read_text())
    assert record["stages"]["compilation"]["status"] == "pass"
    assert record["hardware"]["status"] == "not-run"
    assert record["stages"]["numerical"]["status"] == "not-run"
