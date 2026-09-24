"""Fail-closed DFT-MP-v1 contract and source-matched receipt validator.

This validates retained evidence, not the truth of an external measurement.
The final #1190 reviewer must inspect raw logs and independently reproduce it.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import subprocess
from pathlib import Path

from .freeze_contract import REPO, ROOT, build, canonical, digest, source_digest

OFFICIAL_UPSTREAM_URL = "https://github.com/jinzhezenggroup/vibeqc.git"
OFFICIAL_UPSTREAM_MASTER_REF = "refs/heads/master"

STATUSES = {
    "unknown",
    "unsupported",
    "not-run",
    "running",
    "failed",
    "timed-out",
    "pass",
}
TIMINGS = ("cold", "warm", "changed_geometry")
COMPONENTS = {
    "prepare",
    "scf",
    "final_verification",
    "forces",
    "transfers",
    "synchronization",
    "selection",
    "conversion",
    "other",
}


class InvalidEvidence(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidEvidence(message)


def _parse_official_master_oid(advertisement: bytes) -> str | None:
    """Accept exactly one canonical SHA-1 advertisement for upstream master."""

    lines = advertisement.splitlines()
    if len(lines) != 1:
        return None
    fields = lines[0].split(b"\t")
    if len(fields) != 2 or fields[1] != OFFICIAL_UPSTREAM_MASTER_REF.encode("ascii"):
        return None
    try:
        oid = fields[0].decode("ascii")
    except UnicodeDecodeError:
        return None
    if (
        len(oid) != 40
        or oid != oid.lower()
        or any(c not in "0123456789abcdef" for c in oid)
    ):
        return None
    return oid


def _official_master_oid(repository: Path = REPO) -> str | None:
    """Fetch the advertised official master object without updating local refs/FETCH_HEAD."""

    try:
        advertised = subprocess.run(
            [
                "git",
                "ls-remote",
                "--exit-code",
                OFFICIAL_UPSTREAM_URL,
                OFFICIAL_UPSTREAM_MASTER_REF,
            ],
            cwd=repository,
            check=False,
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if advertised.returncode != 0:
        return None
    oid = _parse_official_master_oid(advertised.stdout)
    if oid is None:
        return None
    try:
        fetched = subprocess.run(
            [
                "git",
                "fetch",
                "--quiet",
                "--no-write-fetch-head",
                OFFICIAL_UPSTREAM_URL,
                oid,
            ],
            cwd=repository,
            check=False,
            capture_output=True,
            timeout=60,
        )
        verified = subprocess.run(
            ["git", "cat-file", "-e", f"{oid}^{{commit}}"],
            cwd=repository,
            check=False,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return oid if fetched.returncode == 0 and verified.returncode == 0 else None


def _git_is_ancestor(repository: Path, ancestor: str, descendant: str) -> bool:
    try:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=repository,
            check=False,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(type(value) is dict, f"expected JSON object: {path}")
    return value


def checked_file(base: Path, record: dict, *, key: str = "path") -> Path:
    require(
        type(record) is dict
        and type(record.get(key)) is str
        and type(record.get("sha256")) is str,
        "missing file path/hash",
    )
    path = Path(record[key])
    if not path.is_absolute():
        path = base / path
    require(path.is_file(), f"missing evidence file: {path}")
    require(
        digest(path.read_bytes()) == record["sha256"],
        f"stale or modified evidence: {path}",
    )
    return path


def checked_capture(base: Path, capture: dict) -> None:
    require(
        type(capture) is dict and set(capture) == {"stdout", "stderr", "progress"},
        "raw attempt capture missing",
    )
    for record in capture.values():
        checked_file(base, record)


def manifest() -> dict:
    value = read_json(ROOT / "manifest.json")
    claim = value.pop("contract_sha256", None)
    require(claim == digest(canonical(value)), "contract hash mismatch")
    require(
        value["schema_version"] == 1 and value["contract_id"] == "DFT-MP-v1",
        "unsupported contract",
    )
    require(
        canonical(value) == canonical(build()),
        "manifest inventory or basis-expanded AO count drift",
    )
    value["contract_sha256"] = claim
    require(
        value["basis"]["j_k"] == "direct"
        and value["basis"]["representation"] == "real_spherical",
        "basis/domain drift",
    )
    require(
        source_digest((REPO / "python/vibeqc/data/basis_pack.json").read_bytes())
        == value["basis"]["basis_pack_sha256"],
        "basis pack drift requires a new contract",
    )
    for case_id, case in value["cases"].items():
        path = ROOT / case["input"]
        require(
            path.is_file() and source_digest(path.read_bytes()) == case["input_sha256"],
            f"geometry drift: {case_id}",
        )
        changed = ROOT / case["changed_input"]
        require(
            changed.is_file()
            and source_digest(changed.read_bytes()) == case["changed_input_sha256"],
            f"changed geometry drift: {case_id}",
        )
        atoms = read_json(path)["atoms"]
        require(len(atoms) == case["atom_count"], f"atom count drift: {case_id}")
    ids = [row["id"] for row in value["rows"]]
    require(
        len(ids) == len(set(ids))
        and all(type(row.get("required")) is bool for row in value["rows"])
        and any(row["required"] for row in value["rows"]),
        "duplicate row or invalid required/optional inventory",
    )
    required_cases = {
        "water",
        "water_dimer",
        "benzene",
        "oh",
        "o2",
        "water8",
        "water16",
        "water32",
        "caffeine",
        "ace_glygly_nme",
    }
    require(required_cases <= value["cases"].keys(), "mandatory workload missing")
    return value


def _finite(value: object, label: str, *, nonnegative: bool = False) -> float:
    require(
        type(value) in (int, float) and math.isfinite(value), f"{label} must be finite"
    )
    result = float(value)
    require(not nonnegative or result >= 0, f"{label} must be nonnegative")
    return result


def _sha(value: object, label: str) -> None:
    require(
        type(value) is str
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value),
        f"invalid {label}",
    )


def _check_run(
    run: dict,
    campaign: dict,
    row: dict,
    contract: dict,
    base: Path,
    capture: dict | None = None,
) -> None:
    require(run.get("status") == "pass", "result schema status missing")
    require(run.get("row_id") == row["id"], "row mismatch")
    require(
        run.get("contract_sha256") == contract["contract_sha256"],
        "stale contract evidence",
    )
    require(
        run.get("source_commit") == campaign["source_commit"],
        "stale or cross-source evidence",
    )
    require(
        run.get("library_sha256") == campaign["library"]["sha256"],
        "cross-library evidence",
    )
    require(
        run.get("artifact_sha256") == campaign["artifact"]["sha256"],
        "cross-artifact evidence",
    )
    case = contract["cases"][row["case"]]
    model = contract["model"]
    require(
        run.get("input_sha256") == case["input_sha256"]
        and run.get("basis_pack_sha256") == contract["basis"]["basis_pack_sha256"],
        "input/basis mismatch",
    )
    require(
        run.get("grid_identity") == case["grid_identity"]
        and run.get("method_version") == model["methods"][row["method"]]["version"],
        "scientific model mismatch",
    )
    require(
        run.get("ao_count") == case["ao_count_spherical"]
        and run.get("atom_count") == case["atom_count"],
        "actual AO/atom count mismatch",
    )
    require(
        run.get("backend") == row["backend"]
        and run.get("provider") == row["provider"]
        and run.get("j_k") == row["j_k"],
        "provider mismatch",
    )
    require(
        run.get("spin") == row["spin"]
        and run.get("charge") == case["charge"]
        and run.get("multiplicity") == case["multiplicity"],
        "spin/charge/multiplicity mismatch",
    )
    require(
        run.get("basis_representation") == "real_spherical"
        and run.get("auxiliary") is None
        and run.get("ecp") is None
        and run.get("periodic") is False,
        "direct all-electron basis domain mismatch",
    )
    require(
        run.get("attained_state") and run.get("reference_attained_state"),
        "missing attained SCF states",
    )
    require(
        run["attained_state"] == run["reference_attained_state"],
        "root/state selection mismatch",
    )
    scf = run.get("scf")
    require(type(scf) is dict and scf.get("converged") is True, "unconverged SCF")
    require(
        scf.get("energy_tolerance_eh") == model["scf"]["energy_tolerance_eh"]
        and scf.get("density_tolerance") == model["scf"]["density_tolerance"],
        "SCF target drift",
    )
    require(
        _finite(scf.get("physical_residual"), "physical residual", nonnegative=True)
        <= model["scf"]["physical_residual_max"],
        "physical residual failed",
    )
    require(
        scf.get("max_iterations") == model["scf"]["max_iterations"]
        and scf.get("screening_thresholds") == model["scf"]["screening_thresholds"],
        "SCF iteration/screening controls drift",
    )
    require(
        scf.get("final_forces_state") == run["attained_state"],
        "forces used another state",
    )
    energy = _finite(run.get("energy_eh"), "total energy")
    require(math.isfinite(energy), "nonfinite energy")
    oracle = run.get("independent_oracle")
    require(
        type(oracle) is dict
        and oracle.get("provider") not in (None, "vibeqc", "native-dft"),
        "independent oracle missing",
    )
    checked_file(base, oracle.get("raw"))
    require(
        oracle.get("method_version") == run["method_version"]
        and oracle.get("basis_pack_sha256") == run["basis_pack_sha256"],
        "oracle scientific model mismatch",
    )
    _sha(oracle.get("source_sha256"), "oracle source hash")
    oracle_energy = _finite(
        oracle.get("total_energy_error_eh"),
        "oracle total energy error",
        nonnegative=True,
    )
    require(
        oracle_energy
        <= min(
            _finite(
                oracle.get("energy_gate_eh"), "oracle energy gate", nonnegative=True
            ),
            contract["gates"]["independent_total_energy_abs_eh"],
        ),
        "independent energy gate failed",
    )
    if row["product"] == "energy+analytic_forces":
        forces = run.get("forces_eh_per_bohr")
        require(
            type(forces) is list
            and len(forces) == case["atom_count"]
            and all(type(f) is list and len(f) == 3 for f in forces),
            "missing analytic force vector",
        )
        for force in forces:
            for component in force:
                _finite(component, "force component")
        require(
            _finite(
                oracle.get("force_component_max_error_eh_per_bohr"),
                "oracle force error",
                nonnegative=True,
            )
            <= min(
                _finite(
                    oracle.get("force_gate_eh_per_bohr"),
                    "oracle force gate",
                    nonnegative=True,
                ),
                contract["gates"]["independent_force_component_max_eh_per_bohr"],
            ),
            "independent force gate failed",
        )
        for key in (
            "grid_convergence",
            "finite_difference",
            "changed_geometry",
            "warm_replay",
            "batch_isolation",
            "failure_recovery",
        ):
            check = run.get("checks", {}).get(key)
            require(
                type(check) is dict and check.get("status") == "pass",
                f"missing {key} check",
            )
            checked_file(base, check.get("raw"))
        fd = run["checks"]["finite_difference"]
        require(
            type(fd.get("step_bohr")) is list
            and len(fd["step_bohr"]) >= 2
            and all(_finite(step, "FD step") > 0 for step in fd["step_bohr"]),
            "multi-step finite differences missing",
        )
        require(
            len(set(fd["step_bohr"])) == len(fd["step_bohr"]),
            "finite-difference steps must be distinct",
        )
        require(
            fd.get("reconverged_each_displacement") is True,
            "finite differences did not reconverge",
        )
        require(
            run["checks"]["grid_convergence"].get("independent_finer_grid") is True,
            "shared coarse grid cannot certify convergence",
        )
        changed = run["checks"]["changed_geometry"]
        require(
            changed.get("input_sha256") == case["changed_input_sha256"]
            and changed.get("grid_identity") == case["changed_grid_identity"]
            and changed.get("complete_energy_forces") is True,
            "changed-geometry E+force evidence absent",
        )
    precision = run.get("precision")
    require(
        type(precision) is dict and type(precision.get("native_provenance")) is dict,
        "native precision provenance missing",
    )
    native = precision["native_provenance"]
    require(
        native.get("operator_work_counters_valid") is True,
        "native operator work counters invalid",
    )
    for name in (
        "mixed_stage_fock_builds",
        "strict_stage_fock_builds",
        "post_scf_fock_builds",
        "refinement_iterations",
        "execution_retries",
        "final_residual_audits",
    ):
        require(
            type(native.get(name)) is int and native[name] >= 0,
            f"missing native {name}",
        )
    require(
        type(precision.get("operators")) is list and precision["operators"],
        "executed operator census missing",
    )
    timeline = precision.get("scf_fock_timeline")
    require(type(timeline) is list and timeline, "SCF/Fock event timeline missing")
    previous_iteration = -1
    for sequence, event in enumerate(timeline):
        require(type(event) is dict, "malformed SCF/Fock event")
        require(
            event.get("kind")
            in (
                "mixed_fock",
                "strict_fock",
                "post_scf_fock",
                "final_audit",
                "retry",
                "fallback",
                "conversion",
            ),
            "unknown SCF/Fock event",
        )
        require(
            event.get("sequence") == sequence
            and event.get("count") == 1
            and type(event.get("iteration")) is int
            and event["iteration"] >= 0,
            "each SCF/Fock call requires one ordered event",
        )
        require(
            event["iteration"] >= previous_iteration
            and type(event.get("state")) is str
            and event["state"],
            "SCF/Fock event iteration or state missing",
        )
        previous_iteration = event["iteration"]
    for kind, name in (
        ("mixed_fock", "mixed_stage_fock_builds"),
        ("strict_fock", "strict_stage_fock_builds"),
        ("post_scf_fock", "post_scf_fock_builds"),
        ("final_audit", "final_residual_audits"),
        ("retry", "execution_retries"),
    ):
        require(
            sum(event["count"] for event in timeline if event["kind"] == kind)
            == native[name],
            f"{kind} timeline/provenance mismatch",
        )
    refinement = [
        event
        for event in timeline
        if event["kind"] == "strict_fock" and event.get("phase") == "refinement"
    ]
    require(
        len(refinement) == native["refinement_iterations"],
        "strict refinement iterations are not tied to Fock events",
    )
    mixed_events = [event for event in timeline if event["kind"] == "mixed_fock"]
    if mixed_events:
        require(
            refinement
            and max(event["sequence"] for event in mixed_events)
            < min(event["sequence"] for event in refinement),
            "mixed Fock work occurred after strict refinement began",
        )
    fock_indices = [
        event["sequence"] for event in timeline if event["kind"].endswith("fock")
    ]
    final_audits = [event for event in timeline if event["kind"] == "final_audit"]
    require(
        fock_indices
        and final_audits
        and final_audits[-1]["sequence"] > max(fock_indices)
        and final_audits[-1]["state"] == run["attained_state"],
        "final physical audit must follow refinement/Fock on returned state",
    )
    for operator in precision["operators"]:
        require(type(operator) is dict, "malformed operator record")
        require(
            all(
                operator.get(name) in ("fp64", "fp32", "tf32", "fp16", "bf16")
                for name in ("storage", "compute", "accumulation", "reduction")
            ),
            "unknown arithmetic dtype",
        )
        require(
            type(operator.get("name")) is str
            and type(operator.get("count")) is int
            and operator["count"] >= 0,
            "invalid operator count",
        )
        require(
            operator.get("arithmetic_mode")
            in ("strict", "mixed", "tf32", "fp16", "bf16"),
            "arithmetic mode missing",
        )
    require(
        precision.get("operator_inventory_complete") is True, "partial operator census"
    )
    for kind, field in (
        ("conversion", "conversion_count"),
        ("fallback", "fallback_count"),
    ):
        require(
            type(precision.get(field)) is int
            and precision[field] >= 0
            and precision[field]
            == sum(event["count"] for event in timeline if event["kind"] == kind),
            f"{kind} count mismatch",
        )
    if row["level"].startswith("fp64"):
        require(
            precision.get("requested_mode") == "fp64"
            and native["mixed_stage_fock_builds"] == 0
            and all(
                o["compute"] == "fp64" and o["arithmetic_mode"] == "strict"
                for o in precision["operators"]
            ),
            "strict FP64 row used reduced arithmetic",
        )
    else:
        require(
            precision.get("requested_mode") in ("auto", "mixed")
            and native["mixed_stage_fock_builds"] > 0,
            "AUTO fallback is not mixed execution",
        )
        require(
            all(
                o["compute"] in ("fp64", "fp32")
                and o["arithmetic_mode"] in ("strict", "mixed")
                for o in precision["operators"]
            ),
            "TF32/FP16/BF16 are separate, unadmitted arithmetic domains",
        )
        require(
            any(
                o["compute"] == "fp32"
                and o["arithmetic_mode"] == "mixed"
                and o["count"] > 0
                for o in precision["operators"]
            ),
            "no executed FP32 mixed operator",
        )
        require(
            native["strict_stage_fock_builds"] > 0
            and native["refinement_iterations"] > 0
            and native["final_residual_audits"] > 0
            and native.get("strict_refinement_applied") is True,
            "final strict refinement/audit absent",
        )
        require(
            run.get("strict_comparator", {}).get("source_commit")
            == campaign["source_commit"]
            and run["strict_comparator"].get("state") == run["attained_state"],
            "unmatched strict comparator",
        )
        require(
            run["strict_comparator"].get("library_sha256")
            == campaign["library"]["sha256"]
            and run["strict_comparator"].get("schedule_identity")
            == run.get("schedule_identity")
            and run.get("schedule_identity"),
            "strict comparator must use matched binary and schedule",
        )
        require(
            run["strict_comparator"].get("physical_residual", float("inf"))
            <= model["scf"]["physical_residual_max"],
            "strict comparator residual failed",
        )
        require(
            run["strict_comparator"].get("energy_tolerance_eh")
            == model["scf"]["energy_tolerance_eh"]
            and run["strict_comparator"].get("density_tolerance")
            == model["scf"]["density_tolerance"]
            and run["strict_comparator"].get("max_iterations")
            == model["scf"]["max_iterations"]
            and run["strict_comparator"].get("screening_thresholds")
            == model["scf"]["screening_thresholds"],
            "strict and mixed final SCF targets differ",
        )
        error = run.get("mixed_vs_strict", {})
        require(
            _finite(
                error.get("total_energy_abs_eh"),
                "mixed total energy error",
                nonnegative=True,
            )
            <= contract["gates"]["mixed_energy_abs_eh"],
            "mixed total energy error failed",
        )
        require(
            _finite(
                error.get("force_component_max_eh_per_bohr"),
                "mixed maximum force error",
                nonnegative=True,
            )
            <= contract["gates"]["mixed_force_component_max_eh_per_bohr"],
            "mixed force error failed",
        )
        require(
            error.get("matched_model_and_grid") is True,
            "mixed/strict mathematical model mismatch",
        )
    if row["product"] == "energy+analytic_forces":
        require(
            run.get("online_scientific_cuda_compiles") == 0,
            "online scientific CUDA compilation",
        )
    if row["level"] == "promoted":
        require(
            capture is not None and run.get("attempt_ledger") == capture["progress"],
            "performance ledger is not the captured raw progress",
        )
        for check_name in ("batch_throughput", "strict_fallback"):
            check = run.get("checks", {}).get(check_name)
            require(
                type(check) is dict and check.get("status") == "pass",
                f"missing {check_name} check",
            )
            checked_file(base, check.get("raw"))
        require(
            run["checks"]["batch_throughput"].get("batch_size")
            == contract["scope"]["batch_throughput"]
            and run["checks"]["batch_throughput"].get("complete_energy_forces") is True,
            "batch throughput is not complete E+force at declared size",
        )
        require(
            run.get("profiled_runs_separate") is True,
            "profiler diagnostics mixed into latency campaign",
        )
        require(
            run.get("ablations")
            and set(run["ablations"]) == set(contract["required_ablations"]),
            "missing ablations",
        )
        for ablation in run["ablations"].values():
            require(type(ablation) is dict, "invalid ablation")
            checked_file(base, ablation.get("raw"))
            for boundary in ("cold", "changed_geometry"):
                require(
                    _finite(ablation.get(f"{boundary}_ms"), "ablation endpoint time")
                    > 0,
                    "ablation must measure complete E+force",
                )
        require(
            type(run.get("peak_simultaneous_bytes")) is int
            and 0
            < run["peak_simultaneous_bytes"]
            <= contract["scope"]["hardware"]["memory_budget_bytes"],
            "memory budget exceeded",
        )
        for boundary in TIMINGS:
            pairs = run.get("timing", {}).get(boundary)
            require(
                type(pairs) is list
                and len(pairs) >= contract["gates"]["minimum_paired_samples"],
                f"insufficient {boundary} pairs",
            )
            for index, pair in enumerate(pairs):
                require(
                    pair.get("order")
                    == (["strict", "mixed"] if index % 2 == 0 else ["mixed", "strict"]),
                    "pair order is not interleaved AB/BA",
                )
                require(
                    pair.get("unprofiled") is True
                    and pair.get("complete_endpoint") is True,
                    "profiled or incomplete timing",
                )
                require(
                    pair.get("conditions_sha256") == campaign["conditions_sha256"],
                    "operating conditions changed",
                )
                for route in ("strict", "mixed"):
                    item = pair.get(route)
                    require(
                        type(item) is dict
                        and set(item.get("components_ms", {})) == COMPONENTS,
                        "timing components missing",
                    )
                    total = _finite(item.get("total_ms"), "complete endpoint time")
                    require(
                        total > 0
                        and abs(
                            sum(
                                _finite(x, "component", nonnegative=True)
                                for x in item["components_ms"].values()
                            )
                            - total
                        )
                        <= 0.01 * total,
                        "timing total/component mismatch",
                    )
        ledger_path = checked_file(base, run["attempt_ledger"])
        try:
            attempts = [
                json.loads(line)
                for line in ledger_path.read_text(encoding="utf-8").splitlines()
            ]
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise InvalidEvidence(f"invalid timing attempt ledger: {error}") from error
        expected_attempts = [
            (boundary, pair_index, route, pair[route]["total_ms"])
            for boundary in TIMINGS
            for pair_index, pair in enumerate(run["timing"][boundary])
            for route in pair["order"]
        ]
        require(
            len(attempts) == len(expected_attempts),
            "timing attempts include missing/failed/dropped samples",
        )
        for index, (attempt, expected_attempt) in enumerate(
            zip(attempts, expected_attempts, strict=True)
        ):
            boundary, pair_index, route, total_ms = expected_attempt
            require(
                type(attempt) is dict
                and attempt.get("attempt_id") == index
                and attempt.get("boundary") == boundary
                and attempt.get("pair_index") == pair_index
                and attempt.get("route") == route
                and attempt.get("status") == "pass"
                and attempt.get("conditions_sha256") == campaign["conditions_sha256"]
                and attempt.get("total_ms") == total_ms,
                "timing ledger/sample mismatch or failed attempt",
            )


def _geomean(values: list[float]) -> float:
    return math.exp(statistics.mean(math.log(v) for v in values))


def _performance_gate(records: dict[str, dict], contract: dict) -> list[str]:
    errors = []
    for method in ("pbe", "r2scan", "pbe0"):
        for boundary in TIMINGS:
            grouped = []
            for case in contract["gates"]["performance_cases"]:
                key = f"{method}/rks/{case}/promoted"
                if key not in records:
                    errors.append(f"missing promoted {key}")
                    continue
                samples = records[key]["timing"][boundary]
                ratios = [
                    p["strict"]["total_ms"] / p["mixed"]["total_ms"] for p in samples
                ]
                if statistics.median(ratios) < 1 / (
                    1 + contract["gates"]["maximum_unexplained_case_regression"]
                ):
                    errors.append(f"{key} {boundary}: >5% regression")
                grouped.append(ratios)
            if len(grouped) != len(contract["gates"]["performance_cases"]):
                continue
            if boundary == "warm":
                continue
            point = _geomean([statistics.median(ratios) for ratios in grouped])
            rng = random.Random(1185)  # noqa: S311 - deterministic bootstrap, not security
            draws = []
            for _ in range(10000):
                draws.append(
                    _geomean(
                        [
                            statistics.median([rng.choice(ratios) for _ in ratios])
                            for ratios in grouped
                        ]
                    )
                )
            lower = sorted(draws)[249]
            if (
                point < contract["gates"]["minimum_geomean_speedup"]
                or lower <= contract["gates"]["minimum_ci95_lower_exclusive"]
            ):
                errors.append(
                    f"{method} {boundary}: speed {point:.4f} CI lower {lower:.4f} below gate"
                )
    return errors


def audit(receipt_path: Path, *, final: bool = False) -> dict:
    contract = manifest()
    receipt = read_json(receipt_path)
    base = receipt_path.resolve().parent
    require(
        receipt.get("schema_version") == 1
        and receipt.get("contract_sha256") == contract["contract_sha256"],
        "receipt contract mismatch",
    )
    campaign = receipt.get("campaign")
    require(
        type(campaign) is dict
        and campaign.get("execution_kind") == "installed_production",
        "not installed production evidence",
    )
    source = campaign.get("source_commit")
    require(
        type(source) is str
        and len(source) == 40
        and all(c in "0123456789abcdef" for c in source),
        "source commit missing",
    )
    for key in ("library", "artifact", "adapter", "build_record", "conditions"):
        checked_file(base, campaign.get(key))
    _sha(campaign.get("conditions_sha256"), "conditions hash")
    require(
        campaign["conditions"]["sha256"] == campaign["conditions_sha256"],
        "operating conditions hash mismatch",
    )
    hardware = campaign.get("hardware", {})
    require(
        hardware.get("device") == "RTX 5090"
        and hardware.get("sm") == 120
        and all(
            hardware.get(k)
            for k in ("driver", "toolchain", "build_profile", "device_uuid")
        ),
        "unqualified hardware identity",
    )
    require(
        campaign.get("build_source_commit") == source
        and campaign.get("library_source_commit") == source,
        "build/source mismatch",
    )
    build_record = read_json(checked_file(base, campaign["build_record"]))
    require(
        build_record.get("source_commit") == source
        and build_record.get("library_sha256") == campaign["library"]["sha256"]
        and build_record.get("artifact_sha256") == campaign["artifact"]["sha256"],
        "unbound build record",
    )
    require(
        build_record.get("scientific_cuda_artifacts_prebuilt") is True,
        "offline scientific CUDA build unproven",
    )
    rows = contract["rows"]
    expected = {row["id"]: row for row in rows}
    required_ids = {row["id"] for row in rows if row["required"]}
    entries = receipt.get("rows")
    require(
        type(entries) is list and len(entries) == len(expected),
        "missing or extra capability rows",
    )
    require(all(type(entry) is dict for entry in entries), "malformed capability row")
    require(
        {entry.get("id") for entry in entries} == set(expected),
        "duplicate or missing capability row",
    )
    passed = {}
    failures = []
    optional_findings = []
    for entry in entries:
        key = entry["id"]
        status = entry.get("status")
        require(status in STATUSES, f"invalid status: {key}")
        if status != "pass":
            if status == "running":
                failures.append(f"{key}: unresolved active adapter attempt")
                continue
            if status != "not-run":
                try:
                    checked_capture(base, entry.get("capture"))
                    if entry.get("partial_progress") is not None:
                        require(
                            entry["partial_progress"] == entry["capture"]["progress"],
                            "partial progress does not match raw capture",
                        )
                except (InvalidEvidence, KeyError, TypeError) as error:
                    (
                        failures if expected[key]["required"] else optional_findings
                    ).append(f"{key}: unretained optional/required attempt: {error}")
                    continue
            require(
                type(entry.get("reason")) is str and entry["reason"],
                f"reason missing: {key}",
            )
            finding = f"{key}: {status}: {entry['reason']}"
            (failures if expected[key]["required"] else optional_findings).append(
                finding
            )
            continue
        try:
            checked_capture(base, entry.get("capture"))
            require(
                entry.get("evidence") == entry["capture"]["stdout"],
                "pass evidence is not captured stdout",
            )
            path = checked_file(base, entry.get("evidence"))
            run = read_json(path)
            _check_run(
                run, campaign, expected[key], contract, path.parent, entry["capture"]
            )
            passed[key] = run
        except (InvalidEvidence, KeyError, TypeError, ValueError) as error:
            failures.append(f"{key}: invalid pass: {error}")
    if required_ids <= passed.keys():
        failures.extend(_performance_gate(passed, contract))
    if final:
        acceptance = receipt.get("final_acceptance", {})
        if (
            acceptance.get("issue") != 1190
            or acceptance.get("status") != "PASS"
            or acceptance.get("source_commit") != source
            or not acceptance.get("review_url")
        ):
            failures.append("#1190 source-matched reviewed PASS missing")
        else:
            require(
                acceptance["review_url"].startswith(
                    "https://github.com/jinzhezenggroup/vibeqc/"
                ),
                "review URL is not in upstream repo",
            )
            raw_acceptance = read_json(
                checked_file(base, acceptance.get("raw_receipt"))
            )
            require(
                raw_acceptance.get("contract_sha256") == contract["contract_sha256"]
                and raw_acceptance.get("source_commit") == source
                and raw_acceptance.get("status") == "PASS",
                "#1190 raw receipt mismatch",
            )
        official_master = _official_master_oid(REPO)
        source_is_official = official_master is not None and _git_is_ancestor(
            REPO, source, official_master
        )
        if official_master is None:
            failures.append("could not resolve and fetch official upstream master")
        elif not source_is_official:
            failures.append("source revision is not in official upstream master")
        if source_is_official:
            source_contract = subprocess.run(
                ["git", "show", f"{source}:tools/dft_mp_v1/manifest.json"],
                cwd=REPO,
                check=False,
                capture_output=True,
            )
            try:
                recorded = json.loads(source_contract.stdout)
            except (json.JSONDecodeError, UnicodeDecodeError):
                recorded = {}
            if (
                source_contract.returncode != 0
                or recorded.get("contract_sha256") != contract["contract_sha256"]
            ):
                failures.append("merged source does not contain this exact contract")
    return {
        "contract_sha256": contract["contract_sha256"],
        "source_commit": source,
        "passed_rows": len(passed),
        "passed_required_rows": len(required_ids & passed.keys()),
        "required_rows": len(required_ids),
        "optional_rows": len(expected) - len(required_ids),
        "optional_findings": optional_findings,
        "product_status": "PASS" if final and not failures else "BLOCKED",
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", nargs="?", type=Path)
    parser.add_argument(
        "--final",
        action="store_true",
        help="require #1190 and upstream merged revision",
    )
    args = parser.parse_args()
    if args.receipt is None:
        print(
            json.dumps(
                {
                    "contract_sha256": manifest()["contract_sha256"],
                    "status": "valid contract; scientific rows not-run",
                },
                indent=2,
            )
        )
        return
    try:
        outcome = audit(args.receipt, final=args.final)
    except (InvalidEvidence, KeyError, TypeError, ValueError) as error:
        outcome = {"product_status": "BLOCKED", "failures": [str(error)]}
    print(json.dumps(outcome, indent=2))
    if outcome["product_status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
