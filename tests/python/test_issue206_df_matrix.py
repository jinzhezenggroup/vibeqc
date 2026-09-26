"""Hardware-free checks of failure retention and the Slurm launch contract."""

import hashlib
import json
import os
import subprocess
import sys
import typing
from pathlib import Path

import pytest

from benchmarks import issue206_df_matrix as matrix

if typing.TYPE_CHECKING:
    from typing_extensions import Self


def test_manifest_hashes_requested_library_in_bounded_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A preflight identifies file bytes without loading or whole-file reads."""
    library = tmp_path / "libvibeqc.so"
    content = b"protocol-only native library" * 100_000
    library.write_bytes(content)
    alias = tmp_path / "selected-library.so"
    alias.symlink_to(library)
    original_open = Path.open
    reads = []

    class BoundedReader:
        def __enter__(self) -> "Self":
            self.stream = original_open(library, "rb")
            return self

        def __exit__(self, *args: object) -> None:
            self.stream.close()

        def read(self, size: int) -> bytes:
            assert 0 < size <= 1 << 20
            reads.append(size)
            return self.stream.read(size)

    def open_file(path: Path, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        if path == library:
            return BoundedReader()
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_file)
    payload = matrix.manifest_payload(
        cases=matrix.MATRIX,
        repeats=5,
        python=sys.executable,
        library=alias,
        output_dir=tmp_path / "output",
    )
    assert payload["source"]["native_library"] == {
        "path": str(library.resolve()),
        "status": "recorded",
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }
    assert len(reads) >= 3
    assert all(row["status"] == "pending" for row in payload["matrix"])


def test_manifest_only_cli_records_unbuilt_library_without_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Planning before compilation cannot be mistaken for qualified evidence."""
    output = tmp_path / "plan"
    library = tmp_path / "not-built.so"
    monkeypatch.setattr(
        sys,
        "argv",
        ["matrix", "--library", str(library), "--output-dir", str(output)],
    )
    matrix.main()
    payload = json.loads((output / "manifest.json").read_text())
    assert payload["schema"] == "vibeqc.issue206.df_matrix"
    assert payload["version"] == 1
    assert payload["source"]["native_library"] == {
        "path": str(library),
        "status": "missing",
        "sha256": None,
        "size_bytes": None,
    }
    assert len(payload["matrix"]) == 4
    assert all(row["result"] is None for row in payload["matrix"])


def test_library_identity_changes_when_same_path_is_rebuilt(tmp_path: Path) -> None:
    library = tmp_path / "libvibeqc.so"
    library.write_bytes(b"original")
    before = matrix._native_library_metadata(library)
    library.write_bytes(b"rebuilt")
    after = matrix._native_library_metadata(library)
    assert before["path"] == after["path"]
    assert before["sha256"] != after["sha256"]


@pytest.mark.parametrize("method", ("rhf", "uhf"))
@pytest.mark.parametrize("forces", (False, True))
@pytest.mark.parametrize("mode", ("reuse", "mixed", "forced", "cold_correction"))
def test_final_state_work_ablation_requires_current_physics_and_all_spin_leaves(
    method: typing.Any, forces: typing.Any, mode: typing.Any
) -> None:
    """Zero final solves must not let missing validation or W pass the ledger."""
    import copy

    from benchmarks.df_host_workloads import validate_final_state_counts

    corrections, corrected = {
        "reuse": (0, 0),
        "mixed": (3, 1),
        "forced": (5, 2),
        "cold_correction": (5, 2),
    }[mode]
    for reference in (False, True) if mode == "forced" else (False,):
        count = corrections * (2 if method == "uhf" else 1)
        record = {
            "eigensolves_by_reason": {
                "final_fock": {"calls": count if reference else 0}
            },
            "device_eigensolves_by_reason": {
                "final_fock": {"calls": 0 if reference else count}
            },
            "exclusive_phases": {
                "final_state_read": {"calls": 2},
                "final_state_fock_build": {"calls": 2 + corrections},
                "final_state_validation": {"calls": 2 + corrections},
                "strict_final_correction": {"calls": corrections},
                "final_state_corrected": {"calls": corrected},
                "final_state_reuse": {"calls": 2 - corrected},
                "final_state_weighted_density": {"calls": 2 if forces else 0},
                "force_response": {"calls": 2 if forces else 0},
            },
        }
        kwargs = {
            "batch_size": 2,
            "method": method,
            "force": mode == "forced",
            "reference": reference,
            "compute_forces": forces,
        }
        validate_final_state_counts(record, **kwargs)
        for group, name in [
            ("exclusive_phases", name) for name in record["exclusive_phases"]
        ] + [
            ("eigensolves_by_reason", "final_fock"),
            ("device_eigensolves_by_reason", "final_fock"),
            ("eigensolves_by_reason", "fallback"),
        ]:
            changed = copy.deepcopy(record)
            changed[group].setdefault(name, {"calls": 0})["calls"] += 1
            with pytest.raises(RuntimeError, match="final state ablation"):
                validate_final_state_counts(changed, **kwargs)
        changed = copy.deepcopy(record)
        changed["exclusive_phases"] = {}
        with pytest.raises(RuntimeError, match="current-F"):
            validate_final_state_counts(changed, **kwargs)
        # A force-rebuild flag cannot silently accept two reused candidates.
        if mode == "reuse":
            with pytest.raises(RuntimeError, match="correction"):
                validate_final_state_counts(record, **{**kwargs, "force": True})


@pytest.mark.parametrize("flag", ("--final-state-ablation", "--combined-host-ablation"))
def test_new_host_ablation_cli_rejects_ambiguous_or_non_host_requests(
    flag: typing.Any, monkeypatch: typing.Any
) -> None:
    """Reject protocol ambiguity before any hardware or output is touched."""
    for extra in ([], ["--host-workloads", "--final-eigen-ablation"]):
        monkeypatch.setattr(sys, "argv", ["issue206_df_matrix.py", flag, *extra])
        with pytest.raises(SystemExit) as error:
            matrix.main()
        assert error.value.code == 2


@pytest.mark.parametrize("method", ("rhf", "uhf"))
@pytest.mark.parametrize("reference", (False, True))
def test_final_state_fixed_point_reuses_rejected_probe_without_hiding_solves(
    method: typing.Any, reference: typing.Any
) -> None:
    """Two accepted probes plus one promoted rejection require three solves.

    The promoted probe also accounts for the sole density correction. Counting
    it again would invent work; omitting accepted probes would hide real work.
    """
    import copy

    from benchmarks.df_host_workloads import validate_final_state_counts

    count = 3 * (2 if method == "uhf" else 1)
    group = "eigensolves_by_reason" if reference else "device_eigensolves_by_reason"
    record = {
        "eigensolves_by_reason": {},
        "device_eigensolves_by_reason": {},
        "exclusive_phases": {
            name: {"calls": calls}
            for name, calls in {
                "final_state_read": 2,
                "final_state_fock_build": 3,
                "final_state_validation": 3,
                "strict_final_correction": 1,
                "final_state_corrected": 1,
                "final_state_reuse": 1,
                "final_state_weighted_density": 2,
                "force_response": 2,
                "final_state_fixed_point": 3,
                "final_state_fixed_point_promotion": 1,
            }.items()
        },
    }
    record[group]["final_fock"] = {"calls": count}
    kwargs = {
        "batch_size": 2,
        "method": method,
        "force": False,
        "reference": reference,
        "compute_forces": True,
    }
    validate_final_state_counts(record, **kwargs)
    for name in ("final_state_fixed_point", "final_state_fixed_point_promotion"):
        broken = copy.deepcopy(record)
        broken["exclusive_phases"][name]["calls"] += 1
        with pytest.raises(RuntimeError, match="fixed-point"):
            validate_final_state_counts(broken, **kwargs)
    for wrong in (count - 1, count + 1):
        broken = copy.deepcopy(record)
        broken[group]["final_fock"]["calls"] = wrong
        with pytest.raises(RuntimeError, match="spin providers"):
            validate_final_state_counts(broken, **kwargs)


@pytest.mark.parametrize("method", ("rhf", "uhf"))
def test_strict_provider_ablation_requires_actual_fock_validation_and_correction(
    method: typing.Any,
) -> None:
    """Several real corrections are valid; missing current-F work never is."""
    import copy

    from benchmarks.df_host_workloads import validate_final_eigen_counts

    record = {
        "eigensolves_by_reason": {},
        "device_eigensolves_by_reason": {
            "final_fock": {"calls": 5 * (2 if method == "uhf" else 1)}
        },
        "exclusive_phases": {
            "final_state_fock_build": {"calls": 7},
            "strict_final_correction": {"calls": 5},
            "final_state_corrected": {"calls": 2},
            "final_state_validation": {"calls": 7},
        },
    }
    kwargs = {
        "batch_size": 2,
        "method": method,
        "reference": False,
        "strict_final_state": True,
    }
    validate_final_eigen_counts(record, **kwargs)
    for phase in record["exclusive_phases"]:
        changed = copy.deepcopy(record)
        changed["exclusive_phases"][phase]["calls"] = 0
        with pytest.raises(RuntimeError, match="strict rebuilding"):
            validate_final_eigen_counts(changed, **kwargs)
    changed = copy.deepcopy(record)
    changed["exclusive_phases"] = {}
    changed["device_eigensolves_by_reason"]["final_fock"]["calls"] = 2
    with pytest.raises(RuntimeError, match="strict rebuilding"):
        validate_final_eigen_counts(changed, **kwargs)


@pytest.mark.parametrize("reference", (False, True))
@pytest.mark.parametrize("method", ("rhf", "uhf"))
def test_final_provider_ablation_rejects_missing_or_unexpected_solves(
    reference: typing.Any, method: typing.Any
) -> None:
    """A disabled trace hook or silent oracle fallback cannot pass promotion."""
    import copy

    from benchmarks.df_host_workloads import validate_final_eigen_counts

    count = 4 * (2 if method == "uhf" else 1)
    record = {
        "eigensolves_by_reason": {"final_fock": {"calls": count if reference else 0}},
        "device_eigensolves_by_reason": {
            "final_fock": {"calls": 0 if reference else count}
        },
    }
    validate_final_eigen_counts(
        record, batch_size=4, method=method, reference=reference
    )
    for group, name in (
        ("eigensolves_by_reason", "final_fock"),
        ("eigensolves_by_reason", "fallback"),
        ("device_eigensolves_by_reason", "final_fock"),
    ):
        changed = copy.deepcopy(record)
        changed[group].setdefault(name, {"calls": 0})["calls"] += 1
        with pytest.raises(RuntimeError, match="declared provider"):
            validate_final_eigen_counts(
                changed, batch_size=4, method=method, reference=reference
            )
    missing = copy.deepcopy(record)
    group, name = (
        ("eigensolves_by_reason", "final_fock")
        if reference
        else ("device_eigensolves_by_reason", "final_fock")
    )
    missing[group][name]["calls"] = 0
    with pytest.raises(RuntimeError, match="declared provider"):
        validate_final_eigen_counts(
            missing, batch_size=4, method=method, reference=reference
        )


@pytest.mark.parametrize("failure", ["exit", "launch", "missing_result", "gate"])
@pytest.mark.parametrize("energy_only", [False, True])
def test_matrix_retains_failures_and_finishes_remaining_cases(
    tmp_path: typing.Any,
    monkeypatch: typing.Any,
    failure: typing.Any,
    energy_only: typing.Any,
) -> None:
    # subprocess.run is replaced throughout: these tests never execute CUDA.
    monkeypatch.setenv("SLURM_JOB_ID", "protocol-test")
    monkeypatch.setattr(matrix, "_git", lambda *args: "")
    output = tmp_path / "endpoints"
    output.mkdir()
    (output / "96ao-b1.json").write_text('{"previous_attempt": true}')
    manifest = tmp_path / "manifest.json"
    payload = matrix.manifest_payload(
        cases=matrix.MATRIX[:2],
        repeats=1,
        python=sys.executable,
        library=tmp_path / "lib.so",
        output_dir=output,
        memory_budget_bytes=32 << 20,
        energy_only=energy_only,
    )
    calls = []

    def endpoint(command: typing.Any, **kwargs: typing.Any) -> typing.Any:
        calls.append(command)
        active = json.loads(manifest.read_text())["matrix"][len(calls) - 1]
        assert active["status"] == "running" and active["command"] == command
        assert command[
            command.index("--density-fitting-memory-budget-bytes") + 1
        ] == str(32 << 20)
        assert "--reference-full-fock" in command
        assert kwargs["env"].get("CUDA_VISIBLE_DEVICES") == os.environ.get(
            "CUDA_VISIBLE_DEVICES"
        )
        assert ("--energy-only" in command) == energy_only
        assert ("--maximum-force-error" in command) != energy_only
        assert command[command.index("--maximum-energy-error") + 1] == "1e-9"
        path = Path(command[command.index("--output") + 1])
        if len(calls) == 1:
            if failure == "launch":
                raise FileNotFoundError("missing interpreter")
            if failure in ("exit", "gate"):
                if failure == "gate":
                    path.write_text('{"gate": {"passed": false}}')
                return subprocess.CompletedProcess(command, 2, "", "endpoint failed")
            return subprocess.CompletedProcess(command, 0, "", "")
        path.write_text('{"converged": true}')
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(matrix.subprocess, "run", endpoint)
    # Exercise only the visibility preflight; no device state is changed and
    # the child launcher above is a pure Python test double.
    monkeypatch.setattr(
        matrix.os, "environ", {**os.environ, "CUDA_VISIBLE_DEVICES": "protocol-test"}
    )
    with pytest.raises(SystemExit, match="DF matrix failed"):
        matrix.run_matrix(
            payload,
            manifest_path=manifest,
            python=sys.executable,
            library=tmp_path / "lib.so",
            output_dir=output,
        )
    rows = json.loads(manifest.read_text())["matrix"]
    assert len(calls) == 2
    assert [row["status"] for row in rows] == ["failed", "passed"]
    if failure == "gate":
        assert (
            json.loads(Path(rows[0]["result"]).read_text())["gate"]["passed"] is False
        )
    else:
        assert rows[0]["result"] is None
    assert Path(rows[0]["log"]).is_file()
    assert Path(rows[1]["result"]).is_file()


def test_slurm_runner_is_kept_out_of_repository_root() -> None:
    root = Path(matrix.ROOT)
    assert not list(root.glob("*.slurm"))
    assert (root / "benchmarks" / "run_issue206_df.slurm").is_file()


def test_sbatch_spool_copy_uses_submission_checkout(tmp_path: typing.Any) -> None:
    root = Path(matrix.ROOT)
    spool = tmp_path / "slurm_script"
    spool.write_text((root / "benchmarks" / "run_issue206_df.slurm").read_text())
    # A harmless interpreter stub reports argv; even --run never reaches Python.
    interpreter = tmp_path / "python-stub"
    interpreter.write_text('#!/bin/bash\nprintf "%s\\n" "$PWD" "$@"\n')
    interpreter.chmod(0o755)
    environment = {
        **os.environ,
        "SLURM_SUBMIT_DIR": str(root),
        "ISSUE206_PYTHON": str(interpreter),
        "ISSUE206_OUTPUT_DIR": str(tmp_path / "results"),
    }
    environment.pop("ISSUE206_ROOT", None)
    completed = subprocess.run(
        ["bash", str(spool)],
        env=environment,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.splitlines()[:2] == [
        str(root),
        "benchmarks/issue206_df_matrix.py",
    ]


def test_published_archive_uses_verified_repository_format() -> None:
    from tools.unpack_evidence import unpack

    assert unpack(matrix.ROOT / "benchmarks/results/issue206-df-a") == 9


@pytest.mark.parametrize(
    "field,value",
    [("iterations", 3), ("warm_start_used", False), ("warm_start_fallback", True)],
)
def test_eager_lazy_timing_rejects_iteration_or_retry_changes(
    field: typing.Any, value: typing.Any
) -> None:
    """Equal endpoints cannot hide a different amount of SCF work."""
    from benchmarks.df_host_workloads import (
        AblationBranchMismatch,
        validate_ablation_branches,
    )

    converged = {"iterations": 2, "warm_start_used": True, "warm_start_fallback": False}
    rows = [
        {
            "workload": "unchanged-geometry",
            "selection": side,
            "diagnostics": {"convergence": [dict(converged)]},
        }
        for side in ("baseline", "candidate")
    ]
    validate_ablation_branches(rows)
    rows[1]["diagnostics"]["convergence"][0][field] = value
    with pytest.raises(AblationBranchMismatch, match="matching SCF") as failure:
        validate_ablation_branches(rows)
    evidence = failure.value.evidence
    assert evidence["status"] == "rejected"
    assert evidence["samples"] == rows
    assert evidence["timing_assessment"] is None
    assert set(evidence["mismatched_branches"]) == {"unchanged-geometry"}


def test_host_branch_rejection_retains_raw_samples_and_failed_manifest(
    monkeypatch: typing.Any, tmp_path: typing.Any
) -> None:
    """A mocked measurement checks the CLI publication boundary without a GPU."""
    from benchmarks import df_host_workloads as host

    rows = [
        {
            "workload": "cold-start",
            "selection": selection,
            "diagnostics": {
                "convergence": [
                    {
                        "iterations": iterations,
                        "warm_start_used": False,
                        "warm_start_fallback": False,
                    }
                ]
            },
        }
        for selection, iterations in (
            ("baseline", 20),
            ("baseline", 21),
            ("candidate", 20),
        )
    ]
    with pytest.raises(host.AblationBranchMismatch) as caught:
        host.validate_ablation_branches(rows)
    failure = caught.value
    failure.evidence.update(
        source={"fixture": "mock measurement"}, inputs={"fixture": "no GPU execution"}
    )

    def reject(**kwargs: typing.Any) -> typing.Any:
        raise failure

    # The CLI assigns VIBEQC_LIBRARY directly; restore it after this test.
    monkeypatch.setenv("VIBEQC_LIBRARY", os.environ.get("VIBEQC_LIBRARY", ""))
    monkeypatch.setattr(host, "host_workloads", reject)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "issue206_df_matrix.py",
            "--run",
            "--host-workloads",
            "--combined-host-ablation",
            "--case",
            "water-tetramer-def2-svp-spherical",
            "--batch",
            "1",
            "--output-dir",
            str(tmp_path),
        ],
    )
    with pytest.raises(host.AblationBranchMismatch):
        matrix.main()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    entry = manifest["matrix"][0]
    assert entry["status"] == "failed" and entry["result"] is None
    rejected = Path(entry["rejected_result"])
    assert rejected.name == "host-96ao-b1.rejected.json"
    assert json.loads(rejected.read_text()) == json.loads(json.dumps(failure.evidence))
    assert not (tmp_path / "host-96ao-b1.json").exists()


@pytest.mark.parametrize("ablation", ("lazy-core", "overlap-cache", "combined"))
@pytest.mark.parametrize(
    "workload", ("cold-start", "unchanged-geometry", "changed-geometry")
)
@pytest.mark.parametrize("selection", ("baseline", "candidate"))
def test_preparation_ablation_rejects_wrong_actual_counts(
    ablation: typing.Any, workload: typing.Any, selection: typing.Any
) -> None:
    """Reinstated solves or stale changed-item cache hits must fail promotion."""
    import copy

    from benchmarks.df_host_workloads import (
        preparation_policies,
        validate_preparation_counts,
    )

    eager, rebuild = preparation_policies(ablation)[selection]
    overlap = (
        4
        if rebuild or workload == "cold-start"
        else int(workload == "changed-geometry")
    )
    core = 4 if eager or workload == "cold-start" else 0
    misses = 0 if rebuild else overlap
    hits = 0 if rebuild else 4 - misses
    components = {
        "eigensolves_by_reason": {
            "core_guess": {"calls": core},
            "overlap": {"calls": overlap},
        },
        "exclusive_phases": {
            "initial_density": {"calls": 4},
            "overlap_cache_miss": {"calls": misses},
            "overlap_cache_hit": {"calls": hits},
        },
    }

    def validate(value: typing.Any) -> None:
        validate_preparation_counts(
            value, batch_size=4, workload=workload, eager=eager, rebuild=rebuild
        )

    validate(components)
    for group, name in (
        ("eigensolves_by_reason", "core_guess"),
        ("eigensolves_by_reason", "overlap"),
        ("exclusive_phases", "initial_density"),
        ("exclusive_phases", "overlap_cache_hit"),
        ("exclusive_phases", "overlap_cache_miss"),
    ):
        broken = copy.deepcopy(components)
        broken[group][name]["calls"] += 1
        with pytest.raises(RuntimeError, match="declared solve/cache policy"):
            validate(broken)


@pytest.mark.parametrize("reference", (False, True))
@pytest.mark.parametrize(
    "workload,overlap,core",
    (("cold-start", 4, 4), ("energy-only", 0, 0), ("changed-geometry", 1, 0)),
)
def test_setup_provider_counts_reject_wrong_provider(
    reference: typing.Any, workload: typing.Any, overlap: typing.Any, core: typing.Any
) -> None:
    """Missing or unexpectedly substituted leaves cannot pass setup promotion."""
    import copy

    from benchmarks.df_host_workloads import validate_setup_eigen_counts

    record = {
        key: {
            "overlap": {"calls": overlap if selected else 0},
            "core_guess": {"calls": core if selected else 0},
        }
        for key, selected in (
            ("eigensolves_by_reason", reference),
            ("device_eigensolves_by_reason", not reference),
        )
    }
    validate_setup_eigen_counts(
        record, batch_size=4, workload=workload, reference=reference
    )
    for key in record:
        changed = copy.deepcopy(record)
        changed[key]["overlap"]["calls"] += 1
        with pytest.raises(RuntimeError, match="declared provider"):
            validate_setup_eigen_counts(
                changed, batch_size=4, workload=workload, reference=reference
            )
