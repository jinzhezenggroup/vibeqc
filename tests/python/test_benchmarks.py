import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _benchmark_support_module():
    """Load benchmark helpers without turning the scripts into a package."""

    path = REPOSITORY_ROOT / "benchmarks" / "_support.py"
    spec = importlib.util.spec_from_file_location("vibeqc_benchmark_support", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _shell_histogram_module():
    """Load the pure shell-work planner without requiring PySCF."""

    path = REPOSITORY_ROOT / "benchmarks" / "shell_class_histogram.py"
    spec = importlib.util.spec_from_file_location("vibeqc_shell_histogram", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _batch_comparison_module():
    """Load pure batch-comparison helpers without importing CUDA packages."""

    benchmark_directory = REPOSITORY_ROOT / "benchmarks"
    path = benchmark_directory / "compare_gpu4pyscf_batch.py"
    spec = importlib.util.spec_from_file_location("vibeqc_batch_comparison", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(benchmark_directory))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def _results_summary_module():
    """Load the artifact-to-Markdown generator as a pure helper module."""

    path = REPOSITORY_ROOT / "benchmarks" / "generate_results_summary.py"
    spec = importlib.util.spec_from_file_location("vibeqc_results_summary", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _aot_shell_gate_module():
    """Load the AOT endpoint helpers without importing a GPU backend."""

    path = REPOSITORY_ROOT / "benchmarks" / "aot_shell_batch_gate.py"
    spec = importlib.util.spec_from_file_location("vibeqc_aot_shell_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_batch_benchmark_writes_reproducible_json(tmp_path):
    """Keep benchmark artifacts tied to raw samples and exact source state."""

    output = tmp_path / "batch.json"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "python")
    environment["VIBEQC_LIBRARY"] = str(REPOSITORY_ROOT / "build" / "libvibeqc.so")
    completed = subprocess.run(
        (
            sys.executable,
            str(REPOSITORY_ROOT / "benchmarks" / "batch_throughput.py"),
            "--device",
            "cpu",
            "--batch",
            "2",
            "--repeats",
            "2",
            "--output",
            str(output),
        ),
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "JSON result:" in completed.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["benchmark"] == "batch_throughput"
    assert payload["settings"]["batch_size"] == 2
    assert len(payload["result"]["timings_seconds"]["warm_batches"]) == 2
    assert payload["result"]["executed_backend"] == "cpu_reference"
    assert payload["environment"]["git"]["commit"]
    assert payload["environment"]["packages"]["numpy"]


def test_benchmark_source_status_ignores_only_pending_result_json():
    support = _benchmark_support_module()
    assert support._source_status_payload(
        "",
        "benchmarks/results/new-point.json\n",
    ) == {
        "dirty": False,
        "pending_generated_benchmark_artifacts": 1,
    }
    assert support._source_status_payload(
        " M src/scf/rhf.cpp",
        "benchmarks/results/new-point.json\nnotes.txt\n",
    ) == {
        "dirty": True,
        "pending_generated_benchmark_artifacts": 1,
    }


def test_gpu_comparison_help_does_not_require_an_allocated_device():
    """Keep benchmark discovery usable on scheduler login nodes."""

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "python")
    environment["CUDA_VISIBLE_DEVICES"] = ""
    for script in (
        "compare_gpu4pyscf.py",
        "compare_gpu4pyscf_batch.py",
    ):
        completed = subprocess.run(
            (
                sys.executable,
                str(REPOSITORY_ROOT / "benchmarks" / script),
                "--help",
            ),
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        assert "--output" in completed.stdout
        assert "oh-def2-svp-uhf" in completed.stdout
        assert "oh-def2-svp-spherical-uhf" in completed.stdout
        assert "water-def2-svp-spherical" in completed.stdout
        assert "water-def2-tzvp" in completed.stdout
        assert "water-def2-tzvp-spherical" in completed.stdout
        assert "water-tetramer-def2-svp-spherical" in completed.stdout
        assert "water-octamer-s4-def2-svp-spherical" in completed.stdout
        if script == "compare_gpu4pyscf_batch.py":
            assert "--minimum-speedup" in completed.stdout
            assert "--maximum-energy-error" in completed.stdout
            assert "--maximum-force-error" in completed.stdout
            assert "--energy-tolerance" in completed.stdout
            assert "--density-tolerance" in completed.stdout
            assert "--reference-gradient-tolerance" in completed.stdout
            assert "--screening-tolerance" in completed.stdout
            assert "--max-iterations" in completed.stdout


def test_aot_endpoint_order_and_class_parser_are_deterministic():
    """Keep the endpoint's pair count and typo rejection independent of CUDA."""

    endpoint = _aot_shell_gate_module()
    assert endpoint.interleaved_selection_order(2) == (
        "baseline",
        "candidate",
        "candidate",
        "baseline",
    )
    order = endpoint.interleaved_selection_order(5, "abba")
    assert len(order) == 10
    assert order.count("baseline") == 5
    assert order.count("candidate") == 5
    assert endpoint.interleaved_selection_order(3, "ab") == (
        "baseline",
        "candidate",
        "baseline",
        "candidate",
        "baseline",
        "candidate",
    )
    assert endpoint._class_list(" dppp, dpds ") == ("dppp", "dpds")
    for value in ("", "   ", "dppp,", ",dppp", "dppp,,dpds", "dppp,dppp"):
        with pytest.raises(Exception, match="non-empty and unique"):
            endpoint._class_list(value)
    assert endpoint._capacity_fock_selection(None, ("psps",)) is None
    assert endpoint._capacity_fock_selection(("dppp",), None) is None
    assert endpoint._capacity_fock_selection(
        ("dppp", "dpds"), ("dpds", "psps")
    ) == ("dppp", "dpds", "psps")


def test_aot_endpoint_default_fock_selection_ignores_ambient_filter(monkeypatch):
    """Make an omitted Fock CLI selection mean the reproducible full registry."""

    endpoint = _aot_shell_gate_module()
    monkeypatch.setenv("VIBEQC_AOT_FOCK_SHELL_CLASSES", "ambient-only")
    with endpoint._aot_selection(("dppp",), None):
        assert os.environ["VIBEQC_AOT_SHELL_CLASSES"] == "dppp"
        assert "VIBEQC_AOT_FOCK_SHELL_CLASSES" not in os.environ
    assert os.environ["VIBEQC_AOT_FOCK_SHELL_CLASSES"] == "ambient-only"


def test_aot_endpoint_environment_overrides_parse_and_restore(monkeypatch):
    """Keep side-specific runtime env changes isolated within each replay."""

    endpoint = _aot_shell_gate_module()
    assert endpoint._parse_environment_overrides(
        ("VIBEQC_TEST_A=one=two", "VIBEQC_TEST_B=")
    ) == {
        "VIBEQC_TEST_A": "one=two",
        "VIBEQC_TEST_B": "",
    }
    parser = endpoint._parser()
    arguments = parser.parse_args(
        (
            "--baseline-env",
            "VIBEQC_PSPS_RESIDENT_BRA=0",
            "--candidate-env",
            "VIBEQC_PSPS_RESIDENT_BRA=1",
        )
    )
    endpoint._validate_arguments(parser, arguments)
    assert arguments.baseline_environment_overrides == {
        "VIBEQC_PSPS_RESIDENT_BRA": "0"
    }
    assert arguments.candidate_environment_overrides == {
        "VIBEQC_PSPS_RESIDENT_BRA": "1"
    }

    monkeypatch.setenv("VIBEQC_TEST_A", "outside")
    monkeypatch.delenv("VIBEQC_TEST_NEW", raising=False)
    with endpoint._aot_selection(
        ("dppp",),
        environment_overrides={
            "VIBEQC_TEST_A": "inside",
            "VIBEQC_TEST_NEW": "created",
        },
    ):
        assert os.environ["VIBEQC_TEST_A"] == "inside"
        assert os.environ["VIBEQC_TEST_NEW"] == "created"
    assert os.environ["VIBEQC_TEST_A"] == "outside"
    assert "VIBEQC_TEST_NEW" not in os.environ

    with pytest.raises(RuntimeError, match="restore"), endpoint._aot_selection(
        ("dppp",),
        environment_overrides={"VIBEQC_TEST_A": "during-error"},
    ):
        raise RuntimeError("restore")
    assert os.environ["VIBEQC_TEST_A"] == "outside"

    for values, message in (
        (("VIBEQC_TEST_DUP=1", "VIBEQC_TEST_DUP=2"), "duplicate"),
        (("VIBEQC_TEST_MALFORMED",), "NAME=VALUE"),
        (("=missing-name",), "non-empty"),
        (("VIBEQC_AOT_SHELL_CLASSES=bad",), "reserved"),
        (("VIBEQC_AOT_FOCK_SHELL_CLASSES=bad",), "reserved"),
    ):
        with pytest.raises(ValueError, match=message):
            endpoint._parse_environment_overrides(values)


def test_aot_endpoint_freezes_after_one_cold_baseline_and_records_schema():
    """Verify fixed-dm0 control flow with a fake prepared batch."""

    endpoint = _aot_shell_gate_module()

    class FakeStream:
        @staticmethod
        def synchronize():
            return None

    fake_cupy = SimpleNamespace(
        cuda=SimpleNamespace(Stream=SimpleNamespace(null=FakeStream()))
    )

    class FakeItem:
        converged = True
        iterations = 1
        energy_change = 0.0
        density_rms = 0.0
        warm_start_used = True
        warm_start_fallback = False
        executed_backend = "cuda"
        bucket_id = 0
        energy = -1.0
        forces = np.zeros((1, 3))

    class FakeResult:
        items = (FakeItem(),)
        energies = np.asarray([-1.0])

    class FakeBatch:
        def __init__(self):
            self.executions = []
            self.freeze_calls = []

        def execute(self, *, strict):
            self.executions.append(
                (
                    strict,
                    os.environ.get("VIBEQC_AOT_SHELL_CLASSES"),
                    os.environ.get("VIBEQC_AOT_FOCK_SHELL_CLASSES"),
                    os.environ.get("VIBEQC_PSPS_RESIDENT_BRA"),
                )
            )
            return FakeResult()

        def set_warm_start_updates(self, enabled):
            self.freeze_calls.append(enabled)

    batch = FakeBatch()
    measurement, warmups = endpoint._fixed_dm0_measurement(
        batch,
        fake_cupy,
        ("dppp",),
        ("dppp", "ppps"),
        2,
        order_style="abba",
        warmups=1,
        baseline_environment_overrides={"VIBEQC_PSPS_RESIDENT_BRA": "0"},
        candidate_environment_overrides={"VIBEQC_PSPS_RESIDENT_BRA": "1"},
        maximum_energy_error=1.0e-12,
        maximum_force_error=1.0e-12,
        minimum_speedup=0.1,
    )

    assert batch.freeze_calls == [False]
    assert len(warmups) == 1
    # cold baseline, one unmeasured warmup, then four ABBA samples
    assert [entry[1] for entry in batch.executions] == [
        "dppp",
        "dppp",
        "dppp",
        "dppp,ppps",
        "dppp,ppps",
        "dppp",
    ]
    assert all(entry[0] for entry in batch.executions)
    assert [entry[3] for entry in batch.executions] == [
        "0",
        "0",
        "0",
        "1",
        "1",
        "0",
    ]
    assert [
        sample["environment_overrides"]
        for sample in measurement["raw_samples"]
    ] == [
        {"VIBEQC_PSPS_RESIDENT_BRA": "0"},
        {"VIBEQC_PSPS_RESIDENT_BRA": "1"},
        {"VIBEQC_PSPS_RESIDENT_BRA": "1"},
        {"VIBEQC_PSPS_RESIDENT_BRA": "0"},
    ]
    assert measurement["fixed_dm0"] == {
        "enabled": True,
        "source": "measured baseline cold result",
        "warm_start_updates": False,
    }
    assert measurement["measurement_order"] == [
        "baseline",
        "candidate",
        "candidate",
        "baseline",
    ]
    assert len(measurement["raw_samples"]) == 4
    assert len(measurement["pairwise_accuracy"]) == 2
    assert measurement["accuracy"]["maximum_energy_error_hartree"] == 0.0
    assert measurement["gate"]["passed"]


def test_aot_endpoint_pairwise_accuracy_and_median_speedup():
    """Check branch-aware parity and robust median speedup arithmetic."""

    endpoint = _aot_shell_gate_module()

    def sample(seconds, energy, iterations):
        return {
            "seconds": seconds,
            "energies_hartree": [energy],
            "forces_hartree_per_bohr": [[[0.0, 0.0, energy]]],
            "iteration_branches": [iterations],
            "convergence": [{"converged": True}],
            "shell_classes": ["dppp"],
        }

    baseline = [sample(3.0, -1.0, 1), sample(1.0, -1.0, 2)]
    candidate = [sample(2.0, -1.0 + 2.0e-12, 1), sample(4.0, -1.0, 3)]
    pairs = endpoint.pairwise_accuracy(baseline, candidate)
    assert pairs[0]["iteration_branches_match"]
    assert not pairs[1]["iteration_branches_match"]
    assert pairs[0]["maximum_energy_error_hartree"] == pytest.approx(2.0e-12)
    assert pairs[0]["maximum_force_error_hartree_per_bohr"] == pytest.approx(
        2.0e-12
    )
    assert endpoint.timing_summary(baseline)["median_seconds"] == 2.0
    assert endpoint.timing_summary(candidate)["median_seconds"] == 3.0
    measurement = {
        "baseline_samples": baseline,
        "candidate_samples": candidate,
        "pairwise_accuracy": pairs,
        "iteration_branches": {
            "baseline": [[1], [2]],
            "candidate": [[1], [3]],
        },
        "timing_summary": {
            "baseline": endpoint.timing_summary(baseline),
            "candidate": endpoint.timing_summary(candidate),
            "speedup": 2.0 / 3.0,
        },
    }
    endpoint._gate_measurement(
        measurement,
        maximum_energy_error=1.0e-9,
        maximum_force_error=1.0e-9,
        minimum_speedup=0.1,
    )
    assert not measurement["gate"]["passed"]
    assert "SCF iteration branch parity" in measurement["gate"]["failures"]


def test_aot_endpoint_dry_run_does_not_import_gpu_packages(tmp_path):
    """Make --dry-run safe on login nodes with no CUDA/PySCF installation."""

    script = REPOSITORY_ROOT / "benchmarks" / "aot_shell_batch_gate.py"
    output = tmp_path / "aot-plan.json"
    code = """
import builtins
import runpy
import sys

real_import = builtins.__import__
blocked = ("cupy", "vibeqc", "gpu4pyscf", "pyscf", "_cases", "_support")
output_path = sys.argv[1]
script_path = sys.argv[2]

def guarded_import(name, *args, **kwargs):
    if name in blocked or name.startswith(tuple(item + "." for item in blocked)):
        raise AssertionError("GPU package imported during dry-run: " + name)
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
sys.argv = [
    sys.argv[0],
    "--dry-run",
    "--batch",
    "1",
    "--repeats",
    "2",
    "--baseline-env",
    "VIBEQC_PSPS_RESIDENT_BRA=0",
    "--candidate-env",
    "VIBEQC_PSPS_RESIDENT_BRA=1",
    "--output",
    output_path,
]
runpy.run_path(script_path, run_name="__main__")
"""
    # Pass the script path as a separate argv item to keep the guard readable.
    completed = subprocess.run(
        (sys.executable, "-c", code, str(output), str(script)),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "JSON result:" in completed.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["protocol"] == "fixed_dm0_interleaved_ab"
    assert payload["baseline_selection"]["shell_classes"] == ["dppp", "dpds"]
    assert payload["candidate_selection"]["shell_classes"][-1] == "dspp"
    assert payload["baseline_selection"]["environment_overrides"] == {
        "VIBEQC_PSPS_RESIDENT_BRA": "0"
    }
    assert payload["candidate_selection"]["environment_overrides"] == {
        "VIBEQC_PSPS_RESIDENT_BRA": "1"
    }
    assert payload["measurement_order"].count("baseline") == 2
    assert payload["measurement_order"].count("candidate") == 2


def test_real_molecule_gate_has_four_explicit_dry_run_points(tmp_path):
    """Lock the 96/192-AO, batch-1/batch-4 acceptance matrix in CI."""

    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((
        str(REPOSITORY_ROOT / "python"),
        str(REPOSITORY_ROOT / "benchmarks"),
    ))
    completed = subprocess.run(
        (
            sys.executable,
            str(REPOSITORY_ROOT / "benchmarks" / "real_molecule_gate.py"),
            "--dry-run",
            "--output-directory",
            str(tmp_path),
        ),
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    commands = completed.stdout.splitlines()
    assert len(commands) == 4
    assert sum("water-tetramer-def2-svp-spherical" in line for line in commands) == 2
    assert sum("water-octamer-s4-def2-svp-spherical" in line for line in commands) == 2
    assert sum("--batch 1" in line for line in commands) == 2
    assert sum("--batch 4" in line for line in commands) == 2
    assert all("--repeats 5" in line for line in commands)
    assert sum("--minimum-speedup 1.0" in line for line in commands) == 2
    assert all("--max-iterations 100" in line for line in commands)
    assert all("--energy-tolerance 1e-12" in line for line in commands)
    assert all("--density-tolerance 1e-10" in line for line in commands)
    assert sum(
        "--reference-gradient-tolerance 1e-09" in line
        for line in commands
    ) == 2
    assert sum(
        "--reference-gradient-tolerance 1e-08" in line
        for line in commands
    ) == 2
    assert all("--screening-tolerance 1e-14" in line for line in commands)
    assert sum("--maximum-energy-error 3e-11" in line for line in commands) == 2
    assert sum("--maximum-force-error 3e-11" in line for line in commands) == 2
    assert sum("--maximum-energy-error 1e-10" in line for line in commands) == 2
    assert sum("--maximum-force-error 5e-10" in line for line in commands) == 2


def test_gpu_comparison_gate_reports_all_threshold_failures():
    """Keep allocated benchmark gates deterministic and independently testable."""

    support = _benchmark_support_module()
    failures = support.benchmark_gate_failures(
        speedup=3.0,
        maximum_energy_error=4.0e-12,
        maximum_force_error=5.0e-12,
        minimum_speedup=4.0,
        maximum_energy_error_limit=2.0e-12,
        maximum_force_error_limit=3.0e-12,
    )
    assert len(failures) == 3
    assert "speedup" in failures[0]
    assert "energy error" in failures[1]
    assert "force error" in failures[2]
    assert support.benchmark_gate_failures(
        speedup=4.0,
        maximum_energy_error=2.0e-12,
        maximum_force_error=3.0e-12,
        minimum_speedup=4.0,
        maximum_energy_error_limit=2.0e-12,
        maximum_force_error_limit=3.0e-12,
    ) == []

    convergence_failures = support.benchmark_gate_failures(
        speedup=4.0,
        maximum_energy_error=2.0e-12,
        maximum_force_error=3.0e-12,
        vibeqc_converged=False,
        reference_converged=False,
    )
    assert convergence_failures == [
        "one or more VIBEQC systems did not converge",
        "one or more GPU4PySCF reference systems did not converge",
    ]


def test_batch_comparison_pairs_each_timing_with_convergence_state():
    """Preserve every replay's SCF diagnostics for straggler analysis."""

    comparison = _batch_comparison_module()
    result = SimpleNamespace(items=(
        SimpleNamespace(
            converged=True,
            iterations=2,
            energy_change=1.5e-12,
            density_rms=2.0e-13,
            warm_start_used=True,
            warm_start_fallback=False,
        ),
        SimpleNamespace(
            converged=True,
            iterations=3,
            energy_change=5.0e-13,
            density_rms=4.0e-14,
            warm_start_used=True,
            warm_start_fallback=False,
        ),
    ))

    payload = comparison.convergence_payload(result)
    assert [item["iterations"] for item in payload] == [2, 3]
    assert payload[0]["energy_change_hartree"] == 1.5e-12
    assert payload[1]["density_rms"] == 4.0e-14
    assert payload[0]["final_residuals"]["energy_change_hartree"] == 1.5e-12
    assert payload[0]["warm_start"] == {"used": True, "fallback": False}


def test_batch_comparison_records_fixed_post_cold_warm_policy():
    """Keep each engine on one dm0 and document the unmeasured priming pass."""

    comparison = _batch_comparison_module()
    assert comparison.fixed_warm_start_policy() == {
        "vibeqc": "engine-local fixed post-cold converged density snapshot",
        "gpu4pyscf": "engine-local fixed post-cold converged density snapshot",
        "cross_engine_density_identity": (
            "not asserted because backend AO conventions are independent"
        ),
    }

    def sample(seconds, iterations):
        return {
            "seconds": seconds,
            "convergence": [{"iterations": value} for value in iterations],
        }

    metadata = comparison.warm_start_priming_metadata(
        sample(1.25, (2, 3)), sample(2.5, (2, 3))
    )
    assert metadata["performed"] is True
    assert metadata["measured"] is False
    assert metadata["engine_order"] == ["vibeqc", "gpu4pyscf"]
    assert metadata["vibeqc"] == {
        "seconds": 1.25,
        "iteration_branch": [2, 3],
    }
    assert metadata["gpu4pyscf"] == {
        "seconds": 2.5,
        "iteration_branch": [2, 3],
    }


def test_batch_comparison_uses_exact_abba_counts_and_iteration_matching():
    comparison = _batch_comparison_module()

    order = comparison.interleaved_engine_order(5)
    assert order == (
        "vibeqc",
        "gpu4pyscf",
        "gpu4pyscf",
        "vibeqc",
        "vibeqc",
        "gpu4pyscf",
        "gpu4pyscf",
        "vibeqc",
        "vibeqc",
        "gpu4pyscf",
    )
    assert order.count("vibeqc") == order.count("gpu4pyscf") == 5

    def sample(seconds, iterations):
        return {
            "seconds": seconds,
            "convergence": [
                {"iterations": value} for value in iterations
            ],
        }

    matched = comparison.iteration_matched_summary(
        [sample(2.0, (2, 2)), sample(3.0, (3, 2)), sample(2.2, (2, 2))],
        [sample(4.0, (2, 2)), sample(4.2, (2, 2)), sample(5.0, (4, 2))],
    )
    assert matched["iteration_branch"] == [2, 2]
    assert matched["vibeqc_median_seconds"] == pytest.approx(2.1)
    assert matched["gpu4pyscf_median_seconds"] == pytest.approx(4.1)
    assert matched["speedup"] == pytest.approx(4.1 / 2.1)


def test_gpu_cycle_tracker_retains_explicit_final_residuals():
    comparison = _batch_comparison_module()
    tracker = comparison.GpuCycleTracker()
    tracker({"cycle": 0, "e_tot": -10.0, "norm_ddm": 0.2})
    tracker({
        "cycle": 1,
        "e_tot": -10.25,
        "norm_ddm": 1.0e-7,
        "norm_gorb": 2.0e-8,
    })
    assert tracker.iterations == 2
    assert tracker.energy_change_hartree == pytest.approx(0.25)
    assert tracker.density_rms == pytest.approx(1.0e-7)
    assert tracker.orbital_gradient_norm == pytest.approx(2.0e-8)


def test_accuracy_gate_prefers_iteration_matched_repeat_pairs():
    comparison = _batch_comparison_module()
    summary = comparison.accuracy_gate_summary([
        {
            "iteration_branches_match": False,
            "maximum_energy_error_hartree": 1.0e-9,
            "maximum_force_error_hartree_per_bohr": 2.0e-9,
        },
        {
            "iteration_branches_match": True,
            "maximum_energy_error_hartree": 3.0e-12,
            "maximum_force_error_hartree_per_bohr": 4.0e-12,
        },
    ])
    assert summary == {
        "selection": "iteration_matched_pairs",
        "pair_count": 1,
        "maximum_energy_error_hartree": 3.0e-12,
        "maximum_force_error_hartree_per_bohr": 4.0e-12,
    }

    unmatched = comparison.accuracy_gate_summary([
        {
            "iteration_branches_match": False,
            "maximum_energy_error_hartree": 1.0e-9,
            "maximum_force_error_hartree_per_bohr": 2.0e-9,
        },
        {
            "iteration_branches_match": False,
            "maximum_energy_error_hartree": 5.0e-12,
            "maximum_force_error_hartree_per_bohr": 6.0e-12,
        },
    ])
    assert unmatched == {
        "selection": "final_pair_unmatched_labeled",
        "pair_count": 1,
        "maximum_energy_error_hartree": 5.0e-12,
        "maximum_force_error_hartree_per_bohr": 6.0e-12,
    }


def test_results_summary_selects_latest_clean_five_repeat_artifacts(tmp_path):
    summary = _results_summary_module()
    readme = tmp_path / "README.md"
    readme.write_text(
        f"before\n{summary.BEGIN_MARKER}\nstale\n{summary.END_MARKER}\nafter\n",
        encoding="utf-8",
    )

    for index, ((ao_count, batch_size), case) in enumerate(
        summary.PARITY_CASES.items()
    ):
        payload = {
            "schema_version": 2,
            "benchmark": "compare_gpu4pyscf_batch",
            "environment": {
                "timestamp_utc": f"2026-08-27T00:00:0{index}+00:00",
                "git": {"commit": f"{index + 1:040x}", "dirty": False},
            },
            "workload": {
                "case": case,
                "ao_count": ao_count,
                "batch_size": batch_size,
            },
            "settings": {"repeats_per_engine": 5},
            "accuracy": {
                "maximum_energy_error_hartree": 1.0e-12,
                "maximum_force_error_hartree_per_bohr": 2.0e-12,
                "gate_selection": {
                    "maximum_energy_error_hartree": 5.0e-13,
                    "maximum_force_error_hartree_per_bohr": 7.0e-13,
                },
            },
            "timing_summary": {
                "ordinary": {
                    "vibeqc_median_seconds": 1.0,
                    "gpu4pyscf_median_seconds": 2.0,
                },
                "iteration_matched": {
                    "iteration_branch": [2] * batch_size,
                    "speedup": 2.0,
                },
            },
            "gate": {"passed": True},
        }
        (tmp_path / f"point-{ao_count}-{batch_size}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    selected = summary.accepted_parity_artifacts(tmp_path.glob("*.json"))
    section = summary.render_parity_section(selected)
    assert "five interleaved warm samples" in section
    assert "192 | 4" in section
    assert summary.update_readme(readme, section)
    assert not summary.update_readme(readme, section)
    with pytest.raises(ValueError, match="stale"):
        summary.update_readme(readme, section + "\nchanged", check=True)


def test_shell_class_histogram_matches_direct_pair_symmetry():
    histogram = _shell_histogram_module()
    shells = [
        histogram.ShellWork(angular=2, ao_count=6, primitive_count=1),
        histogram.ShellWork(angular=1, ao_count=3, primitive_count=2),
        histogram.ShellWork(angular=0, ao_count=1, primitive_count=3),
    ]
    rows = histogram.summarize_shell_classes(shells, angular_order=5)
    assert {row["class"] for row in rows} == {"dppp", "dpds", "ddps"}
    assert sum(row["primitive_quartets"] for row in rows) > 0
    assert sum(row["primitive_work_fraction"] for row in rows) == pytest.approx(1.0)


def test_gpu4pyscf_rys_ip1_canonicalization_merges_all_orientations():
    """Map pair/within-pair Rys directions to one generic class key."""

    histogram = _shell_histogram_module()
    assert histogram.gpu4pyscf_rys_ip1_shell_class(
        "rys_ejk_ip1_1110"
    ) == (1, 1, 1, 0)
    assert histogram.gpu4pyscf_rys_ip1_shell_class(
        "rys_ejk_ip1_1011"
    ) == (1, 1, 1, 0)
    assert histogram.gpu4pyscf_rys_ip1_shell_class(
        "namespace::rys_vjk_ip1_0011"
    ) == (1, 1, 0, 0)
    assert histogram.gpu4pyscf_rys_ip1_shell_class(
        "rys_ejk_ip1_kernel"
    ) is None

    aggregate = histogram.aggregate_gpu4pyscf_rys_ip1_sqlite
    # The SQLite helper is tested below; this compact input also locks the
    # canonical transformation independently from Nsight's schema.
    assert histogram.shell_class_label(
        histogram.gpu4pyscf_rys_ip1_shell_class("rys_ejk_ip1_1011")
    ) == "ppps"
    with pytest.raises(ValueError, match="unsupported angular digit"):
        histogram.gpu4pyscf_rys_ip1_shell_class("rys_ejk_ip1_9999")
    assert callable(aggregate)


def test_gpu4pyscf_rys_ip1_sqlite_aggregation_sums_canonical_directions(tmp_path):
    """Aggregate Nsight rows by canonical class, not raw kernel suffix."""

    histogram = _shell_histogram_module()
    database = tmp_path / "capture.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
    connection.execute(
        """
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            start INTEGER NOT NULL,
            end INTEGER NOT NULL,
            demangledName INTEGER NOT NULL,
            shortName INTEGER NOT NULL
        )
        """
    )
    names = {
        1: "rys_ejk_ip1_1110(RysIntEnvVars)",
        2: "rys_ejk_ip1_1011(RysIntEnvVars)",
        3: "rys_ejk_ip1_1100(RysIntEnvVars)",
        4: "unrelated_kernel",
    }
    connection.executemany(
        "INSERT INTO StringIds(id, value) VALUES (?, ?)", names.items()
    )
    connection.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL"
        "(start, end, demangledName, shortName) VALUES (?, ?, ?, ?)",
        [
            (0, 2_000_000, 1, 1),
            (3_000_000, 6_000_000, 2, 2),
            (7_000_000, 8_000_000, 3, 3),
            (9_000_000, 100_000_000, 4, 4),
        ],
    )
    connection.commit()
    connection.close()

    profile = histogram.aggregate_gpu4pyscf_rys_ip1_sqlite(database)
    assert profile == {
        "ppps": {
            "kernel_time_milliseconds": 5.0,
            "launches": 2,
            "kernel_names": sorted(names[index] for index in (1, 2)),
        },
        "ppss": {
            "kernel_time_milliseconds": 1.0,
            "launches": 1,
            "kernel_names": [names[3]],
        },
    }


def test_active_shell_class_histogram_ranks_screened_primitive_work():
    histogram = _shell_histogram_module()
    entries = [
        SimpleNamespace(
            label="dppp",
            shell_angular=(2, 1, 1, 1),
            shell_quartets=3,
            tiles=5,
            ao_quartets=900,
            primitive_quartets=1800,
        ),
        SimpleNamespace(
            label="dpds",
            shell_angular=(2, 1, 2, 0),
            shell_quartets=2,
            tiles=4,
            ao_quartets=700,
            primitive_quartets=2100,
        ),
        SimpleNamespace(
            label="pppp",
            shell_angular=(1, 1, 1, 1),
            shell_quartets=10,
            tiles=10,
            ao_quartets=1000,
            primitive_quartets=9000,
        ),
    ]
    rows = histogram.summarize_active_shell_classes(entries, angular_order=5)
    assert [row["class"] for row in rows] == ["dpds", "dppp"]
    assert sum(row["primitive_work_fraction"] for row in rows) == pytest.approx(1.0)
    assert sum(row["tile_fraction"] for row in rows) == pytest.approx(1.0)

    all_rows = histogram.summarize_active_shell_classes(
        entries, angular_order=None
    )
    assert [row["class"] for row in all_rows] == ["pppp", "dpds", "dppp"]
    assert sum(row["primitive_work_fraction"] for row in all_rows) == pytest.approx(
        1.0
    )


def test_ppps_queue_summary_labels_block_orientation_and_overflow_buckets():
    histogram = _shell_histogram_module()
    profile = SimpleNamespace(
        descriptor_slots=10,
        non_empty_descriptors=4,
        empty_descriptors=6,
        hole_rate=0.6,
        tasks=40,
        primitive_work=400,
        ket_count_min=2,
        ket_count_median=8,
        ket_count_p90=16,
        ket_count_p99=16,
        ket_count_max=16,
        lane_efficiency=(0.5, 0.25, 0.125, 0.0625),
        primitive_warp_efficiency=0.75,
        task_tail_imbalance=(0.1, 0.2, 0.3, 0.4),
        primitive_tail_imbalance=(0.2, 0.3, 0.4, 0.5),
        orientation_tasks=(30, 10),
        orientation_primitive_work=(250, 150),
        bra_primitive_tasks=(0,) * 2 + (40,) + (0,) * 62,
        bra_primitive_work=(0,) * 2 + (400,) + (0,) * 62,
        ket_primitive_tasks=(0,) * 64 + (40,),
        ket_primitive_work=(0,) * 64 + (400,),
    )

    summary = histogram.summarize_ppps_queue_profile(profile)
    assert summary["lane_efficiency"] == {
        "32": 0.5,
        "64": 0.25,
        "128": 0.125,
        "256": 0.0625,
    }
    assert summary["orientation"]["1011"] == {
        "tasks": 10,
        "primitive_work": 150,
    }
    assert summary["bra_primitive_pair_groups"] == [
        {"primitive_pairs": 2, "tasks": 40, "primitive_work": 400}
    ]
    assert summary["ket_primitive_pair_groups"] == [
        {"primitive_pairs": "64+", "tasks": 40, "primitive_work": 400}
    ]


def test_shell_histogram_runtime_switch_matches_native_opt_out():
    histogram = _shell_histogram_module()
    assert histogram.runtime_switch_enabled(None)
    assert histogram.runtime_switch_enabled("1")
    assert histogram.runtime_switch_enabled("enabled")
    assert not histogram.runtime_switch_enabled("0")
    assert not histogram.runtime_switch_enabled("none")
    assert histogram.ppps_block_threads(None) == 256
    assert histogram.ppps_block_threads("32") == 32
    assert histogram.ppps_block_threads("64") == 64
    assert histogram.ppps_block_threads("128") == 128
    assert histogram.ppps_block_threads("256") == 256
    assert histogram.ppps_block_threads("96") == 0
