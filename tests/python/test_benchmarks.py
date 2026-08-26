import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

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
