"""Reference qualification must not weaken or silently replace historical gates."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "benchmarks"))
    return importlib.import_module("real_molecule_gate")


def test_reference_override_changes_only_requested_reference(gate, tmp_path):
    for point in gate.real_molecule_gate_points():
        kwargs = {
            "repeats": 7,
            "output": tmp_path / "point.json",
            "density_fitting_memory_budget_bytes": 0,
        }
        historical = gate._point_command(point, **kwargs)
        tightened = gate._point_command(
            point, reference_gradient_overrides={96: 1e-11}, **kwargs
        )
        expected = historical.copy()
        if point.expected_ao_count == 96:
            expected[expected.index("--reference-gradient-tolerance") + 1] = "1e-11"
        assert tightened == expected
        assert tightened[tightened.index("--maximum-force-error") + 1] == str(
            point.maximum_force_error
        )


@pytest.mark.parametrize(
    "value", ["96=nan", "96=inf", "96=-1e-11", "96=0", "0=1e-11", "96", "x=1e-11"]
)
def test_invalid_reference_override_rejected(gate, value):
    with pytest.raises(gate.argparse.ArgumentTypeError):
        gate._reference_override(value)


def test_reference_loosening_rejected(gate):
    point = gate.real_molecule_gate_points()[0]
    with pytest.raises(ValueError, match="only tighten"):
        gate._reference_tolerance(
            point, {point.expected_ao_count: point.reference_gradient_tolerance * 10}
        )


@pytest.mark.parametrize("overrides", [["96=1e-11", "96=1e-12"], ["384=1e-11"]])
def test_ambiguous_or_unused_overrides_rejected(gate, monkeypatch, tmp_path, overrides):
    argv = ["gate", "--dry-run", "--output-directory", str(tmp_path)]
    for value in overrides:
        argv += ["--reference-gradient-tolerance", value]
    monkeypatch.setattr(gate.sys, "argv", argv)
    with pytest.raises(ValueError):
        gate.main()


def test_summary_records_both_policies_and_preserves_failed_point(
    gate, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        gate.sys,
        "argv",
        [
            "gate",
            "--repeats",
            "7",
            "--output-directory",
            str(tmp_path),
            "--reference-gradient-tolerance",
            "96=1e-11",
        ],
    )
    calls = []

    def run(command, check):
        calls.append(command)
        failed = len(calls) == 1
        output = Path(command[command.index("--output") + 1])
        output.write_text(
            json.dumps(
                {
                    "gate": {
                        "passed": not failed,
                        "failures": ["retained"] if failed else [],
                    }
                }
            )
        )
        return SimpleNamespace(returncode=2 if failed else 0)

    monkeypatch.setattr(gate.subprocess, "run", run)
    with pytest.raises(SystemExit) as error:
        gate.main()
    assert error.value.code == 2
    result = json.loads((tmp_path / "summary.json").read_text())
    assert len(calls) == 4
    assert not result["passed"]
    assert result["reference_convergence_policy"] == "explicit_tightening"
    assert result["reference_gradient_overrides"] == {"96": 1e-11}
    for point in result["points"]:
        if point["ao_count"] == 96:
            assert point["historical_reference_gradient_tolerance"] == 1e-9
            assert point["reference_gradient_tolerance"] == 1e-11
            assert point["maximum_force_error_hartree_per_bohr"] == 3e-11
        else:
            assert (
                point["historical_reference_gradient_tolerance"]
                == point["reference_gradient_tolerance"]
                == 1e-8
            )


def test_full_fock_control_is_scoped_to_selected_reference(gate, tmp_path):
    for point in gate.real_molecule_gate_points():
        command = gate._point_command(
            point,
            repeats=7,
            output=tmp_path / "x.json",
            density_fitting_memory_budget_bytes=0,
            reference_full_fock_sizes={96},
        )
        assert ("--reference-full-fock" in command) == (point.expected_ao_count == 96)


@pytest.mark.parametrize("full_fock", [False, True])
def test_reference_fock_setting_does_not_change_numerical_gates(gate, full_fock):
    comparison = importlib.import_module("compare_gpu4pyscf_batch")
    engine = SimpleNamespace()
    comparison._configure_reference_scf(
        engine,
        energy_tolerance=1e-12,
        gradient_tolerance=1e-11,
        max_iterations=100,
        full_fock=full_fock,
    )
    assert engine.direct_scf is (not full_fock)
    assert engine.conv_tol == 1e-12
    assert engine.conv_tol_grad == 1e-11
    assert engine.direct_scf_tol == 1e-14
    assert engine.max_cycle == 100
