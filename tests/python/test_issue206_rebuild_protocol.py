"""Every recorded phase must qualify; changed endpoints include reset work."""

import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import issue206_rebuild as runner


def _engine(clock: list[float], events: list[str], converged: bool) -> SimpleNamespace:
    def reset(molecule: object) -> None:
        clock[0] += 7.0
        events.append("reset")

    def kernel(*, dm0: object) -> float:
        clock[0] += 3.0
        events.append("kernel")
        return -1.0

    def gradient() -> np.ndarray:
        clock[0] += 2.0
        events.append("gradient")
        return np.zeros((2, 3))

    return SimpleNamespace(
        mol=SimpleNamespace(set_geom_=lambda *args, **kwargs: object()),
        reset=reset,
        kernel=kernel,
        converged=converged,
        nuc_grad_method=lambda: SimpleNamespace(kernel=gradient),
    )


def test_changed_reference_timer_includes_geometry_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock, events = [0.0], []
    monkeypatch.setattr(runner, "_sync", lambda cp: events.append("sync"))
    monkeypatch.setattr(runner.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(runner, "gpu_convergence_payload", lambda *args: [])
    engine = _engine(clock, events, True)
    seed = np.eye(2)
    result = runner._stock_sample(
        [engine],
        [seed],
        SimpleNamespace(asnumpy=np.asarray),
        systems=[[("H", (0, 0, 0)), ("H", (0, 0, 1))]],
        coordinates=[np.zeros((2, 3))],
    )
    assert result["seconds"] == 12.0
    assert events == ["sync", "reset", "kernel", "gradient", "sync"]
    np.testing.assert_array_equal(seed, np.eye(2))


def test_nonconverged_reference_is_rejected_before_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    monkeypatch.setattr(runner, "_sync", lambda cp: None)
    monkeypatch.setattr(runner, "gpu_convergence_payload", lambda *args: [])
    engine = _engine([0.0], events, False)
    with pytest.raises(RuntimeError, match="converg"):
        runner._stock_sample([engine], [np.eye(2)], SimpleNamespace(asnumpy=np.asarray))
    assert "gradient" not in events


def test_streamed_route_does_not_satisfy_explicit_fallback() -> None:
    from benchmarks.issue206_resident_sentinel import validate_response_record

    record = {
        "operation": "force_response",
        "nbf": 4,
        "naux": 5,
        "source_backed": True,
        "counters": {},
        "tiles": [],
    }
    with pytest.raises(RuntimeError, match="fallback"):
        validate_response_record(record, expected_policy="fallback")
    assert (
        validate_response_record(record, expected_policy="streamed")["policy"][
            "residency"
        ]
        == "streamed"
    )


@pytest.mark.parametrize(
    "upload,scatter", [("drain", ""), ("packed", ""), ("", "sharded")]
)
def test_host_fallback_rejects_unexecuted_attribution_probes(
    upload: str, scatter: str
) -> None:
    from benchmarks import issue308_response_timeline as timeline

    with pytest.raises(RuntimeError, match="fallback"):
        timeline.validate_host_fallback_probes(upload, scatter)
    timeline.validate_host_fallback_probes("", "")


@pytest.mark.parametrize(
    "malformed", ["batch", "atoms", "coordinates", "scalar", "complex", "nan"]
)
def test_endpoint_comparison_rejects_incomplete_or_nonphysical_arrays(
    malformed: str,
) -> None:
    reference = {
        "energies_hartree": np.array([-1.0, -1.0]),
        "forces_hartree_per_bohr": np.zeros((2, 2, 3)),
    }
    candidate = {key: value.copy() for key, value in reference.items()}
    if malformed == "batch":
        candidate = {key: value[:1] for key, value in candidate.items()}
    elif malformed == "atoms":
        candidate["forces_hartree_per_bohr"] = np.zeros((2, 1, 3))
    elif malformed == "coordinates":
        candidate["forces_hartree_per_bohr"] = np.zeros((2, 2, 1))
    elif malformed == "scalar":
        candidate["energies_hartree"] = np.array(-1.0)
    elif malformed == "complex":
        candidate["energies_hartree"] = candidate["energies_hartree"].astype(complex)
    else:
        candidate["forces_hartree_per_bohr"][0, 0, 0] = np.nan
    with pytest.raises((ValueError, TypeError), match="endpoint"):
        runner._paired_errors(candidate, reference)


def test_endpoint_comparison_preserves_complete_numerical_errors() -> None:
    """Retain maxima from different batch items and the last force coordinate."""
    reference = {
        "energies_hartree": [-1.0, -2.0],
        "forces_hartree_per_bohr": np.zeros((2, 2, 3)),
    }
    candidate = {
        "energies_hartree": [-1.25, -2.0],
        "forces_hartree_per_bohr": np.zeros((2, 2, 3)),
    }
    candidate["forces_hartree_per_bohr"][1, 1, 2] = -0.5
    assert runner._paired_errors(candidate, reference) == {
        "maximum_energy_error_hartree": 0.25,
        "maximum_force_error_hartree_per_bohr": 0.5,
    }


@pytest.mark.parametrize("cold_error", ["none", "energy", "force"])
def test_cold_endpoint_participates_in_cli_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cold_error: str
) -> None:
    """Exercise the real CLI gate with fake engines, not a real GPU or library.

    All replays agree exactly, so only the saved cold result can reject this
    campaign. This protects the full result wiring, not just pair subtraction.
    """
    correct = {
        "seconds": 1.0,
        "energies_hartree": [-1.0],
        "forces_hartree_per_bohr": np.zeros((1, 2, 3)).tolist(),
        "convergence": [{"converged": True, "iterations": 2}],
    }
    cold_forces = np.zeros((2, 3))
    if cold_error == "force":
        cold_forces[1, 2] = 1e-3
    cold_result = SimpleNamespace(
        energies=np.array([-1.0 + (1e-3 if cold_error == "energy" else 0.0)]),
        items=[SimpleNamespace(forces=cold_forces)],
    )
    batch = SimpleNamespace(
        execute=lambda *args, **kwargs: cold_result,
        last_density_fitting_metric_diagnostics=list,
        set_warm_start_updates=lambda enabled: None,
    )
    calculator = SimpleNamespace(
        prepare_batch=lambda *args, **kwargs: nullcontext(batch)
    )
    monkeypatch.setattr("vibeqc.Calculator", lambda **kwargs: calculator)
    cp = SimpleNamespace(asnumpy=np.asarray)
    monkeypatch.setitem(sys.modules, "cupy", cp)
    monkeypatch.setitem(sys.modules, "gpu4pyscf", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "gpu4pyscf.scf", SimpleNamespace(uhf=object()))
    monkeypatch.setitem(
        sys.modules,
        "pyscf",
        SimpleNamespace(
            df=SimpleNamespace(
                addons=SimpleNamespace(
                    make_auxmol=lambda *args: SimpleNamespace(nao_nr=lambda: 2)
                )
            ),
            gto=object(),
            scf=object(),
        ),
    )
    case = SimpleNamespace(
        method="rhf", basis_representation="spherical", charge=0, multiplicity=1
    )
    basis = SimpleNamespace(name="fake-shared-basis")
    systems = [[("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.0))]]
    coordinates = [np.array([position for _, position in systems[0]])]
    engine = SimpleNamespace(
        mol=SimpleNamespace(nao_nr=lambda: 2),
        kernel=lambda: -1.0,
        converged=True,
        nuc_grad_method=lambda: SimpleNamespace(kernel=lambda: np.zeros((2, 3))),
        make_rdm1=lambda: np.eye(2),
    )
    monkeypatch.setattr(runner, "benchmark_cases", lambda: {"fake": case})
    monkeypatch.setattr(
        runner, "_case_inputs", lambda *args: (basis, basis, basis, basis, {})
    )
    monkeypatch.setattr(
        runner, "_coordinates", lambda *args: (systems, coordinates, coordinates)
    )
    monkeypatch.setattr(runner, "_stock_engines", lambda *args: [engine])
    monkeypatch.setattr(runner, "_reset_stock", lambda *args: None)
    monkeypatch.setattr(runner, "_sync", lambda *args: None)
    monkeypatch.setattr(runner, "_native_sample", lambda *args: correct)
    monkeypatch.setattr(runner, "_stock_sample", lambda *args, **kwargs: correct)
    monkeypatch.setattr(
        runner, "convergence_payload", lambda *args: correct["convergence"]
    )
    monkeypatch.setattr(
        runner, "gpu_convergence_payload", lambda *args: correct["convergence"]
    )
    monkeypatch.setattr(
        runner,
        "native_build_metadata",
        lambda *args: {"probe": {"source_identity": "fake"}, "library_sha256": "fake"},
    )
    # Both GPU imports and native construction are replaced above. These env
    # values exercise the CLI preflight only; no physical device is accessed.
    monkeypatch.setenv("SLURM_JOB_ID", "fake")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "fake")
    library, output = tmp_path / "fake.so", tmp_path / "result.json"
    library.touch()
    # main selects this library globally; register the key so teardown restores
    # the previous selection rather than poisoning later native tests.
    monkeypatch.setenv("VIBEQC_LIBRARY", str(library))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "issue206_rebuild",
            "--case",
            "fake",
            "--library",
            str(library),
            "--output",
            str(output),
        ],
    )
    if cold_error == "none":
        runner.main()
    else:
        with pytest.raises(SystemExit) as stopped:
            runner.main()
        assert stopped.value.code == 2
    payload = json.loads(output.read_text())
    assert payload["status"] == (
        "pass" if cold_error == "none" else "numerical-gate-failed"
    )
    assert payload["version"] == 3
    assert payload["accuracy"]["cold_pair"][
        "maximum_energy_error_hartree"
    ] == pytest.approx(1e-3 if cold_error == "energy" else 0.0)
    assert payload["accuracy"]["cold_pair"][
        "maximum_force_error_hartree_per_bohr"
    ] == pytest.approx(1e-3 if cold_error == "force" else 0.0)


@pytest.mark.parametrize("selected_library", [None, "/existing/native-library.so"])
@pytest.mark.parametrize("cold_error", ["none", "energy", "force"])
def test_cold_cli_stub_does_not_leak_library_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selected_library: str | None,
    cold_error: str,
) -> None:
    """A fake CLI run must not redirect later native tests to its empty .so."""
    if selected_library is None:
        monkeypatch.delenv("VIBEQC_LIBRARY", raising=False)
    else:
        monkeypatch.setenv("VIBEQC_LIBRARY", selected_library)
    with pytest.MonkeyPatch.context() as isolated:
        test_cold_endpoint_participates_in_cli_acceptance(
            tmp_path, isolated, cold_error
        )
    assert os.environ.get("VIBEQC_LIBRARY") == selected_library
