"""Hardware-free protocol checks for the #206 force ledger."""

import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from benchmarks import issue206_df_force_probe as probe


@pytest.fixture
def protocol(tmp_path, monkeypatch):
    """Use an identifiable fake binary; sample calls never touch CUDA."""
    library = tmp_path / "libvibeqc.so"
    library.write_bytes(b"protocol-only native library")
    # main() selects this binary through the process environment. Register it
    # with monkeypatch so later native tests recover their original library.
    monkeypatch.setenv("VIBEQC_LIBRARY", str(library))
    output = tmp_path / "ledger.json"
    monkeypatch.setenv("SLURM_JOB_ID", "protocol-test")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe",
            "--case",
            probe.CASES[0],
            "--library",
            str(library),
            "--output",
            str(output),
        ],
    )
    energy = {
        "seconds": 2.0,
        "iterations": 4,
        "converged": True,
        "energy_hartree": -1.0,
        "has_forces": False,
    }
    force = {**energy, "seconds": 5.5, "has_forces": True}
    return library, output, energy, force


@pytest.mark.parametrize("variable", ["SLURM_JOB_ID", "CUDA_VISIBLE_DEVICES"])
def test_probe_requires_slurm_and_cuda_visibility(protocol, monkeypatch, variable):
    monkeypatch.delenv(variable)
    with pytest.raises(SystemExit):
        probe.main()


def test_probe_writes_validated_force_increment_and_binary_identity(
    protocol, monkeypatch
):
    library, output, energy, force = protocol
    samples = iter([energy, force])

    def sample(case, properties, selected_library):
        assert selected_library == library.resolve()
        return next(samples)

    monkeypatch.setattr(probe, "_sample", sample)
    probe.main()
    payload = json.loads(output.read_text())
    record = payload["records"][0]
    assert payload["schema"] == "vibeqc.issue206.df_force_ledger"
    assert payload["version"] == 2
    assert record["force_increment_seconds"] == 3.5
    assert record["validation"]["valid"] is True
    assert record["validation"]["energy_tolerance_hartree"] == 1e-10
    assert payload["source"]["native_library"] == str(library.resolve())
    assert (
        payload["source"]["native_library_sha256"]
        == hashlib.sha256(library.read_bytes()).hexdigest()
    )
    assert isinstance(payload["source"]["git_dirty"], bool)


@pytest.mark.parametrize(
    "which,field,value,match",
    [
        (0, "converged", False, "converge"),
        (1, "converged", False, "converge"),
        (1, "iterations", 5, "iteration"),
        (1, "energy_hartree", -1.001, "energy parity"),
        (0, "has_forces", True, "force outputs"),
        (1, "has_forces", False, "force outputs"),
        (0, "energy_hartree", float("nan"), "finite energies"),
        (1, "seconds", float("inf"), "finite positive"),
        (0, "seconds", 0.0, "finite positive"),
    ],
)
def test_invalid_pairs_never_publish_a_ledger(
    protocol, monkeypatch, which, field, value, match
):
    _, output, energy, force = protocol
    samples = copy.deepcopy([energy, force])
    samples[which][field] = value
    sample_iter = iter(samples)
    monkeypatch.setattr(probe, "_sample", lambda *args: next(sample_iter))
    with pytest.raises(ValueError, match=match):
        probe.main()
    assert not output.exists()


def test_changed_binary_never_publishes_a_ledger(protocol, monkeypatch):
    library, output, energy, force = protocol
    samples = iter([energy, force])

    def sample(*args):
        library.write_bytes(b"different build during timing")
        return next(samples)

    monkeypatch.setattr(probe, "_sample", sample)
    with pytest.raises(RuntimeError, match="changed during"):
        probe.main()
    assert not output.exists()


def test_unrequested_trace_cannot_contaminate_unprofiled_evidence(
    protocol, monkeypatch
):
    monkeypatch.setenv("VIBEQC_DF_TRACE", "unexpected.jsonl")
    with pytest.raises(SystemExit):
        probe.main()


@pytest.mark.parametrize("omit_force", [False, True])
def test_trace_protocol_preserves_raw_evidence_and_requires_force_components(
    protocol, monkeypatch, omit_force
):
    library, output, energy, force = protocol
    directory = output.parent / "traces"
    monkeypatch.setattr(
        sys, "argv", [*sys.argv, "--component-trace-dir", str(directory)]
    )

    def sample(case, properties, selected_library):
        assert selected_library == library.resolve()
        operations = ["ri_j", "ri_k"]
        if "forces" in properties and not omit_force:
            operations.extend(["force_response", "one_electron_response"])
        rows = []
        for index, operation in enumerate(operations):
            rows.append(
                {
                    "schema": "vibeqc.df_trace",
                    "version": 1,
                    "id": index,
                    "operation": operation,
                    "execution": "stream",
                    "valid": True,
                    "cuda_error": 0,
                    "nvtx": True,
                    "systems": 1,
                    "system_offset": 0,
                    "nbf": 2,
                    "naux": 2,
                    "source_backed": True,
                    "streamed": True,
                    "final_synchronization_ms": 1,
                    "host_completion_ms": 3,
                    "profiler_event_count": 2,
                    "dropped_regions": 0,
                    "dropped_tiles": 0,
                    "regions": [
                        {"name": operation, "parent": -1, "host_ms": 2, "gpu_ms": 2}
                    ],
                    "counters": {},
                    "tiles": [],
                }
            )
        Path(os.environ["VIBEQC_DF_TRACE"]).write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        return dict(force if "forces" in properties else energy)

    monkeypatch.setattr(probe, "_sample", sample)
    if omit_force:
        with pytest.raises(ValueError, match="missing executed"):
            probe.main()
        assert not output.exists()
    else:
        probe.main()
        payload = json.loads(output.read_text())
        assert payload["execution"]["profiled"] is True
        raw = payload["records"][0]["energy_plus_force"]["components"]["raw_trace"]
        assert (
            raw["sha256"] == hashlib.sha256(Path(raw["path"]).read_bytes()).hexdigest()
        )
        assert "force_attribution" in payload["records"][0]
        with pytest.raises(FileExistsError):
            probe.main()
    assert "VIBEQC_DF_TRACE" not in os.environ
