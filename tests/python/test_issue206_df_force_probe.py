"""Hardware-free protocol checks for the #206 force ledger."""

import copy
import hashlib
import json
import sys

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
