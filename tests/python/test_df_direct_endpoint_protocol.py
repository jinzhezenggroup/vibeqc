"""Complete endpoint evidence cannot hide a failed or mismatched force sample."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.compare_df_direct_endpoint import check_endpoint, reference_work_counter
from tools.render_hf_acceptance_benchmarks import (
    checked_run,
    compact_table,
    digest,
    write_collection,
)


@pytest.mark.parametrize(
    "diagnostics",
    [[], [{"phase": "warm", "counters": {"bytes": 8769110016, "extra_jk_builds": 0}}]],
)
def test_warm_retention_preserves_missing_counts_and_work(
    tmp_path: Path, diagnostics: list[dict]
) -> None:
    """Compact rows must preserve null work counts, FP64 values and large counters."""
    metadata = {"aos": 768, "diagnostic_work": diagnostics}
    samples = [
        {"energy": -2431.1197119999997, "scf_jk_builds": None, "gate": True},
        {"gate": True, "energy": -2431.1197120000006, "scf_jk_builds": 2},
    ]
    path = tmp_path / "samples.json"
    write_collection(path, [{**metadata, "samples": compact_table(samples)}])
    record = json.loads(path.read_text())[0]
    assert record["diagnostic_work"] == diagnostics
    assert [
        dict(zip(record["samples"]["columns"], row, strict=True))
        for row in record["samples"]["rows"]
    ] == samples
    assert metadata == {"aos": 768, "diagnostic_work": diagnostics}


def test_warm_retention_rejects_nonfinite_diagnostics(tmp_path: Path) -> None:
    """An invalid work counter cannot survive a seemingly valid scalar sample."""
    with pytest.raises(ValueError):
        write_collection(
            tmp_path / "samples.json",
            [
                {
                    "diagnostic_work": [{"counter": float("nan")}],
                    "samples": compact_table([{"gate": True}]),
                }
            ],
        )


@pytest.mark.parametrize(
    "fault",
    ["shape", "nonfinite", "energy", "force", "convergence", "status", "oracle"],
)
def test_all_endpoint_samples_must_pass(fault: str) -> None:
    item = SimpleNamespace(
        energy=-76.0, forces=np.zeros((3, 3)), succeeded=True, converged=True
    )
    reference = {"energy": -76.0, "forces": np.zeros((3, 3)), "converged": True}
    assert check_endpoint(item, reference)["gate"]
    if fault == "shape":
        item.forces = np.zeros((1, 3))
    elif fault == "nonfinite":
        item.forces[0, 0] = np.nan
    elif fault == "energy":
        item.energy += 1e-7
    elif fault == "force":
        item.forces[0, 0] = 1e-6
    elif fault == "convergence":
        item.converged = False
    elif fault == "status":
        item.succeeded = False
        item.forces = None
    else:
        reference["converged"] = False
    assert not check_endpoint(item, reference)["gate"]


@pytest.mark.parametrize("owned", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_reference_counts_initial_fock_and_restores_engine(
    owned: bool, fails: bool
) -> None:
    """A cycle count must not hide the initial J/K build or corrupt later runs."""

    class Engine:
        callback = None

        def get_veff(self, density: np.ndarray) -> np.ndarray:
            return density

    engine = Engine()
    if owned:
        engine.get_veff = lambda density: 2 * density
    original = engine.get_veff
    try:
        with reference_work_counter(engine) as work:
            density = np.eye(2)
            engine.get_veff(density)  # Pre-loop energy baseline.
            engine.get_veff(density)  # One reported iteration.
            assert work["scf_jk_builds"] == 2
            if fails:
                raise RuntimeError("probe failure")
    except RuntimeError:
        assert fails
    assert engine.get_veff == original
    assert ("get_veff" in vars(engine)) == owned
    assert engine.callback is None


@pytest.mark.parametrize(
    "fault", ["cold_force", "early_warm", "missing_repeat", "identity"]
)
def test_readme_reduction_rejects_incomplete_or_failed_evidence(
    tmp_path: Path, fault: str
) -> None:
    """The figure must not hide cold failures or select only the best last repeat."""
    oracle = {"energy": -76.0, "forces": [[0.0, 0.0, 0.0]], "converged": True}
    protocol = {"aos": 24}
    reference = {
        "identity": {"protocol": protocol},
        "records": [
            {
                **oracle,
                "geometry": i,
                "phase": phase,
                "iterations": 1,
                "scf_jk_builds": 2,
                "complete_seconds": 1.0,
            }
            for i, phase in enumerate(("cold", "moved"))
        ],
        "reference_samples": [
            {
                **oracle,
                "geometry": i,
                "phase": phase,
                "iterations": 1,
                "scf_jk_builds": 2,
                "complete_seconds": 1.0,
            }
            for i, phase in enumerate(("warm", "moved-warm"))
            for _ in range(2)
        ],
    }
    ref_path = tmp_path / "reference.json"
    ref_path.write_text(json.dumps(reference))
    native = {
        "identity": {"protocol": protocol, "reference_sha256": digest(ref_path)},
        "records": [
            {**row, "status": 0, "gate": True, "diagnostic": False}
            for row in reference["records"] + reference["reference_samples"]
        ],
    }
    path = tmp_path / "native.json"
    path.write_text(json.dumps(native))
    checked_run(path, ref_path, 2)
    if fault == "cold_force":
        native["records"][0]["forces"] = [[0.0, 0.0, 1e-5]]
    elif fault == "early_warm":
        native["records"][2]["energy"] += 1e-5
    elif fault == "missing_repeat":
        native["records"].pop()
    else:
        native["identity"]["reference_sha256"] = "different reference"
    path.write_text(json.dumps(native))
    with pytest.raises(ValueError):
        checked_run(path, ref_path, 2)
