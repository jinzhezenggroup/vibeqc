"""Every supported endpoint must survive compact evidence partitioning."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest


@pytest.fixture(scope="module")
def renderer() -> ModuleType:
    # Plotting is optional and outside this test: load the exact reducer while
    # replacing only the unused plotting imports, not its reduction functions.
    path = Path(__file__).resolve().parents[2] / "tools/render_readme_benchmarks.py"
    spec = importlib.util.spec_from_file_location("reviewed_readme_reducer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    matplotlib = ModuleType("matplotlib")
    matplotlib.use = lambda *args: None
    with patch.dict(
        sys.modules,
        {"matplotlib": matplotlib, "matplotlib.pyplot": ModuleType("matplotlib.pyplot")},
    ):
        spec.loader.exec_module(module)
    return module


def _hf(*, forces: bool, passed: bool) -> dict:
    convergence = [{"iterations": 2, "converged": True}]
    engine = {
        "cold_seconds": 0.5,
        "cold_convergence": convergence,
        "warm_samples": [{"seconds": 0.25, "convergence": convergence}],
    }
    return {
        "workload": {
            "properties": ["energy", "forces"] if forces else ["energy"],
            "density_fitting": "cuda",
            "geometries": [[[1, 0, 0, 0], [1, 0, 0, 1]]],
            "ao_count": 2,
            "auxiliary_basis": "fixture",
        },
        "gate": {"passed": passed},
        "accuracy": {"gate_selection": {}, "paired_warm_repeats": []},
        "settings": {"gates": {}, "density_fitting_metric_diagnostics": {}},
        "vibeqc": engine,
        "gpu4pyscf": engine,
    }


def _run(
    renderer: ModuleType, monkeypatch: pytest.MonkeyPatch, root: Path, destination: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["reduce", "--raw-directory", str(root), "--destination", str(destination)],
    )
    # Test the real reducer, file partition and manifest, not Matplotlib rendering.
    monkeypatch.setattr(renderer, "figures", lambda *args: None)
    renderer.main()


@pytest.mark.parametrize("passed", [True, False])
def test_hf_energy_and_force_records_are_both_retained(
    renderer: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, passed: bool
) -> None:
    root, destination = tmp_path / "raw", tmp_path / "out"
    (root / "hf").mkdir(parents=True)
    for name, forces in (("energy", False), ("force", True)):
        (root / "hf" / f"{name}.json").write_text(
            json.dumps(_hf(forces=forces, passed=passed))
        )
    _run(renderer, monkeypatch, root, destination)
    manifest = json.loads((destination / "summary.json").read_text())
    rows = []
    for part in manifest["samples"]:
        data = (destination / part["path"]).read_bytes()
        selected = json.loads(data)
        assert part["sha256"] == hashlib.sha256(data).hexdigest()
        assert part["bytes"] == len(data)
        assert part["records"] == len(selected)
        rows.extend(selected)
    assert {row["family"] for row in rows} == {"hf", "hf_energy"}
    assert len(rows) == 2
    for row in rows:
        assert row["status"] == ("measured" if passed else "failed")
        assert row["raw_sha256"] == hashlib.sha256(
            (root / row["raw_file"]).read_bytes()
        ).hexdigest()
        assert row["engines"]["VibeQC"]["samples"][0]["ms"] == 250
    assert {row["endpoint"] for row in rows} == {
        "SCF energy",
        "energy + analytic forces",
    }


@pytest.mark.parametrize(
    "record",
    [{"family": "future"}, {"family": "dft", "method": "future-rks"}],
)
def test_unknown_partition_is_rejected_not_silently_dropped(
    renderer: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, record: dict
) -> None:
    root, destination = tmp_path / "raw", tmp_path / "out"
    root.mkdir()
    (root / "stopped-points.json").write_text(json.dumps([record]))
    with pytest.raises(ValueError, match="unassigned benchmark records"):
        _run(renderer, monkeypatch, root, destination)
    assert not destination.exists()


def test_explicit_dft_atom_count_does_not_require_arguments(
    renderer: ModuleType, tmp_path: Path
) -> None:
    raw = tmp_path / "point.json"
    raw.write_text(json.dumps({"method": "pbe-rks", "atoms": 3, "status": "failed"}))
    record, _ = renderer.reduce_point(raw, tmp_path)
    assert record["atoms"] == 3
    assert record["status"] == "failed"
    assert record["engines"] == {}
