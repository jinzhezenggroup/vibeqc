"""Scientific consumers of the live, explicitly source-scoped force experiment.

Recompute all-repeat numerical gates and producer conservation, rather than
retaining a historical report solely for a checksum or sample-count assertion.
"""

from pathlib import Path

import numpy as np
import pytest

from tools.generativeqc_validation.record import load_publication_record

DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/results/pbe0-ao-density-force-20261004"
)


@pytest.mark.parametrize(
    ("campaign", "atoms"),
    [
        (campaign, atoms)
        for campaign in ("master", "prototype")
        for atoms in (3, 6, 12, 24, 48, 96)
    ]
    + [("prototype", 4)],
)
def test_frozen_force_screening_all_repeat_scientific_gates(
    campaign: str, atoms: int
) -> None:
    """No timing/iteration filter may remove an inaccurate energy or force row."""
    entries = load_publication_record(
        DIRECTORY, role="samples", name=f"{campaign}-{atoms}.json.gz"
    )
    reference = entries["reference"]
    phases = [
        ("cold", 0, 0),
        *[("warm", 0, repeat) for repeat in range(5)],
        ("moved", 1, 0),
        *[("moved-warm", 1, repeat) for repeat in range(5)],
    ]
    for entry in entries.values():
        assert [
            (row["phase"], row["geometry"], row["repeat"])
            for row in entry["result"]["records"]
        ] == phases
    xc_objects = reference["receipt"]["reference_xc_objects"]
    assert xc_objects and all(item["on_gpu"] for item in xc_objects)
    for name, entry in entries.items():
        if name == "reference":
            continue
        native = entry["result"]
        assert native["protocol"] == reference["result"]["protocol"]
        assert entry["outcome"]["exit_code"] == 0
        for row in native["records"]:
            assert row["status"] == 0 and row["converged"]
            assert np.isfinite(row["complete_seconds"]) and row["complete_seconds"] > 0
            force = np.asarray(row["forces"])
            assert force.shape == (atoms, 3) and np.isfinite(force).all()
            for oracle in reference["result"]["records"]:
                if oracle["geometry"] != row["geometry"]:
                    continue
                assert abs(row["energy"] - oracle["energy"]) <= 1e-8
                np.testing.assert_allclose(force, oracle["forces"], atol=1e-7, rtol=0)


def test_observed_generic_force_work_and_independent_source_arrays() -> None:
    """The producer gates conserve work; both source channels remain accurate."""
    record = load_publication_record(
        DIRECTORY, role="samples", name="producer-sources.json.gz"
    )
    expected = np.asarray(record["accepted_snapshot"]["derivative_sources"]).reshape(-1)
    rows = record["rows"]
    assert [row["label"] for row in rows if row["kind"] == "replay"] == [
        "work-0",
        "work-1",
    ]
    for row in rows:
        if row["kind"] == "replay":
            assert row["instrumented"]
            np.testing.assert_allclose(row["derivatives"], expected, atol=1e-9, rtol=0)
        elif row["kind"] == "actual-generic-force-work":
            assert row["angular_order"] >= 4
            assert row["decoded"] == sum(
                row[key]
                for key in (
                    "schwarz_rejected",
                    "density_rejected",
                    "zero_weight_rejected",
                    "admitted",
                )
            )
            if row["angular_order"] <= 6:
                assert row["explicit_gradient_evaluations"] == row["admitted"]
                assert row["dual3_gradient_evaluations"] == 0
            else:
                assert row["explicit_gradient_evaluations"] == 0
                assert row["dual3_gradient_evaluations"] <= 3 * row["admitted"]
