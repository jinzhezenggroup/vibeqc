"""Independently verify complete matched AO endpoints and semantic work counts."""

import argparse
import hashlib
import json
import lzma
import math
import statistics
from pathlib import Path

import numpy as np

SOURCE = "edb9bf98d454f356ef3dea4488a6a763343510bad25efb4d67f96dc877c27421"
LIBRARY = "473f05abacad690caa7a3e149aa309d723d021e1ca4fd4ec91b78512e16268f2"
HARNESS = "7af86061955304c6e7dbf1b834043309364591520ea99c982f535f09707b1ce9"
BASE = "b2ee9dd7fc6f82427ab56218d7b9d803687259be"
JOBS = ("5565", "5566", "5567")
CAMPAIGNS = {
    "frozen": (SOURCE, LIBRARY, BASE, JOBS, ""),
    "current": (
        "46852852006dc81b164796e994bc335c4f336da3f03e38ba76bec3c3e0269111",
        "c2f7c6e192a83ff09816af2e0834c3a22fb0190009e7592a9f2e5c3025b04254",
        "a718695de66d04af272addb61b2b28ebd6700eb4",
        ("5573", "5574", "5578"),
        "current-",
    ),
}
PHASES = ("cold", "warm", "moved", "moved-warm")
ROWS = [("cold", 0, 0)] + [("warm", 0, repeat) for repeat in range(5)]
ROWS += [("moved", 1, 0)] + [("moved-warm", 1, repeat) for repeat in range(5)]
WORK_SIGNATURE = (
    "point_ao_visits",
    "point_ao_square_sum",
    "dense_point_ao_square_sum",
    "active_aos_min",
    "active_aos_max",
    "active_aos_sum",
    "tile_count",
    "empty_tile_count",
    "retained_map_bytes",
    "budget_bytes",
)


_MEMBERS = {}


def digest(path: Path) -> str:
    return hashlib.sha256(_MEMBERS[path.as_posix()].encode()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(_MEMBERS[path.as_posix()])


def complete_rows(record: dict, atoms: int) -> list[dict]:
    """Reject missing repeats, failed convergence, or nonfinite physical values."""
    assert record["status"] == "measured" and record["stage"] == "complete"
    rows = record["records"]
    assert [(row["phase"], row["geometry"], row["repeat"]) for row in rows] == ROWS
    for row in rows:
        assert row["converged"] and row["status"] == 0 and row["gate"] is True
        assert math.isfinite(row["energy"])
        forces = np.asarray(row["forces"])
        assert forces.shape == (atoms, 3) and np.isfinite(forces).all()
        assert math.isfinite(row["complete_seconds"]) and row["complete_seconds"] > 0
    return rows


def verify_variant(
    directory: Path,
    atoms: int,
    mode: str,
    reference: dict,
    reference_hash: str,
) -> dict:
    """Check all oracle pairs, exact work, budget bounds and geometry reuse."""
    path = directory / f"{mode}.json"
    record = load(path)
    campaign = load(path.with_suffix(".campaign.json"))
    assert load(path.with_suffix(".outcome")) == {"exit_code": 0}
    assert record["protocol"] == reference["protocol"]
    assert record["reference_sha256"] == reference_hash
    assert record["protocol"]["energy_gate"] == 1e-8
    assert record["protocol"]["force_gate"] == 1e-7
    assert record["engine"] == "native"
    assert campaign["source_identity"] == SOURCE
    assert campaign["library_sha256"] == LIBRARY
    assert campaign["harness_sha256"] == HARNESS
    assert campaign["base_commit"] == BASE and campaign["mode"] == mode
    job = JOBS[2] if mode == "zero-budget" else JOBS[1] if atoms == 96 else JOBS[0]
    assert campaign["job"] == job
    assert campaign["cuda_visible_devices"] is not None
    assert record["scheduler"]["SLURM_JOB_ID"] == job
    assert record["scheduler"]["SLURM_JOB_PARTITION"] == "main"
    assert record["native_build"]["library_sha256"] == LIBRARY
    assert record["native_build"]["probe"]["source_identity"] == SOURCE
    rows = complete_rows(record, atoms)
    calls = campaign["force_calls"]
    assert len(calls) == len(rows) == 12
    points = atoms * 48 * 16 * 32
    aos = record["protocol"]["aos"]
    assert aos == atoms * 8
    pair_visits = (1 + 2 * points) * atoms * (atoms - 1) // 2
    errors = {"energy": 0.0, "force": 0.0}
    for row, call in zip(rows, calls, strict=True):
        for oracle in reference["records"]:
            if oracle["geometry"] == row["geometry"]:
                errors["energy"] = max(
                    errors["energy"], abs(row["energy"] - oracle["energy"])
                )
                errors["force"] = max(
                    errors["force"],
                    float(np.max(np.abs(np.asarray(row["forces"]) - oracle["forces"]))),
                )
        grid = call["grid_work_plan"]
        components = row["native_force_components"]
        assert grid == components["grid_work_plan"]
        assert grid["grid_points"] == call["xc_points"] == points
        assert grid["grid_pair_visits"] == call["grid_pair_visits"] == pair_visits
        assert grid["tile_points"] == 256 and grid["tile_count"] == points // 256
        assert components["work_counts"]["executed"] == {
            "semilocal_geometry_points": points,
            "partition_grid_pair_visits": pair_visits,
        }
        assert call["grid_density_source"] == "exact-final-scf-device-binding"
        for space in ("device", "host"):
            bound = f"additional_{space}_" + (
                "peak_bound" if space == "device" else "numeric_bound"
            )
            budget = f"additional_{space}_budget"
            assert call[bound] == components["resource_bounds"][bound]
            assert (
                0 < call[bound] <= call[budget] == components["resource_bounds"][budget]
            )
            assert call[budget] == (512 << 20 if space == "device" else 256 << 20)
        selection = call["resident_ao_selection"]
        assert (
            selection["full_ao_capacity"] == aos and selection["derivative_order"] == 2
        )
        work = selection["work"]
        if mode == "dense":
            assert selection["mode"] == "disabled" and selection["cutoff"] is None
            assert selection["cache_host_reserve_bytes"] == 0 and work is None
            continue
        assert selection["mode"] == "explicit-sampled-jet-cutoff"
        assert selection["cutoff"] == 1e-16
        assert 0 <= work["numeric_peak_bound_bytes"] <= work["budget_bytes"]
        assert work["budget_bytes"] == selection["cache_host_reserve_bytes"]
        assert work["lookups"] == work["tile_count"] == grid["tile_count"]
        assert work["discovery_density_contractions"] == 0
        assert work["dense_capability_tiles"] == 0
        assert work["dense_point_ao_square_sum"] == points * aos**2
        assert 0 <= work["active_aos_min"] <= work["active_aos_max"] <= aos
        assert 0 <= work["empty_tile_count"] <= work["tile_count"]
        assert work["point_ao_visits"] == 256 * work["active_aos_sum"]
        assert 0 <= work["point_ao_square_sum"] <= points * aos**2
        if mode == "zero-budget":
            assert (
                selection["cache_budget_requested_bytes"] == work["budget_bytes"] == 0
            )
            assert work["discoveries"] == work["cache_hits"] == 0
            assert work["discovery_seconds"] == work["discovery_ao_jet_values"] == 0
            assert work["retained_map_bytes"] == work["numeric_peak_bound_bytes"] == 0
            assert work["dense_budget_tiles"] == grid["tile_count"]
            assert work["active_aos_min"] == work["active_aos_max"] == aos
            assert work["point_ao_square_sum"] == points * aos**2
        else:
            assert (
                selection["cache_budget_requested_bytes"]
                == work["budget_bytes"]
                == 16 << 20
            )
            assert work["dense_budget_tiles"] == 0
            assert (
                work["retained_map_bytes"] + work["transient_reserve_bytes"]
                == work["numeric_peak_bound_bytes"]
            )
            assert work["empty_tile_count"] > 0
            assert work["point_ao_square_sum"] < points * aos**2
            refresh = row["phase"] in ("cold", "moved")
            assert work["discoveries"] == (grid["tile_count"] if refresh else 0)
            assert work["cache_hits"] == (0 if refresh else grid["tile_count"])
            assert work["discovery_ao_jet_values"] == (
                points * aos * 10 if refresh else 0
            )
            assert (work["discovery_seconds"] > 0) is refresh
            initial = calls[0 if row["geometry"] == 0 else 6]["resident_ao_selection"][
                "work"
            ]
            assert all(work[key] == initial[key] for key in WORK_SIGNATURE)
    assert errors["energy"] <= 1e-8 and errors["force"] <= 1e-7, (atoms, mode, errors)
    return {
        "all_reference_pair_maximum_errors": errors,
        "phase_medians_seconds": {
            phase: statistics.median(
                row["complete_seconds"] for row in rows if row["phase"] == phase
            )
            for phase in PHASES
        },
        "phase_iterations": {
            phase: [row["iterations"] for row in rows if row["phase"] == phase]
            for phase in PHASES
        },
        "sha256": digest(path),
        "campaign_sha256": digest(path.with_suffix(".campaign.json")),
        "source_identity": SOURCE,
        "library_sha256": LIBRARY,
        "job": job,
        "cuda_visible_devices": campaign["cuda_visible_devices"],
        "source_file_sha256": record["source_file_sha256"],
        "grid_work_plan": calls[0]["grid_work_plan"],
        "resources": rows[0]["native_force_components"]["resource_bounds"],
        "force_calls": calls,
    }


def load_bundle(campaign: str = "frozen") -> dict[str, str]:
    """Use separately pinned identities; never infer them from retained claims."""
    global _MEMBERS, SOURCE, LIBRARY, BASE, JOBS
    SOURCE, LIBRARY, BASE, JOBS, prefix = CAMPAIGNS[campaign]
    directory = Path(__file__).resolve().parent
    storage = json.loads((directory / f"{prefix}storage.json").read_text())
    assert storage["schema"] == "generativeqc.lossless-utf8-evidence.v1"
    assert storage["source_identity"] == SOURCE
    assert storage["library_sha256"] == LIBRARY
    assert storage["base_commit"] == BASE
    packed = (directory / f"{prefix}campaign.json.xz").read_bytes()
    assert hashlib.sha256(packed).hexdigest() == storage["sha256"]
    raw = lzma.decompress(packed)
    assert hashlib.sha256(raw).hexdigest() == storage["uncompressed_sha256"]
    members = json.loads(raw)
    assert set(members) == set(storage["members"]) and len(members) == 56
    for name, value in members.items():
        assert hashlib.sha256(value.encode()).hexdigest() == storage["members"][name]
    assert (
        hashlib.sha256(members["receipts/run-ao-endpoint.py"].encode()).hexdigest()
        == HARNESS
    )
    _MEMBERS = members
    return members


def verify_all() -> dict:
    """Keep every phase and compare only same-binary matched native variants."""
    summary = {
        "scope": "Same frozen candidate binary and 256-point/default-budget dense versus force-only local AO. SCF remains dense.",
        "reference_scope": "Retained independent numerical oracles only; no fresh reference timing comparison.",
        "cold_caveat": "Ordered processes share disk compiler/artifact caches; cold timings are not isolated compilation measurements.",
        "cases": [],
    }
    for atoms in (3, 6, 12, 24, 48, 96):
        directory = Path(".") / "endpoints" / str(atoms)
        reference_path = directory / "reference.json"
        reference = load(reference_path)
        complete_rows(reference, atoms)
        variants = {
            mode: verify_variant(
                directory, atoms, mode, reference, digest(reference_path)
            )
            for mode in ("dense", "local")
        }
        dense, local = variants["dense"], variants["local"]
        for key in (
            "source_identity",
            "library_sha256",
            "job",
            "cuda_visible_devices",
            "source_file_sha256",
            "grid_work_plan",
        ):
            assert dense[key] == local[key], (atoms, key)
        for dense_call, local_call in zip(
            dense["force_calls"], local["force_calls"], strict=True
        ):
            assert (
                dense_call["additional_device_peak_bound"]
                == local_call["additional_device_peak_bound"]
            )
            assert (
                local_call["additional_host_numeric_bound"]
                - dense_call["additional_host_numeric_bound"]
                == local_call["resident_ao_selection"]["cache_host_reserve_bytes"]
            )
        work = local["force_calls"][1]["resident_ao_selection"]["work"]
        summary["cases"].append(
            {
                "atoms": atoms,
                "variants": variants,
                "reference_sha256": digest(reference_path),
                "warm_endpoint_reduction": 1
                - local["phase_medians_seconds"]["warm"]
                / dense["phase_medians_seconds"]["warm"],
                "force_contraction_G_M2_fraction": work["point_ao_square_sum"]
                / work["dense_point_ao_square_sum"],
                "force_active_aos_mean": work["active_aos_sum"] / work["tile_count"],
            }
        )
    directory = Path(".") / "zero-budget" / "3"
    reference_path = directory / "reference.json"
    reference = load(reference_path)
    complete_rows(reference, 3)
    summary["zero_budget_control"] = verify_variant(
        directory, 3, "zero-budget", reference, digest(reference_path)
    )
    summary["native_endpoints_checked"] = 24 * len(summary["cases"]) + 12
    return summary


def main() -> None:
    """Recompute the retained summary instead of trusting its accepted flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", choices=CAMPAIGNS, default="frozen")
    args = parser.parse_args()
    load_bundle(args.campaign)
    summary = verify_all()
    assert summary == load(Path("receipts/ao-summary.json"))
    print(
        "PASS:",
        summary["native_endpoints_checked"],
        "complete endpoints, exact identities, all-repeat gates and AO work",
    )
    for case in summary["cases"]:
        print(
            case["atoms"],
            "warm reduction",
            case["warm_endpoint_reduction"],
            "force GM2 fraction",
            case["force_contraction_G_M2_fraction"],
        )


if __name__ == "__main__":
    main()
