"""Verify retained direct-HF qualification without relabelling native failures."""

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text())
    decoded = {}
    for record in manifest["files"]:
        path = (root / record["stored_path"]).resolve()
        require(path.is_relative_to(root), "retained payload escapes the bundle")
        stored = path.read_bytes()
        require(len(stored) == record["stored_bytes"], f"stored size: {path.name}")
        require(
            hashlib.sha256(stored).hexdigest() == record["stored_sha256"],
            f"stored checksum: {path.name}",
        )
        require(record["encoding"] == "gzip", "unsupported payload encoding")
        data = gzip.decompress(stored)
        require(len(data) == record["bytes"], f"decoded size: {path.name}")
        require(
            hashlib.sha256(data).hexdigest() == record["sha256"],
            f"decoded checksum: {path.name}",
        )
        name = record["logical_path"]
        require(name not in decoded, f"duplicate logical record: {name}")
        decoded[name] = data

    def read(name: str) -> dict[str, Any]:
        return json.loads(decoded[name])

    candidate = read("validation/manifest.json")
    comparison = read("baseline/comparison.json")
    summary = read("validation/matrix/summary.json")
    require(candidate["revision"] == manifest["candidate_source"], "candidate revision")
    require(
        comparison["baseline_source"] == manifest["baseline_source"], "base revision"
    )
    require(candidate["status"] == "failed", "raw native-suite failure was relabelled")
    require(
        candidate["direct_accuracy_passed"] and candidate["matrix_passed"],
        "force acceptance",
    )
    require(comparison["all_shared_native_results_match"], "shared native regression")
    require(not comparison["new_candidate_failures"], "new native failure")
    require(not comparison["raw_suites_relabelled"], "baseline suite was relabelled")
    require(comparison["candidate_native_count"] == 65, "candidate test count")
    require(comparison["baseline_native_count"] == 64, "base test count")
    require(
        comparison["candidate_only_tests"] == ["vibeqc_cuda_force_convergence_tests"],
        "unexpected added native test",
    )
    require(
        comparison["candidate_failures"] == comparison["baseline_failures"]
        and len(comparison["baseline_failures"]) == 3,
        "baseline failure disposition",
    )
    require(
        comparison["baseline_public_oh_exit"] == 1,
        "baseline OH regression did not fail",
    )
    require(
        candidate["library_sha256"] == comparison["candidate_library_sha256"],
        "candidate library identity",
    )
    require(
        comparison["baseline_library_sha256"] != candidate["library_sha256"],
        "baseline binary was substituted",
    )
    qualification = root.parent / "reference-qualification/full-fock-diagnostic.json"
    require(
        hashlib.sha256(qualification.read_bytes()).hexdigest()
        == manifest["reference_qualification_sha256"],
        "reference qualification changed",
    )
    require(summary["passed"] and len(summary["points"]) == 4, "complete matrix")
    require(summary["repeats"] == 7, "paired repetition count")
    pair_count = 0
    for point in summary["points"]:
        record = read("validation/matrix/" + Path(point["artifact"]).name)
        pairs = record["accuracy"]["paired_warm_repeats"]
        selected = record["accuracy"]["gate_selection"]
        require(
            len(pairs) == 7 and len({p["repeat"] for p in pairs}) == 7,
            "missing paired sample",
        )
        require(
            selected["selection"] == "all_measured_pairs", "selective accuracy verdict"
        )
        require(selected["pair_count"] == 7, "selected pair count")
        require(
            record["native_build"]["library_sha256"] == candidate["library_sha256"],
            "mixed binary",
        )
        require(
            record["gate"]["passed"] and not record["gate"]["failures"], "failed point"
        )
        for key in (
            "maximum_energy_error_hartree",
            "maximum_force_error_hartree_per_bohr",
        ):
            require(
                max(p[key] for p in pairs) <= selected[key] <= point[key],
                f"accuracy gate: {key}",
            )
        if point["ao_count"] == 96:
            require(
                point["reference_gradient_tolerance"] == 1e-11,
                "96-AO reference tolerance",
            )
            require(
                not point["reference_incremental_fock"], "96-AO reference Fock policy"
            )
            require(
                point["maximum_force_error_hartree_per_bohr"] == 3e-11,
                "96-AO force gate",
            )
            matched = record["timing_summary"]["iteration_matched"]
            require(
                matched is not None and matched["speedup"] >= 1.0,
                "96-AO matched speed gate",
            )
        else:
            require(point["ao_count"] == 192, "unexpected endpoint")
            require(
                point["reference_gradient_tolerance"] == 1e-8,
                "192-AO reference tolerance",
            )
            require(point["reference_incremental_fock"], "192-AO reference Fock policy")
            require(
                point["maximum_force_error_hartree_per_bohr"] == 5e-10,
                "192-AO force gate",
            )
            require(point["minimum_speedup"] is None, "invented 192-AO speed gate")
        pair_count += len(pairs)
    require(pair_count == 28, "incomplete paired campaign")
    print(
        f"Verified {len(decoded)} byte-exact records and all {pair_count} paired observations; "
        "three reproduced base native failures remain explicitly failed."
    )


if __name__ == "__main__":
    main()
