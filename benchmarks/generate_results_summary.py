"""Generate the headline parity table from accepted benchmark artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

BEGIN_MARKER = "<!-- BEGIN GENERATED PARITY TABLE -->"
END_MARKER = "<!-- END GENERATED PARITY TABLE -->"
PARITY_CASES = {
    (96, 1): "water-tetramer-def2-svp-spherical",
    (96, 4): "water-tetramer-def2-svp-spherical",
    (192, 1): "water-octamer-s4-def2-svp-spherical",
    (192, 4): "water-octamer-s4-def2-svp-spherical",
}


def _repeat_count(payload: dict[str, Any]) -> int:
    settings = payload.get("settings", {})
    return int(settings.get("repeats_per_engine", settings.get("repeats", 0)))


def _is_direct_density_fitting_artifact(payload: dict[str, Any]) -> bool:
    """Keep the historical parity table scoped to direct-SCF artifacts.

    Schema-v1/v2 direct artifacts predate the explicit density-fitting field,
    so a missing value is treated as ``none``.  CUDA-DF artifacts use the same
    workload keys and must not replace direct-SCF evidence merely because they
    have a newer timestamp.
    """

    return payload.get("workload", {}).get("density_fitting", "none") == "none"


def accepted_parity_artifacts(
    paths: Iterable[Path],
) -> dict[tuple[int, int], tuple[Path, dict[str, Any]]]:
    """Select the newest clean five-repeat schema-v2 artifact per gate point."""

    selected: dict[tuple[int, int], tuple[Path, dict[str, Any]]] = {}
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        workload = payload.get("workload", {})
        key = (workload.get("ao_count"), workload.get("batch_size"))
        if (
            payload.get("schema_version") != 2
            or payload.get("benchmark") != "compare_gpu4pyscf_batch"
            or key not in PARITY_CASES
            or workload.get("case") != PARITY_CASES[key]
            or not _is_direct_density_fitting_artifact(payload)
            or _repeat_count(payload) < 5
            or not payload.get("gate", {}).get("passed", False)
            or payload.get("environment", {}).get("git", {}).get("dirty") is not False
        ):
            continue
        timestamp = payload.get("environment", {}).get("timestamp_utc", "")
        previous = selected.get(key)
        previous_timestamp = (
            ""
            if previous is None
            else previous[1].get("environment", {}).get("timestamp_utc", "")
        )
        if timestamp > previous_timestamp:
            selected[key] = (path, payload)
    return selected


def _milliseconds(value: float) -> str:
    return f"{value * 1.0e3:.3f} ms"


def render_parity_section(
    selected: dict[tuple[int, int], tuple[Path, dict[str, Any]]],
) -> str:
    """Render the generated Markdown section in stable AO/batch order."""

    missing = [key for key in PARITY_CASES if key not in selected]
    if missing:
        points = ", ".join(f"{ao}-AO batch-{batch}" for ao, batch in missing)
        raise ValueError(f"missing accepted schema-v2 artifacts for {points}")

    lines = [
        BEGIN_MARKER,
        "## Current 96/192-AO parity matrix",
        "",
        "This table is generated from the newest clean accepted artifacts with at least five interleaved warm samples per engine. Ordinary and iteration-matched medians are deliberately reported separately.",
        "",
        "| AO | Batch | Artifact | Source | Samples | Ordinary VibeQC / GPU4PySCF | Iteration-matched branch | Matched speedup | Max dE | Max dF |",
        "| ---: | ---: | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for key in sorted(selected):
        path, payload = selected[key]
        ordinary = payload["timing_summary"]["ordinary"]
        matched = payload["timing_summary"].get("iteration_matched")
        accuracy = payload["accuracy"]
        gate_accuracy = accuracy.get("gate_selection", accuracy)
        commit = payload["environment"]["git"]["commit"]
        if matched is None:
            branch = "unavailable (unmatched)"
            speedup = "n/a"
        else:
            branch = "/".join(str(item) for item in matched["iteration_branch"])
            speedup = f"{matched['speedup']:.3f}x"
        lines.append(
            "| "
            f"{key[0]} | {key[1]} | [`{path.stem}`]({path.name}) | "
            f"`{commit[:7]}` | {_repeat_count(payload)} | "
            f"{_milliseconds(ordinary['vibeqc_median_seconds'])} / "
            f"{_milliseconds(ordinary['gpu4pyscf_median_seconds'])} | "
            f"{branch} | {speedup} | "
            f"{gate_accuracy['maximum_energy_error_hartree']:.2e} Eh | "
            f"{gate_accuracy['maximum_force_error_hartree_per_bohr']:.2e} Eh/bohr |"
        )
    lines.extend(
        (
            "",
            "When a shared branch exists, the speed gate uses its iteration-matched median. Otherwise the ordinary median is explicitly labeled unmatched and remains a timing observation rather than a parity claim.",
            END_MARKER,
        )
    )
    return "\n".join(lines)


def update_readme(readme: Path, section: str, *, check: bool = False) -> bool:
    """Replace the marked section and optionally fail when it is stale."""

    original = readme.read_text(encoding="utf-8")
    begin = original.find(BEGIN_MARKER)
    end = original.find(END_MARKER)
    if begin < 0 or end < begin:
        raise ValueError("README parity markers are missing or out of order")
    updated = original[:begin] + section + original[end + len(END_MARKER) :]
    if updated == original:
        return False
    if check:
        raise ValueError(f"{readme} is stale; run generate_results_summary.py")
    readme.write_text(updated, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-directory",
        type=Path,
        default=Path(__file__).with_name("results"),
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=Path(__file__).with_name("results") / "README.md",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    selected = accepted_parity_artifacts(args.results_directory.glob("*.json"))
    section = render_parity_section(selected)
    changed = update_readme(args.readme, section, check=args.check)
    print("benchmark summary is current" if not changed else f"updated {args.readme}")


if __name__ == "__main__":
    main()
