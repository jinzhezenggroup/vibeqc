"""Retain reviewable endpoint evidence and raw-input hashes, excluding bulk traces.

Run from the repository root after qualification. Measurements retain every
energy/force array, timing, numerical gate and executed branch. Repeated basis
metadata is interned once per run; original files remain in local artifacts.
"""

import copy
import hashlib
import json
import statistics
from pathlib import Path

SOURCE = Path(".artifacts/issue399")
DESTINATION = Path("benchmarks/results/issue399-density-seed")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compact_json(value, level=0):
    """Keep numeric rows and flat event records on one line without losing data."""

    def numeric_tensor(item):
        return isinstance(item, (int, float)) or (
            isinstance(item, list) and all(numeric_tensor(child) for child in item)
        )

    if isinstance(value, dict):
        if all(not isinstance(item, (dict, list)) for item in value.values()):
            return json.dumps(value, allow_nan=False)
        fields = [
            json.dumps(str(key)) + ": " + compact_json(item, level + 2)
            for key, item in value.items()
        ]
        opening, closing = "{", "}"
    elif isinstance(value, list):
        if numeric_tensor(value) or all(
            not isinstance(item, (dict, list)) for item in value
        ):
            return json.dumps(value, allow_nan=False)
        fields = [compact_json(item, level + 2) for item in value]
        opening, closing = "[", "]"
    else:
        return json.dumps(value, allow_nan=False)
    return (
        opening
        + "\n"
        + ",\n".join(" " * (level + 2) + item for item in fields)
        + "\n"
        + " " * level
        + closing
    )


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = compact_json(value)
    assert json.loads(encoded) == value
    path.write_text(encoded + "\n")


def retain(path, version):
    data = json.loads(path.read_text())
    samples = data["samples"]
    expected = 1 if "verification" in path.stem else 5 * len(data["policies"])
    assert len(samples) == expected, path
    assert all(s["iterations"] == [3] for s in samples), path
    assert all(
        s["maximum_energy_error"] <= 1e-9
        and (s["maximum_force_error"] is None or s["maximum_force_error"] <= 1e-8)
        for s in samples
    ), path
    result = copy.deepcopy(data)
    result["original_json_sha256"] = digest(path)
    result["original_json_bytes"] = path.stat().st_size
    result["basis_metadata"] = result["initialization_convergence"][0]["basis_metadata"]
    for rows in [
        result["initialization_convergence"],
        result.get("cold_convergence") or [],
        *(s["convergence"] for s in result["samples"] + result.get("diagnostics", [])),
    ]:
        for row in rows:
            assert row.pop("basis_metadata") == result["basis_metadata"]
    # Large nested traces are retained as reduced component records separately.
    components = []
    for sample in result["samples"] + result.get("diagnostics", []):
        if "components" in sample:
            components.append(
                {
                    "policy": sample["policy"],
                    "repeat": sample["repeat"],
                    "components": sample.pop("components"),
                    "process_device_resident_bytes": sample[
                        "process_device_resident_bytes"
                    ],
                }
            )
    write(DESTINATION / "measurements" / (version + "-" + path.name), result)
    if components:
        write(DESTINATION / "components" / (version + "-" + path.name), components)
    return {
        policy: {
            "seconds": [s["seconds"] for s in samples if s["policy"] == policy],
            "median_seconds": statistics.median(
                s["seconds"] for s in samples if s["policy"] == policy
            ),
            "iterations": [s["iterations"] for s in samples if s["policy"] == policy],
            "maximum_energy_error": max(
                s["maximum_energy_error"] for s in samples if s["policy"] == policy
            ),
            "maximum_force_error": max(
                (s["maximum_force_error"] or 0)
                for s in samples
                if s["policy"] == policy
            ),
            "scope": samples[0]["scope"],
        }
        for policy in data["policies"]
    }


def main():
    """Publish only complete runs; incomplete/diagnostic timings never become wins."""
    timings = {}
    for version in ("v1", "v2"):
        directory = SOURCE / version
        for path in sorted(directory.glob("*.json")):
            if not path.stem[:3].isdigit():
                continue
            timings[version + "-" + path.stem] = retain(path, version)
        patch = directory / "source.patch"
        if patch.exists():
            (DESTINATION / "reproduction" / (version + "-source.patch")).write_bytes(
                patch.read_bytes()
            )
    write(DESTINATION / "timings.json", timings)
    journals = {}
    for path in sorted(SOURCE.glob("v*/*journal.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        # #399 owns SCF/final-state work. The subsequent derivative kernel
        # telemetry is duplicated by the component ledger and can be very large.
        stop = next(
            (i for i, row in enumerate(rows) if row.get("name") == "force_response"),
            len(rows),
        )
        journals[str(path.relative_to(SOURCE))] = {
            "sha256": digest(path),
            "original_rows": len(rows),
            "scope": "SCF and final-state prefix, before force_response",
            "records": rows[:stop],
        }
    write(DESTINATION / "journals.json", journals)


if __name__ == "__main__":
    main()
