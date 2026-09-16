"""Retain complete numerical/timing rows and compact parsed compiler evidence.

Raw compiler streams and profiler databases remain local artifacts. Their
digests bind the reduced reports; no timing or failed numerical row is removed.
"""

import gzip
import hashlib
import json
import statistics
from pathlib import Path

DEST = Path("benchmarks/results/issue404-407-df")


def compact_json(value, level=0):
    """Preserve numeric arrays and flat measurement records on a single line."""

    def numeric_tensor(item):
        return isinstance(item, (int, float)) or (
            isinstance(item, list) and all(numeric_tensor(child) for child in item)
        )

    if isinstance(value, dict):
        if (
            "milliseconds" in value
            or "artifact_key" in value
            or ("angular" in value and "shell_frequency" in value)
        ):
            return json.dumps(value, allow_nan=False)
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


def write(name, value):
    path = DEST / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(compact_json(value) + "\n")
    assert path.stat().st_size < 1024**2, path


def batch(source, prefix):
    path = Path(source) / "report.json"
    payload = json.loads(path.read_text())
    payload["original_report_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    for row in payload["compiled"]:
        for field in ("stdout", "stderr"):
            text = row["compile"].pop(field)
            row["compile"][field + "_sha256"] = hashlib.sha256(
                text.encode()
            ).hexdigest()
        # Parsed resource/returncode/timeout/rejection fields remain complete.
    results = payload.pop("results")
    payload["results_files"] = []
    for profile in sorted({r["profile"] for r in results}):
        name = f"{prefix}-{profile}-results.json"
        write(name, [r for r in results if r["profile"] == profile])
        payload["results_files"].append(name)
    write(f"{prefix}-batch.json", payload)


def main():
    for path in Path(".artifacts/df404407/v2").glob("*-40[467].json"):
        write(path.name, json.loads(path.read_text()))
    for path in Path(".artifacts/df404407/final").glob("*-default.json"):
        payload = json.loads(path.read_text())
        assert len(payload["samples"]) == 10 and len(payload["diagnostics"]) == 2
        write(path.name, payload)
    batch(".artifacts/df404/final-batch", "derivative")
    batch(".artifacts/df405/final-batch", "value")
    batch(".artifacts/df405/batch1", "initial-value")
    for version, name in (("v2", "384-resident"), ("v4", "384-smoke")):
        path = Path(f".artifacts/df405/{version}/{name}.json")
        write(f"rejected-value-{version}.json", json.loads(path.read_text()))
    snapshots = {}
    for version, source in (
        ("qualification", ".artifacts/df404407/v2"),
        ("value-v2", ".artifacts/df405/v2"),
        ("value-v4", ".artifacts/df405/v4"),
        ("batch", ".artifacts/df404/final-batch"),
        ("final", ".artifacts/df404407/final"),
    ):
        raw = (Path(source) / "source.patch").read_bytes()
        name = f"reproduction/{version}-source.patch.gz"
        compressed = gzip.compress(raw, mtime=0)
        (DEST / name).write_bytes(compressed)
        snapshots[name] = {
            "base": "e41c1dcd448df9cb66775dde617ceb42cf1570c4",
            "uncompressed_sha256": hashlib.sha256(raw).hexdigest(),
            "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
            "uncompressed_bytes": len(raw),
        }
    write("source-snapshots.json", snapshots)
    for path in (DEST / "profiles").glob("*.json"):
        write(str(path.relative_to(DEST)), json.loads(path.read_text()))
    timings = {}
    for pattern in ("*-40[467].json", "*-default.json"):
        for path in sorted(DEST.glob(pattern)):
            samples = {}
            for row in json.loads(path.read_text())["samples"]:
                samples.setdefault(row["policy"], []).append(row["seconds"])
            timings[path.stem] = {
                "samples": samples,
                "medians": {p: statistics.median(t) for p, t in samples.items()},
            }
    write("timings.json", timings)


if __name__ == "__main__":
    main()
