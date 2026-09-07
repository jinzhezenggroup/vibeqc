"""Summarize measured CUDA kernel durations from an Nsight Systems SQLite export.

Generic kernels batch classes by total angular order. Orders 9--12 necessarily
contain f; orders 3--8 mix f and s/p/d work. Preserve this uncertainty as an
interval instead of inventing a time split from primitive-work counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from pathlib import Path


def classify_kernel(name: str) -> tuple[str | None, str, str | None]:
    """Return consumer, exact f/non-f/mixed attribution, and any exact class."""
    consumer = "fock" if "fock" in name else "force" if "force" in name else None
    generated = re.search(r"generated_(?:sm\d+_)?([spdf]{4})_", name)
    if generated:
        shell_class = generated[1]
        return consumer, "f" if "f" in shell_class else "non_f", shell_class
    angular = re.search(
        r"(?:build_fock_direct_quartet|two_electron_force_quartet)\w*"
        r"<\s*(?:true|false|\(bool\)[01]),\s*(?:\(unsigned int\))?(\d+)[uU]?",
        name,
    )
    if angular:
        order = int(angular[1])
        return consumer, "f" if order > 8 else "non_f" if order < 3 else "mixed", None
    # The special low-order workers have an explicit angular class/order.
    if any(marker in name for marker in ("psss", "ppps", "order2", "order5")):
        # Generic order-five work can include fsss/fpss/fsps; a specialized
        # named ppps entry has only s/p work even though both have order three.
        return consumer, "mixed" if "order5" in name else "non_f", None
    if "two_electron" in name or "direct_packed" in name or "direct_quartet" in name:
        return consumer, "mixed", None
    # Nuclear/one-electron force and matrix assembly cannot be attributed to
    # a four-center f class, but remain in their consumer's denominator.
    return consumer, "overhead", None


def summarize(rows: list[dict]) -> dict:
    """Retain exact measured durations and explicit lower/upper f-time bounds."""
    result = {
        c: {key: 0 for key in ("f", "non_f", "mixed", "overhead")}
        for c in ("fock", "force")
    }
    classes = {}
    for row in rows:
        consumer, attribution, shell_class = classify_kernel(row["name"])
        if row["nanoseconds"] < 0 or row["launches"] < 1:
            raise ValueError("invalid measured kernel duration or launch count")
        row.update(consumer=consumer, attribution=attribution, shell_class=shell_class)
        if consumer is not None:
            result[consumer][attribution] += row["nanoseconds"]
            if shell_class is not None:
                key = f"{shell_class}/{consumer}"
                classes[key] = classes.get(key, 0) + row["nanoseconds"]
    for consumer, counts in result.items():
        total = sum(counts.values())
        if total == 0:
            raise ValueError(f"profile has no {consumer} device durations")
        counts.update(
            total_nanoseconds=total,
            f_fraction_lower=counts["f"] / total,
            f_fraction_upper=(counts["f"] + counts["mixed"]) / total,
        )
    return {
        "schema": "vibeqc.f_shell_device_time",
        "schema_version": 1,
        "method": "summed measured kernel durations; mixed generic angular groups remain unresolved",
        "consumers": result,
        "exact_class_nanoseconds": classes,
        "all_kernel_nanoseconds": sum(r["nanoseconds"] for r in rows),
        "kernels": rows,
    }


def read_trace(path: Path) -> dict:
    """Read only the exported kernel timeline; no runtime GPU access is needed."""
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as connection:
        rows = [
            {"name": name, "nanoseconds": duration, "launches": count}
            for name, duration, count in connection.execute(
                "SELECT s.value, SUM(k.end-k.start), COUNT(*) "
                "FROM CUPTI_ACTIVITY_KIND_KERNEL AS k JOIN StringIds AS s "
                "ON k.demangledName=s.id GROUP BY s.value ORDER BY SUM(k.end-k.start) DESC"
            )
        ]
    result = summarize(rows)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    result["sqlite_sha256"] = digest.hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = read_trace(args.trace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
