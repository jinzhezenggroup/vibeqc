"""Prepare compact #409 review evidence without publishing transient traces.

Clean samples retain their original numbers, ordering and changed SCF branches.
Intrusive counters retain their original scope. Collection does not select a
representation, certify a speedup or turn an incomplete campaign into a pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

LIMIT = 1 << 20


def main():
    """Reject incomplete clean cells unless explicitly preparing a partial draft."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    source, output = args.artifacts.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "scope": __doc__,
        "partial_draft": args.partial,
        "files": [],
        "incomplete": [],
    }

    def retain(path, name, *, value=None):
        """Keep exact input hashes even when whitespace-only JSON compaction is needed."""
        original = path.read_bytes()
        data = (
            original
            if value is None
            else (json.dumps(value, separators=(",", ":")) + "\n").encode()
        )
        if len(data) > LIMIT and path.suffix == ".json":
            data = (json.dumps(json.loads(data), separators=(",", ":")) + "\n").encode()
        if len(data) > LIMIT:
            raise ValueError(
                f"{path}: review record exceeds 1 MiB; select semantic summaries explicitly"
            )
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        manifest["files"].append(
            {
                "path": name,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "original_path": str(path),
                "original_bytes": len(original),
                "original_sha256": hashlib.sha256(original).hexdigest(),
            }
        )

    try:
        cells = [
            (
                source / "clean-warm" / f"{n}-{observable}.json",
                f"warm/{n}-{observable}.json",
            )
            for n in (96, 192, 384, 768)
            for observable in (("forces", "energy") if n >= 384 else ("forces",))
        ]
        cells += [
            (
                source / "rebuild-v2" / f"{n}-{phase}-clean.json",
                f"rebuild/{n}-{phase}.json",
            )
            for n in (384, 768)
            for phase in ("cold", "changed")
        ]
        cells += [
            (
                source / "constrained-clean/768-12GiB-forces.json",
                "constrained/768-12GiB-forces.json",
            )
        ]
        cells += [
            (source / "unequal-v2" / f"{case}-measure.json", f"unequal/{case}.json")
            for case in (
                "water-def2-svp-spherical",
                "water-tetramer-def2-svp-spherical",
                "water-hexadecamer-2s4-def2-svp-spherical",
            )
        ]
        if not (source / "unequal-v2/manifest.json").exists():
            manifest["incomplete"].append(
                {
                    "path": "unequal-v2/manifest.json",
                    "reason": "unequal campaign not complete",
                }
            )
        order = [
            (i, p)
            for i in range(7)
            for p in (("dense", "packed") if i % 2 == 0 else ("packed", "dense"))
        ]
        for path, name in cells:
            if not path.exists():
                manifest["incomplete"].append({"path": str(path), "reason": "missing"})
                continue
            data = json.loads(path.read_text())
            samples = data.get("samples", [])
            if (
                data.get("failure")
                or [(s["repeat"], s["policy"]) for s in samples] != order
                or (
                    not name.startswith("constrained/")
                    and len(data.get("diagnostics", [])) != 2
                )
            ):
                manifest["incomplete"].append(
                    {
                        "path": str(path),
                        "reason": data.get("failure", "incomplete interleaving"),
                    }
                )
                continue
            if any(
                not math.isfinite(s["seconds"])
                or s["seconds"] <= 0
                or not math.isfinite(s["maximum_energy_error"])
                or not math.isfinite(s["maximum_force_error"] or 0)
                or not all(c["converged"] for c in s["convergence"])
                or s["maximum_energy_error"] > 1e-9
                or (s["maximum_force_error"] or 0) > 1e-8
                for s in samples
            ):
                raise ValueError(f"{path}: unchanged numerical gate failed")
            retain(path, name)
        if manifest["incomplete"] and not args.partial:
            raise ValueError(
                "campaign is incomplete; inspect manifest before publication"
            )
        for name in (
            "phasea-summary.json",
            "clean-warm/work-reconciliation.json",
            "capacity/summary.json",
        ):
            if (source / name).exists():
                retain(source / name, name)
        for path in sorted((source / "capacity").glob("*-12GiB-*.json")):
            # Process samples are numerical memory evidence; raw CUDA/journal
            # traces stay in the artifact directory and remain hash-addressed.
            retain(path, f"capacity/{path.name}")
        for path in sorted((source / "rebuild-v2").glob("*-changed-reference.json")):
            retain(path, f"references/{path.name}")
        for path in sorted((source / "unequal-v2").glob("*-reference.json")):
            retain(path, f"references/{path.name}")
        for path in sorted(
            (source / "projection/trials-column").glob("*-summary.json")
        ):
            retain(path, f"projection/{path.name}")
        for path in sorted(
            (source / "projection/trials-column").glob("*-samples.jsonl")
        ):
            retain(
                path,
                f"projection/{path.stem}.json",
                value=[json.loads(line) for line in path.read_text().splitlines()],
            )
        for path in sorted((source / "projection/trials-column").glob("*-plan.json")):
            retain(path, f"projection/{path.name}")
        for name, target in (
            ("projection/producer-validation/report.json", "producer/initial.json"),
            (
                "projection/producer-followup/report.json",
                "producer/truncated-and-unequal.json",
            ),
            (
                "native/validated-stage1/manifest.json",
                "qualification/native-stage1.json",
            ),
            (
                "native/validated-stage1/path-coverage.json",
                "qualification/native-paths.json",
            ),
            ("stage2/validation.json", "qualification/high-level.json"),
        ):
            retain(source / name, target)
        for name in (
            "manifest.json",
            "build-flags.json",
            "build_identity.hpp",
            "source.patch",
        ):
            target = name + ".txt" if name.endswith(".hpp") else name
            retain(source / "stage2/frozen" / name, f"reproduction/{target}")
        for name in (
            "native/validated-stage1/memcheck.log",
            "native/validated-stage1/density-memcheck.log",
            "native/validated-stage1/initcheck.log",
            "native/validated-stage1/synccheck.log",
            "native/validated-stage1/response-leakcheck.log",
            "stage2/molecular-memcheck.log",
            "stage2/failed-neighbors.log",
        ):
            path = source / name
            log = path.read_text()
            # Preserve observed totals and the complete log's identity, without
            # publishing routine stdout or manufacturing missing exit statuses.
            totals = {
                "scope": "Observed log summaries; test/library identity is in the qualification manifests.",
                "source_log_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "error_counts": [
                    int(n) for n in re.findall(r"ERROR SUMMARY: (\d+) errors", log)
                ],
                "leaks": [
                    {"bytes": int(b), "allocations": int(a)}
                    for b, a in re.findall(
                        r"LEAK SUMMARY: (\d+) bytes leaked in (\d+) allocations", log
                    )
                ],
                "pytest_totals": re.findall(
                    r"\d+ passed(?:, \d+ skipped)? in [\d.]+s", log
                ),
            }
            retain(path, f"qualification/{path.stem}.json", value=totals)
        for name in (
            "rebuild-endpoints.py",
            "changed-reference.py",
            "capacity-probe.py",
            "unequal-endpoints.py",
            "run-clean-warm.py",
            "run-rebuilds.py",
            "run-capacity-probes.sh",
            "run-unequal.py",
            "constrained-clean.py",
            "run-constrained-clean.py",
            "summarize-phasea.py",
            "reconcile-work.py",
            "run-tests.py",
            "run-molecular-memcheck.sh",
            "run-failed-neighbors.sh",
        ):
            # .txt preserves measured bytes through future format hooks. The
            # portable current runner is maintained separately as real source.
            retain(source / "stage2" / name, f"reproduction/measured-{name}.txt")
        retain(
            source / "stage2/measured-df-policy-endpoint.py.txt",
            "reproduction/measured-df-policy-endpoint.py.txt",
        )
        retain(
            source / "stage2/constrained-clean.derivation.json",
            "reproduction/constrained-clean.derivation.json",
        )
        retain(Path(__file__), "collector.py.txt")
    except Exception as error:
        manifest["collection_failure"] = repr(error)
        raise
    finally:
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Prepared {len(manifest['files'])} records; incomplete cells: {len(manifest['incomplete'])}"
    )


if __name__ == "__main__":
    main()
