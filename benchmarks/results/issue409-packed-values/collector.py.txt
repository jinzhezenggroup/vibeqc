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


def bind_changed_diagnostics(clean_path: Path, companion_directory: Path):
    """Join completed clean samples to a separately completed diagnostic run.

    A timeout after clean timing does not invalidate completed measurements.
    The companion must contain no new clean samples and must prove the same
    binary, geometry, physical model and immutable starting density. The caller
    retains both originals; this derived record never overwrites either input.
    """
    campaign_path = companion_directory / "manifest.json"
    if not campaign_path.exists():
        return None
    campaign = json.loads(campaign_path.read_text())
    if campaign.get("status") != "passed":
        return None
    clean = json.loads(clean_path.read_text())
    companion_path = companion_directory / "768-changed-clean.json"
    companion = json.loads(companion_path.read_text())

    def require(condition, reason):
        if not condition:
            raise ValueError(f"invalid changed diagnostic companion: {reason}")

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    require(campaign.get("diagnostics_only") is True, "not a diagnostics-only run")
    require(campaign.get("exit_code") == 0, "unsuccessful companion process")
    require(
        campaign.get("slurm_job_id") == companion.get("slurm_job_id"),
        "companion process identity",
    )
    require(campaign.get("predecessor_sha256") == sha(clean_path), "clean file hash")
    require(not clean.get("failure") and not companion.get("failure"), "failed input")
    order = [
        (i, p)
        for i in range(7)
        for p in (("dense", "packed") if i % 2 == 0 else ("packed", "dense"))
    ]
    require(
        [(s["repeat"], s["policy"]) for s in clean["samples"]] == order,
        "incomplete original clean series",
    )
    require(companion.get("samples") == [], "companion contains clean repeats")
    require(clean["phase"] == "changed" and clean["aos"] == 768, "wrong endpoint")
    for key in (
        "phase",
        "aos",
        "case",
        "native_identity",
        "library_sha256",
        "source_patch_sha256",
        "reference_sha256",
        "checkpoint_sha256",
        "changed_reference_sha256",
        "frozen_density_sha256",
        "coordinates_bohr",
        "changed_coordinates_bohr",
        "controls",
    ):
        require(key in clean and key in companion and clean[key] == companion[key], key)
    require(bool(clean["frozen_density_sha256"]), "missing frozen density")
    diagnostics = companion["diagnostics"]
    require(
        [d["policy"] for d in diagnostics] == ["dense", "packed"], "diagnostic arms"
    )
    require(
        all(
            math.isfinite(d["seconds"])
            and d["seconds"] > 0
            and math.isfinite(d["maximum_energy_error"])
            and d["maximum_energy_error"] <= 1e-9
            and math.isfinite(d["maximum_force_error"])
            and d["maximum_force_error"] <= 1e-8
            and len(d["convergence"]) == 1
            and all(item["converged"] for item in d["convergence"])
            for d in diagnostics
        ),
        "diagnostic numerical gate",
    )
    provenance = {
        "scope": "Original seven clean pairs plus separate diagnostics; no new or pooled clean samples.",
        "clean_original_sha256": sha(clean_path),
        "diagnostic_original_sha256": sha(companion_path),
        "diagnostic_campaign_sha256": sha(campaign_path),
        "clean_slurm_job_id": clean["slurm_job_id"],
        "diagnostic_slurm_job_id": companion["slurm_job_id"],
        "diagnostic_runner_sha256": companion["runner_sha256"],
        "diagnostic_iterations_observed_in_clean": {
            d["policy"]: any(
                s["policy"] == d["policy"] and s["iterations"] == d["iterations"]
                for s in clean["samples"]
            )
            for d in diagnostics
        },
    }
    return {
        **clean,
        "diagnostics": diagnostics,
        "diagnostic_companion": provenance,
    }


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
        else:
            campaign = source / "unequal-v2/manifest.json"
            retain(campaign, "campaigns/unequal.json")
            if any(
                check["exit_code"]
                for check in json.loads(campaign.read_text())["checks"]
            ):
                manifest["incomplete"].append(
                    {
                        "path": str(campaign),
                        "reason": "unequal campaign failed; original failed cell retained",
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
            composed = None
            if (
                name == "rebuild/768-changed.json"
                and len(data.get("diagnostics", [])) < 2
            ):
                companion_directory = source / "rebuild-768-changed-diagnostics"
                composed = bind_changed_diagnostics(path, companion_directory)
                if composed is not None:
                    # Preserve the timed-out original exactly, including its
                    # scheduler identity and any interrupted diagnostic work.
                    retain(path, "partial/" + name)
                    retain(
                        companion_directory / "768-changed-clean.json",
                        "rebuild/768-changed-diagnostics.json",
                    )
                    retain(
                        companion_directory / "manifest.json",
                        "campaigns/768-changed-diagnostic-companion.json",
                    )
                    for script in (
                        "rebuild-diagnostic-companion.py",
                        "run-768-changed-followup.py",
                        "account-changed-response.py",
                    ):
                        retain(
                            source / "stage2" / script,
                            f"reproduction/measured-{script}.txt",
                        )
                    retain(
                        source / "stage2/rebuild-diagnostic-companion.derivation.json",
                        "reproduction/rebuild-diagnostic-companion.derivation.json",
                    )
                    attribution = companion_directory / "response-attribution.json"
                    if attribution.exists():
                        if (
                            json.loads(attribution.read_text())["binding"]
                            != composed["diagnostic_companion"]
                        ):
                            raise ValueError("changed-response attribution is stale")
                        retain(
                            attribution, "rebuild/768-changed-response-attribution.json"
                        )
                    data = composed
            samples = data.get("samples", [])
            if (
                data.get("failure")
                or [(s["repeat"], s["policy"]) for s in samples] != order
                or (
                    not name.startswith("constrained/")
                    and len(data.get("diagnostics", [])) != 2
                )
            ):
                # Preserve failed and interrupted observations without admitting
                # them as completed qualification cells or pooling their samples.
                retained_name = (
                    "failed/" if data.get("failure") else "partial/"
                ) + name
                retain(path, retained_name)
                manifest["incomplete"].append(
                    {
                        "path": str(path),
                        "reason": "endpoint failed"
                        if data.get("failure")
                        else "incomplete interleaving or diagnostics",
                        "retained": retained_name,
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
            retain(path, name, value=composed)
        if manifest["incomplete"] and not args.partial:
            raise ValueError(
                "campaign is incomplete; inspect manifest before publication"
            )
        for name in (
            "phasea-summary.json",
            "storage-dataflow.json",
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
        for path in sorted((source / "rebuild-v2").glob("terminal-*.json")):
            retain(path, f"campaigns/{path.name}")
        for path in sorted((source / "terminal-observer").glob("terminal-*.json")):
            retain(path, f"campaigns/{path.name}")
        profile_summary = source / "nsys-dataflow-summary.json"
        if profile_summary.exists():
            profiles = json.loads(profile_summary.read_text())
            retain(profile_summary, "nsys-dataflow-summary.json")
            campaigns = set()
            for item in profiles["profiles"]:
                account = source / "dataflow-accounts" / Path(item["path"]).name
                if hashlib.sha256(account.read_bytes()).hexdigest() != item["sha256"]:
                    raise ValueError(f"{account}: stale Nsight account hash")
                data = json.loads(account.read_text())
                campaign, name = data["campaign"], data["name"]
                if (
                    campaign
                    not in {
                        "dataflow-profiles",
                        "unequal-dataflow-profiles",
                        "cold-dataflow-profiles",
                    }
                    or Path(name).name != name
                ):
                    raise ValueError(f"{account}: unexpected profile identity")
                observation = source / campaign / (name + ".json")
                if (
                    hashlib.sha256(observation.read_bytes()).hexdigest()
                    != data["source_record_sha256"]
                ):
                    raise ValueError(f"{account}: changed profiled numerical record")
                retain(account, f"dataflow/{account.name}")
                retain(observation, f"dataflow/observations/{campaign}-{name}.json")
                campaigns.add(campaign)
            for campaign in sorted(campaigns):
                retain(
                    source / campaign / "manifest.json", f"campaigns/{campaign}.json"
                )
            scripts = {"account-nsys.py"}
            if "dataflow-profiles" in campaigns:
                scripts.update(("dataflow-probe.py", "run-dataflow-profiles.py"))
            if "unequal-dataflow-profiles" in campaigns:
                scripts.update(("unequal-dataflow-probe.py", "run-unequal-dataflow.py"))
            if "cold-dataflow-profiles" in campaigns:
                scripts.update(
                    (
                        "cold-dataflow-probe.py",
                        "unequal-cold-dataflow-probe.py",
                        "run-cold-dataflow.py",
                    )
                )
            for script in sorted(scripts):
                retain(
                    source / "stage2" / script, f"reproduction/measured-{script}.txt"
                )
            if profiles["incomplete_campaigns"]:
                manifest["incomplete"].append(
                    {
                        "path": str(profile_summary),
                        "reason": "remaining dataflow profile campaigns",
                        "campaigns": profiles["incomplete_campaigns"],
                    }
                )
        else:
            manifest["incomplete"].append(
                {"path": str(profile_summary), "reason": "missing dataflow accounts"}
            )
        for path in sorted((source / "unequal-v2").glob("*-reference.json")):
            retain(path, f"references/{path.name}")
        oracle = source / "unequal-reference-diagnosis/manifest.json"
        if oracle.exists():
            diagnosis = json.loads(oracle.read_text())
            # Independent-oracle completion explains a retained failure; it
            # never changes the original campaign's admission verdict.
            if diagnosis.get("status") == (
                "completed diagnosis; original GPU gate verdicts unchanged"
            ):
                retain(oracle, "failed/unequal/independent-oracle-diagnosis.json")
                retain(
                    source / "stage2/diagnose-unequal-reference.py",
                    "reproduction/measured-diagnose-unequal-reference.py.txt",
                )
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
            (
                "stage2/format-verification.json",
                "qualification/format-verification.json",
            ),
            (
                "publishing-cuda-ownership.json",
                "qualification/publishing-cuda-ownership.json",
            ),
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
        final_directory = source / "final-qualified"
        final_manifest = final_directory / "manifest.json"
        if final_manifest.exists():
            final = json.loads(final_manifest.read_text())
            if final.get("status") == "passed":
                if not all(
                    c.get("status") == "passed" and c.get("exit_code") == 0
                    for c in final["checks"]
                ):
                    raise ValueError(
                        "final qualification contains an unsuccessful check"
                    )
                for check in final["checks"]:
                    log_path = final_directory / (check["name"] + ".log")
                    if (
                        hashlib.sha256(log_path.read_bytes()).hexdigest()
                        != check["log_sha256"]
                    ):
                        raise ValueError("final qualification log hash changed")
                retain(final_manifest, "qualification/final-manifest.json")
                retain(
                    final_directory / "build-resources.txt",
                    "qualification/final-incremental-build-resources.txt",
                )
                retain(
                    final_directory / "compiled-kernel-resources.log",
                    "qualification/final-compiled-kernel-resources.json",
                    value={
                        "scope": "Exact cuobjdump output in a JSON string preserves trailing spaces through text hooks.",
                        "output": (
                            final_directory / "compiled-kernel-resources.log"
                        ).read_text(),
                    },
                )
                log_summaries = []
                for check in final["checks"]:
                    log = (final_directory / (check["name"] + ".log")).read_text()
                    log_summaries.append(
                        {
                            "name": check["name"],
                            "exit_code": check["exit_code"],
                            "source_log_sha256": check["log_sha256"],
                            "error_counts": [
                                int(n)
                                for n in re.findall(r"ERROR SUMMARY: (\d+) errors", log)
                            ],
                            "leaks": [
                                {"bytes": int(b), "allocations": int(a)}
                                for b, a in re.findall(
                                    r"LEAK SUMMARY: (\d+) bytes leaked in (\d+) allocations",
                                    log,
                                )
                            ],
                            "pytest_totals": re.findall(
                                r"\d+ passed(?:, \d+ skipped)? in [\d.]+s", log
                            ),
                        }
                    )
                retain(
                    final_manifest,
                    "qualification/final-check-summaries.json",
                    value=log_summaries,
                )
                portable = final_directory / "portable-warm-192/192-forces.json"
                portable_data = json.loads(portable.read_text())
                if (
                    portable_data["library_sha256"]
                    != final["sha256"]["build/cuda/libvibeqc.so"]
                    or portable_data["native_source_identity"]
                    != final["native_identity"]
                ):
                    raise ValueError("portable endpoint used a different final library")
                retain(portable, "qualification/final-portable-192-forces.json")
                retain(
                    final_directory / "portable-warm-192/manifest.json",
                    "qualification/final-portable-manifest.json",
                )
                retain(
                    source / "stage2/final-qualification.py",
                    "reproduction/measured-final-qualification.py.txt",
                )
            else:
                manifest["incomplete"].append(
                    {
                        "path": str(final_manifest),
                        "reason": "final qualification unfinished",
                    }
                )
        else:
            manifest["incomplete"].append(
                {"path": str(final_manifest), "reason": "missing final qualification"}
            )
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
            "account-dataflow.py",
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
        if manifest["incomplete"] and not args.partial:
            raise ValueError(
                "campaign is incomplete; inspect manifest before publication"
            )
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
