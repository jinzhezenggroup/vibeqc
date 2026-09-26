"""Verify an accepted local bundle in fresh processes inside a GPU allocation.

The input archive must come from the real CLI tuning workflow. This check uses
an isolated cache and retains its diagnostics, complete energy/force replays,
and failure reports in the output directory; it never changes the user's cache.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
from vibeqc.profiles import atomic_json, file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True)
    env = {**os.environ, "VIBEQC_PROFILE_CACHE": str(output / "cache")}
    for name in (
        "VIBEQC_PROFILE",
        "VIBEQC_AOT_SHELL_CLASSES",
        "VIBEQC_AOT_FOCK_SHELL_CLASSES",
        "VIBEQC_AOT_MIXED_FOCK_SHELL_CLASSES",
    ):
        env.pop(name, None)

    def command(arguments, name, *, overrides=None, expect=0):
        result = subprocess.run(
            [sys.executable, "-m", "vibeqc", *map(str, arguments)],
            env={**env, **(overrides or {})},
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        (output / f"{name}.log").write_text(result.stdout + result.stderr)
        assert result.returncode == expect, (name, result.stdout, result.stderr)
        return result.stdout

    installed = Path(
        command(["profile", "install", args.archive.resolve()], "install").strip()
    )
    diagnostics = json.loads(command(["profile", "diagnose"], "diagnose"))
    assert diagnostics["source"] == "local", diagnostics
    assert Path(diagnostics["directory"]) == installed
    with zipfile.ZipFile(args.archive) as archive:
        evidence = json.loads(archive.read("evidence.json"))
    workload = evidence["workload"]
    atomic_json(output / "workload.json", workload)
    results = {}
    for label in ("baseline", "local"):
        replay_env = dict(env)
        if label == "baseline":
            replay_env["VIBEQC_PROFILE"] = "off"
        replay = subprocess.run(
            [
                sys.executable,
                "-m",
                "vibeqc._autotune_worker",
                str(output / "workload.json"),
                str(output / f"{label}.json"),
            ],
            env=replay_env,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        (output / f"{label}.log").write_text(replay.stdout + replay.stderr)
        assert replay.returncode == 0, (label, replay.stdout, replay.stderr)
        results[label] = json.loads((output / f"{label}.json").read_text())
    assert results["local"]["profile"]["source"] == "local"
    assert all(
        r["converged"] and set(r["backend"]) == {"cuda"} for r in results.values()
    )
    energy_error = float(
        np.max(
            np.abs(
                np.asarray(results["baseline"]["energies"])
                - results["local"]["energies"]
            )
        )
    )
    force_error = float(
        np.max(
            np.abs(
                np.asarray(results["baseline"]["forces"]) - results["local"]["forces"]
            )
        )
    )
    assert energy_error < 1e-9 and force_error < 1e-7

    # Failed and inapplicable searches must preserve the active immutable bundle.
    atoms = workload["atoms"]
    xyz = output / "input.xyz"
    xyz.write_text(
        f"{len(atoms)}\nBohr\n"
        + "".join(
            f"{symbol} {' '.join(map(str, position))}\n" for symbol, position in atoms
        )
    )
    active = output / "cache/active.json"
    active_before = active.read_bytes()
    no_work = json.loads(
        command(
            ["autotune", "--quick", xyz, "--units", "bohr", "--basis", "sto-3g"],
            "no-work",
        )
    )
    assert no_work["installed"] is None and no_work["failure"] is None
    assert active.read_bytes() == active_before
    failed = json.loads(
        command(
            [
                "autotune",
                "--quick",
                xyz,
                "--units",
                "bohr",
                "--basis",
                workload["basis"],
                "--portable-baseline",
                "--budget-seconds",
                "1",
            ],
            "budget",
            expect=1,
        )
    )
    assert failed["failure"] and failed["installed"] is None
    assert active.read_bytes() == active_before

    incompatible = output / "incompatible"
    shutil.copytree(installed, incompatible)
    profile = json.loads((incompatible / "profile.json").read_text())
    profile["identity"]["device"]["major"] += 1
    atomic_json(incompatible / "profile.json", profile)
    rejected = json.loads(
        command(
            ["profile", "diagnose"],
            "incompatible",
            overrides={"VIBEQC_PROFILE": str(incompatible)},
        )
    )
    assert rejected["source"] != "local" and rejected["rejected"]
    assert active.read_bytes() == active_before
    command(["profile", "clear"], "clear")
    cleared = json.loads(command(["profile", "diagnose"], "cleared"))
    assert cleared["source"] != "local"
    assert (installed / "libvibeqc.so").exists()
    report = {
        "passed": True,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "archive_sha256": file_hash(args.archive),
        "profile_identity": diagnostics["identity"],
        "diagnostics": diagnostics,
        "maximum_energy_error": energy_error,
        "maximum_force_error": force_error,
        "budget_failure": failed,
        "no_work": no_work,
        "incompatible_diagnostics": rejected,
        "cleared_diagnostics": cleared,
        "active_bundle_preserved_on_failure": True,
    }
    atomic_json(output / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
