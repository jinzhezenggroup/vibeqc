"""Reproduce paired warm packed/full molecular endpoints under finite Slurm.

Existing checkpoints reproduce a declared frozen input. Without --seed-dir,
the first cold dense solve supplies a fresh density that is frozen for both
policies and observables. Fresh seeds can change SCF branches and do not claim
the historical checkpoint identity. No tracing or sampling overlaps clean calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

_BENCHMARKS_DIR = next(
    parent for parent in Path(__file__).resolve().parents if parent.name == "benchmarks"
)
sys.path.insert(0, str(_BENCHMARKS_DIR))
from _retention import raw_output_path

ROOT = Path(__file__).resolve().parents[3]


def main():
    """Keep the existing endpoint admission and numerical gates authoritative."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=raw_output_path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument(
        "--aos",
        type=int,
        nargs="+",
        choices=(96, 192, 384, 768),
        default=(96, 192, 384, 768),
    )
    parser.add_argument("--df-budget", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--seed-dir", type=Path, help="Directory of 96.checkpoint, 192.checkpoint, etc."
    )
    parser.add_argument("--forces-only", action="store_true")
    parser.add_argument("--without-components", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or args.repeats < 7 or args.df_budget < 0:
        parser.error(
            "requires finite Slurm, at least seven repeats and nonnegative budget"
        )
    library = args.library.resolve(strict=True)
    output = args.output.resolve()
    if len(set(args.aos)) != len(args.aos):
        parser.error("duplicate AO sizes would overwrite evidence")
    seeds = {}
    if args.seed_dir:
        seeds = {
            n: (args.seed_dir / f"{n}.checkpoint").resolve(strict=True)
            for n in args.aos
        }
    output.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items() if not k.startswith("VIBEQC_")}
    env.update(
        PYTHONPATH=os.pathsep.join((str(ROOT / "python"), str(ROOT))),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        VIBEQC_LIBRARY=str(library),
        VIBEQC_DF_REFERENCE_FINAL_VALIDATION="0",
        VIBEQC_DF_FINAL_PROJECTION="auto",
        VIBEQC_DF_FORCE_SCREEN_ABS="off",
        VIBEQC_DF_EXCHANGE="auto",
        VIBEQC_DF_SEED_EXCHANGE="auto",
        VIBEQC_DF_FINAL_EXCHANGE="auto",
        VIBEQC_DF_RESIDENT_EXCHANGE="auto",
        VIBEQC_DF_SHELL_POLICY="auto",
    )
    # The parent imports no CUDA package and owns no device context. The child
    # receives the scheduler's original CUDA_VISIBLE_DEVICES unchanged.
    sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
    from benchmarks.df_policy_endpoint import CASES

    manifest = {
        "scope": __doc__,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "df_budget_bytes": args.df_budget,
        "checks": [],
    }

    def save():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    for n in args.aos:
        seed = seeds.get(n)
        for observable in (
            ("forces",) if args.forces_only or n < 384 else ("forces", "energy")
        ):
            name = f"{n}-{observable}"
            command = [
                sys.executable,
                "-m",
                "benchmarks.df_policy_endpoint",
                "--aos",
                str(n),
                "--output",
                str(output / f"{name}.json"),
                "--repeats",
                str(args.repeats),
                "--df-budget",
                str(args.df_budget),
                "--control",
                "VIBEQC_DF_VALUE_STORAGE",
                "--policies",
                "dense",
                "packed",
                "--reference",
                str(
                    ROOT
                    / "benchmarks/results/issue377-379-df/gpu4pyscf"
                    / f"{CASES[n]}.json"
                ),
            ]
            if seed is not None:
                command += ["--skip-cold", "--warm-checkpoint-in", str(seed)]
            else:
                seed = output / f"{n}.checkpoint"
                command += ["--warm-checkpoint-out", str(seed)]
            if observable == "energy":
                command.append("--energy-only")
            if not args.without_components:
                command.append("--components-after")
            print("starting", name, flush=True)
            with (output / f"{name}.log").open("x") as stream:
                run = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            manifest["checks"].append(
                {"case": name, "command": command, "exit_code": run.returncode}
            )
            save()
            if run.returncode:
                raise SystemExit(run.returncode)


if __name__ == "__main__":
    main()
