"""Run the separate source and release-compile tiers of the f-shell matrix."""

# Source-tree CLI bootstrap for transitive compiler clients.
import sys as _compiler_sys
from pathlib import Path as _CompilerPath

_compiler_sys.path.insert(
    0, str(_CompilerPath(__file__).resolve().parents[1] / "python")
)

import argparse
import json
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.vibeqc_validation.f_shell import (
    F_SHELL_CLASSES,
    SMOKE_CLASSES,
    catalog,
    compile_matrix,
    write_archive,
)


def main() -> int:
    """Keep ordinary source checks independent of CUDA and reference packages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tier", choices=("source", "compile", "numerical"), default="source"
    )
    parser.add_argument("--architecture", default="sm_120")
    parser.add_argument("--shell-class", action="append", choices=F_SHELL_CLASSES)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="only the five representative CUDA smoke classes",
    )
    parser.add_argument("--nvcc", type=Path, default=Path("nvcc"))
    parser.add_argument("--cache", type=Path, default=Path("build/f-shell-cache"))
    parser.add_argument("--compile-jobs", type=int, default=2)
    parser.add_argument("--compile-timeout", type=float, default=600)
    parser.add_argument(
        "--slurm-time",
        default="00:10:00",
        help="finite main/gpu:5090:1 allocation per numerical class",
    )
    parser.add_argument("--runtime-timeout", type=int, default=900)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--archive",
        type=Path,
        help="also write hashed per-class evidence and a compact index",
    )
    args = parser.parse_args()
    if args.tier != "source":
        resolved = shutil.which(str(args.nvcc))
        if resolved is None:
            parser.error(f"CUDA compiler not found: {args.nvcc}")
        args.nvcc = Path(resolved).resolve()
    if args.smoke and args.shell_class:
        parser.error("--smoke and --shell-class are mutually exclusive")
    names = args.shell_class or (SMOKE_CLASSES if args.smoke else F_SHELL_CLASSES)
    report = catalog(architecture=args.architecture, names=names)
    if args.tier in ("compile", "numerical"):
        report = compile_matrix(
            report,
            nvcc=args.nvcc,
            cache=args.cache,
            jobs=args.compile_jobs,
            timeout=args.compile_timeout,
            progress=lambda row: print(
                row["shell_class"],
                row["compilation"]["status"],
                "cached"
                if row["compilation"]["cache_hit"]
                else f"{row['compilation']['seconds']:.2f}s",
                flush=True,
            ),
        )
    if args.tier == "numerical":
        from tools.vibeqc_validation.f_shell_numerics import numerical_matrix

        report = numerical_matrix(
            report,
            nvcc=args.nvcc,
            cache=args.cache,
            slurm_time=args.slurm_time,
            timeout=args.runtime_timeout,
            progress=lambda row: print(
                row["shell_class"], "numerical", row["numerical"]["status"], flush=True
            ),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    if args.archive is not None:
        write_archive(report, args.archive)
    stages = (
        ("source",) if args.tier == "source" else ("source", "compilation", "resources")
    )
    if args.tier == "numerical":
        stages += ("numerical",)
    return (
        0
        if all(
            row[stage]["status"] == "pass" for row in report["rows"] for stage in stages
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
