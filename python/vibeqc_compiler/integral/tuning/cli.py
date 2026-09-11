"""Parse autotuning command-line options and invoke the shared execution driver."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from ..cuda_schedule import (
    ScheduleKind,
)
from ..ir import KernelConsumer
from .driver import _run_autotune
from .shared import _PRODUCTION_MANIFEST_PATH


def argument_parser() -> argparse.ArgumentParser:
    """Shared CLI contract for developer and supported user-local tuning."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nvcc",
        type=Path,
        default=Path(os.environ.get("VIBEQC_NVCC", shutil.which("nvcc") or "nvcc")),
    )
    parser.add_argument("--architecture", default="sm_120")
    parser.add_argument("--srun", default="srun")
    parser.add_argument("--partition", default="main")
    parser.add_argument("--gres", default="gpu:1")
    parser.add_argument(
        "--slurm-time",
        default="00:10:00",
        help="finite Slurm allocation time used for the benchmark process",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="run directly on an already allocated/visible GPU",
    )
    parser.add_argument(
        "--shell-class",
        action="append",
        help="shell class to tune; repeat for a single-process batch",
    )
    parser.add_argument(
        "--shell-class-file",
        action="append",
        type=Path,
        metavar="PATH",
        help=(
            "file containing shell classes (one per line or comma-separated); "
            "repeat to combine lists"
        ),
    )
    parser.add_argument(
        "--consumer",
        choices=tuple(item.value for item in KernelConsumer),
        default=KernelConsumer.FORCE.value,
    )
    parser.add_argument(
        "--schedule-kind",
        action="append",
        choices=tuple(item.value for item in ScheduleKind),
        help=(
            "restrict the search to one schedule family; repeat to combine "
            "families when tuning several classes with the same strategy"
        ),
    )
    parser.add_argument("--tasks", type=int, default=512)
    parser.add_argument("--primitives", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--compile-jobs", type=int, default=2)
    parser.add_argument("--compile-timeout", type=float, default=300.0)
    parser.add_argument("--minimum-speedup", type=float, default=1.0)
    parser.add_argument("--absolute-tolerance", type=float, default=2.0e-10)
    parser.add_argument("--relative-tolerance", type=float, default=2.0e-10)
    parser.add_argument("--max-registers", type=int)
    parser.add_argument("--max-packed-registers", type=int)
    parser.add_argument("--max-stack-bytes", type=int)
    parser.add_argument("--max-shared-bytes", type=int)
    parser.add_argument(
        "--allow-experimental-subgroup-winner",
        action="store_true",
        help=(
            "allow subgroup schedules to become manifest winners after the "
            "synthetic gate; normally they require a separate end-to-end "
            "production acceptance"
        ),
    )
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--work-directory", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_PRODUCTION_MANIFEST_PATH,
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        help="write winners into this schema-v2 manifest path",
    )
    parser.add_argument(
        "--require-all-winners",
        action="store_true",
        help=(
            "write --manifest-output only when every requested shell class "
            "passes all autotune gates"
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--max-candidates",
        type=int,
        help="bound schedule candidates per class (plus a required baseline)",
    )
    return parser


def main() -> None:
    parser = argument_parser()
    arguments = parser.parse_args()
    if arguments.compile_jobs < 1:
        parser.error("--compile-jobs must be positive")
    if arguments.compile_timeout <= 0:
        parser.error("--compile-timeout must be positive")
    if not (arguments.shell_class or arguments.shell_class_file):
        parser.error("autotune requires --shell-class or --shell-class-file")
    report = _run_autotune(arguments)
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(output, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(output, encoding="utf-8")
    if len(report["winners"]) != len(set(arguments.shell_class)):
        raise SystemExit(4)
