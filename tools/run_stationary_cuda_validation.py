"""Run the opt-in stationary CUDA qualification through the existing Slurm profile.

No GPU probe occurs in this launcher. Configure/build libvibeqc separately in
build-cuda, then run this script with optional pytest selectors or --sanitizer.
The child inherits Slurm device visibility unchanged.
"""

import argparse
import os
import subprocess
import sys
import typing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from vibeqc_compiler.common.cuda_adapter import resolve_cuda_execution_profile


def main() -> typing.Any:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sanitizer",
        nargs="?",
        const="memcheck",
        choices=("memcheck", "initcheck", "synccheck"),
        help="run Compute Sanitizer on the actual device test (default: memcheck)",
    )
    parser.add_argument("--library", type=Path, help="explicit current native library")
    parser.add_argument("--cache", type=Path, help="ignored generated-artifact cache")
    parser.add_argument("--evidence", type=Path, help="local evidence output directory")
    parser.add_argument("--cpu-regression", action="store_true")
    parser.add_argument("--full-fd", action="store_true")
    parser.add_argument("--time", default="00:20:00")
    args, pytest_args = parser.parse_known_args()
    toolkit = Path(os.environ.get("CUDA_HOME", "/group/software/cuda-12.9.1"))
    if not (toolkit / "bin/nvcc").is_file():
        raise SystemExit("set CUDA_HOME to a complete CUDA 12.9 toolkit")
    env = dict(os.environ)
    env.update(
        CUDA_HOME=str(toolkit),
        CUDACXX=str(toolkit / "bin/nvcc"),
        PATH=str(toolkit / "bin") + os.pathsep + env.get("PATH", ""),
        LD_LIBRARY_PATH=str(toolkit / "lib64")
        + os.pathsep
        + env.get("LD_LIBRARY_PATH", ""),
        PYTHONPATH=os.pathsep.join(
            (str(ROOT), str(ROOT / "python"), str(ROOT / "tests/python"))
        ),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        CCACHE_DIR="/tmp/vibeqc-163-cuda-ccache",
        CCACHE_TEMPDIR="/tmp/vibeqc-163-cuda-ccache/tmp",
        VIBEQC_LIBRARY=str(
            args.library
            or os.environ.get("VIBEQC_LIBRARY")
            or ROOT
            / (
                "build-cpu-regression/libvibeqc.so"
                if args.cpu_regression
                else "build-cuda/libvibeqc.so"
            )
        ),
        VIBEQC_DFT_CUDA_TEST="1",
        VIBEQC_STATIONARY_CACHE=str(
            args.cache
            or os.environ.get("VIBEQC_STATIONARY_CACHE")
            or ROOT / ".cache/stationary-cuda"
        ),
        VIBEQC_STATIONARY_EVIDENCE=str(
            args.evidence
            or os.environ.get("VIBEQC_STATIONARY_EVIDENCE")
            or ROOT / "build-cuda/stationary-evidence"
        ),
    )
    profile = resolve_cuda_execution_profile(local=False, slurm_time=args.time)
    if args.full_fd:
        env["VIBEQC_STATIONARY_FULL_FD"] = "1"
    test = "tests/python/test_dft_complete_cuda.py"
    if args.cpu_regression:
        test = "tests/python/test_dft_complete_cpu.py"
    subprocess.run(
        ["squeue", "-o", "%.18i %.9P %.20j %.8u %.2t %.10M %R"], check=True, env=env
    )
    command = [
        sys.executable,
        "-m",
        "pytest",
        test,
        "-q",
        "-s",
        *pytest_args,
    ]
    if args.cpu_regression:
        if args.sanitizer:
            parser.error("--sanitizer requires the CUDA suite, not --cpu-regression")
        # CPU regressions do not consume a GPU allocation.
        return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode
    if args.sanitizer:
        command = [
            str(toolkit / "bin/compute-sanitizer"),
            "--tool",
            args.sanitizer,
            "--error-exitcode",
            "99",
            *command,
        ]
    print("Slurm profile:", profile.to_dict(), flush=True)
    return subprocess.run(
        profile.wrap(command), cwd=ROOT, env=env, check=False
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
