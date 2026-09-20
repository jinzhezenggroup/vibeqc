"""Run issue #662 stationary CUDA timeline benchmarks through the Slurm profile."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from vibeqc_compiler.common.cuda_adapter import resolve_cuda_execution_profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--time", default="00:30:00")
    args, benchmark_args = parser.parse_known_args()
    toolkit = Path(os.environ.get("CUDA_HOME", "/group/software/cuda-12.9.1"))
    if not (toolkit / "bin/nvcc").is_file():
        raise SystemExit("set CUDA_HOME to a complete CUDA 12.9 toolkit")
    library = args.library or Path(
        os.environ.get("VIBEQC_LIBRARY", ROOT / "build-cuda/libvibeqc.so")
    )
    output = args.output or ROOT / "build-cuda/stationary-662/timeline.json"
    cache = args.cache or ROOT / ".cache/stationary-662"
    env = dict(os.environ)
    env.update(
        CUDA_HOME=str(toolkit),
        CUDACXX=str(toolkit / "bin/nvcc"),
        PATH=str(toolkit / "bin") + os.pathsep + env.get("PATH", ""),
        LD_LIBRARY_PATH=str(toolkit / "lib64")
        + os.pathsep
        + env.get("LD_LIBRARY_PATH", ""),
        PYTHONPATH=os.pathsep.join((str(ROOT), str(ROOT / "python"))),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        CCACHE_DIR="/tmp/vibeqc-662-cuda-ccache",
        CCACHE_TEMPDIR="/tmp/vibeqc-662-cuda-ccache/tmp",
        VIBEQC_LIBRARY=str(library),
    )
    profile = resolve_cuda_execution_profile(local=False, slurm_time=args.time)
    command = [
        sys.executable,
        "benchmarks/stationary_cuda_timeline.py",
        "--output",
        str(output),
        "--cache",
        str(cache),
        *benchmark_args,
    ]
    print("Slurm profile:", profile.to_dict(), flush=True)
    print("Evidence:", output, flush=True)
    return subprocess.run(
        profile.wrap(command), cwd=ROOT, env=env, check=False
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
