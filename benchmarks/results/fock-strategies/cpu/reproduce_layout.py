"""Reproduce the host DF code-layout diagnosis without modifying a checkout.

Run inside a finite Slurm GPU allocation, passing a production #202 checkout
and its build. These temporary host-only DSOs interpose the pre-fix CPU DF
implementation at three alignments. This is a controlled diagnosis, not a
candidate scientific algorithm or a production selector.
"""

import argparse
import json
import os
import statistics
import subprocess
from pathlib import Path


def capture(argv, **kwargs):
    """Capture a checked argument vector without shell interpolation."""
    return subprocess.check_output(argv, text=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler", default="c++")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("run inside a finite Slurm allocation")
    root, build, output = (
        args.head.resolve(),
        args.build.resolve(),
        args.output.resolve(),
    )
    output.mkdir(parents=True, exist_ok=False)
    before = "6fc010e2adbb2d132b45c869eaf2355d284faf4b"
    source = capture(
        ["git", "-C", str(root), "show", before + ":src/scf/density_fitting.cpp"]
    )
    probe = (root / "benchmarks/fock_dispatch_probe.cpp").read_text()
    condition = '#if __has_include("scf/fock_provider.hpp")'
    assert probe.count(condition) == 1
    probe = probe.replace(condition, "#if 0")
    (output / "raw.cpp").write_text(probe)
    flags = [
        args.compiler,
        "-O3",
        "-DNDEBUG",
        "-std=c++20",
        "-I",
        str(root / "include"),
        "-I",
        str(root / "src"),
    ]
    link = ["-L", str(build), "-Wl,-rpath," + str(build), "-lvibeqc"]
    subprocess.run(
        [*flags, str(output / "raw.cpp"), *link, "-o", str(output / "raw")], check=True
    )
    for alignment in (16, 32, 64):
        text = source.replace(
            "std::vector<double> build_exchange(",
            f"[[gnu::aligned({alignment})]] std::vector<double> build_exchange(",
        )
        path = output / f"align-{alignment}.cpp"
        path.write_text(text)
        subprocess.run(
            [
                *flags,
                "-shared",
                "-fPIC",
                str(path),
                *link,
                "-o",
                str(output / f"align-{alignment}.so"),
            ],
            check=True,
        )
    rows, reference = [], None
    for cycle in range(4):
        order = (16, 32, 64) if cycle % 2 == 0 else (64, 32, 16)
        for alignment in order:
            env = dict(os.environ, LD_PRELOAD=str(output / f"align-{alignment}.so"))
            result = json.loads(capture([str(output / "raw")], env=env))
            if reference is None:
                reference = result["rows"]
            for a, b in zip(reference, result["rows"], strict=True):
                assert a["coulomb"] == b["coulomb"] and a["exchange"] == b["exchange"]
            rows.append({"cycle": cycle, "alignment": alignment, "result": result})
            print(
                cycle,
                alignment,
                [
                    (r["nbf"], r["approximation"], statistics.median(r["seconds"]))
                    for r in result["rows"]
                ],
                flush=True,
            )
    record = {
        "source_revision": before,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "compiler": capture([args.compiler, "--version"]),
        "records": rows,
    }
    (output / "comparison.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
