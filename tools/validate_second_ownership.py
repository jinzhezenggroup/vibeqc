"""Reproduce native second-derivative lifetime checks under CPU/CUDA sanitizers.

The executable drives the actual generated ABI and shared storage owner.
Scientific error gates belong to validate_second_derivatives.py and pytest;
this checks memory safety, invalid/empty chunks and transactional replay.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from vibeqc_compiler.common.compiler_process import run_compiler
from vibeqc_compiler.integral.second_derivatives import (
    build_eri_second_ir,
    build_one_electron_second_ir,
    build_second_derivative_kernel,
)
from vibeqc_compiler.integral.second_derivatives_native import (
    SECOND_RECORD_TAG,
    emit_second_derivative_runtime,
)

DRIVER = r"""
int main() {
  constexpr double exponents[] = {0.6, 0.8, 1.1, 0.9};
  constexpr double positions[4][3] = {{.13,-.31,.24},{-.43,.27,.51},{.68,-.14,-.22},{-.21,.48,-.63}};
  SecondRecord records[2]{};
  for (auto& record : records) {
    auto& b = record.primitive;
    b.kind = @TAG@U;
    b.weights[0] = 1;
    for (unsigned c = 0; c < 4; ++c) {
      b.exponents[c] = c < @SHELLS@ ? exponents[c] : 1;
      for (unsigned a = 0; a < 3; ++a) b.centers[c][a] = c < @CENTERS@ ? positions[c][a] : 0;
    }
    for (unsigned i = 0; i < @WEIGHTS@; ++i) record.weights[i] = @RAW@ ? 1 : i % 2 ? -0.31 : 0.73;
    for (unsigned i = 0; i < @DIRECTIONS@; ++i) record.direction[i] = (i + 1.0) / 12;
  }
  char detail[1024]{};
  void* handle = nullptr;
  auto check = [&](bool condition) {
    if (!condition) { std::fprintf(stderr, "ownership failure: %s\n", detail); std::exit(2); }
  };
  check(vibeqc_second_create_v1(0, @MAJOR@, 0, 2, 1, 4096, &handle, detail, sizeof(detail)) == VIBEQC_STATUS_SUCCESS);
  double output[@OUTPUTS@]{};
  auto run = [&](std::size_t count, std::size_t stride = sizeof(SecondRecord)) {
    return vibeqc_second_run_v1(handle, count ? records : nullptr, count, stride, 1, output, 0, detail, sizeof(detail));
  };
  check(run(2) == VIBEQC_STATUS_SUCCESS);
  for (double value : output) check(std::isfinite(value));
  for (double& value : output) value = 173;
  records[1].primitive.exponents[0] = records[1].primitive.exponents[1] = 1.7e308;
  check(run(2) == VIBEQC_STATUS_NUMERICAL_FAILURE);
  for (double value : output) check(value == 173);
  records[1] = records[0];
  check(run(2, sizeof(SecondRecord) - 8) == VIBEQC_STATUS_INVALID_ARGUMENT);
  for (double value : output) check(value == 173);
  records[1].primitive.kind = 0;
  check(run(2) == VIBEQC_STATUS_INVALID_ARGUMENT);
  for (double value : output) check(value == 173);
  records[1] = records[0];
  check(run(1) == VIBEQC_STATUS_SUCCESS);
  check(run(0) == VIBEQC_STATUS_SUCCESS);
  for (double value : output) check(value == 0);
  vibeqc_second_destroy_v1(handle);
  std::puts("native ownership/replay passed");
}
"""


def main():
    """Compile finite fixtures, then execute only inside the declared backend."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--compiler")
    parser.add_argument("--sanitizer", default="compute-sanitizer")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cuda = args.backend == "cuda"
    if cuda and not args.compile_only and not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("real CUDA sanitizer execution requires a Slurm allocation")
    args.output.mkdir(parents=True, exist_ok=False)
    cases = (
        (
            "attraction_hvp",
            build_one_electron_second_ir(
                "nuclear_attraction", (1, 2), output="weighted_hvp"
            ),
            (0, 4),
            tuple(range(9)),
        ),
        ("eri_hvp", build_eri_second_ir((2, 1, 0, 1)), (0, 4), tuple(range(12))),
        (
            "ffff_raw",
            build_eri_second_ir((3, 3, 3, 3), output="raw_hessian"),
            (0,),
            (0,),
        ),
    )
    for name, integral, components, outputs in cases:
        kernel = build_second_derivative_kernel(
            integral, components, output_indices=outputs
        )
        source = (
            emit_second_derivative_runtime(kernel, backend=args.backend)
            + "\n#include <cstdlib>\n"
            + DRIVER
        )
        for token, value in {
            "@TAG@": SECOND_RECORD_TAG,
            "@SHELLS@": len(integral.signature.shells),
            "@CENTERS@": len(integral.operator.centers),
            "@WEIGHTS@": len(components),
            "@OUTPUTS@": len(outputs),
            "@RAW@": int(integral.contractions[0].weights is None),
            "@DIRECTIONS@": 3 * len(integral.requested_derivative_centers)
            if integral.contractions[0].output == "weighted_hvp"
            else 0,
            "@MAJOR@": 12 if cuda else 0,
        }.items():
            source = source.replace(token, str(value))
        path = args.output / (name + (".cu" if cuda else ".cpp"))
        path.write_text(source)
        executable = (args.output / name).resolve()
        flags = (
            ["-arch=sm_120", "-O3", "--fmad=false"]
            if cuda
            else [
                "-O1",
                "-g",
                "-ffp-contract=off",
                "-fsanitize=address,undefined",
                "-fno-omit-frame-pointer",
            ]
        )
        result = run_compiler(
            [
                args.compiler or ("nvcc" if cuda else "c++"),
                "-std=c++17",
                *flags,
                f"-I{ROOT / 'src'}",
                f"-I{ROOT / 'include'}",
                str(path),
                *(["-lcublas"] if cuda else []),
                "-o",
                str(executable),
            ],
            300,
            label=name,
        )
        (args.output / (name + "-compiler.log")).write_text(
            result.stdout + result.stderr
        )
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        if args.compile_only:
            print(executable, flush=True)
            continue
        command = (
            [
                args.sanitizer,
                "--tool",
                "memcheck",
                "--leak-check",
                "full",
                "--error-exitcode",
                "99",
            ]
            if cuda
            else []
        ) + [str(executable)]
        environment = {
            **os.environ,
            "ASAN_OPTIONS": "detect_leaks=1:halt_on_error=1",
            "UBSAN_OPTIONS": "halt_on_error=1:print_stacktrace=1",
        }
        subprocess.run(command, env=environment, check=True, timeout=180)
        print(name + " passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
