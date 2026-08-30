"""Capture the current 768-AO bounded-Fock class timing profile.

The normal endpoint keeps class timing disabled.  This focused diagnostic
first converges one cold batch, freezes its density, and then asks the native
backend to execute only the bounded Fock construction on that fixed ``dm0``.
The native library writes one ``class/launches/gpu_ms/share`` row per selected
class to stderr.  Keeping the count diagnostic opt-in is important: counting
every retained descriptor uses the compacting path and is substantially more
expensive than the steady streaming worker being measured here.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="water-32mer-4s4-def2-svp-spherical")
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--fock-classes",
        help=(
            "comma-separated Fock classes to profile during warm replays; "
            "the cold convergence still uses the complete production set"
        ),
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help=(
            "also enable the expensive retained-task count diagnostic; "
            "normally leave this disabled for class timing"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.max_iterations < 1 or args.repeats < 1:
        parser.error("--max-iterations and --repeats must be positive")
    fock_classes = None
    if args.fock_classes is not None:
        fields = tuple(item.strip() for item in args.fock_classes.split(","))
        if not fields or any(not item for item in fields):
            parser.error("--fock-classes must contain non-empty names")
        if len(set(fields)) != len(fields):
            parser.error("--fock-classes must not contain duplicates")
        fock_classes = ",".join(fields)

    # Keep --help usable on scheduler login nodes.  CuPy and the native Python
    # binding are imported only once an allocated GPU job is actually running.
    import cupy as cp
    from vibeqc import Calculator

    from benchmarks._cases import benchmark_cases

    case = benchmark_cases()[args.case]
    calculator = Calculator(
        method=case.method,
        basis=case.vibeqc_basis,
        basis_representation=case.basis_representation,
        device="cuda",
        max_iterations=args.max_iterations,
        energy_tolerance=1.0e-12,
        density_tolerance=1.0e-10,
        screening_tolerance=1.0e-14,
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "benchmark": "issue52_current_fock_profile",
        "case": args.case,
        "max_iterations": args.max_iterations,
        "repeats": args.repeats,
        "count_diagnostic": args.count,
        "warm_profile_seconds": [],
    }

    with calculator.prepare_batch(
        [case.atoms],
        charges=[case.charge],
        multiplicities=[case.multiplicity],
        warm_start=True,
    ) as batch:
        start = time.perf_counter()
        cold = batch.execute(strict=True)
        cp.cuda.Stream.null.synchronize()
        payload["cold_seconds"] = time.perf_counter() - start
        payload["cold_iterations"] = cold.items[0].iterations
        batch.set_warm_start_updates(False)

        # These switches are sampled per execution by the CUDA plan.  The
        # fixed post-cold density makes each repeat a comparable Fock replay.
        diagnostic_environment = {
            "VIBEQC_BOUNDED_DIRECT_FOCK_ONLY_DIAGNOSTIC": "1",
            "VIBEQC_BOUNDED_DIRECT_AOT_ONLY_DIAGNOSTIC": "1",
            "VIBEQC_BOUNDED_DIRECT_FOCK_CLASS_PROFILE": "1",
        }
        if args.count:
            diagnostic_environment["VIBEQC_BOUNDED_DIRECT_COUNT_DIAGNOSTIC"] = "1"
        if fock_classes is not None:
            diagnostic_environment["VIBEQC_AOT_FOCK_SHELL_CLASSES"] = fock_classes
        previous_environment = {
            name: os.environ.get(name) for name in diagnostic_environment
        }
        for name, value in diagnostic_environment.items():
            os.environ[name] = value

        warm_seconds: list[float] = []
        try:
            for _ in range(args.repeats):
                cp.cuda.Stream.null.synchronize()
                start = time.perf_counter()
                # Fock-only mode intentionally reports NOT_CONVERGED after
                # the diagnostic on older libraries and success on newer
                # ones; strict=False preserves the timing output.
                result = batch.execute(strict=False)
                cp.cuda.Stream.null.synchronize()
                warm_seconds.append(time.perf_counter() - start)
                if result.items[0].status_message not in {
                    "SCF did not converge",
                    "success",
                }:
                    raise RuntimeError(
                        "Fock diagnostic returned unexpected status: "
                        f"{result.items[0].status_message}"
                    )
        finally:
            for name, value in previous_environment.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        payload["warm_profile_seconds"] = warm_seconds

    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
        print(f"JSON result: {args.output}")


if __name__ == "__main__":
    main()
