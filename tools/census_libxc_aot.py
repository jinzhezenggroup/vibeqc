#!/usr/bin/env python3
"""Offline Libxc compiler census; compile probes only with explicit --compile.

Examples (from an uninstalled checkout)::

    python tools/census_libxc_aot.py --output /tmp/xc-census.json
    python tools/census_libxc_aot.py --name LDA_C_VWN_4 --backend cpu \
        --compile --max-artifacts 2 --output /tmp/xc-compile.json

No report from this tool grants numerical, GPU-runtime or public DFT admission.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from vibeqc_compiler.common.compiler_work import compiler_work_budget
from vibeqc_compiler.common.provenance import atomic_json
from vibeqc_compiler.xc import bulk_aot, libxc_bulk


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--name", action="append", help="registration; repeat to select more"
    )
    parser.add_argument("--spin", action="append", choices=("polarized", "unpolarized"))
    parser.add_argument(
        "--derivative-order", type=int, action="append", choices=(0, 1, 2)
    )
    parser.add_argument("--backend", action="append", choices=bulk_aot.BACKENDS)
    parser.add_argument("--work-limit", type=int, default=1_000_000)
    parser.add_argument("--max-artifacts", type=int, default=64)
    parser.add_argument("--max-source-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--catalog", type=Path, default=libxc_bulk.CATALOG_PATH)
    parser.add_argument(
        "--compile", action="store_true", help="explicit offline object probes"
    )
    parser.add_argument("--cc", help="C compiler executable (default: cc)")
    parser.add_argument("--nvcc", help="CUDA compiler executable (default: nvcc)")
    parser.add_argument("--cuda-arch", default="sm_80")
    parser.add_argument("--compile-timeout", type=float, default=60)
    args = parser.parse_args(argv)
    try:
        plan = bulk_aot.census_catalog(
            names=args.name,
            spins=args.spin or ("polarized", "unpolarized"),
            derivative_orders=args.derivative_order or (1,),
            backends=args.backend or bulk_aot.BACKENDS,
            work_limit=args.work_limit,
            budget=bulk_aot.PackageBudget(args.max_artifacts, args.max_source_bytes),
            source_root=args.source_root,
            catalog_path=args.catalog,
        )
        # Retain the completed census even when an optional probe fails later.
        atomic_json(args.output, plan)
        if args.compile:

            def load_variant(record: dict) -> bulk_aot.SourceVariant:
                with compiler_work_budget(args.work_limit):
                    program = libxc_bulk.build_bulk_program(
                        record["name"],
                        spin=record["spin"],
                        source_root=args.source_root,
                        catalog_path=args.catalog,
                    )
                    return bulk_aot.inspect_program(
                        program,
                        record["derivative_order"],
                        record["backend"],
                        domain=libxc_bulk.BULK_SEMANTICS,
                    )

            plan["measurements"] = bulk_aot.measure_plan(
                plan,
                load_variant,
                compilers={"cpu": args.cc, "cuda": args.nvcc},
                timeout=args.compile_timeout,
                cuda_arch=args.cuda_arch,
            )
            atomic_json(args.output, plan)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(2, f"census failed: {error}\n")
    summary = plan["summary"]
    print(
        f"{summary['registration_variants']} emitted variants; "
        f"{summary['unique_artifacts']} exact-source groups; "
        f"{summary['selected_artifacts']} selected build tasks; "
        f"{len(plan['blocked_imports'])} graph-blocked registrations"
    )
    failures = sum(row["status"] != "emitted" for row in plan["observations"])
    failures += sum(
        row["status"] not in ("compiled", "not-selected")
        for row in plan.get("measurements", {}).values()
    )
    if failures:
        print(
            f"{failures} requested emission/compile probes lack successful evidence",
            file=sys.stderr,
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
