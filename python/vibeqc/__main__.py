"""Command-line entry point for explicit tuning and validated profile management."""

import argparse
import json
import sys
from pathlib import Path

from . import _native, profiles


def parser() -> argparse.ArgumentParser:
    """Expose stable quick/full, diagnostic, and cluster installation commands."""
    root = argparse.ArgumentParser(prog="vibeqc")
    commands = root.add_subparsers(dest="command", required=True)
    resources = commands.add_parser(
        "resources", help="dry-run HF resource estimates without scientific execution"
    )
    resources.add_argument("input", type=Path)
    resources.add_argument("--basis", default="sto-3g")
    resources.add_argument("--auxiliary-basis")
    resources.add_argument("--representation", choices=("cartesian", "spherical"))
    resources.add_argument("--method", choices=("rhf", "uhf"), default="rhf")
    resources.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    resources.add_argument(
        "--density-fitting", choices=("none", "cpu", "cuda", "auto"), default="none"
    )
    resources.add_argument("--charge", type=int, default=0)
    resources.add_argument("--multiplicity", type=int, default=1)
    resources.add_argument("--units", choices=("angstrom", "bohr"), default="angstrom")
    resources.add_argument("--batch", type=int, default=1)
    resources.add_argument("--host-bytes", type=int)
    resources.add_argument("--device-bytes", type=int)
    resources.add_argument("--pinned-host-bytes", type=int)
    resources.add_argument("--host-reserve-bytes", type=int, default=0)
    resources.add_argument("--device-reserve-bytes", type=int, default=0)
    resources.add_argument("--headroom-fraction", type=float, default=0)
    resources.add_argument("--diis-history", type=int, default=8)
    resources.add_argument("--max-iterations", type=int, default=100)
    tune = commands.add_parser(
        "autotune", help="explicitly tune a representative XYZ workload"
    )
    modes = tune.add_mutually_exclusive_group()
    modes.add_argument(
        "--quick", action="store_true", help="bounded hotspot search (default)"
    )
    modes.add_argument(
        "--full", action="store_true", help="broader class and schedule coverage"
    )
    modes.add_argument(
        "--show-profile",
        action="store_true",
        help="show stored profiles without probing a GPU",
    )
    modes.add_argument(
        "--clear-profile", action="store_true", help="deactivate all local profiles"
    )
    tune.add_argument("input", type=Path, nargs="?")
    tune.add_argument("--basis", default="def2-svp")
    tune.add_argument("--method", choices=("rhf", "uhf"), default="rhf")
    tune.add_argument("--charge", type=int, default=0)
    tune.add_argument("--multiplicity", type=int, default=1)
    tune.add_argument("--units", choices=("angstrom", "bohr"), default="angstrom")
    tune.add_argument(
        "--representation", choices=("cartesian", "spherical"), default="cartesian"
    )
    tune.add_argument("--batch", type=int, default=1)
    tune.add_argument("--coverage", type=float)
    tune.add_argument("--max-classes", type=int)
    tune.add_argument("--max-candidates", type=int)
    tune.add_argument("--compile-jobs", type=int, default=2)
    tune.add_argument("--budget-seconds", type=int, default=7200)
    tune.add_argument("--repeats", type=int, default=6)
    tune.add_argument(
        "--retune", action="store_true", help="also search officially covered consumers"
    )
    tune.add_argument(
        "--portable-baseline",
        action="store_true",
        help="explicitly tune against generic CUDA even on an officially tuned device",
    )
    tune.add_argument(
        "--source-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="matching VibeQC source checkout for native/codegen compilation",
    )
    tune.add_argument("--nvcc", type=Path)
    tune.add_argument(
        "--export", type=Path, help="export the accepted binary/evidence bundle"
    )
    tune.set_defaults(device_id=0)
    profile = commands.add_parser(
        "profile", help="manage validated local CUDA profiles"
    )
    operations = profile.add_subparsers(dest="operation", required=True)
    install = operations.add_parser(
        "install", help="validate and install a cluster bundle on the allocated GPU"
    )
    install.add_argument("archive", type=Path)
    export = operations.add_parser(
        "export", help="export an accepted bundle and its evidence"
    )
    export.add_argument("directory", type=Path)
    export.add_argument("destination", type=Path)
    operations.add_parser("show", help="show stored profiles without probing hardware")
    operations.add_parser(
        "clear", help="deactivate profiles; retain binaries used by live processes"
    )
    operations.add_parser(
        "diagnose", help="probe the allocated GPU and explain runtime selection"
    )
    return root


def main() -> int:
    """Leave the working profile intact after errors; diagnostics remain actionable."""
    arguments = parser()
    args = arguments.parse_args()
    try:
        if args.command == "resources":
            from .autotune import read_xyz
            from .resources import ResourceBudget
            from .resources_hf import estimate_hf_resources

            if not 1 <= args.batch <= 100000:
                raise ValueError("resource dry-run batch must be in [1, 100000]")
            atoms = read_xyz(args.input, units=args.units)
            budget = ResourceBudget(
                host_bytes=args.host_bytes,
                device_bytes=args.device_bytes,
                pinned_host_bytes=args.pinned_host_bytes,
                host_reserve_bytes=args.host_reserve_bytes,
                device_reserve_bytes=args.device_reserve_bytes,
                headroom_fraction=args.headroom_fraction,
            )
            plan = estimate_hf_resources(
                [atoms] * args.batch,
                budget=budget,
                basis=args.basis,
                auxiliary_basis=args.auxiliary_basis,
                basis_representation=args.representation,
                method=args.method,
                backend=args.backend,
                density_fitting=args.density_fitting,
                charges=[args.charge] * args.batch,
                multiplicities=[args.multiplicity] * args.batch,
                diis_history=args.diis_history,
                max_iterations=args.max_iterations,
            )
            print(json.dumps(plan.to_dict(), indent=2))
            return 0 if plan.status == "feasible" else 2
        operation = (
            args.operation
            if args.command == "profile"
            else (
                "show" if args.show_profile else "clear" if args.clear_profile else None
            )
        )
        if operation == "show":
            path = profiles.cache_root() / "active.json"
            print(
                json.dumps(
                    {
                        "cache": str(path.parent),
                        "active": json.loads(path.read_text()) if path.exists() else {},
                    },
                    indent=2,
                )
            )
        elif operation == "clear":
            profiles.clear_profiles()
            print("Local profiles deactivated.")
        elif operation == "export":
            profiles.export_bundle(args.directory, args.destination)
            print(args.destination)
        elif operation == "install":
            probe = profiles.probe_device(_native.load_library())
            nvcc = profiles.find_nvcc()
            installed = profiles.import_bundle(
                args.archive, probe, profiles.toolchain_identity(nvcc) if nvcc else None
            )
            print(installed)
        elif operation == "diagnose":
            library = _native.load_library(device="cuda")
            print(json.dumps(library._vibeqc_profile_diagnostics, indent=2))
        else:
            if args.input is None:
                arguments.error(
                    "autotune needs an XYZ input or a profile management option"
                )
            args.coverage = (
                args.coverage
                if args.coverage is not None
                else (1.0 if args.full else 0.97)
            )
            args.max_classes = (
                args.max_classes
                if args.max_classes is not None
                else (55 if args.full else 8)
            )
            args.max_candidates = (
                args.max_candidates
                if args.max_candidates is not None
                else (16 if args.full else 4)
            )
            if (
                not 0 < args.coverage <= 1
                or min(
                    args.batch,
                    args.max_classes,
                    args.max_candidates,
                    args.compile_jobs,
                    args.budget_seconds,
                    args.multiplicity,
                )
                < 1
                or args.repeats < 4
            ):
                arguments.error(
                    "positive budgets/counts, coverage in (0,1], and at least four repeats are required"
                )
            from .autotune import run

            report = run(args)
            print(
                json.dumps(
                    {
                        "installed": report["installed"],
                        "report": report["directory"] + "/report.json",
                        "failure": report.get("failure"),
                        "reason": report.get("reason"),
                    },
                    indent=2,
                )
            )
            return int("failure" in report)
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"vibeqc: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
