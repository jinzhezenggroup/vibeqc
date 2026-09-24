"""Capture complete warm DF forces, with separate clean and intrusive runs.

Use Nsight's cudaProfilerApi capture range to exclude preparation and priming.
The component trace is optional; the progress trace is deliberately prohibited
because it fences each region and changes pageable-copy/stream interactions.
This runner checks every complete force against the retained independent #206
reference. It never interprets inclusive host copy time as DMA duration.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

try:
    from benchmarks._retention import raw_output_path
except ModuleNotFoundError:
    from _retention import raw_output_path
from vibeqc import Calculator, _native
from vibeqc.autotune import source_identity

from benchmarks._cases import benchmark_cases
from benchmarks.compare_gpu4pyscf_batch import convergence_payload, scaled_geometries
from benchmarks.df_component_ledger import (
    aggregate,
    aggregate_host,
    read_host_trace,
    read_trace,
)
from benchmarks.issue206_resident_sentinel import validate_response_record

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    19: "oh-def2-svp-spherical-uhf",
    192: "water-octamer-s4-def2-svp-spherical",
    384: "water-hexadecamer-2s4-def2-svp-spherical",
    768: "water-32mer-4s4-def2-svp-spherical",
}
CANDIDATES = {
    "generic": ("generic", "warp", "scalar", "pageable"),
    "shell-warp": ("shell", "warp", "scalar", "pageable"),
    "shell-packed": ("shell", "packed", "scalar", "pageable"),
    "shell-compact": ("shell", "compact", "scalar", "pageable"),
    "blas": ("generic", "warp", "blas", "pageable"),
    "pinned": ("generic", "warp", "blas", "pinned-panels"),
    "combined-warp": ("shell", "warp", "blas", "pinned-panels"),
    "combined-packed": ("shell", "packed", "blas", "pinned-panels"),
    "combined-compact": ("shell", "compact", "blas", "pinned-panels"),
}
CANDIDATE_CONTROLS = (
    "VIBEQC_DF_WEIGHTED_EXECUTION",
    "VIBEQC_DF_SHELL_SCHEDULE",
    "VIBEQC_DF_RESPONSE_ALGEBRA",
    "VIBEQC_DF_RAW_STAGING",
)


def validate_three_center_response_work(
    counters: Mapping[str, int], nbf: int, naux: int
) -> dict[str, int | str]:
    """Validate and describe dense or packed three-center response work.

    ``three_center_derivative_weights`` counts the public derivative weights
    consumed by the response producer. Dense AO-pair routes consume
    ``naux * nbf * nbf`` weights, while symmetric packed-pair routes consume
    ``naux * nbf * (nbf + 1) // 2``. The timeline must accept both exact
    routes; comparing every trace against the dense shape would reject the
    qualified packed route at the larger endpoint.
    """

    packed_pairs = counters.get("response_packed_pairs", 0)
    if type(packed_pairs) is not int or packed_pairs not in (0, 1):
        raise RuntimeError("response_packed_pairs must be 0 or 1")
    pair_storage = "packed" if packed_pairs else "dense"
    pair_stride = nbf * (nbf + 1) // 2 if packed_pairs else nbf * nbf
    expected = naux * pair_stride
    observed = counters.get("three_center_derivative_weights")
    observed_bytes = counters.get("three_center_derivative_weight_bytes")
    # The device response producer publishes both counters.  The retained
    # host-weight fallback predates the element counter and publishes only
    # exact FP64 byte work, so recover the logical count without weakening the
    # completeness check or accepting a rounded byte total.
    if observed is None and observed_bytes is not None:
        if type(observed_bytes) is not int or observed_bytes % 8:
            raise RuntimeError(
                "three-center response weight bytes are not integral FP64 work"
            )
        observed = observed_bytes // 8
    if observed is None:
        observed = 0
    if observed != expected:
        raise RuntimeError(
            "incomplete three-center response work for "
            f"{pair_storage} pairs: {observed} != {expected}"
        )
    if observed_bytes is not None and observed_bytes != expected * 8:
        raise RuntimeError(
            "three-center response weight bytes disagree with pair storage: "
            f"{observed_bytes} != {expected * 8}"
        )
    return {
        "pair_storage": pair_storage,
        "pair_stride": pair_stride,
        "expected_three_center_derivative_weights": expected,
        "observed_three_center_derivative_weights": observed,
    }


def validate_host_fallback_probes(upload: str, scatter: str) -> None:
    """Host fallback has no device upload/scatter attribution probe owner."""
    if upload or scatter:
        raise RuntimeError(
            "host fallback cannot qualify requested device attribution probes"
        )


def validate_metric_response_work(counters: Mapping[str, int], naux: int) -> int:
    """Validate metric derivative work from element or exact FP64-byte counts."""

    expected = naux * naux
    observed = counters.get("metric_derivative_weights")
    observed_bytes = counters.get("metric_derivative_weight_bytes")
    if observed is None and observed_bytes is not None:
        if type(observed_bytes) is not int or observed_bytes % 8:
            raise RuntimeError(
                "metric response weight bytes are not integral FP64 work"
            )
        observed = observed_bytes // 8
    if observed != expected:
        raise RuntimeError(
            f"incomplete metric response work: {observed or 0} != {expected}"
        )
    if observed_bytes is not None and observed_bytes != expected * 8:
        raise RuntimeError(
            "metric response weight bytes disagree with auxiliary shape: "
            f"{observed_bytes} != {expected * 8}"
        )
    return observed


def main() -> None:
    """Require source/binary agreement and fresh artifacts before any GPU work."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aos", type=int, choices=CASES, default=384)
    parser.add_argument("--batch", type=int, choices=(1, 4), default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=raw_output_path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        help="Retained matched comparator JSON; required for cases without a committed reference",
    )
    parser.add_argument(
        "--density-fitting-memory-budget-bytes",
        type=int,
        default=0,
        help="Native DF allowance; positive force requests split it between values and response",
    )
    parser.add_argument(
        "--response-memory-budget-bytes",
        type=int,
        help="Override only resident DF response scratch, keeping the raw value provider fixed",
    )
    parser.add_argument("--component-trace", action="store_true")
    parser.add_argument("--nsys", action="store_true")
    parser.add_argument(
        "--candidate",
        action="append",
        choices=CANDIDATES,
        help="Sweep named force consumers on one fixed post-cold density; repeat to select order",
    )
    parser.add_argument(
        "--expected-response-policy",
        choices=("auto", "resident", "streamed", "fallback"),
        default="auto",
        help="optional structural route gate for traced force responses",
    )
    parser.add_argument("--max-h2d-bytes", type=int)
    parser.add_argument("--max-d2h-bytes", type=int)
    parser.add_argument("--max-transformed-tile-productions", type=int)
    args = parser.parse_args()
    if args.density_fitting_memory_budget_bytes < 0:
        parser.error("DF memory allowance must be nonnegative")
    if args.response_memory_budget_bytes is not None:
        if (
            args.response_memory_budget_bytes <= 0
            or args.density_fitting_memory_budget_bytes
        ):
            parser.error(
                "response override requires positive bytes and a zero public DF allowance"
            )
        os.environ["VIBEQC_DF_RESPONSE_BUDGET_BYTES"] = str(
            args.response_memory_budget_bytes
        )
    elif os.environ.get("VIBEQC_DF_RESPONSE_BUDGET_BYTES"):
        parser.error(
            "select the response override explicitly with --response-memory-budget-bytes"
        )
    if args.repeats < 1:
        parser.error("repeats must be positive")
    for name in (
        "max_h2d_bytes",
        "max_d2h_bytes",
        "max_transformed_tile_productions",
    ):
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if args.nsys and args.repeats != 1:
        parser.error("capture one warm replay per profiler invocation")
    if args.nsys and args.candidate and len(args.candidate) != 1:
        parser.error("capture one candidate per profiler invocation")
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("run inside a finite Slurm GPU allocation")
    for key in ("VIBEQC_DF_TRACE", "VIBEQC_DF_HOST_TRACE", "VIBEQC_DF_PROGRESS_TRACE"):
        if os.environ.get(key):
            parser.error(f"unset ambient {key}; select instrumentation explicitly")
    if args.candidate:
        if any(
            os.environ.get(k)
            for k in (
                "VIBEQC_DF_RESPONSE_UPLOAD_PROBE",
                "VIBEQC_DF_RESPONSE_SCATTER_PROBE",
            )
        ):
            parser.error("candidate sweeps cannot be combined with attribution probes")
        os.environ.update(
            zip(CANDIDATE_CONTROLS, CANDIDATES[args.candidate[0]], strict=True)
        )
    args.output.mkdir(parents=True, exist_ok=False)
    library_path = args.library.resolve()
    os.environ["VIBEQC_LIBRARY"] = str(library_path)
    library = _native.load_library()
    library.vibeqc_get_source_identity.restype = ctypes.c_char_p
    identity = library.vibeqc_get_source_identity().decode()
    if identity != source_identity(ROOT):
        raise RuntimeError(
            "native library does not match the current scientific source"
        )
    reference_path = args.reference or (
        ROOT
        / "benchmarks/results/issue206-metric-gemv/measurements"
        / f"{args.aos}ao-b{args.batch}-forces-blas.json"
    )
    reference_path = reference_path.resolve()
    if not reference_path.is_file():
        parser.error("supply --reference with a retained matched complete-force result")
    reference = json.loads(reference_path.read_text())
    case = benchmark_cases()[CASES[args.aos]]
    geometries = scaled_geometries(case.atoms, args.batch)
    inputs = reference["workload"]
    serialized_geometries = [
        [
            {"element": element, "coordinates_bohr": list(position)}
            for element, position in atoms
        ]
        for atoms in geometries
    ]
    if inputs["geometries"] != serialized_geometries:
        raise RuntimeError("retained independent reference has different coordinates")
    expected_settings = {
        "case": CASES[args.aos],
        "ao_count": args.aos,
        "batch_size": args.batch,
        "method": case.method,
        "charge": case.charge,
        "multiplicity": case.multiplicity,
        "basis_representation": case.basis_representation,
        "auxiliary_basis": "same as orbital basis",
        "density_fitting_relative_threshold": 1e-10,
        "density_tolerance": 1e-10,
        "energy_tolerance": 1e-12,
        "max_iterations": 100,
        "vibeqc_screening_tolerance": 1e-12,
    }
    if any(inputs[key] != value for key, value in expected_settings.items()):
        raise RuntimeError(
            "retained independent reference has different scientific settings"
        )
    payload = {
        "scope": "intrusive attribution"
        if args.nsys or args.component_trace
        else "clean endpoint",
        "case": CASES[args.aos],
        "aos": args.aos,
        "batch": args.batch,
        "source_identity": identity,
        "library_sha256": hashlib.sha256(library_path.read_bytes()).hexdigest(),
        "library": str(library_path),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "process_id": os.getpid(),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "controls": {k: v for k, v in os.environ.items() if k.startswith("VIBEQC_")},
        "reference": str(reference_path),
        "density_fitting_memory_budget_bytes": args.density_fitting_memory_budget_bytes,
        "response_memory_budget_bytes": args.response_memory_budget_bytes,
        "reference_density_fitting_memory_budget_bytes": inputs[
            "density_fitting_memory_budget_bytes"
        ],
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "scientific_settings": expected_settings,
        "warm_start_policy": "fixed post-cold density, one untimed replay before measurement",
        "samples": [],
        "candidate_order": args.candidate,
    }
    patch = subprocess.check_output(["git", "diff", "--binary", "HEAD"])
    (args.output / "source.patch").write_bytes(patch)
    (args.output / "runner.py").write_bytes(Path(__file__).read_bytes())
    payload["source_patch_sha256"] = hashlib.sha256(patch).hexdigest()
    cudart = ctypes.CDLL("libcudart.so.12") if args.nsys else None

    def save() -> None:
        (args.output / "result.json").write_text(json.dumps(payload, indent=2) + "\n")

    save()
    calculator = Calculator(
        method=case.method,
        basis=case.vibeqc_basis,
        basis_representation=case.basis_representation,
        device="cuda",
        density_fitting="cuda",
        auxiliary_basis=case.vibeqc_basis,
        density_fitting_memory_budget_bytes=args.density_fitting_memory_budget_bytes,
        screening_tolerance=1e-12,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        max_iterations=100,
    )
    with calculator.prepare_batch(
        geometries,
        charges=[case.charge] * args.batch,
        multiplicities=[case.multiplicity] * args.batch,
    ) as batch:
        # Prime the same prepared state; cold work is outside every timed sample.
        cold_start = time.perf_counter()
        cold = batch.execute(strict=True, properties=("energy", "forces"))
        payload["cold_seconds"] = time.perf_counter() - cold_start
        payload["cold_iterations"] = [item.iterations for item in cold.items]
        payload["cold_convergence"] = convergence_payload(cold)
        batch.set_warm_start_updates(False)
        if not args.candidate:
            prime = batch.execute(strict=True, properties=("energy", "forces"))
            payload["prime_iterations"] = [item.iterations for item in prime.items]
        # Named candidates each receive their own prime below, including the
        # first one. Avoid an additional duplicate replay on large endpoints.
        save()
        runs = [
            (candidate, repeat)
            for candidate in (args.candidate or [None])
            for repeat in range(args.repeats)
        ]
        expected_branch = None
        for candidate, repeat in runs:
            if candidate and repeat == 0:
                os.environ.update(
                    zip(CANDIDATE_CONTROLS, CANDIDATES[candidate], strict=True)
                )
                # The force execution selector changes no SCF equation. Warm
                # snapshots stay frozen; this untimed replay primes allocations
                # and records the branch before measuring the selected consumer.
                candidate_prime = batch.execute(
                    strict=True, properties=("energy", "forces")
                )
                payload.setdefault("candidate_prime_iterations", {})[candidate] = [
                    item.iterations for item in candidate_prime.items
                ]
            prefix = f"{candidate}-" if candidate else ""
            trace_path = args.output / f"{prefix}warm-{repeat}.cuda.jsonl"
            host_path = args.output / f"{prefix}warm-{repeat}.host.jsonl"
            if args.component_trace:
                os.environ["VIBEQC_DF_TRACE"] = str(trace_path.resolve())
                os.environ["VIBEQC_DF_HOST_TRACE"] = str(host_path.resolve())
            if cudart:
                payload["capture_start_monotonic_ns"] = time.monotonic_ns()
                if cudart.cudaProfilerStart() != 0:
                    raise RuntimeError("cudaProfilerStart failed")
            start = time.perf_counter()
            try:
                result = batch.execute(strict=True, properties=("energy", "forces"))
                seconds = time.perf_counter() - start
            finally:
                if cudart:
                    if cudart.cudaProfilerStop() != 0:
                        raise RuntimeError("cudaProfilerStop failed")
                    payload["capture_end_monotonic_ns"] = time.monotonic_ns()
                os.environ.pop("VIBEQC_DF_TRACE", None)
                os.environ.pop("VIBEQC_DF_HOST_TRACE", None)
            energies = np.array([item.energy for item in result.items])
            forces = np.array([item.forces for item in result.items])
            ref = reference["gpu4pyscf"]
            energy_error = float(np.max(np.abs(energies - ref["energies_hartree"])))
            force_error = float(np.max(np.abs(forces - ref["forces_hartree_per_bohr"])))
            sample = {
                "candidate": candidate,
                "controls": {k: os.environ.get(k) for k in CANDIDATE_CONTROLS},
                "seconds": seconds,
                "iterations": [item.iterations for item in result.items],
                "convergence": convergence_payload(result),
                "warm_start_used": [item.warm_start_used for item in result.items],
                "warm_start_fallback": [
                    item.warm_start_fallback for item in result.items
                ],
                "energies_hartree": energies.tolist(),
                "forces_hartree_per_bohr": forces.tolist(),
                "maximum_energy_error_hartree": energy_error,
                "maximum_force_error_hartree_per_bohr": force_error,
                "metric": [
                    d.to_dict() for d in batch.last_density_fitting_metric_diagnostics()
                ],
            }
            if candidate:
                if expected_branch is None:
                    expected_branch = sample["iterations"]
                if (
                    sample["iterations"] != expected_branch
                    or not all(sample["warm_start_used"])
                    or any(sample["warm_start_fallback"])
                ):
                    raise RuntimeError(
                        "candidate sweep changed the fixed-density warm SCF branch"
                    )
            if args.component_trace:
                records = read_trace(trace_path)
                responses = [r for r in records if r["operation"] == "force_response"]
                if len(responses) != args.batch:
                    raise RuntimeError("missing per-item force response trace")
                probe = os.environ.get("VIBEQC_DF_RESPONSE_UPLOAD_PROBE", "")
                scatter_probe = os.environ.get("VIBEQC_DF_RESPONSE_SCATTER_PROBE", "")
                sample["response_plans"] = []
                for response in responses:
                    structural = validate_response_record(
                        response,
                        expected_policy=args.expected_response_policy,
                        max_h2d_bytes=args.max_h2d_bytes,
                        max_d2h_bytes=args.max_d2h_bytes,
                        max_transformed_tile_productions=args.max_transformed_tile_productions,
                    )
                    n, a, counters = (
                        response["nbf"],
                        response["naux"],
                        response["counters"],
                    )
                    if n != args.aos or a != args.aos:
                        raise RuntimeError("unexpected response dimensions")
                    three_center_work = validate_three_center_response_work(
                        counters, n, a
                    )
                    validate_metric_response_work(counters, a)
                    if candidate:
                        execution, _, algebra, staging = CANDIDATES[candidate]
                        if bool(counters.get("three_center_shell_panels")) != (
                            execution == "shell"
                        ):
                            raise RuntimeError(
                                "selected generated shell consumer did not execute"
                            )
                        if bool(counters.get("response_charge_blas_dots")) != (
                            algebra == "blas"
                        ):
                            raise RuntimeError(
                                "selected response algebra did not execute"
                            )
                        if bool(counters.get("raw_panel_pinned_host_bytes")) != (
                            staging == "pinned-panels"
                            and not response["source_backed"]
                            and not counters.get("response_borrowed_jk_bytes", 0)
                            and not counters.get("response_borrowed_whitened_bytes", 0)
                        ):
                            raise RuntimeError(
                                "selected raw panel staging did not execute"
                            )
                    # The legacy host-weight compatibility route does not
                    # publish auxiliary-panel counters. It does publish the
                    # exact host raw reuse work, so keep that fallback
                    # auditable without applying resident-only panel rules.
                    host_fallback_reuse = counters.get("raw_value_reuse_bytes", 0)
                    if (
                        structural["policy"]["residency"] == "fallback"
                        and host_fallback_reuse
                        and not counters.get("response_auxiliary_blocks")
                    ):
                        validate_host_fallback_probes(probe, scatter_probe)
                        if host_fallback_reuse % (n * n * 8):
                            raise RuntimeError(
                                "host fallback raw reuse is not integral N^2 work"
                            )
                        sample["response_plans"].append(
                            {
                                "source_backed": response["source_backed"],
                                "borrowed_jk_bytes": 0,
                                "auxiliary_weight_tile": None,
                                "auxiliary_blocks": None,
                                "raw_value_slices": None,
                                "raw_value_bytes": 0,
                                "raw_value_reuse_bytes": host_fallback_reuse,
                                "response_scratch_bytes": counters.get(
                                    "response_scratch_bytes", 0
                                ),
                                "three_center_pair_storage": three_center_work[
                                    "pair_storage"
                                ],
                                "three_center_pair_stride": three_center_work[
                                    "pair_stride"
                                ],
                                "three_center_derivative_weights": three_center_work[
                                    "observed_three_center_derivative_weights"
                                ],
                                "pinned_host_bytes": counters.get(
                                    "raw_panel_pinned_host_bytes", 0
                                ),
                                "selected_policy": structural["policy"],
                                "structural_invariants": structural,
                            }
                        )
                        continue
                    # Source providers regenerate raw columns on the device.
                    # Validate their logical tile work rather than pretending
                    # absent H2D copies are a missing or free response.
                    source_backed = response["source_backed"]
                    byte_key = (
                        "raw_value_bytes" if source_backed else "raw_value_upload_bytes"
                    )
                    forbidden_key = (
                        "raw_value_upload_bytes" if source_backed else "raw_value_bytes"
                    )
                    if counters.get(forbidden_key, 0):
                        raise RuntimeError(
                            "response used an unexpected raw value provider"
                        )
                    slices, remainder = divmod(counters.get(byte_key, 0), n * n * 8)
                    borrowed_jk_bytes = counters.get("response_borrowed_jk_bytes", 0)
                    borrowed_whitened_bytes = counters.get(
                        "response_borrowed_whitened_bytes", 0
                    )
                    if borrowed_jk_bytes and borrowed_whitened_bytes:
                        raise RuntimeError(
                            "response published multiple borrowed value owners"
                        )
                    borrowed_bytes = borrowed_jk_bytes or borrowed_whitened_bytes
                    if remainder or (slices == 0 and not borrowed_bytes):
                        raise RuntimeError("invalid raw response byte count")
                    panels = counters["response_auxiliary_blocks"]
                    if borrowed_bytes:
                        # A full-rank whitened owner is one N^2*Naux tensor;
                        # the older J/K scratch owner is three such tensors.
                        # Both are resident response owners, but the former
                        # need not publish a resident auxiliary tile because
                        # its panel consumer is shell-aligned.
                        expected_borrowed_bytes = (
                            n * n * a * 8
                            if borrowed_whitened_bytes
                            else 3 * n * n * a * 8
                        )
                        tile = counters.get("response_resident_auxiliary_tile", 0)
                        # A borrowed response may use an already resident raw
                        # owner (zero H2D slices) or stage the complete host
                        # tensor once. Both are resident paths; only the
                        # latter publishes one bulk upload.
                        resident_raw_owner = slices == 0 and not counters.get(
                            "raw_value_bulk_uploads", 0
                        )
                        resident_raw_upload = (
                            slices == a and counters.get("raw_value_bulk_uploads") == 1
                        )
                        if (
                            borrowed_bytes != expected_borrowed_bytes
                            or not (resident_raw_owner or resident_raw_upload)
                            or counters.get("raw_panel_host_gather_elements", 0)
                            or (
                                structural["policy"]["exchange"] == "dense"
                                and counters.get("response_ao_matrix_products", 0)
                                != 2 * a * (2 if case.method == "uhf" else 1)
                            )
                        ):
                            raise RuntimeError(
                                "resident response did not reuse its raw/projected tensors"
                            )
                    else:
                        reused, remainder = divmod(
                            counters["raw_value_reuse_bytes"], n * n * 8
                        )
                        # Charge retains the first tile; each later exchange
                        # panel reuses its own raw columns and rereads the rest.
                        tile = reused - a
                        if remainder or slices != (panels + 1) * a - tile:
                            raise RuntimeError(
                                "raw response reads disagree with panel reuse"
                            )
                    if borrowed_bytes:
                        # Some resident-whitened traces omit a tile because
                        # the shell-aligned panel width is not a public plan
                        # field. The sentinel still enforces the shape-only
                        # [1, Naux] panel bound; use the tighter tile bound
                        # whenever the trace publishes one.
                        if not 1 <= panels <= a:
                            raise RuntimeError(
                                "invalid resident response panel count: "
                                f"{panels} not in [1, {a}]"
                            )
                        panel_lower_bound = (a + tile - 1) // tile if tile else 1
                        if tile and panels < panel_lower_bound:
                            raise RuntimeError(
                                "resident response panel count is below its tile bound"
                            )
                    else:
                        panel_lower_bound = (a + tile - 1) // tile
                        if not 1 <= tile <= a or not panel_lower_bound <= panels <= a:
                            raise RuntimeError(
                                "invalid response consumer panel count: "
                                f"{panels} not in [{panel_lower_bound}, {a}]"
                            )
                    sample["response_plans"].append(
                        {
                            "source_backed": source_backed,
                            "borrowed_jk_bytes": borrowed_jk_bytes,
                            "borrowed_whitened_bytes": borrowed_whitened_bytes,
                            "auxiliary_weight_tile": tile,
                            "auxiliary_blocks": panels,
                            "raw_value_slices": slices,
                            "raw_value_bytes": slices * n * n * 8,
                            "response_scratch_bytes": counters[
                                "response_scratch_bytes"
                            ],
                            "three_center_pair_storage": three_center_work[
                                "pair_storage"
                            ],
                            "three_center_pair_stride": three_center_work[
                                "pair_stride"
                            ],
                            "three_center_derivative_weights": three_center_work[
                                "observed_three_center_derivative_weights"
                            ],
                            "pinned_host_bytes": counters.get(
                                "raw_panel_pinned_host_bytes", 0
                            ),
                            "selected_policy": structural["policy"],
                            "structural_invariants": structural,
                        }
                    )
                    if counters.get("raw_probe_prior_stream_drains", 0) != (
                        slices if probe else 0
                    ):
                        raise RuntimeError(
                            "upload drain ablation did not execute as requested"
                        )
                    if counters.get("raw_probe_gather_elements", 0) != (
                        slices * n * n if probe == "packed" else 0
                    ):
                        raise RuntimeError(
                            "upload gather ablation did not execute as requested"
                        )
                    if counters.get("derivative_probe_gradient_copies", 0) != (
                        128 if scatter_probe == "sharded" else 0
                    ):
                        raise RuntimeError(
                            "gradient destination ablation did not execute as requested"
                        )
                sample["components"] = aggregate(records)
                sample["trace_sha256"] = hashlib.sha256(
                    trace_path.read_bytes()
                ).hexdigest()
                sample["host_components"] = aggregate_host(read_host_trace(host_path))
                if sample["host_components"]["reference_eigensolves"]:
                    raise RuntimeError(
                        "warm replay executed a CPU-reference eigensolve"
                    )
                sample["host_trace_sha256"] = hashlib.sha256(
                    host_path.read_bytes()
                ).hexdigest()
            payload["samples"].append(sample)
            save()
            if not (energy_error < 1e-9 and force_error < 1e-8):
                raise RuntimeError(
                    "complete endpoint failed independent reference parity"
                )
            print(
                args.aos,
                args.batch,
                candidate,
                repeat,
                seconds,
                energy_error,
                force_error,
                flush=True,
            )


if __name__ == "__main__":
    main()
