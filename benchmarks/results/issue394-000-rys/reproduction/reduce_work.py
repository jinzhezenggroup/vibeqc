"""Retain independently reconstructed work and compare identical shell domains.

Detailed source counters and CUDA-event intervals are intrusive diagnostics.
Clean kernel timings are reduced separately from the Nsight captures.
"""

import json
import subprocess
import sys

from collect import DESTINATION, SOURCE, digest, write

from benchmarks.df_component_ledger import read_trace


def main():
    """Reject changed primitive/component/panel work before reporting a saving."""
    compared = {}
    fields = (
        "shell_tasks",
        "active_shell_tasks",
        "primitive_products",
        "geometry_preparations",
        "active_component_products",
        "public_weight_loads",
        "public_nonzero_weights",
        "expansion_term_products",
        "folding_shared_atomics",
        "folding_direct_stores",
        "gradient_atomics_a",
        "gradient_atomics_b",
        "gradient_atomics_c",
        "gradient_atomics_shared_atom",
        "gradient_atomics_distinct_atom",
        "orbital_local_accumulations",
    )
    for aos in (384, 768):
        arms, components = {}, {}
        for policy in ("polynomial", "rys"):
            trace = SOURCE / "v1" / f"{aos}-work.0-{policy}.jsonl"
            output = SOURCE / "v1" / f"{aos}-{policy}-ledger.json"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "benchmarks.df_shell_work_ledger",
                    "--trace",
                    str(trace),
                    "--measurement",
                    str(SOURCE / "v1" / f"{aos}-work.json"),
                    "--generated-header",
                    "build/cuda-release-sm120/generated/generated_df_shell_derivatives.cuh",
                    "--output",
                    str(output),
                ],
                check=True,
            )
            arms[policy] = json.loads(output.read_text())
            write(DESTINATION / "work" / output.name, arms[policy])
            basic = SOURCE / "v1" / f"{aos}-endpoint.diagnostic-0-{policy}.jsonl"
            records = [
                r for r in read_trace(basic) if r["operation"] == "force_response"
            ]
            assert len(records) == 1
            record = records[0]
            regions = [
                r
                for r in record["regions"]
                if r["name"] == "three_center_derivative_contraction"
            ]
            components[policy] = {
                "trace_sha256": digest(basic),
                "scope": "Basic component/resource diagnostics; no detailed work atomics. Event intervals include host gaps and are not clean kernel times.",
                "three_center_panel_count": len(regions),
                "three_center_derivative_contraction_gpu_inclusive_ms": sum(
                    r["gpu_ms"] for r in regions
                ),
                "000_gpu_inclusive_ms": sum(
                    r["gpu_ms"]
                    for r in record["regions"]
                    if r["name"] == "shell_000_packet"
                ),
                "000_resources": {
                    key: value
                    for key, value in record["counters"].items()
                    if key.startswith("shell_000_")
                    and key.endswith(
                        (
                            "registers",
                            "shared_bytes",
                            "thread_limit",
                            "block_limit",
                            "selected",
                        )
                    )
                },
            }
        before, after = arms["polynomial"], arms["rys"]
        assert before["host_reconstruction"] == after["host_reconstruction"]
        assert len(before["classes"]) == len(after["classes"])
        for b, a in zip(before["classes"], after["classes"], strict=True):
            assert b["angular"] == a["angular"]
            assert all(b["work"][field] == a["work"][field] for field in fields)
            assert b["lowering"] == "polynomial"
            assert a["lowering"] == (
                "rys" if a["angular"] == [0, 0, 0] else "polynomial"
            )
        assert (
            components["polynomial"]["three_center_panel_count"]
            == components["rys"]["three_center_panel_count"]
        )
        compared[str(aos)] = {
            "equal_work_fields": list(fields),
            "host_reconstruction_sha256": before["host_reconstruction_sha256"],
            "totals": {policy: arm["totals"] for policy, arm in arms.items()},
            "000": {
                policy: next(r for r in arm["classes"] if r["angular"] == [0, 0, 0])
                for policy, arm in arms.items()
            },
            "basic_components": components,
        }
    write(DESTINATION / "work-comparison.json", compared)


if __name__ == "__main__":
    main()
