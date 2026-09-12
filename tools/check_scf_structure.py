"""Enforce the dependency direction of shared native SCF modules."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# These are implementation boundaries, independent of #231's scientific CUDA
# ownership inventory. A reference oracle must not acquire a method driver or
# generated backend dependency merely because a future consumer needs it.
ALLOWED = {
    "reference": ("scf/reference/",),
    "initial_guess": ("scf/reference/", "scf/initial_guess/", "core/", "integrals/"),
    "gradient": ("scf/gradient/", "scf/reference/", "integrals/", "core/"),
    "solver": (
        "scf/solver/",
        "scf/gradient/",
        "scf/reference/",
        "scf/initial_guess/",
        "scf/types.hpp",
        "scf/proposals.hpp",
        "scf/fock_prepared.hpp",
        "scf/fock_build.hpp",
        "core/",
        "integrals/",
        "runtime/",
    ),
}
# CUDA planning and linear algebra have separate rebuild ownership. Enumerate
# these extracted owners rather than exempting all historical cuda/ fragments.
CUDA_MODULES = {
    "cuda_planning": (
        "arena",
        "checked_layout",
        "direct_constants",
        "direct_metadata",
        "packed_basis",
        "topology",
        "queue_plan",
        "queue_profile",
    ),
    "cuda_eigensolver": (
        "eigensolver",
        "eigensolver_kernels",
        "eigensolver_types",
        "matrix_index",
        "device_timer",
        "launch_geometry",
    ),
}
CUDA_MODULES["cuda_df_source"] = (
    "df_source",
    "df_source_setup",
    "df_source_internal",
    "df_source_kernels",
    "metadata_upload",
    "df_integral_export",
    "df_integral_export_batch",
)
CUDA_ALLOWED = {
    "cuda_planning": (
        "scf/cuda/arena.",
        "scf/cuda/checked_layout.",
        "scf/cuda/direct_constants.",
        "scf/cuda/integral_limits.",
        "scf/cuda/scf_constants.",
        "scf/cuda/direct_metadata.",
        "scf/cuda/packed_basis.",
        "scf/cuda/topology.",
        "scf/cuda/queue_plan.",
        "scf/cuda/queue_profile.",
        "scf/cuda/eigensolver_types.",
        "scf/cuda/launch_geometry.",
        "scf/cuda/rhf_policy.hpp",
        "scf/cuda_batch.hpp",
        "scf/direct_task_layout.hpp",
        "scf/generated_shell_task.hpp",
        "scf/aot_shell_registry.hpp",
        "molecule/",
        "core/",
    ),
    "cuda_eigensolver": (
        "scf/cuda/eigensolver.",
        "scf/cuda/eigensolver_kernels.",
        "scf/cuda/eigensolver_types.",
        "scf/cuda/matrix_index.",
        "scf/cuda/device_timer.",
        "scf/cuda/launch_geometry.",
        "scf/cuda_batch.hpp",
    ),
}
CUDA_ALLOWED["cuda_df_source"] = (
    "scf/cuda/df_source.",
    "scf/cuda/df_source_setup.",
    "scf/cuda/df_source_internal.",
    "scf/cuda/metadata_upload.",
    "scf/cuda/rhf_policy.hpp",
    "scf/cuda/df_source_kernels.",
    "scf/cuda/df_integral_export.",
    "scf/cuda/df_integral_export_batch.",
    "scf/cuda/packed_basis.",
    "scf/cuda/topology.",
    "scf/cuda/checked_layout.",
    "scf/cuda_density_fitting.hpp",
    "scf/cuda_density_fitting_integrals.hpp",
    "molecule/",
    "runtime/",
    "core/",
)
# DF plan/provider ownership is separate from direct HF queues and from the
# generic SCF driver. Kernel owners cannot acquire host plan or solver state.
CUDA_MODULES["cuda_df_runtime"] = (
    "df_plan",
    "df_plan_setup",
    "df_plan_internal",
    "df_setup_internal",
    "df_plan_lifetime",
    "df_runtime",
    "df_jk",
    "df_jk_internal",
    "df_coulomb",
    "df_exchange",
    "df_force_response",
    "df_scf_state",
    "df_scf_library",
    "df_rhf_scf",
    "df_uhf_scf",
)
CUDA_ALLOWED["cuda_df_runtime"] = tuple(
    "scf/cuda/" + stem + "." for stem in CUDA_MODULES["cuda_df_runtime"]
) + (
    "scf/cuda/df_metric_kernels.",
    "scf/cuda/df_jk_kernels.",
    "scf/cuda/df_scf_kernels.",
    "scf/cuda_density_fitting.hpp",
    "scf/cuda_df_gradient.hpp",
    "scf/density_fitting.hpp",
    "molecule/basis.hpp",
    "runtime/",
)
CUDA_MODULES["cuda_df_kernels"] = (
    "df_metric_kernels",
    "df_jk_kernels",
    "df_scf_kernels",
)
CUDA_ALLOWED["cuda_df_kernels"] = tuple(
    "scf/cuda/" + stem + "." for stem in CUDA_MODULES["cuda_df_kernels"]
)
CUDA_MODULES["cuda_scf_kernels"] = (
    "scf_constants",
    "scf_state_kernels",
    "scf_matrix_kernels",
    "scf_density_kernels",
    "scf_diis_kernels",
    "scf_convergence_kernels",
    "basis_transform_kernels",
)
CUDA_ALLOWED["cuda_scf_kernels"] = tuple(
    "scf/cuda/" + stem + "." for stem in CUDA_MODULES["cuda_scf_kernels"]
) + ("scf/cuda/matrix_index.",)
CUDA_MODULES["cuda_resources"] = ("resources",)
CUDA_ALLOWED["cuda_resources"] = (
    "scf/cuda/resources.",
    "scf/cuda/eigensolver.",
    "scf/cuda/matrix_library.",
    "runtime/resource_cuda.cuh",
    "runtime/allocation_measurement.hpp",
)
CUDA_MODULES["cuda_matrix_library"] = ("matrix_library", "runtime_support")
CUDA_ALLOWED["cuda_matrix_library"] = (
    "scf/cuda/matrix_library.",
    "scf/cuda/runtime_support.",
    "scf/cuda/scf_matrix_kernels.",
    "scf/cuda/launch_geometry.",
)
# Device queue owners consume borrowed metadata and screening contracts. They
# cannot acquire host bucket/graph ownership or integral recurrence code.
CUDA_MODULES["cuda_direct_queues"] = (
    "direct_queue_index",
    "direct_screening",
    "direct_task_encoding",
    "direct_page_screening",
    "direct_queue_profile",
    "direct_tile_validation",
    "direct_density_bounds",
    "direct_tile_compaction",
    "direct_generated_tasks",
    "direct_resident_tasks",
    "direct_bounded_pages",
    "direct_bounded_tasks",
    "direct_queue_scan",
    "direct_queue_diagnostics",
)
CUDA_ALLOWED["cuda_direct_queues"] = tuple(
    "scf/cuda/" + stem + "." for stem in CUDA_MODULES["cuda_direct_queues"]
) + (
    "scf/cuda/direct_metadata.",
    "scf/cuda/direct_constants.",
    "scf/cuda/matrix_index.",
    "scf/cuda/packed_basis.",
    "scf/cuda/device_timer.",
)
# Provider host APIs own staging and lifetime while borrowing kernel launches.
# A retained recurrence fragment must not enter a host implementation.
CUDA_MODULES["cuda_direct_provider_host"] = ("direct_jk", "direct_jk_plan")
CUDA_ALLOWED["cuda_direct_provider_host"] = (
    "scf/cuda/direct_jk.",
    "scf/cuda/direct_jk_plan.",
    "scf/cuda/direct_jk_kernels.hpp",
    "scf/cuda/packed_basis.",
    "scf/cuda/checked_layout.",
    "scf/cuda/metadata_upload.",
    "scf/cuda/topology.",
    "scf/cuda_direct_jk.hpp",
    "runtime/",
)
CUDA_MODULES["cuda_one_electron_export"] = (
    "one_electron_export",
    "one_electron_export_batch",
    "one_electron_view",
)
CUDA_ALLOWED["cuda_one_electron_export"] = tuple(
    "scf/cuda/" + stem + "." for stem in CUDA_MODULES["cuda_one_electron_export"]
) + (
    "scf/cuda/one_electron_export_kernels.hpp",
    "scf/cuda/one_electron_values.cuh",
    "scf/cuda/packed_basis.",
    "scf/cuda/rhf_policy.hpp",
    "scf/cuda/runtime_support.",
    "scf/cuda/topology.",
    "scf/cuda_density_fitting_integrals.hpp",
    "molecule/basis.hpp",
    "runtime/",
)
CUDA_MODULES["cuda_provider_kernel_interfaces"] = (
    "direct_jk_kernels.hpp",
    "one_electron_export_kernels.hpp",
)
CUDA_ALLOWED["cuda_provider_kernel_interfaces"] = ("scf/cuda/packed_basis.",)
# Retained numerical primitives have no queue policy or host plan dependency.
# One-electron consumers share only these bounded scientific building blocks.
CUDA_MODULES["cuda_integral_numerics"] = (
    "integral_limits",
    "scalar_math",
    "gaussian_geometry",
    "cartesian_angular",
    "boys_table",
    "hermite_recurrence",
    "coulomb_auxiliary",
)
CUDA_ALLOWED["cuda_integral_numerics"] = tuple(
    "scf/cuda/" + stem + "." for stem in CUDA_MODULES["cuda_integral_numerics"]
) + ("scf/cuda/packed_basis.", "molecule/basis.hpp")
CUDA_MODULES["cuda_one_electron_native"] = (
    "one_electron_reference",
    "one_electron_force_reference",
    "one_electron_force_workspace",
    "one_electron_native_overlap",
    "one_electron_native_attraction",
    "one_electron_native_attraction_gradient",
    "one_electron_native_contraction",
    "one_electron_native_force",
)
CUDA_ALLOWED["cuda_one_electron_native"] = (
    tuple("scf/cuda/" + stem + "." for stem in CUDA_MODULES["cuda_one_electron_native"])
    + tuple("scf/cuda/" + stem + "." for stem in CUDA_MODULES["cuda_integral_numerics"])
    + (
        "scf/cuda/one_electron_export_kernels.hpp",
        "scf/cuda/packed_basis.",
        "scf/cuda/matrix_index.",
    )
)
CUDA_MODULES["cuda_nuclear_kernels"] = ("nuclear_kernels",)
CUDA_ALLOWED["cuda_nuclear_kernels"] = (
    "scf/cuda/nuclear_kernels.",
    "scf/cuda/one_electron_export_kernels.hpp",
    "scf/cuda/gaussian_geometry.",
    "scf/cuda/packed_basis.",
)
CUDA_MODULES["cuda_direct_pair_cache"] = ("direct_pair_cache",)
CUDA_ALLOWED["cuda_direct_pair_cache"] = (
    "scf/cuda/direct_pair_cache.",
    "scf/cuda/gaussian_geometry.",
    "scf/cuda/packed_basis.",
)
# Retained direct numerics and fused consumers have separate dependency/rebuild
# boundaries. Exact .hpp entries keep host launch contracts independent of the
# device implementations that share their basename.
CUDA_MODULES["cuda_direct_numerics"] = (
    "direct_native_cartesian",
    "direct_native_contraction",
    "direct_native_dsss_gradient",
    "direct_native_eri_order2",
    "direct_native_eri_order3",
    "direct_native_eri_order4",
    "direct_native_gradient_types",
    "direct_native_high_order_coulomb",
    "direct_native_order01_gradient",
    "direct_native_order2_gradient",
    "direct_native_order2_shell",
    "direct_native_order3_gradient",
    "direct_native_order456_gradient",
    "direct_native_pair_high_order_gradient",
    "direct_native_pair_order2",
    "direct_native_pair_order2_gradient",
    "direct_native_pair_order3",
    "direct_native_pair_order3_gradient",
    "direct_native_ppss_gradient",
    "direct_native_psps_gradient",
    "direct_native_psss",
    "direct_native_shell_class",
    "direct_native_shell_pair_hermite",
    "direct_native_source_contraction",
)
CUDA_ALLOWED["cuda_direct_numerics"] = (
    tuple("scf/cuda/" + stem + ".cuh" for stem in CUDA_MODULES["cuda_direct_numerics"])
    + tuple("scf/cuda/" + stem + "." for stem in CUDA_MODULES["cuda_integral_numerics"])
    + ("scf/cuda/packed_basis.hpp", "scf/cuda/direct_queue_index.cuh")
)
CUDA_MODULES["cuda_direct_contractions"] = (
    "eri_tensor_index",
    "direct_eri_symmetry",
    "direct_fock_accumulation",
    "direct_fock_quartet",
    "direct_fock_psss",
    "direct_fock_order2",
    "direct_force_density",
    "direct_force_low_order",
    "direct_force_order2",
    "direct_force_quartet",
    "direct_bounded_contraction",
)
CUDA_ALLOWED["cuda_direct_contractions"] = (
    CUDA_ALLOWED["cuda_direct_numerics"]
    + tuple(
        "scf/cuda/" + stem + ".cuh" for stem in CUDA_MODULES["cuda_direct_contractions"]
    )
    + (
        "scf/cuda/direct_constants.hpp",
        "scf/cuda/direct_metadata.hpp",
        "scf/cuda/packed_basis.hpp",
        "scf/cuda/direct_queue_index.cuh",
        "scf/cuda/direct_screening.cuh",
        "scf/cuda/direct_task_encoding.cuh",
        "scf/cuda/direct_page_screening.cuh",
        "scf/cuda/direct_queue_profile.cuh",
        "scf/cuda/matrix_index.cuh",
        "scf/cuda/device_timer.cuh",
    )
)
CUDA_MODULES["cuda_direct_consumers"] = (
    "direct_cached_tensor_kernels.cu",
    "direct_schwarz_kernels.cu",
    "direct_packed_fock_kernels.cu",
    "direct_angular_fock.cu",
    "direct_reference_force.cu",
    "direct_bounded_dddd.cu",
    "direct_bounded_exact_force.cu",
    "direct_bounded_fallback.cu",
    "direct_angular_force.cu",
    "direct_jk_kernels.cu",
    "weighted_eri_kernels.cu",
)
CUDA_ALLOWED["cuda_direct_consumers"] = (
    CUDA_ALLOWED["cuda_direct_contractions"]
    + tuple(
        "scf/cuda/" + Path(name).stem + ".hpp"
        for name in CUDA_MODULES["cuda_direct_consumers"]
    )
    + ("scf/cuda_weighted_eri.hpp",)
)
CUDA_MODULES["cuda_direct_kernel_interfaces"] = (
    "direct_cached_tensor_kernels.hpp",
    "direct_schwarz_kernels.hpp",
    "direct_packed_fock_kernels.hpp",
    "direct_angular_fock.hpp",
    "direct_reference_force.hpp",
    "direct_bounded_dddd.hpp",
    "direct_bounded_exact_force.hpp",
    "direct_bounded_fallback.hpp",
    "direct_angular_force.hpp",
    "weighted_eri_kernels.hpp",
)
CUDA_ALLOWED["cuda_direct_kernel_interfaces"] = (
    "scf/cuda/direct_metadata.hpp",
    "scf/cuda/packed_basis.hpp",
    "scf/cuda_weighted_eri.hpp",
)
# The legacy host driver coordinates policy and lifetime through explicit
# interfaces. Keep recurrence and kernel implementation includes out of C++.
CUDA_MODULES["cuda_hf_driver"] = ("scf/cuda_rhf.cpp",)
CUDA_ALLOWED["cuda_hf_driver"] = (
    "molecule/basis.hpp",
    "runtime/resource_cuda.cuh",
    "runtime/resource_usage.hpp",
    "scf/aot_shell_registry.hpp",
    "scf/cuda/arena.hpp",
    "scf/cuda/basis_transform_kernels.hpp",
    "scf/cuda/checked_layout.hpp",
    "scf/cuda/direct_bounded_pages.hpp",
    "scf/cuda/direct_bounded_tasks.hpp",
    "scf/cuda/direct_constants.hpp",
    "scf/cuda/direct_angular_fock.hpp",
    "scf/cuda/direct_angular_force.hpp",
    "scf/cuda/direct_bounded_dddd.hpp",
    "scf/cuda/direct_bounded_exact_force.hpp",
    "scf/cuda/direct_bounded_fallback.hpp",
    "scf/cuda/direct_cached_tensor_kernels.hpp",
    "scf/cuda/direct_packed_fock_kernels.hpp",
    "scf/cuda/direct_reference_force.hpp",
    "scf/cuda/direct_schwarz_kernels.hpp",
    "scf/cuda/direct_density_bounds.hpp",
    "scf/cuda/direct_generated_tasks.hpp",
    "scf/cuda/direct_jk_kernels.hpp",
    "scf/cuda/weighted_eri_kernels.hpp",
    "scf/cuda/direct_metadata.hpp",
    "scf/cuda/direct_pair_cache.hpp",
    "scf/cuda/direct_queue_diagnostics.hpp",
    "scf/cuda/direct_queue_scan.hpp",
    "scf/cuda/direct_resident_tasks.hpp",
    "scf/cuda/direct_tile_compaction.hpp",
    "scf/cuda/direct_tile_validation.hpp",
    "scf/cuda/eigensolver.hpp",
    "scf/cuda/matrix_library.hpp",
    "scf/cuda/metadata_upload.hpp",
    "scf/cuda/nuclear_kernels.hpp",
    "scf/cuda/one_electron_derivatives.cuh",
    "scf/cuda/one_electron_export_kernels.hpp",
    "scf/cuda/one_electron_force_reference.hpp",
    "scf/cuda/one_electron_force_workspace.hpp",
    "scf/cuda/one_electron_values.cuh",
    "scf/cuda/one_electron_view.hpp",
    "scf/cuda/packed_basis.hpp",
    "scf/cuda/queue_plan.hpp",
    "scf/cuda/resources.hpp",
    "scf/cuda/rhf_policy.hpp",
    "scf/cuda/runtime_support.hpp",
    "scf/cuda/scf_convergence_kernels.hpp",
    "scf/cuda/scf_density_kernels.hpp",
    "scf/cuda/scf_diis_kernels.hpp",
    "scf/cuda/scf_matrix_kernels.hpp",
    "scf/cuda/scf_state_kernels.hpp",
    "scf/cuda/topology.hpp",
    "scf/cuda_density_fitting.hpp",
    "scf/cuda_density_fitting_integrals.hpp",
    "scf/cuda_direct_jk.hpp",
    "scf/cuda_eigensolver_policy.hpp",
    "scf/cuda_weighted_eri.hpp",
    "scf/direct_task_layout.hpp",
    "scf/generated_shell_task.hpp",
    "scf/rhf.hpp",
)
# Upstream physical-reference export is a host bridge for post-HF clients.
CUDA_ALLOWED["cuda_hf_driver"] += (
    "posthf/capacity.hpp",
    "runtime/allocation_measurement.hpp",
    "scf/cuda/reference_export.cuh",
    "scf/mean_field.hpp",
    "tensor/metrics.hpp",
)
CUDA_MODULES["cuda_reference_export"] = ("reference_export",)
CUDA_ALLOWED["cuda_reference_export"] = (
    "posthf/capacity.hpp",
    "scf/mean_field.hpp",
    "tensor/cuda_error.hpp",
)
SUFFIXES = {".cpp", ".hpp", ".cu", ".cuh"}
ROOT = Path(__file__).resolve().parents[1]


def audit_scf_structure(root: Path = ROOT) -> dict:
    """Check quoted/angle source includes, including relative-path spellings.

    Standard-library headers are outside this source graph. Comments do not
    introduce edges. Report source sizes without treating a small line count as
    evidence that the full HF decomposition has been completed.
    """
    source = (root / "src").resolve()
    errors, edges, modules = [], [], []
    groups = [
        (owner, allowed, sorted((source / "scf" / owner).rglob("*")))
        for owner, allowed in ALLOWED.items()
    ]
    for owner, stems in CUDA_MODULES.items():
        paths = []
        for stem in stems:
            # A full source-relative path selects a legacy root owner; an
            # explicit suffix separates implementation and interface rules.
            base = (
                source / stem if stem.startswith("scf/") else source / "scf/cuda" / stem
            )
            paths.extend(
                [base]
                if base.suffix in SUFFIXES
                else [Path(str(base) + suffix) for suffix in sorted(SUFFIXES)]
            )
        groups.append((owner, CUDA_ALLOWED[owner], paths))
    for owner, allowed, paths in groups:
        for path in paths:
            if path.suffix not in SUFFIXES or not path.is_file():
                continue
            content = path.read_text()
            relative = path.relative_to(source).as_posix()
            modules.append(
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "lines": len(content.splitlines()),
                }
            )
            # Preserve line numbers when dropping multiline comments.
            text = re.sub(
                r"/\*.*?\*/|//[^\n]*",
                lambda m: "\n" * m[0].count("\n"),
                content,
                flags=re.DOTALL,
            )
            for match in re.finditer(
                r'^\s*#\s*include\s*["<]([^">]+)[">]', text, re.MULTILINE
            ):
                header = match[1]
                candidate = source / header
                if not candidate.is_file():
                    candidate = path.parent / header
                if not candidate.is_file():
                    continue
                try:
                    target = candidate.resolve().relative_to(source).as_posix()
                except ValueError:
                    continue
                edges.append({"from": relative, "to": target})
                if not target.startswith(allowed):
                    line = text.count("\n", 0, match.start()) + 1
                    errors.append(
                        f"{relative}:{line}: forbidden {owner} dependency on {target}"
                    )
    return {"modules": modules, "edges": edges, "errors": errors}


def main():
    """Return failure for a dependency violation; expose an optional JSON inventory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = audit_scf_structure()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for error in report["errors"]:
            print(error)
        print(
            f"Checked {len(report['modules'])} shared SCF modules; "
            f"{len(report['errors'])} dependency errors"
        )
    return bool(report["errors"])


if __name__ == "__main__":
    raise SystemExit(main())
