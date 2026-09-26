"""Routing contracts for bounded Direct-J/K generated streaming."""

from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_primary_streaming_route_partitions_paged_generated_classes() -> None:
    """Selected generated classes must stream instead of being double-counted."""

    source = (REPOSITORY_ROOT / "src/scf/cuda_rhf.cpp").read_text()
    assert "host_primary_streaming_fock_shell_class_mask" in source
    assert "host_primary_streaming_fock_flags" not in source
    assert "generated::preferred_streaming_fock_shell_class_mask()" in source
    assert "primary_streaming_fock_override.value_or(" in source
    registry_header = (
        REPOSITORY_ROOT / "src/scf/aot_shell_registry.hpp"
    ).read_text()
    assert (
        "std::uint64_t preferred_streaming_fock_shell_class_mask() noexcept;"
        in registry_header
    )

    page_begin = source.index("const auto launch_bounded_paged_generated_fock")
    page_end = source.index("const auto launch_bounded_generic_fock", page_begin)
    page_source = source[page_begin:page_end]
    assert "host_primary_streaming_fock_shell_class_mask" in page_source
    assert "host_generated_streaming_fock_shell_class_mask" in source
    mask_begin = source.index(
        "const std::uint64_t host_primary_streaming_fock_shell_class_mask"
    )
    mask_end = source.index(";", mask_begin)
    assert "!mixed_precision_fock" in source[mask_begin:mask_end]

    route_begin = source.index("if (!bounded_direct_count_diagnostic)")
    route_end = source.index("cudaError_t paged_error", route_begin)
    reset = source[route_begin:route_end]
    assert "launch_reset_bounded_generated_streaming_flags_kernel(" in reset
    assert "cudaMemcpyAsync" not in reset

    # Both the cache owner and its defensive driver guard must agree that
    # changing a selector requires recapturing the paged/streaming partition.
    owner = (REPOSITORY_ROOT / "src/scf/cuda/rhf_bucket.cpp").read_text()
    assert "(*plan)->primary_streaming_fock_mask !=" in owner
    assert (
        "plan.primary_streaming_fock_mask != requested_primary_streaming_fock_mask"
        in source
    )

    policy = (REPOSITORY_ROOT / "src/scf/cuda/rhf_policy.cpp").read_text()
    assert "VIBEQC_BOUNDED_DIRECT_PRIMARY_STREAMING_MASK" in policy
    assert "std::strtoull(selection, &end, 0)" in policy
