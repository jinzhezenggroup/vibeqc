"""Routing contracts for bounded Direct-J/K generated streaming."""

from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_primary_streaming_route_partitions_paged_generated_classes() -> None:
    """Selected generated classes must stream instead of being double-counted."""

    source = (REPOSITORY_ROOT / "src/scf/cuda_rhf.cpp").read_text()
    assert "host_primary_streaming_fock_shell_class_mask" in source
    assert "host_primary_streaming_fock_flags" in source
    assert "cudaMemcpyAsync(" in source

    page_begin = source.index("const auto launch_bounded_paged_generated_fock")
    page_end = source.index("const auto launch_bounded_generic_fock", page_begin)
    page_source = source[page_begin:page_end]
    assert "host_primary_streaming_fock_shell_class_mask" in page_source
    assert "host_generated_streaming_fock_shell_class_mask" in source
    assert (
        "!mixed_precision_fock && requested_primary_streaming_fock_mask.has_value()"
        in source
    )

    policy = (REPOSITORY_ROOT / "src/scf/cuda/rhf_policy.cpp").read_text()
    assert "VIBEQC_BOUNDED_DIRECT_PRIMARY_STREAMING_MASK" in policy
    assert "std::strtoull(selection, &end, 0)" in policy
