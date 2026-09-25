"""Structural gate for retiring persistent-ERI handwritten Direct-HF force kernels."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_small_hf_force_requests_skip_persistent_eri_force_consumers() -> None:
    source = (ROOT / "src/scf/cuda_rhf.cpp").read_text(encoding="utf-8")
    assert (
        "!options.compute_forces && !options.export_physical_reference && "
        "nbf <= kPersistentEriAoLimit"
    ) in source
    force_dispatch = source[
        source.index("if (!options.compute_forces)")
        : source.index("if (reuse_converged_fock", source.index("if (!options.compute_forces)"))
    ]
    assert "launch_two_electron_force_kernel" not in force_dispatch
    assert "launch_two_electron_uhf_force_kernel" not in force_dispatch
    assert "launch_generated_shell_class_forces" in force_dispatch

    identity = (ROOT / "src/scf/cuda/rhf_bucket_internal.hpp").read_text(
        encoding="utf-8"
    )
    assert "first.compute_forces == second.compute_forces" in identity


def test_reference_force_file_keeps_only_matrix_direct_fallback() -> None:
    source = (ROOT / "src/scf/cuda/direct_reference_force.cu").read_text(encoding="utf-8")
    header = (ROOT / "src/scf/cuda/direct_reference_force.hpp").read_text(encoding="utf-8")
    assert "__global__ void two_electron_force_kernel(" not in source
    assert "__global__ void two_electron_uhf_force_kernel(" not in source
    assert "launch_two_electron_force_kernel(" not in header
    assert "launch_two_electron_uhf_force_kernel(" not in header
    assert "__global__ void two_electron_force_direct_kernel(" in source
    assert "__global__ void two_electron_uhf_force_direct_kernel(" in source


def test_reference_force_retirement_ledger_marks_fallback_only() -> None:
    ledger = json.loads(
        (ROOT / "docs/cuda_ownership/direct_hf_retirement.json").read_text(
            encoding="utf-8"
        )
    )
    entry = next(
        item
        for item in ledger["families"]
        if item["id"] == "retained-reference-force-consumer"
    )
    assert entry["status"] == "unsupported-fallback"
    assert "compute_forces disables persistent_eri" in entry["current_selector"]
