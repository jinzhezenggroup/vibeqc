"""End-to-end CPU promotion gate for one non-curated Libxc GGA."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from tools.qualify_libxc_compiled_cpu import qualify_compiled_cpu
from tools.qualify_libxc_molecular_scf import qualify_molecular_scf
from tools.qualify_libxc_production_domain import qualify_functional

NAME = "GGA_X_PBE_SOL"


def test_pbesol_reaches_exact_molecular_scf_evidence() -> None:
    build_dir_value = os.environ.get("VIBEQC_BUILD_DIR")
    if not build_dir_value:
        pytest.skip("native VibeQC build directory is unavailable")
    pyscf = pytest.importorskip("pyscf")
    from pyscf.dft import libxc

    compiled = qualify_compiled_cpu(
        NAME,
        evidence="test://libxc-pbesol/compiled-cpu",
    )
    assert compiled["stage_evidence"]["status"] == "pass"

    production = qualify_functional(
        NAME,
        evidence="test://libxc-pbesol/production-domain",
        pyscf_version=pyscf.__version__,
        libxc=libxc,
    )
    assert production["stage_evidence"]["status"] == "pass"

    molecular = qualify_molecular_scf(
        NAME,
        compiled_cpu_evidence=compiled["stage_evidence"],
        production_domain_evidence=production["stage_evidence"],
        build_dir=Path(build_dir_value),
        evidence="test://libxc-pbesol/molecular-scf",
    )

    assert molecular["stage_evidence"]["status"] == "pass"
    assert molecular["stage_evidence"]["qualification"]["coverage"] == [
        {"backend": "cpu", "spin": "polarized", "products": ["energy"]},
        {"backend": "cpu", "spin": "unpolarized", "products": ["energy"]},
    ]
    details = molecular["details"]
    assert len(details) == 6
    assert all(row["status"] == "pass" for row in details)
    assert {
        (row["spin"], row["phase"]) for row in details
    } == {
        (spin, phase)
        for spin in ("polarized", "unpolarized")
        for phase in ("cold", "warm-replay", "changed-geometry")
    }
