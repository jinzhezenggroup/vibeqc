"""Ownership tests for compiler-generated Direct-Fock scatter math."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from vibeqc_compiler.integral.lowering.fock_accumulation import (
    emit_direct_fock_accumulation_header,
    emit_direct_force_density_coefficient,
    emit_generated_shell_fock_accumulation,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_direct_fock_scatter_has_one_compiler_equation_owner() -> None:
    """Keep RHF/UHF coefficients out of retained handwritten CUDA."""

    shared = (
        REPOSITORY_ROOT
        / "python/vibeqc_compiler/integral/lowering/fock_accumulation.py"
    ).read_text(encoding="utf-8")
    native = (REPOSITORY_ROOT / "src/scf/cuda/direct_fock_accumulation.cuh").read_text(
        encoding="utf-8"
    )
    shell_lowering = (
        REPOSITORY_ROOT / "python/vibeqc_compiler/integral/lowering/fock.py"
    ).read_text(encoding="utf-8")

    assert "-0.5 * density_bd * integral" in shared
    assert "total_cd * integral" in shared
    assert "-0.5 * density_bd * integral" not in native
    assert "total_cd * integral" not in native
    assert "-0.5 * density_bd * integral" not in shell_lowering
    assert "total_cd * integral" not in shell_lowering
    assert '#include "generated_direct_fock_accumulation.cuh"' in native


def test_direct_force_density_has_one_compiler_equation_owner() -> None:
    """Keep the exact RHF/UHF force density contraction out of native CUDA."""

    generated = emit_direct_force_density_coefficient()
    native = (REPOSITORY_ROOT / "src/scf/cuda/direct_force_density.cuh").read_text(
        encoding="utf-8"
    )
    for equation in (
        "0.5 * total_ab * total_cd",
        "0.25 * density[physical_offset + ac]",
        "unique_eri_symmetry_permutation",
    ):
        assert equation in generated
        assert equation not in native
    assert '#include "generated_direct_fock_accumulation.cuh"' in native
    assert "direct_force_density_coefficient" in emit_direct_fock_accumulation_header()


def test_generated_shell_and_native_scatter_share_spin_semantics() -> None:
    """Render both adapters from the same compiler-owned contraction."""

    native = emit_direct_fock_accumulation_header()
    generated = emit_generated_shell_fock_accumulation()
    for equation in (
        "const double total_cd = alpha_cd + beta_cd;",
        "-alpha_bd * integral",
        "-beta_bd * integral",
        "-0.5 * density_bd * integral",
    ):
        assert equation in native
        assert equation in generated


def test_direct_fock_scatter_cli_is_deterministic(tmp_path: Path) -> None:
    """Make the build-time compatibility header reproducible from checkout."""

    output = tmp_path / "generated_direct_fock_accumulation.cuh"
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "tools/generate_shell_kernels.py"),
        "--direct-fock-accumulation-output",
        str(output),
    ]
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
    first = output.read_text(encoding="utf-8")
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
    assert output.read_text(encoding="utf-8") == first
    assert first == emit_direct_fock_accumulation_header()
