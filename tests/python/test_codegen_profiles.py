"""Tests for safe generated-shell profile resolution and portable CUDA output."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.generate_portable_shell_registry import write_portable_bundle
from tools.vibeqc_codegen.profile import (
    PORTABLE_PROFILE,
    normalize_cuda_architecture,
    resolve_profile,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _manifest(path: Path) -> Path:
    payload = {
        "schema_version": 2,
        "default_architecture": "sm_120",
        "architectures": {
            "sm_80": {
                "cuda_toolkit": "12.0",
                "kernels": [{"shell_class": "dppp"}],
            },
            "sm_120": {
                "cuda_toolkit": "12.9.1",
                "kernels": [{"shell_class": "dppp"}],
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("raw", "normalized"),
    (
        ("80", "sm_80"),
        ("sm_89", "sm_89"),
        ("90-real", "sm_90"),
        ("90-virtual", "sm_90"),
        ("native", "native"),
    ),
)
def test_normalize_cuda_architecture(raw: str, normalized: str) -> None:
    assert normalize_cuda_architecture(raw) == normalized


def test_exact_profile_is_selected_without_changing_the_schedule(
    tmp_path: Path,
) -> None:
    resolution = resolve_profile(_manifest(tmp_path / "manifest.json"), "80")
    assert resolution.requested_architecture == "sm_80"
    assert resolution.selected_profile == "sm_80"
    assert resolution.tuned
    assert "exact measured" in resolution.reason


def test_unknown_architecture_uses_zero_aot_portable_profile(
    tmp_path: Path,
) -> None:
    resolution = resolve_profile(_manifest(tmp_path / "manifest.json"), "sm_90")
    assert resolution.requested_architecture == "sm_90"
    assert resolution.selected_profile == PORTABLE_PROFILE
    assert not resolution.tuned
    assert "generated AOT kernels disabled" in resolution.reason


def test_strict_mode_rejects_an_untuned_architecture(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no tuned generated-shell profile"):
        resolve_profile(
            _manifest(tmp_path / "manifest.json"),
            "sm_90",
            require_tuned=True,
        )


def test_explicit_profile_cannot_be_reused_across_compute_capabilities(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="refusing to use CUDA profile"):
        resolve_profile(
            _manifest(tmp_path / "manifest.json"),
            "sm_80",
            requested_profile="sm_120",
        )


def test_resolver_cli_has_a_stable_cmake_contract(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    script = REPOSITORY_ROOT / "tools" / "vibeqc_codegen" / "profile.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest),
            "--architecture",
            "90-real",
            "--profile",
            "auto",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "portable_cuda;sm_90;0"


def test_portable_bundle_is_deterministic_and_exposes_no_generated_kernels(
    tmp_path: Path,
) -> None:
    output = tmp_path / "generated"
    first = write_portable_bundle(output, 4, "sm_90")
    first_bytes = {path.name: path.read_bytes() for path in first}
    second = write_portable_bundle(output, 4, "sm_90")
    assert [path.name for path in first] == [path.name for path in second]
    assert first_bytes == {path.name: path.read_bytes() for path in second}

    header = (output / "vibeqc_generated_shell_registry.hpp").read_text(
        encoding="utf-8"
    )
    source = (output / "vibeqc_generated_shell_registry.cu").read_text(
        encoding="utf-8"
    )
    assert "std::array<ShellKernelMetadata, 0> kShellKernels" in header
    assert 'kProductionShellProfile[] = "portable_cuda"' in header
    assert 'kRequestedCudaArchitecture[] = "sm_90"' in header
    assert "kProductionShellProfileTuned = false" in header
    assert source.count("return 0;") == 2
    assert source.count("return cudaErrorNotSupported;") == 2


def test_cmake_exposes_safe_auto_portable_and_strict_modes() -> None:
    source = (REPOSITORY_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    for marker in (
        "VIBEQC_ENABLE_AOT_SHELLS",
        "VIBEQC_AOT_REQUIRE_TUNED_PROFILE",
        "VIBEQC_AOT_PROFILE",
        "tools/vibeqc_codegen/profile.py",
        "tools/generate_portable_shell_registry.py",
        "VIBEQC_AOT_PROFILE_IS_TUNED",
    ):
        assert marker in source
