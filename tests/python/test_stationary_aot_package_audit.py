from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import audit_stationary_aot_package as audit


def _fake_loader(directory: Path, *, functional: int, spin: str, **_: object) -> object:
    name = {
        (0, "unpolarized"): "lda_rks",
        (0, "polarized"): "lda_uks",
        (1, "unpolarized"): "pbe_rks",
        (1, "polarized"): "pbe_uks",
        (2, "unpolarized"): "r2scan_rks",
        (2, "polarized"): "r2scan_uks",
    }[(functional, spin)]
    library = directory / f"libvibeqc_stationary_{name}.so"
    metadata = {
        "binary_sha256": f"binary-{name}",
        "source_identity": f"source-{name}",
        "contract_identity": f"contract-{name}",
        "plan_identity": f"plan-{name}",
        "key": f"key-{name}",
        "identity": {"target": {"code_kinds": ["cubin"]}},
        "driver_ptx_jit_possible": False,
        "driver_ptx_jit_required": False,
    }
    return SimpleNamespace(library=library, metadata=metadata)


def _layout(root: Path, *, binary_size: int = 10) -> None:
    root.mkdir()
    for _, _, name in audit.QUALIFIED_STATIONARY_AOT:
        (root / f"libvibeqc_stationary_{name}.so").write_bytes(b"x" * binary_size)
        (root / f"vibeqc_stationary_{name}.json").write_text("{}\n")


def test_package_audit_reports_exact_six_artifact_footprint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "build"
    _layout(root, binary_size=10)
    native = root / "libvibeqc.so"
    native.write_bytes(b"n" * 120)
    monkeypatch.setattr(audit, "load_stationary_aot_artifact", _fake_loader)
    monkeypatch.setattr(audit, "_qualified_aot_plan", lambda functional, spin: object())

    result = audit.audit_stationary_aot_directory(
        root, architecture="sm_120", native_library=native
    )

    assert len(result.artifacts) == 6
    assert result.aot_binary_bytes == 60
    assert result.manifest_bytes == 18
    assert result.aot_package_bytes == 78
    assert result.native_library_bytes == 120
    assert result.aot_to_native_ratio == 0.5
    assert all(item.code_kinds == ("cubin",) for item in result.artifacts)
    assert not any(item.driver_ptx_jit_required for item in result.artifacts)


def test_checkout_and_installed_identity_must_match(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkout = tmp_path / "checkout"
    installed = tmp_path / "installed"
    _layout(checkout)
    _layout(installed)
    monkeypatch.setattr(audit, "load_stationary_aot_artifact", _fake_loader)
    monkeypatch.setattr(audit, "_qualified_aot_plan", lambda functional, spin: object())

    left = audit.audit_stationary_aot_directory(checkout, architecture="sm_120")
    right = audit.audit_stationary_aot_directory(installed, architecture="sm_120")
    audit.assert_same_artifact_identity(left, right)

    changed = list(right.artifacts)
    changed[0] = replace(changed[0], binary_sha256="repaired-wheel-binary")
    mismatch = replace(right, artifacts=tuple(changed))
    with pytest.raises(ValueError, match="lda_rks"):
        audit.assert_same_artifact_identity(left, mismatch)


def test_native_cubin_gate_rejects_ptx_only_package(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "build"
    _layout(root)
    monkeypatch.setattr(audit, "load_stationary_aot_artifact", _fake_loader)
    monkeypatch.setattr(audit, "_qualified_aot_plan", lambda functional, spin: object())

    package = audit.audit_stationary_aot_directory(root, architecture="sm_120")
    changed = list(package.artifacts)
    changed[0] = replace(
        changed[0],
        code_kinds=("ptx",),
        driver_ptx_jit_possible=True,
        driver_ptx_jit_required=True,
    )
    ptx_only = replace(package, artifacts=tuple(changed))

    with pytest.raises(ValueError, match="lda_rks"):
        audit.assert_native_cubin_path(ptx_only)


def test_package_audit_cli_runs_from_uninstalled_checkout(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[2]
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, str(root / "tools/audit_stationary_aot_package.py"), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--require-native-cubin" in result.stdout


def test_package_audit_rejects_real_loader_integrity_failures(tmp_path: Path) -> None:
    import json

    from vibeqc_compiler.common.provenance import file_hash
    from vibeqc_compiler.method.stationary_cuda import stationary_aot_contract_identity

    root = tmp_path / "actual-loader"
    root.mkdir()
    for functional, spin, name in audit.QUALIFIED_STATIONARY_AOT:
        library = root / f"libvibeqc_stationary_{name}.so"
        # These bytes are hashed only, never loaded or executed as native code.
        library.write_bytes(f"opaque audit fixture {name}".encode())
        metadata = {
            "schema": "vibeqc.stationary-cuda-aot.v2",
            "functional": functional,
            "spin": spin,
            "plan_identity": audit._qualified_aot_plan(functional, spin).identity,
            "partition_iterations": 3,
            "architectures": ["sm_120"],
            "code_objects": [{"architecture": "sm_120", "kind": "cubin"}],
            "contract_identity": stationary_aot_contract_identity(
                functional, spin=spin
            ),
            "source_identity": f"opaque-fixture-{name}",
            "binary_sha256": file_hash(library),
            "binary_bytes": library.stat().st_size,
            "compile_contract": {"fp64": True, "fmad": False},
        }
        (root / f"vibeqc_stationary_{name}.json").write_text(json.dumps(metadata))
    result = audit.audit_stationary_aot_directory(root, architecture="sm_120")
    assert len(result.artifacts) == 6
    audit.assert_native_cubin_path(result)
    library = root / "libvibeqc_stationary_pbe_uks.so"
    original = library.read_bytes()
    library.write_bytes(original + b"tampered")
    with pytest.raises(ValueError, match="binary integrity"):
        audit.audit_stationary_aot_directory(root, architecture="sm_120")
    library.write_bytes(original)
    manifest = root / "vibeqc_stationary_pbe_uks.json"
    metadata = json.loads(manifest.read_text())
    metadata["contract_identity"] = "foreign-contract"
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="contract_identity"):
        audit.audit_stationary_aot_directory(root, architecture="sm_120")
