"""Fast, offline integrity checks for the native D4 migration assets."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import typing
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generated_table_matches_manifest() -> None:
    folder = ROOT / "src/dft/dispersion"
    manifest = json.loads((folder / "d4_manifest.json").read_text())
    assert manifest["reference_model"] == "gfn2"
    assert manifest["storage"] == "lower-triangle-including-diagonal"
    assert manifest["reference_count"] == 262
    assert manifest["revision"] == "6e1f59c3f39d919a2dbef0601d2576727c8b30e8"
    assert digest(folder / manifest["output"]) == manifest["output_sha256"]


def test_independent_oracle_assets_match_manifest() -> None:
    manifest = json.loads((ROOT / "tests/data/d4/oracle_manifest.json").read_text())
    assert manifest["oracle_version"] == "dftd4 version 4.2.0"
    assert manifest["reference_model"] == "gfn2"
    assert len(manifest["cases"]) == 5
    assert manifest["fixture_sha256"] == digest(
        ROOT / "tests/native/d4_oracle_fixtures.hpp"
    )
    assert manifest["adapter_sha256"] == digest(
        ROOT / "tools/oracle/d4_fixed_charge.f90"
    )
    assert manifest["generator_sha256"] == digest(
        ROOT / "tools/oracle/generate_d4_reference.py"
    )


def load_generator() -> typing.Any:
    path = ROOT / "tools/parameters/generate_d4.py"
    spec = importlib.util.spec_from_file_location("d4_parameter_generator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_packing_refuses_nonsymmetric_matrix() -> None:
    gen = load_generator()
    refs = [
        {"coordination_number": float(i), "charge": 0.0, "gaussian_count": 1}
        for i in range(2)
    ]
    with pytest.raises(gen.D4DataError, match="exactly symmetric"):
        gen.render_header("revision", "digest", [], refs, [1.0, 2.0, 3.0, 4.0])
    packed = gen.render_header("revision", "digest", [], refs, [1.0, 2.0, 2.0, 3.0])
    assert "1.0, 2.0, 3.0," in packed
    assert "kReferenceCount * (kReferenceCount + 1) / 2" in packed


def test_exporter_rejects_nonfinite_table_values() -> None:
    gen = load_generator()
    with pytest.raises(gen.D4DataError, match="NaN or infinity"):
        gen.format_double(float("nan"))


def test_migration_records_original_source_blobs() -> None:
    manifest = json.loads(
        (ROOT / "src/dft/dispersion/xtbloom_manifest.json").read_text()
    )
    assert manifest["revision"] == "2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3"
    assert any(
        record["path"] == "src/backends/cuda/gfn2_d4.cu"
        for record in manifest["sources"]
    )
    assert all(
        len(record["git_blob"]) == 40 and len(record["sha256"]) == 64
        for record in manifest["sources"]
    )


def test_eeq_tables_and_charge_parameters_match_pinned_manifest() -> None:
    folder = ROOT / "src/dft/dispersion"
    manifest = json.loads((folder / "d4_eeq_manifest.json").read_text())
    assert manifest["reference_model"] == "eeq"
    assert manifest["dftd4_revision"] == "6e1f59c3f39d919a2dbef0601d2576727c8b30e8"
    assert (
        manifest["multicharge_revision"] == "6a5d63f9e9e29dcf13cc47cc27f33bf9015681bf"
    )
    assert manifest["mctc_revision"] == "e9de066d89f250d1cfb6de3a33f0c27c0e2f855d"
    assert manifest["profiles"] == {
        "r2scan-3c": {"ga": 2.0, "gc": 1.0},
        "standard": {"ga": 3.0, "gc": 2.0},
    }
    assert manifest["element_count"] == 86
    assert manifest["reference_count"] == 262
    assert set(manifest["outputs"]) == {
        "d4_eeq_data.hpp",
        "d4_eeq_r2scan3c_c6.hpp",
    }
    for name, expected in manifest["outputs"].items():
        assert digest(folder / name) == expected
    bundle_identity = "".join(
        f"{name}:{manifest['outputs'][name]}\n" for name in sorted(manifest["outputs"])
    )
    assert (
        hashlib.sha256(bundle_identity.encode()).hexdigest()
        == manifest["bundle_sha256"]
    )


def test_independent_eeq_oracle_assets_match_manifest() -> None:
    manifest = json.loads((ROOT / "tests/data/d4/eeq_oracle_manifest.json").read_text())
    assert manifest["oracle_version"] == "dftd4 version 4.2.0"
    assert manifest["cases"] == [
        "water_r2scan3c",
        "asymmetric_pbe",
        "charged_zinc_ammonia_pbe",
    ]
    assert manifest["fixture_sha256"] == digest(
        ROOT / "tests/native/d4_eeq_oracle_fixtures.hpp"
    )
    assert manifest["generator_sha256"] == digest(
        ROOT / "tools/oracle/generate_d4_eeq_reference.py"
    )
