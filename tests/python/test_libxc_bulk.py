"""Real bulk imports, independent E/vxc/fxc fixtures and offline source identity."""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.common.array_graph import evaluate_array_graph
from vibeqc_compiler.xc import libxc_bulk
from vibeqc_compiler.xc.libxc_maple import MapleImportError

from tools import import_libxc_bulk

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / libxc_bulk.SOURCE_ASSET
CATALOG = libxc_bulk.read_catalog()
CASES = [
    case
    for path in sorted((ROOT / "tests/data/xc/libxc-bulk").glob("*.json"))
    for case in json.loads(path.read_text())["cases"]
]


def test_every_advertised_import_has_an_independent_both_spin_fixture() -> None:
    imported = set(libxc_bulk.available_functionals())
    assert CATALOG["counts"]["maple_files"] == 285
    assert CATALOG["counts"]["energy_source_files"] == 268
    assert len(imported) == CATALOG["counts"]["graph_imported_registrations"] == 221
    assert {(case["name"], case["spin"]) for case in CASES} == {
        (name, spin) for name in imported for spin in ("polarized", "unpolarized")
    }
    assert len(CASES) == 2 * len(imported)


@pytest.mark.parametrize(
    "case", CASES, ids=lambda case: f"{case['name']}-{case['spin']}"
)
def test_bulk_physical_energy_gradient_hessian_matches_independent_libxc(
    case: dict,
) -> None:
    program = libxc_bulk.build_bulk_program(case["name"], spin=case["spin"])
    features = np.asarray(case["features"], dtype=float).T
    values = evaluate_array_graph(
        program.graph,
        program.roots(2),
        dict(zip(program.features, features, strict=True)),
    )
    actual = np.stack(
        [np.broadcast_to(value, features.shape[1:]) for value in values], axis=1
    )
    np.testing.assert_allclose(actual, case["expected"], rtol=2e-9, atol=2e-10)


def test_catalog_and_human_report_reproduce_offline() -> None:
    regenerated = import_libxc_bulk.make_catalog(SOURCE)
    assert regenerated == CATALOG
    assert (
        import_libxc_bulk.render_report(regenerated)
        == import_libxc_bulk.REPORT.read_text()
    )


def test_unknown_and_blocked_names_do_not_fall_back() -> None:
    blocked = next(
        row["name"]
        for row in CATALOG["registrations"]
        if row["graph_status"] == "blocked"
    )
    for name in ("NOT_A_FUNCTIONAL", blocked):
        with pytest.raises(MapleImportError):
            libxc_bulk.build_bulk_program(name)


def test_bulk_does_not_promote_public_functional_admission() -> None:
    from fractions import Fraction

    from vibeqc_compiler.xc.program import build_program
    from vibeqc_compiler.xc.spec import CATALOG, FunctionalSpec, UnsupportedXC

    spec = FunctionalSpec("BULK_ONLY", (("GGA_X_PBE_SOL", Fraction(1)),))
    assert spec.to_payload()["production_admitted"] is False
    assert "GGA_X_PBE_SOL" not in CATALOG
    with pytest.raises(UnsupportedXC, match="not production-domain admitted"):
        build_program(spec)


def test_source_and_parameter_owner_tampering_is_rejected(tmp_path: Path) -> None:
    copy = tmp_path / "upstream"
    shutil.copytree(SOURCE, copy)
    owner = copy / "src/gga_x_pbe.c"
    original = owner.read_bytes()
    owner.write_bytes(original + b"\n/* changed parameter owner */\n")
    with pytest.raises(MapleImportError, match="SHA-256"):
        libxc_bulk.build_bulk_program("GGA_X_PBE_SOL", source_root=copy)
    owner.write_bytes(original)
    entry = copy / "maple/gga_exc/gga_x_pbe.mpl"
    entry.write_bytes(entry.read_bytes() + b"\n# altered formula\n")
    with pytest.raises(MapleImportError, match="SHA-256"):
        libxc_bulk.build_bulk_program("GGA_X_PBE_SOL", source_root=copy)


def test_projection_paths_do_not_enter_scientific_identity(tmp_path: Path) -> None:
    copy = tmp_path / "elsewhere"
    shutil.copytree(SOURCE, copy)
    first = libxc_bulk.build_bulk_program("GGA_X_PBE_SOL")
    second = libxc_bulk.build_bulk_program("GGA_X_PBE_SOL", source_root=copy)
    assert first.identity == second.identity
    assert first.emit_source(2) == second.emit_source(2)


def test_bad_catalog_and_bad_archive_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "catalog.json"
    wrong = {**CATALOG, "importer_semantics": "not-the-running-importer"}
    path.write_text(json.dumps(wrong))
    with pytest.raises(MapleImportError, match="regeneration"):
        libxc_bulk.read_catalog(path)
    path.write_text(json.dumps({**CATALOG, "registrations": []}))
    with pytest.raises(MapleImportError, match="empty"):
        libxc_bulk.read_catalog(path)
    bad = tmp_path / "wrong.tar.gz"
    bad.write_bytes(b"not the pinned archive")
    with pytest.raises(ValueError, match="SHA-256"):
        import_libxc_bulk.extract_archive(bad, tmp_path / "destination")
    with pytest.raises(ValueError, match="empty"):
        import_libxc_bulk.make_catalog(tmp_path / "absent")


@pytest.mark.parametrize("name", ["LDA_C_VWN_4", "GGA_X_PBE_SOL", "MGGA_X_R2SCAN01"])
def test_existing_emitters_and_compiled_c_hessian(name: str, tmp_path: Path) -> None:
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("C compiler unavailable")
    program = libxc_bulk.build_bulk_program(name)
    source = program.emit_source(2)
    cuda = program.emit_source(2, cuda=True)
    assert "__device__" in cuda and "bulk_xc_point" in cuda
    path = tmp_path / "point.c"
    library_path = tmp_path / "point.so"
    path.write_text(source)
    subprocess.run(
        [
            cc,
            "-std=c99",
            "-O1",
            "-shared",
            "-fPIC",
            str(path),
            "-lm",
            "-o",
            str(library_path),
        ],
        check=True,
        capture_output=True,
    )
    function = ctypes.CDLL(str(library_path)).bulk_xc_point
    ptr = ctypes.POINTER(ctypes.c_double)
    function.argtypes = [ptr, ptr]
    function.restype = None
    case = next(
        case for case in CASES if case["name"] == name and case["spin"] == "polarized"
    )
    for features, expected in zip(case["features"], case["expected"], strict=True):
        inputs = np.asarray(features, dtype=np.float64)
        output = np.empty(len(expected), dtype=np.float64)
        function(inputs.ctypes.data_as(ptr), output.ctypes.data_as(ptr))
        np.testing.assert_allclose(output, expected, rtol=2e-9, atol=2e-10)


def test_packaged_compiler_imports_without_site_runtime_or_network(
    tmp_path: Path,
) -> None:
    package = tmp_path / "site/vibeqc_compiler"
    shutil.copytree(
        ROOT / "python/vibeqc_compiler",
        package,
        ignore=shutil.ignore_patterns("__pycache__", "assets"),
    )
    shutil.copytree(SOURCE, package / "assets" / libxc_bulk.SOURCE_ASSET)
    code = """
import sys
import socket
def no_network(*args, **kwargs):
    raise AssertionError("offline compilation attempted network access")
socket.socket = no_network
socket.create_connection = no_network
from vibeqc_compiler.xc.libxc_bulk import build_bulk_program
program = build_bulk_program("GGA_X_PBE_SOL")
assert program.roots(2)
assert not {"vibeqc", "pyscf", "torch", "cupy", "numpy"}.intersection(sys.modules)
print(program.identity)
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", code],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(package.parent)},
        check=True,
        capture_output=True,
        text=True,
    )
    assert (
        result.stdout.strip() == libxc_bulk.build_bulk_program("GGA_X_PBE_SOL").identity
    )


def test_bulk_source_projection_does_not_shadow_production_adapters() -> None:
    from vibeqc_compiler.xc import b88_vwn_maple, pbe_maple, rsh_maple, scan_maple

    for adapter in (b88_vwn_maple, pbe_maple, rsh_maple, scan_maple):
        assert adapter._libxc_root().resolve() != SOURCE.resolve()
