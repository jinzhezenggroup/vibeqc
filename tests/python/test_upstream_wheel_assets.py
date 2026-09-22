"""A source move must remain usable outside the checkout after wheel projection."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib")
ROOT = Path(__file__).resolve().parents[2]


def test_installed_compiler_contains_pinned_libxc_source_closure(
    tmp_path: Path,
) -> None:
    package = tmp_path / "vibeqc_compiler"
    shutil.copytree(
        ROOT / "python/vibeqc_compiler",
        package,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    mappings = config["tool"]["scikit-build"]["wheel"]["force-include"]
    for source, destination in mappings.items():
        target = tmp_path / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        if (ROOT / source).is_dir():
            shutil.copytree(ROOT / source, target, dirs_exist_ok=True)
        else:
            shutil.copyfile(ROOT / source, target)
    registry = json.loads((ROOT / "upstream/manifest.json").read_text())
    libxc = registry["sources"]["libxc-7.0.0"]
    installed = package / "assets" / libxc["local_root"]
    assert installed.is_dir(), "wheel omitted the canonical Libxc source directory"
    for name in libxc["files"]:
        assert (installed / name).read_bytes() == (
            ROOT / libxc["local_root"] / name
        ).read_bytes()
    xtbloom = registry["sources"]["xtbloom-gfn1-d3"]
    installed_xtbloom = package / "assets" / xtbloom["local_root"]
    for name in xtbloom["files"]:
        assert (installed_xtbloom / name).read_bytes() == (
            ROOT / xtbloom["local_root"] / name
        ).read_bytes()
    assert (installed_xtbloom / "gfn1.json").is_file()
    script = """
from vibeqc_compiler.common.paths import asset_path, source_root
from vibeqc_compiler.integral.expr import Graph
from vibeqc_compiler.xc.libxc_maple import import_maple_file
try:
    source_root()
except ValueError:
    pass
else:
    raise AssertionError('test must not resolve a checkout')
root = asset_path('upstream/libxc/7.0.0')
module = import_maple_file(root, 'gga_c_pbe.mpl', defines={'gga_c_pbe_params'}, support_files=('util.mpl',))
graph = Graph()
energy = module.call(graph, 'f', graph.constant(1), graph.constant(0), graph.constant(0), 0, 0)
assert -1 < graph.evaluate(energy, {}) < 0
assert asset_path('manifests/libxc/7.0.0/manifest.json').is_file()
xtbloom = asset_path('upstream/xtbloom/2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3')
assert (xtbloom / 'gfn1_d3.json').is_file()
assert (xtbloom / 'gfn1.json').is_file()
"""
    completed = subprocess.run(
        [sys.executable, "-S", "-c", script],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
