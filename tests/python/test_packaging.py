"""Python/native packaging regressions."""

import re
import shutil
from pathlib import Path

import pytest
from vibeqc import _native


def test_installed_package_library_finds_wheel_layout(tmp_path, monkeypatch):
    package = tmp_path / "vibeqc"
    library = package / "lib" / "libvibeqc.so"
    library.parent.mkdir(parents=True)
    library.touch()
    monkeypatch.setattr(_native, "PACKAGE_DIR", package)

    assert _native._installed_package_library() == library


def test_native_candidates_prefer_override_then_bundled(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit" / "libvibeqc.so"
    explicit.parent.mkdir(parents=True)
    explicit.touch()
    package = tmp_path / "site" / "vibeqc"
    bundled = package / "lib" / "libvibeqc.so"
    bundled.parent.mkdir(parents=True)
    bundled.touch()
    monkeypatch.setenv("VIBEQC_LIBRARY", str(explicit))
    monkeypatch.setattr(_native, "PACKAGE_DIR", package)

    assert _native._candidate_paths()[:2] == [explicit, bundled]


def test_wheel_compiler_templates_include_local_dependencies(tmp_path, monkeypatch):
    """Exercise JIT source preparation using only the declared wheel payload."""
    tomllib = pytest.importorskip("tomllib")
    from vibeqc_compiler.common import paths
    from vibeqc_compiler.dft.ao_cuda import emit_grid_source
    from vibeqc_compiler.tensor.cuda_execute import tensor_source_identity

    root = Path(__file__).resolve().parents[2]
    package = tmp_path / "vibeqc_compiler"
    shutil.copytree(root / "python/vibeqc_compiler", package)
    config = tomllib.loads((root / "pyproject.toml").read_text())
    for source, destination in config["tool"]["scikit-build"]["wheel"][
        "force-include"
    ].items():
        target = tmp_path / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        if (root / source).is_dir():
            shutil.copytree(root / source, target)
        else:
            shutil.copyfile(root / source, target)
    monkeypatch.setattr(paths, "PACKAGE", package)
    assert tensor_source_identity()
    assert all(header.is_file() for header in emit_grid_source()[2])
    assets = package / "assets"
    # Quoted native includes can resolve beside the template or through the
    # compiler's src/include roots; none may depend on a neighboring checkout.
    for source in (assets / "src").rglob("*"):
        if source.is_file():
            for name in re.findall(
                r'^#include "([^"]+)"', source.read_text(), re.MULTILINE
            ):
                assert any(
                    (base / name).is_file()
                    for base in (source.parent, assets / "src", assets / "include")
                ), (source, name)
