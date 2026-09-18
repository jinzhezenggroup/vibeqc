"""Python/native packaging regressions."""

import re
import shutil
from pathlib import Path

import pytest
from vibeqc import _cuda_runtime, _native


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


def test_cuda_runtime_search_finds_pypi_provider_dirs(tmp_path, monkeypatch):
    site_packages = tmp_path / "site-packages"
    cublas = site_packages / "nvidia" / "cublas" / "lib"
    cusolver = site_packages / "nvidia" / "cusolver" / "lib"
    cublas.mkdir(parents=True)
    cusolver.mkdir(parents=True)

    monkeypatch.setattr(
        _cuda_runtime.site, "getsitepackages", lambda: [str(site_packages)]
    )
    monkeypatch.setattr(_cuda_runtime.site, "getusersitepackages", lambda: None)

    search_dirs = _cuda_runtime._runtime_search_dirs()
    assert cublas.resolve() in search_dirs
    assert cusolver.resolve() in search_dirs


def test_cuda_runtime_preload_uses_curated_sonames_not_driver(tmp_path, monkeypatch):
    provider = tmp_path / "lib"
    provider.mkdir()
    sonames = [group[0] for group in _cuda_runtime._CUDA_RUNTIME_LIBRARY_GROUPS]
    for soname in sonames:
        (provider / soname).touch()

    loaded = []

    def fake_cdll(path, *, mode):
        loaded.append((Path(path).name, mode))
        return object()

    monkeypatch.setattr(_cuda_runtime, "_cuda_runtime_handles", {})
    monkeypatch.setattr(_cuda_runtime, "_runtime_search_dirs", lambda: [provider])
    monkeypatch.setattr(_cuda_runtime.ctypes, "CDLL", fake_cdll)

    assert _cuda_runtime.preload_cuda_runtime_libraries() == tuple(sonames)
    assert [name for name, _ in loaded] == sonames
    assert "libcuda.so.1" not in sonames


def test_cuda_runtime_preload_is_noop_off_linux(monkeypatch):
    monkeypatch.setattr(_cuda_runtime.sys, "platform", "darwin")

    assert _cuda_runtime._runtime_search_dirs() == []
    assert _cuda_runtime.preload_cuda_runtime_libraries() == ()


def test_cuda_runtime_preload_falls_through_loader_errors(tmp_path, monkeypatch):
    broken = tmp_path / "broken"
    working = tmp_path / "working"
    broken.mkdir()
    working.mkdir()
    soname = "libcudart.so.12"
    (broken / soname).touch()
    (working / soname).touch()
    attempts = []

    def fake_cdll(path, *, mode):
        attempts.append((Path(path), mode))
        if Path(path).parent == broken:
            raise OSError("broken provider")
        return object()

    monkeypatch.setattr(_cuda_runtime, "_CUDA_RUNTIME_LIBRARY_GROUPS", ((soname,),))
    monkeypatch.setattr(_cuda_runtime, "_cuda_runtime_handles", {})
    monkeypatch.setattr(
        _cuda_runtime, "_runtime_search_dirs", lambda: [broken, working]
    )
    monkeypatch.setattr(_cuda_runtime.ctypes, "CDLL", fake_cdll)

    assert _cuda_runtime.preload_cuda_runtime_libraries() == (soname,)
    assert [path.parent for path, _ in attempts] == [broken, working]
    assert _cuda_runtime.preload_cuda_runtime_libraries() == ()


def test_cuda_runtime_loader_install_is_idempotent():
    installed = _native.load_library
    _cuda_runtime.install_native_loader()

    assert _native.load_library is installed


def test_package_installs_cuda_runtime_wrapper():
    assert getattr(_native.load_library, "_vibeqc_cuda_runtime_loader", False)


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


def test_project_metadata_declares_cuda_runtime_providers():
    import tomllib

    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    dependencies = project["project"]["dependencies"]
    required = {
        "nvidia-cublas-cu12",
        "nvidia-cusolver-cu12",
        "nvidia-cusparse-cu12",
        "nvidia-cuda-runtime-cu12",
        "nvidia-nvjitlink-cu12",
    }
    assert required <= {dependency.split(">=", 1)[0] for dependency in dependencies}
    assert project["project"]["optional-dependencies"]["cuda12"] == []


def test_cibuildwheel_uses_base_dependencies_for_provider_smoke():
    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github/workflows/wheels.yml").read_text()
    assert "CIBW_TEST_EXTRAS" not in workflow
