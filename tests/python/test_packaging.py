"""Python/native packaging regressions."""

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
