"""Keep native-loader typing and vendored provenance mechanically verifiable."""

from __future__ import annotations

import ctypes
import hashlib
import inspect
import json
from pathlib import Path

import pytest
from vibeqc import _cuda_runtime, _native


def test_implib_sources_match_pinned_manifest() -> None:
    root = Path(__file__).resolve().parents[2]
    vendor = root / "cmake/3rdparty/implib"
    manifest = json.loads((vendor.parent / "implib_manifest.json").read_text())
    assert manifest["files"]
    for record in manifest["files"]:
        data = (vendor / record["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == record["sha256"], record["path"]
        blob = b"blob " + str(len(data)).encode("ascii") + b"\0" + data
        assert hashlib.sha1(blob).hexdigest() == record["git_blob"], record["path"]


def test_cuda_runtime_wrapper_preserves_concrete_signature() -> None:
    assert inspect.signature(
        _native.load_library, follow_wrapped=False
    ) == inspect.signature(_native.load_library)


@pytest.mark.parametrize("selection", [None, (None, 0), ("cpu", 0), ("cuda", 3)])
def test_cuda_runtime_loader_forwards_selection_after_preloading(
    selection: tuple[str | None, int] | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[object] = []
    library = ctypes.CDLL(None)

    def native_loader(*, device: str | None = None, device_id: int = 0) -> ctypes.CDLL:
        calls.append((device, device_id))
        return library

    def preload() -> tuple[str, ...]:
        calls.append("preload")
        return ()

    monkeypatch.setattr(_native, "load_library", native_loader)
    monkeypatch.setattr(_cuda_runtime, "preload_cuda_runtime_libraries", preload)
    _cuda_runtime.install_native_loader()
    installed = _native.load_library
    _cuda_runtime.install_native_loader()
    assert _native.load_library is installed
    if selection is None:
        actual = installed()
    else:
        actual = installed(device=selection[0], device_id=selection[1])
    assert actual is library
    assert calls == ["preload", (None, 0) if selection is None else selection]
