"""Second-integral artifacts cannot cross cache, compiler or layout contexts."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.vibeqc_hessian import analytic


@pytest.fixture
def requests(monkeypatch, tmp_path):
    calls = []
    compiler = object()
    monkeypatch.setattr(analytic, "_COMPILE_CACHE", {})

    def compile_tile(ir, adapter, cache, **kwargs):
        cache = Path(cache).resolve()
        cache.mkdir(parents=True, exist_ok=True)
        library = cache / f"tile-{len(calls)}.so"
        library.write_bytes(b"fake compiled artifact")
        artifact = SimpleNamespace(
            native=SimpleNamespace(library=library),
            adapter=adapter,
            cache=cache,
            **kwargs,
        )
        calls.append(artifact)
        return artifact

    monkeypatch.setattr(analytic, "compile_second_derivative", compile_tile)

    def request(**changes):
        options = {
            "adapter": compiler,
            "cache": tmp_path / "first",
            "output_indices": (0, 1),
            "component_indices": (0, 1),
        }
        options.update(changes)
        return analytic._compile_cached(
            ("overlap", 1, 1), lambda **kwargs: "IR", {}, **options
        )

    return request, calls


def test_same_context_reuses_live_artifact(requests):
    request, calls = requests
    first = request()
    assert request() is first
    assert request(cache=first.cache / ".." / "first") is first
    assert len(calls) == 1


@pytest.mark.parametrize("changed", ["cache", "adapter", "outputs", "components"])
def test_distinct_contexts_never_alias(requests, tmp_path, changed):
    request, calls = requests
    first = request()
    changes = {
        "cache": {"cache": tmp_path / "second"},
        "adapter": {"adapter": object()},
        "outputs": {"output_indices": (1, 0)},
        "components": {"component_indices": (1, 0)},
    }
    second = request(**changes[changed])
    assert second is not first
    assert len(calls) == 2
    assert request(**changes[changed]) is second
    if changed == "outputs":
        assert second.output_indices == (1, 0)
    elif changed == "components":
        assert second.component_indices == (1, 0)


def test_removed_binary_is_rebuilt_not_returned_from_memory(requests):
    request, calls = requests
    first = request()
    first.native.library.unlink()
    second = request()
    assert second is not first
    assert second.native.library.is_file()
    assert len(calls) == 2
