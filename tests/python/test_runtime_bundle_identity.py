"""Executable cache checks for input consumption and local-header topology."""

import ctypes
from pathlib import Path

import pytest
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.common.native_runtime import compile_runtime_bundle


def value(artifact: object, symbol: str = "value") -> int:
    call = getattr(ctypes.CDLL(str(artifact.library)), symbol)
    call.restype = ctypes.c_int
    return call()


def test_bundle_iterator_options_match_published_binary(tmp_path: Path) -> None:
    source = tmp_path / "value.cpp"
    source.write_text(
        '#ifndef VALUE\n#define VALUE 3\n#endif\nextern "C" int value(){return VALUE;}\n'
    )
    compiler = CppCompilerAdapter(Path("c++"))
    flags = ("-DVALUE=19", "-ffp-contract=off")
    first = compile_runtime_bundle(
        compiler, tmp_path / "cache", (source,), options=iter(flags)
    )
    assert value(first) == 19
    replay = compile_runtime_bundle(
        compiler, tmp_path / "cache", (source,), options=flags
    )
    assert first.library == replay.library
    assert value(replay) == 19


def test_local_header_association_participates_in_bundle_identity(
    tmp_path: Path,
) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    headers = (a / "config.hpp", b / "config.hpp")
    headers[0].write_text("#define VALUE 19\n")
    headers[1].write_text("#define VALUE 23\n")
    for folder in (a, b):
        for name in ("first", "second"):
            (folder / f"{name}.cpp").write_text(
                f'#include "config.hpp"\nextern "C" int {name}(){{return VALUE;}}\n'
            )
    compiler = CppCompilerAdapter(Path("c++"))
    first = compile_runtime_bundle(
        compiler,
        tmp_path / "cache",
        (a / "first.cpp", b / "second.cpp"),
        headers=headers,
    )
    swapped = compile_runtime_bundle(
        compiler,
        tmp_path / "cache",
        (b / "first.cpp", a / "second.cpp"),
        headers=headers,
    )
    assert (value(first, "first"), value(first, "second")) == (19, 23)
    assert first.metadata["key"] != swapped.metadata["key"]
    assert (value(swapped, "first"), value(swapped, "second")) == (23, 19)


@pytest.mark.parametrize("libraries", (("m",), iter(("m",))))
def test_bundle_library_arguments_reach_compiler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, libraries: object
) -> None:
    source = tmp_path / "value.cpp"
    source.write_text('extern "C" int value(){return 19;}\n')
    original = CppCompilerAdapter.compile_shared_many

    def checked(
        self: object, sources: object, output: object, **kwargs: object
    ) -> object:
        assert tuple(kwargs["libraries"]) == ("m",)
        return original(self, sources, output, **kwargs)

    monkeypatch.setattr(CppCompilerAdapter, "compile_shared_many", checked)
    assert (
        value(
            compile_runtime_bundle(
                CppCompilerAdapter(Path("c++")),
                tmp_path / "cache",
                (source,),
                libraries=libraries,
            )
        )
        == 19
    )
