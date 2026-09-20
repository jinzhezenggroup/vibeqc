"""Automatic ISA admission must be safe on every processor in the inventory."""

import typing

import pytest
from vibeqc_compiler.common import cpu_dispatch
from vibeqc_compiler.common.cpu_target import CPU_TARGETS


@pytest.mark.parametrize(
    "records,expected",
    [
        (
            "processor: 0\nflags: avx2 fma avx512f\n\nprocessor: 1\nflags: avx2 fma\n",
            ("avx2", "fma"),
        ),
        ("processor: 0\nflags: avx2 fma\n\nprocessor: 1\nflags: fma\n", ("fma",)),
        ("processor: 0\nflags: avx2 fma\n\nprocessor: 1\nmodel name: unknown\n", ()),
        ("processor: 0\nflags: avx2 fma\n\nprocessor: 1\nflags:\n", ()),
    ],
)
def test_linux_dispatch_uses_common_processor_features(
    monkeypatch: typing.Any, records: typing.Any, expected: typing.Any
) -> None:
    monkeypatch.setattr(cpu_dispatch.sys, "platform", "linux")
    monkeypatch.setattr(cpu_dispatch.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(cpu_dispatch.Path, "read_text", lambda self: records)
    monkeypatch.delenv("VIBEQC_CPU_TARGET", raising=False)
    runtime = cpu_dispatch.detect_cpu_features()
    assert runtime.features == expected
    selected = cpu_dispatch.select_cpu_target(CPU_TARGETS, runtime)
    assert selected.selected_target == (
        "x86_64-avx2-fma" if "avx2" in expected else "generic"
    )


def test_linux_unavailable_feature_inventory_is_generic(
    monkeypatch: typing.Any,
) -> None:
    def unavailable(self: typing.Any) -> typing.Any:
        raise OSError("unavailable CPU inventory")

    monkeypatch.setattr(cpu_dispatch.sys, "platform", "linux")
    monkeypatch.setattr(cpu_dispatch.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(cpu_dispatch.Path, "read_text", unavailable)
    assert cpu_dispatch.detect_cpu_features().features == ()
