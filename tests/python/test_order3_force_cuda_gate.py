"""Order-three qualification must not turn execution failures into skips."""

import ctypes
import os
from types import SimpleNamespace
from typing import Self

import numpy as np
import pytest
from vibeqc import Calculator, Primitive, Shell, _native


def _require_cuda(calculator: Calculator) -> None:
    descriptor = _native.ContextDescriptor(
        ctypes.sizeof(_native.ContextDescriptor),
        _native.ABI_VERSION,
        0,
        _native.BACKEND_CUDA,
    )
    context = ctypes.c_void_p()
    library = calculator._library
    status = library.vibeqc_context_create(
        ctypes.byref(descriptor), ctypes.byref(context)
    )
    try:
        if (
            status in (_native.STATUS_NOT_IMPLEMENTED, _native.STATUS_CUDA_ERROR)
            and os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1"
        ):
            pytest.skip("CUDA context unavailable before order-three execution")
        _native.check(library, status)
    finally:
        if context.value:
            library.vibeqc_context_destroy(context)


def test_cuda_order3_fsss_fallback_matches_cpu_and_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise generated FSSS force math through fixed and bounded Direct queues."""
    basis = (
        Shell(0, 0, (Primitive(1.5, 1.0),)),
        Shell(0, 3, (Primitive(0.6, 1.0),)),
        Shell(1, 0, (Primitive(1.2, 1.0),)),
    )
    system = [("He", (0.13, -0.07, -0.72)), ("H", (-0.09, 0.11, 0.68))]
    moved = np.asarray([[0.14, -0.07, -0.72], [-0.09, 0.11, 0.68]])
    common = {
        "method": "rhf",
        "basis": basis,
        "energy_tolerance": 1.0e-10,
        "density_tolerance": 1.0e-8,
        "screening_tolerance": 1.0e-14,
    }
    reference = Calculator(device="cpu", **common).singlepoint(system, charge=1)
    monkeypatch.setenv("VIBEQC_AOT_SHELL_CLASSES", "ssss")
    outputs = {}
    for mode in ("exact", "streaming"):
        if mode == "streaming":
            monkeypatch.setenv("VIBEQC_BOUNDED_DIRECT_STREAMING", "force")
        else:
            monkeypatch.delenv("VIBEQC_BOUNDED_DIRECT_STREAMING", raising=False)
        calculator = Calculator(device="cuda", **common)
        _require_cuda(calculator)
        # After context admission, preparation and replay errors are failures.
        with calculator.prepare_batch(
            [system], charges=[1], warm_start=True
        ) as prepared:
            outputs[mode] = (
                prepared.execute(strict=True).items[0],
                prepared.execute([moved], strict=True).items[0],
            )

    exact, streaming = outputs["exact"][0], outputs["streaming"][0]
    assert exact.energy == pytest.approx(reference.energy, abs=3.0e-10)
    assert np.allclose(exact.forces, reference.forces, atol=3.0e-8)
    assert streaming.iterations == exact.iterations
    assert streaming.energy == pytest.approx(exact.energy, abs=3.0e-13)
    assert np.allclose(streaming.forces, exact.forces, atol=1.0e-11)
    exact_moved, streaming_moved = outputs["exact"][1], outputs["streaming"][1]
    assert streaming_moved.iterations == exact_moved.iterations
    assert streaming_moved.energy == pytest.approx(exact_moved.energy, abs=3.0e-13)
    assert np.allclose(streaming_moved.forces, exact_moved.forces, atol=1.0e-11)


@pytest.mark.parametrize("failure", ["prepare", "cold", "changed"])
def test_order3_gate_preserves_runtime_failures(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Transport fakes check exception handling, not scientific CUDA arithmetic."""

    class BrokenReplay:
        calls = 0

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

        def execute(self, *_args: object, **_kwargs: object) -> SimpleNamespace:
            self.calls += 1
            if failure == "cold" or self.calls == 2:
                raise RuntimeError("injected order-three execution failure")
            return SimpleNamespace(items=[object()])

    class FakeCalculator:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def singlepoint(self, *_args: object, **_kwargs: object) -> object:
            return object()

        def prepare_batch(self, *_args: object, **_kwargs: object) -> BrokenReplay:
            if failure == "prepare":
                raise RuntimeError("injected order-three execution failure")
            return BrokenReplay()

    namespace = test_cuda_order3_fsss_fallback_matches_cpu_and_streaming.__globals__
    monkeypatch.setitem(namespace, "Calculator", FakeCalculator)
    monkeypatch.setitem(namespace, "_require_cuda", lambda _calculator: None)
    with pytest.raises(RuntimeError, match="injected order-three"):
        test_cuda_order3_fsss_fallback_matches_cpu_and_streaming(monkeypatch)
