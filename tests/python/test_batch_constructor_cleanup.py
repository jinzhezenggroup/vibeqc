"""Failed preparation releases owned handles even while its traceback is retained."""

import typing

import pytest
from vibeqc import Calculator, ResourceBudget, _native
from vibeqc.batch import PreparedBatch
from vibeqc.resources import ResourceAllocationError


@pytest.mark.parametrize(
    "resource_aware,failure",
    [
        (False, MemoryError("injected allocation")),
        (False, ValueError("injected preparation validation")),
        (True, None),
    ],
)
def test_preparation_failure_releases_native_owners(
    monkeypatch: typing.Any, resource_aware: typing.Any, failure: typing.Any
) -> None:
    calculator = Calculator(
        device="cpu", resource_budget=ResourceBudget() if resource_aware else None
    )
    library = calculator._library
    native_prepare = library.vibeqc_batch_prepare
    prepared = PreparedBatch.__new__(PreparedBatch)
    allocated = []

    def failed_prepare(*args: typing.Any) -> typing.Any:
        status = native_prepare(*args)
        assert status == _native.STATUS_SUCCESS
        assert prepared._context.value and prepared._batch.value
        allocated.append((prepared._context.value, prepared._batch.value))
        if failure is not None:
            raise failure
        # Exercise the real resource-status conversion, not a mocked exception.
        return _native.STATUS_OUT_OF_MEMORY

    monkeypatch.setattr(library, "vibeqc_batch_prepare", failed_prepare)
    expected = type(failure) if failure is not None else ResourceAllocationError
    try:
        with pytest.raises(expected) as caught:
            prepared.__init__(
                calculator,
                [[("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]],
                warm_start=False,
            )
        assert allocated
        if failure is not None:
            assert caught.value is failure
        else:
            assert caught.value.resource_diagnostics["phase"] == "preparation"
        # Both prepared and the exception traceback remain live here: __del__
        # cannot conceal a constructor leak by eventually collecting the object.
        assert not prepared._batch.value
        assert not prepared._context.value
        assert prepared._resource_ledger is None
        monkeypatch.setattr(library, "vibeqc_batch_prepare", native_prepare)
        with calculator.prepare_batch(
            [[("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]], warm_start=False
        ) as retry:
            assert retry.execute(properties=("energy",), strict=True).items[0].succeeded
    finally:
        prepared.close()
