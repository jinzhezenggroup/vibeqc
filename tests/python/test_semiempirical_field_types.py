"""Reject mutable/non-scalar enum fields at the common IR construction boundary."""

from dataclasses import replace

import numpy as np
import pytest
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.method.semiempirical import (
    InvalidSemiempiricalMethod,
    ParameterResource,
    ParameterSetRef,
    PrimitiveNode,
    ProductSpec,
    SemiempiricalMethodIR,
    StateField,
)

PARAMETERS = ParameterSetRef(
    "test", "test-v1", "source", (1,), (ParameterResource("basis", "v1"),)
)
STATE = StateField("charge", "atom")
PRIMITIVE = PrimitiveNode("basis", "basis", "equation")
PRODUCT = ProductSpec("energy", 0)
METHOD = SemiempiricalMethodIR(
    "test", "test", "restricted", PARAMETERS, (STATE,), (PRIMITIVE,), (PRODUCT,)
)
CASES = (
    (STATE, "dtype"),
    (STATE, "spin_semantics"),
    (STATE, "version"),
    (PRIMITIVE, "category"),
    (PRIMITIVE, "version"),
    (PARAMETERS, "version"),
    (PRODUCT, "version"),
    (METHOD, "version"),
)


@pytest.mark.parametrize(("record", "field"), CASES)
@pytest.mark.parametrize("shape", [(), (1,), (2,)])
def test_enum_fields_reject_arrays(
    record: object, field: str, shape: tuple[int, ...]
) -> None:
    value = np.full(shape, getattr(record, field))
    with pytest.raises(InvalidSemiempiricalMethod, match="canonical string"):
        replace(record, **{field: value})


@pytest.mark.parametrize(("record", "field"), CASES)
def test_canonical_strings_preserve_payload_and_identity(
    record: object, field: str
) -> None:
    clone = replace(record, **{field: getattr(record, field)})
    assert clone.to_payload() == record.to_payload()
    assert canonical_hash(clone.to_payload()) == canonical_hash(record.to_payload())
    with pytest.raises(InvalidSemiempiricalMethod):
        replace(record, **{field: "unsupported-value"})
