"""B3LYP uses the shared KS execution-plan family, including semantic aliases."""

from dataclasses import replace

import pytest
from vibeqc.ks import _native_semilocal_family, ks_coefficients
from vibeqc_compiler.method import compile_ks_execution_plan, resolve_method


@pytest.mark.parametrize("spin,exchange", (("unpolarized", -0.1), ("polarized", -0.2)))
def test_b3lyp_family_and_exchange_come_from_the_resolved_graph(
    spin: str, exchange: float
) -> None:
    method = resolve_method("B3LYP", spin=spin)
    alias = replace(method, identifier="review-b3-family-alias")
    for graph in (method, alias):
        assert _native_semilocal_family(graph) == 3
        assert ks_coefficients(graph) == (1.0, 1.0, exchange)
        assert compile_ks_execution_plan(graph).required_lowerers == (
            "semilocal-xc",
            "full-range-exchange",
        )


def test_b3lyp_family_does_not_admit_range_separated_exchange() -> None:
    with pytest.raises(NotImplementedError, match="short-range-exchange"):
        _native_semilocal_family(resolve_method("CAM-B3LYP"))
