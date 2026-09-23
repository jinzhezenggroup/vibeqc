"""Opt-in bulk representation must not rekey existing mathematical identities."""

from __future__ import annotations

from fractions import Fraction

import pytest
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.method.spec import SemilocalXCPrimitive, resolve_method
from vibeqc_compiler.xc.semilocal_codegen import build_roots, emit_polarized_semilocal
from vibeqc_compiler.xc.spec import FunctionalSpec, UnsupportedXC, functional


@pytest.mark.parametrize("name", ("LDA_XC_PW", "PBE", "B3LYP", "R2SCAN"))
@pytest.mark.parametrize("production", (False, True))
def test_existing_expression_identity_preserves_established_payload(
    name: str, production: bool
) -> None:
    spec = (
        next(
            item.functional
            for item in resolve_method(name, spin="polarized").primitives
            if isinstance(item, SemilocalXCPrimitive)
        )
        if name == "B3LYP"
        else functional(name, spin="polarized")
    )
    outputs = ((), (0,), (0, 0))
    graph, roots, identity = build_roots(spec, outputs, production=production)
    reachable = graph.topological_order(roots)
    indices = {node: index for index, node in enumerate(reachable)}
    # Established pre-bridge serialization. Future changes to actual equations
    # still change the nodes/spec; opting into a new consumer must not change
    # this legacy payload for an ordinary call.
    expected = canonical_hash(
        {
            "spec": spec.to_payload(),
            "outputs": outputs,
            "optimization": "after",
            "nodes": [
                (
                    graph.nodes[node].operation,
                    [indices[arg] for arg in graph.nodes[node].arguments],
                    str(graph.nodes[node].payload),
                )
                for node in reachable
            ],
            "roots": [indices[root.identifier] for root in roots],
        }
    )
    assert identity == expected


@pytest.mark.parametrize("name", ("LDA_C_VWN_4", "GGA_X_PBE_SOL"))
def test_bulk_opt_in_stays_distinct_and_cannot_claim_production(name: str) -> None:
    spec = functional(name, spin="polarized")
    options = {
        "value_type": "BulkValue",
        "function_name": "bulk_point",
        "identity_constant": "kBulkIdentity",
        "pointwise_bulk": True,
    }
    cpu = emit_polarized_semilocal(spec, **options)
    cuda = emit_polarized_semilocal(
        spec, function_qualifier="__device__ inline", **options
    )
    assert (
        cpu.replace(
            "inline BulkValue bulk_point", "__device__ inline BulkValue bulk_point"
        )
        == cuda
    )
    with pytest.raises(ValueError, match="cannot claim production"):
        emit_polarized_semilocal(spec, production=True, **options)
    with pytest.raises(UnsupportedXC, match="not production-domain admitted"):
        build_roots(spec, ((),))


def test_bulk_bridge_rejects_nonunit_weight_and_tau_projection() -> None:
    weighted = FunctionalSpec(
        "weighted-bulk", (("GGA_X_PBE_SOL", Fraction(2)),), spin="polarized"
    )
    with pytest.raises(UnsupportedXC, match="unit-weight"):
        build_roots(weighted, ((),), pointwise_bulk=True)
    with pytest.raises(UnsupportedXC, match="tau/laplacian projection"):
        build_roots(
            functional("MGGA_X_R2SCAN01", spin="polarized"),
            ((),),
            pointwise_bulk=True,
        )
