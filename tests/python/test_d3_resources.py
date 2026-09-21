"""Static resource accounting for the production D3(BJ) owner."""

from vibeqc.resources_d3 import d3_resource_request
from vibeqc_compiler.method import resolve_method


def _decisions(backend: str) -> dict[str, str]:
    graph = resolve_method("PBE-D3(BJ)", spin="unpolarized")
    request = d3_resource_request(
        ((1, 1),),
        method=graph,
        backend=backend,
    )
    assert request.unsupported_reason is None
    assert request.infeasible_reason is None
    assert len(request.candidates) == 1
    return dict(request.candidates[0].decisions)


def test_d3_cpu_resource_formula_matches_native_layout() -> None:
    decisions = _decisions("cpu")

    assert int(decisions["plan_host_bytes"]) == 64
    assert int(decisions["execution_host_bytes"]) == 414
    assert int(decisions["device_bytes"]) == 0
    assert int(decisions["workspace_bytes"]) == 256
    assert int(decisions["table_bytes"]) == 291456


def test_d3_cuda_resource_formula_matches_native_layout() -> None:
    decisions = _decisions("cuda")

    assert int(decisions["plan_host_bytes"]) == 64
    assert int(decisions["execution_host_bytes"]) == 110
    assert int(decisions["device_bytes"]) == 291838
    assert int(decisions["workspace_bytes"]) == 256
    assert int(decisions["table_bytes"]) == 291456


def test_d3_resource_request_fails_closed_outside_element_domain() -> None:
    graph = resolve_method("PBE-D3(BJ)", spin="unpolarized")
    request = d3_resource_request(
        ((1, 87),),
        method=graph,
        backend="cpu",
    )

    assert request.candidates == ()
    assert request.unsupported_reason == (
        "production D3(BJ) supports atomic numbers 1 through 86"
    )


def test_d3_resource_request_applies_local_retained_cap() -> None:
    graph = resolve_method("PBE-D3(BJ)", spin="unpolarized")
    request = d3_resource_request(
        ((1, 1),),
        method=graph,
        backend="cpu",
        maximum_bytes=477,
    )

    assert request.candidates == ()
    assert request.infeasible_reason is not None
    assert "requires 478 bytes" in request.infeasible_reason
