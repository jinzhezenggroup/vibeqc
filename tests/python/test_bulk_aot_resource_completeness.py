"""Incomplete PTXAS function records must not disappear from eligibility."""

import pytest
from vibeqc_compiler.xc import bulk_aot

GOOD = (
    "ptxas info : Function properties for bulk_xc_census_probe\n"
    "    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
    "ptxas info : Used 32 registers, 0 bytes lmem\n"
)
INCOMPLETE = (
    (
        "ptxas info : Function properties for expensive_kernel\n"
        "ptxas info : Used 240 registers, 0 bytes lmem\n"
    ),
    (
        "ptxas info : Function properties for expensive_kernel\n"
        "    4096 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
    ),
    "ptxas info : Function properties for bulk_xc_census_probe\n",
)


@pytest.mark.parametrize("partial", INCOMPLETE)
@pytest.mark.parametrize("prepend", (False, True))
def test_incomplete_function_cannot_be_ignored(partial: str, prepend: bool) -> None:
    log = partial + GOOD if prepend else GOOD + partial
    resources = bulk_aot.ptxas_resources(log)
    gate = bulk_aot.gate_cuda_resources(
        {"status": "compiled", "resources": resources},
        bulk_aot.CudaResourceLimits(64, 0, 0, 0, 0),
    )
    assert gate["status"] == "unavailable"
    assert not gate["package_eligible"]


def test_complete_zero_shared_record_is_eligible() -> None:
    resources = bulk_aot.ptxas_resources(GOOD)
    assert resources["shared_bytes"] == 0
    gate = bulk_aot.gate_cuda_resources(
        {"status": "compiled", "resources": resources},
        bulk_aot.CudaResourceLimits(32, 0, 0, 0, 0),
    )
    assert gate["status"] == "passed"
    assert gate["package_eligible"]


def test_complete_expensive_function_still_exceeds_limit() -> None:
    log = GOOD + GOOD.replace("bulk_xc_census_probe", "expensive_kernel").replace(
        "32 registers", "240 registers"
    )
    gate = bulk_aot.gate_cuda_resources(
        {"status": "compiled", "resources": bulk_aot.ptxas_resources(log)},
        bulk_aot.CudaResourceLimits(64, 0, 0, 0, 0),
    )
    assert gate["status"] == "rejected"
    assert gate["reasons"] == ["register-limit"]


def test_missing_local_memory_is_not_borrowed_from_another_function() -> None:
    log = GOOD + GOOD.replace("bulk_xc_census_probe", "missing_local").replace(
        ", 0 bytes lmem", ""
    )
    resources = bulk_aot.ptxas_resources(log)
    assert resources["local_bytes"] is None
    gate = bulk_aot.gate_cuda_resources(
        {"status": "compiled", "resources": resources},
        bulk_aot.CudaResourceLimits(64, 0, 0, 0, 0),
    )
    assert not gate["package_eligible"]


def test_invalid_arch_is_rejected_before_compiler_lookup() -> None:
    variant = bulk_aot.SourceVariant(
        name="test",
        family="LDA",
        spin="unpolarized",
        import_identity="test",
        domain="test",
        features=("rho",),
        derivative_order=1,
        backend="cpu",
        source="void bulk_xc_point(const double* x, double* y) { y[0] = x[0]; }",
        energy_nodes=1,
        ssa={},
    )
    with pytest.raises(ValueError, match="cuda_arch"):
        bulk_aot.compile_probe(variant, cuda_arch="native")
