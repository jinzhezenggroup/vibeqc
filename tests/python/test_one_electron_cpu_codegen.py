"""Regression gates for the #351 production CPU S/T lowering."""

from vibeqc_compiler.integral.one_electron_cpu import (
    emit_one_electron_st_cpu,
    one_electron_st_cpu_inventory,
)


def test_cpu_st_inventory_is_shared_ir_and_bounded_to_spdf():
    inventory = one_electron_st_cpu_inventory()
    assert inventory["schema"] == "vibeqc.one_electron_st_cpu"
    assert inventory["version"] == 1
    assert inventory["precision"] == "fp64"
    assert inventory["maximum_angular"] == 3
    assert inventory["families"] == ["overlap", "kinetic"]
    assert len(inventory["values"]) == 32
    assert len(inventory["derivatives"]) == 32


def test_cpu_st_emitter_is_deterministic_host_code():
    first = emit_one_electron_st_cpu()
    second = emit_one_electron_st_cpu()
    assert first == second
    assert "namespace vibeqc::integrals::generated_one_electron_cpu" in first
    assert "overlap_kinetic_gradient" in first
    assert "__device__" not in first
    assert "__forceinline__" not in first
    assert "__noinline__" not in first
