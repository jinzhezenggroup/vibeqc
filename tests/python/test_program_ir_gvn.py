"""Post-transform ProgramIR value-numbering coverage for #830."""

from dataclasses import replace

from vibeqc_compiler.common.liveness import EffectKind
from vibeqc_compiler.common.program import PlanCall, ProgramBuffer, ProgramIR
from vibeqc_compiler.common.program_optimize import cleanup_program_ir
from vibeqc_compiler.xc import functional
from vibeqc_compiler.xc.contractions import ContractionProgram
from vibeqc_compiler.xc.program_ir import fixed_density_tile_program


def _redundant_program(*, outputs: tuple[str, ...] = ("out",)) -> ProgramIR:
    buffers = (
        ProgramBuffer("x", 8),
        ProgramBuffer("a", 8),
        ProgramBuffer("b", 8),
        ProgramBuffer("out", 8),
    )
    calls = (
        PlanCall("a", "provider.calc", "calc-v1", ("x",), ("a",)),
        PlanCall("b", "provider.calc", "calc-v1", ("x",), ("b",)),
        PlanCall("finish", "provider.finish", "finish-v1", ("a", "b"), ("out",)),
    )
    return ProgramIR("redundant", buffers, ("x",), calls, outputs)


def test_program_gvn_eliminates_proven_pure_internal_duplicate() -> None:
    result = cleanup_program_ir(
        _redundant_program(),
        effects={"provider.calc": EffectKind.PURE},
    )
    assert tuple(call.name for call in result.program.calls) == ("a", "finish")
    assert result.program.calls[-1].reads == ("a", "a")
    assert tuple(buffer.name for buffer in result.program.buffers) == (
        "x",
        "a",
        "out",
    )
    assert result.eliminated_calls == ("b",)
    assert result.eliminated_buffers == ("b",)
    assert result.value_numbering.reused_values == 1
    assert result.value_numbering.effect_barriers == 1
    assert result.records[0].name == "effect_aware_gvn"
    assert result.records[0].changed
    payload = result.to_payload()
    assert payload["calls_before"] == 3
    assert payload["calls_after"] == 2
    assert payload["buffers_before"] == 4
    assert payload["buffers_after"] == 3
    assert payload["before_identity"] != payload["after_identity"]
    assert payload["effect_policy"] == {"provider.calc": "pure"}


def test_program_gvn_fails_closed_for_opaque_provider() -> None:
    original = _redundant_program()
    result = cleanup_program_ir(original, effects={})
    assert result.program is original
    assert not result.eliminated_calls
    assert result.value_numbering.reused_values == 0
    assert result.value_numbering.effect_barriers == len(original.calls)
    assert not result.records[0].changed


def test_program_gvn_preserves_exported_buffer_abi() -> None:
    program = ProgramIR(
        "exported-duplicates",
        (
            ProgramBuffer("x", 8),
            ProgramBuffer("a", 8),
            ProgramBuffer("b", 8),
        ),
        ("x",),
        (
            PlanCall("a", "provider.calc", "calc-v1", ("x",), ("a",)),
            PlanCall("b", "provider.calc", "calc-v1", ("x",), ("b",)),
        ),
        ("a", "b"),
    )
    result = cleanup_program_ir(
        program,
        effects={"provider.calc": EffectKind.PURE},
    )
    assert result.program is program
    assert result.value_numbering.effect_barriers == 2
    assert not result.eliminated_calls


def _pbe_tile_program() -> ProgramIR:
    contract = ContractionProgram(functional("PBE", spin="polarized")).contract
    return fixed_density_tile_program(
        contract,
        nao=12,
        tile_points=7,
        basis_bytes=100,
        grid_bytes=1000,
        basis_identity="basis-v1",
        native_identity="native-v1",
    )


def test_post_transform_cleanup_restores_real_pbe_tile_program() -> None:
    """A scheduling clone of pure AO collocation is eliminated before lowering."""

    baseline = _pbe_tile_program()
    jets = next(buffer for buffer in baseline.buffers if buffer.name == "jets")
    collocation, xc = baseline.calls
    transformed = ProgramIR(
        baseline.name,
        (*baseline.buffers, replace(jets, name="jets_clone")),
        baseline.inputs,
        (
            collocation,
            replace(
                collocation,
                name="collocation_clone",
                writes=("jets_clone",),
            ),
            replace(
                xc,
                reads=("jets_clone", "density", "quadrature"),
            ),
        ),
        baseline.outputs,
    )
    result = cleanup_program_ir(
        transformed,
        effects={"dft.NativeAO.evaluate": EffectKind.PURE},
    )
    assert result.program == baseline
    assert result.eliminated_calls == ("collocation_clone",)
    assert result.eliminated_buffers == ("jets_clone",)
    assert result.value_numbering.reused_values == 1
    assert result.value_numbering.effect_barriers == 1
