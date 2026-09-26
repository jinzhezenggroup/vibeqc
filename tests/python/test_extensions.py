"""Public programmable extension contracts for #558."""

import typing
from fractions import Fraction

import pytest
from vibeqc.extensions import API_VERSION, method, tensor, xc
from vibeqc_compiler.method import resolve_method


def test_custom_hybrid_and_builtin_share_canonical_method_ir() -> None:
    functional = xc.compose(
        "my-pbe0",
        {"GGA_X_PBE": "3/4", "GGA_C_PBE": 1},
        exact_exchange="1/4",
    )
    custom_ir = method.compose("my-pbe0", xc=functional)
    builtin_ir = method.named("PBE0")

    assert custom_ir.identity == builtin_ir.identity
    assert custom_ir.manifest_identity != builtin_ir.manifest_identity
    assert custom_ir.identity == resolve_method("PBE0").identity
    assert method.inspect(custom_ir)["extension_api_version"] == API_VERSION


def test_public_builders_are_exact_and_fail_closed() -> None:
    with pytest.raises(TypeError, match="exact integer"):
        xc.compose("bad", {"GGA_X_PBE": 0.75})

    ranged = xc.compose(
        "ranged",
        {"GGA_X_PBE": 1},
        range_omega="1/5",
        long_range_exchange="1/4",
    )
    with pytest.raises(method.UnsupportedMethod, match="range-separated"):
        method.compose("ranged", xc=ranged)

    inherited = xc.compose(
        "hybrid",
        {"GGA_X_PBE": "3/4", "GGA_C_PBE": 1},
        exact_exchange="1/4",
    )
    with pytest.raises(method.UnsupportedMethod, match="conflicts"):
        method.compose("conflict", xc=inherited, exact_exchange="1/5")


def test_xc_spin_is_preserved_by_public_method_composition() -> None:
    polarized = xc.compose(
        "polarized-pbe",
        {"GGA_X_PBE": 1, "GGA_C_PBE": 1},
        spin="polarized",
    )
    ir = method.compose("polarized-pbe", xc=polarized)
    assert ir.spin == "polarized"
    assert method.resolve(ir).spin == "polarized"
    with pytest.raises(method.UnsupportedMethod, match="conflicts with XC spin"):
        method.compose("bad-spin", xc=polarized, spin="unpolarized")


def test_component_builder_canonicalizes_fragment_order_and_cancellation() -> None:
    a = xc.compose(
        "a",
        [
            ("GGA_X_PBE", Fraction(1, 2)),
            ("GGA_C_PBE", 1),
            ("GGA_X_PBE", Fraction(1, 4)),
        ],
    )
    b = xc.compose(
        "b",
        {"GGA_X_PBE": "3/4", "GGA_C_PBE": 1},
    )
    assert a.components == b.components
    assert method.compose("a", xc=a).identity == method.compose("b", xc=b).identity


def test_public_method_typecheck_reuses_compiler_capability_contract() -> None:
    custom = method.compose(
        "custom",
        semilocal_components={"GGA_X_PBE": "3/4", "GGA_C_PBE": 1},
        exact_exchange="1/4",
    )
    capability = method.BackendCapability(
        backend="test-cuda",
        dtypes=("float64",),
        spins=("unpolarized",),
        derivative_orders=(0,),
        ingredients=("rho", "sigma"),
        operators=("semilocal-xc", "full-range-exchange"),
    )
    typed = method.verify(custom, capability=capability)
    assert typed.method.identity == method.named("PBE0").identity
    assert typed.backend == "test-cuda"


def test_tensor_extension_surface_is_replayable_and_backend_neutral() -> None:
    space = tensor.IndexSpace("ao", "ao", 2)
    index = tensor.Index("i", space)
    spec = tensor.TensorSpec((index,), role="input")
    node = tensor.input_tensor("x", spec)
    program = tensor.Program({"value": node})

    report = tensor.inspect(program)
    assert report["extension_api_version"] == API_VERSION
    assert report["logical_hash"] == program.logical_hash
    assert report["program"]["schema_version"] == tensor.SCHEMA_VERSION
    assert tensor.Program.loads(program.dumps()).logical_hash == program.logical_hash


@pytest.mark.parametrize("spin", ["unpolarized", "polarized"])
@pytest.mark.parametrize(
    "components",
    [
        {},
        (),
        (("GGA_X_PBE", 1), ("GGA_X_PBE", -1)),
        {"GGA_X_PBE": 0},
    ],
    ids=["empty-mapping", "empty-sequence", "cancelled", "zero-weight"],
)
def test_empty_semilocal_contribution_preserves_exact_exchange(
    components: typing.Any, spin: typing.Any
) -> None:
    ir = method.compose(
        "pure-exchange",
        semilocal_components=components,
        exact_exchange=1,
        spin=spin,
    )
    expected = method.compose("pure-exchange", exact_exchange=1, spin=spin)
    assert ir.identity == expected.identity
    assert ir.requirements["operators"] == ("full-range-exchange",)


@pytest.mark.parametrize(
    "components",
    [
        {},
        (),
        (("GGA_X_PBE", 1), ("GGA_X_PBE", -1)),
        {"GGA_X_PBE": 0},
    ],
    ids=["empty-mapping", "empty-sequence", "cancelled", "zero-weight"],
)
def test_empty_method_and_standalone_xc_still_fail_closed(
    components: typing.Any,
) -> None:
    with pytest.raises(method.UnsupportedMethod, match="empty"):
        method.compose("empty", semilocal_components=components)
    with pytest.raises(xc.UnsupportedXC, match="empty"):
        xc.compose("empty", components)
