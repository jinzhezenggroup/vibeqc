"""Tensor type/legality and data-only replay boundaries, without CUDA."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from tools.vibeqc_tensor import (
    Index,
    IndexSpace,
    Node,
    Program,
    Symmetry,
    TensorSpec,
    add,
    broadcast,
    constant,
    divide,
    einsum,
    execute,
    gather,
    input_tensor,
    multiply,
    reduce_sum,
    reshape,
    slice_tensor,
    transpose,
)
from tools.vibeqc_tensor.examples import example_cases
from tools.vibeqc_tensor.types import SPACE_KINDS


def tensor(name="x", *, size=3, kind="occupied", spin=None, role="input", **kwargs):
    space = IndexSpace(kind, kind, size, spin)
    return input_tensor(name, TensorSpec((Index("i", space),), role=role, **kwargs))


@pytest.mark.parametrize("kind", sorted(SPACE_KINDS))
@pytest.mark.parametrize("spin", [None, "alpha", "beta"])
def test_explicit_space_ranges_and_spin(kind, spin):
    space = IndexSpace("population", kind, 5, spin)
    index = Index("p", space, 1, 4)
    spec = TensorSpec((index,), role="parameter", differentiable=True)
    node = input_tensor("parameter", spec)
    replay = Program.loads(Program({"out": node}).dumps())
    assert replay.outputs["out"].spec == spec
    assert spec.shape == (3,)
    assert index.coordinate(1) == 2


@pytest.mark.parametrize("bad", [-1, True, 2.0, 1 << 63])
def test_invalid_dimensions_rejected_before_allocating(bad):
    with pytest.raises(ValueError):
        IndexSpace("o", "occupied", bad)


def test_size_product_overflow_and_scalar_empty_shapes():
    huge = IndexSpace("huge", "batch", 1 << 40)
    with pytest.raises(ValueError, match="byte count"):
        TensorSpec((Index("x", huge), Index("y", huge)))
    assert TensorSpec().shape == ()
    assert TensorSpec().size == 1
    assert tensor(size=0).spec.size == 0
    with pytest.raises(ValueError, match="budget"):
        execute(Program({"x": tensor()}), {"x": np.ones(3)}, max_bytes=1)


def test_exact_factors_and_dummy_alpha_renaming_have_stable_hashes():
    x = tensor()
    p = Program({"energy": einsum("i,i->", x, x, coefficient=Fraction(2, 8))})
    q = Program({"energy": einsum("p,p->", x, x, coefficient="1/4")})
    assert p.logical_hash == q.logical_hash
    assert p.dumps() == q.dumps()
    assert p.outputs["energy"].attrs["coefficient"] == (1, 4)
    for coefficient in (0.25, True, complex(1)):
        with pytest.raises(TypeError, match="coefficients"):
            add(x, coefficients=(coefficient,))
    with pytest.raises(ValueError, match="reduced"):
        Node("add", (x,), x.spec.result(), (("coefficients", ((2, 4),)),))
    with pytest.raises(ValueError, match="coefficient"):
        Node("add", (x,), x.spec.result(), (("coefficients", ((1, 0),)),))


def test_order_and_provenance_do_not_change_equation_identity():
    x, y = tensor("x"), tensor("y")
    a, b = multiply(x, y), add(x, y)
    provenance = {"versions": ["first"]}
    p = Program({"b": b, "a": a}, definitions=(b, a), provenance=provenance)
    q = Program(
        {"a": a, "b": b}, definitions=(a, b), provenance={"versions": ["second"]}
    )
    assert p.logical_hash == q.logical_hash
    assert tuple(p.debug_names.values()) == tuple(q.debug_names.values())
    provenance["versions"].append("changed")
    p.provenance["versions"].append("also changed")
    assert p.provenance == {"versions": ["first"]}
    with pytest.raises(TypeError):
        p.outputs["a"] = x


def test_hash_is_stable_across_python_hash_seeds():
    source = """
from tools.vibeqc_tensor.examples import example_cases
for case in example_cases():
    print(case.program.logical_hash)
"""
    outputs = [
        subprocess.check_output(
            [sys.executable, "-c", source],
            text=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        for seed in ("11", "97")
    ]
    assert outputs[0] == outputs[1]


def test_input_and_space_names_cannot_hide_semantic_conflicts():
    with pytest.raises(ValueError, match="input name"):
        Program({"a": tensor(), "b": tensor(representation="spin_orbital")})
    a = tensor("a", spin="alpha")
    b = tensor("b", spin="beta")
    with pytest.raises(ValueError, match="space name"):
        Program({"a": a, "b": b})
    with pytest.raises(ValueError, match="index spaces"):
        einsum("i,i->", tensor(), tensor("v", kind="virtual"))
    with pytest.raises(ValueError, match="index spaces"):
        einsum("i,i->", a, b)
    with pytest.raises(ValueError, match="representation"):
        multiply(tensor(), tensor("spin", representation="spin_orbital"))


@pytest.mark.parametrize(
    "factory",
    [
        lambda x: add(x, tensor(size=2)),
        lambda x: add(x, coefficients=(1, 2)),
        lambda x: multiply(x, tensor(kind="virtual")),
        lambda x: divide(x, tensor(dtype="float32")),
        lambda x: transpose(x, (1,)),
        lambda x: transpose(x, (0, 0)),
        lambda x: reshape(x, ()),
        lambda x: slice_tensor(x, ((-1, 2),)),
        lambda x: slice_tensor(x, ((2, 1),)),
        lambda x: slice_tensor(x, ((0, 4),)),
        lambda x: gather(x, 0, (3,)),
        lambda x: gather(x, 0, (-1,)),
        lambda x: gather(x, True, (0,)),
        lambda x: reduce_sum(x, (0, 0)),
        lambda x: broadcast(x, x.spec.indices, ()),
        lambda x: einsum("i", x),
        lambda x: einsum("...i->i", x),
        lambda x: einsum("i->j", x),
        lambda x: einsum("i->ii", x),
        lambda x: einsum("ij->i", x),
        lambda x: einsum("i,i->", x),
    ],
)
def test_illegal_primitives_fail_at_construction(factory):
    with pytest.raises(ValueError):
        factory(tensor())


def test_complex_conjugation_and_mutable_destinations_are_explicitly_unsupported():
    with pytest.raises(ValueError, match="real"):
        TensorSpec(dtype="complex128")
    x = tensor()
    for op in ("conjugate", "scatter_assign", "update", "while", "__import__"):
        with pytest.raises(ValueError, match="unsupported tensor primitive"):
            Node(op, (x,), x.spec.result())
    with pytest.raises(ValueError, match="attributes"):
        Node("multiply", (x, x), x.spec.result(), (("out", "x"),))


def test_runtime_input_contract_and_constant_validation():
    p = Program({"out": tensor()})
    for value in (np.ones(4), np.ones(3, dtype=np.float32), np.ones(3, dtype=complex)):
        with pytest.raises(ValueError, match="real dtype"):
            execute(p, {"x": value})
    with pytest.raises(ValueError, match="missing tensor"):
        execute(p, {})
    with pytest.raises(ValueError, match="non-finite"):
        execute(p, {"x": np.array([1.0, np.nan, 0.0])})
    with pytest.raises(ValueError, match="division by zero"):
        execute(
            Program({"out": divide(tensor("x"), tensor("y"))}),
            {"x": np.ones(3), "y": np.zeros(3)},
        )
    with pytest.raises(ValueError, match="constant length"):
        constant((1, 2))
    with pytest.raises(ValueError, match="constants"):
        TensorSpec(role="constant", differentiable=True)


@pytest.mark.parametrize("field", ["schema_version", "primitive_version"])
def test_incompatible_schema_versions_are_rejected(field):
    payload = example_cases()[0].program.to_payload()
    payload[field] = 99
    with pytest.raises(ValueError, match="version"):
        Program.from_payload(payload)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(logical_hash="0" * 64),
        lambda p: p["conventions"].update(aliasing="in-place"),
        lambda p: p["nodes"][-1].update(op="conjugate"),
        lambda p: p["nodes"][-1]["spec"].update(dtype="complex128"),
        lambda p: p["nodes"][-1]["inputs"].append("missing"),
        lambda p: p["nodes"].reverse(),
        lambda p: p["nodes"].append(p["nodes"][0]),
        lambda p: p["nodes"][0].update(id="unverified_debug_name"),
        lambda p: p["nodes"][-1]["attributes"].update(output=[0, 0]),
        lambda p: p["nodes"][-1]["spec"]["indices"][0].update(stop=1),
        lambda p: p.update(unknown_field="silently ignored"),
    ],
)
def test_corrupted_equations_cannot_be_replayed(mutation):
    payload = json.loads(example_cases()[0].program.dumps())
    mutation(payload)
    with pytest.raises(ValueError):
        Program.from_payload(payload)


def test_duplicate_json_fields_and_executable_strings_are_data_only(tmp_path):
    with pytest.raises(ValueError, match="duplicate JSON"):
        Program.loads('{"schema": "one", "schema": "two"}')
    marker = tmp_path / "should-not-exist"
    payload = json.loads(example_cases()[0].program.dumps())
    payload["nodes"][-1]["op"] = f"__import__('pathlib').Path('{marker}').touch()"
    with pytest.raises(ValueError, match="unsupported"):
        Program.from_payload(payload)
    assert not marker.exists()


def test_symmetry_declarations_must_preserve_populations():
    o, v = IndexSpace("o", "occupied", 2), IndexSpace("v", "virtual", 2)
    with pytest.raises(ValueError, match="different index domains"):
        TensorSpec((Index("i", o), Index("a", v)), symmetries=(Symmetry((1, 0)),))
    with pytest.raises(ValueError, match="unique"):
        TensorSpec((Index("i", o), Index("i", v)))
    with pytest.raises(ValueError, match="declared transpose result"):
        x = input_tensor("x", TensorSpec((Index("i", o), Index("j", o)), role="input"))
        Node(
            "transpose",
            (x,),
            replace(x.spec.result(), symmetries=(Symmetry((1, 0)),)),
            (("axes", (0, 1)),),
        )
