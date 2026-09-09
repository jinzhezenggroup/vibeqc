"""Lower the audited spatial inventory into the existing typed TensorIR."""

from collections import defaultdict

from tools.vibeqc_tensor import (
    Index,
    IndexSpace,
    PackedLayout,
    Program,
    Symmetry,
    TensorSpec,
    add,
    einsum,
    input_tensor,
)
from tools.vibeqc_validation.schema import canonical_hash

from .inventory import TERMS

VERSION = 1
BLOCKS = ("ovov", "ovvo", "oovv", "ovvv", "ovoo")


def amplitude_specs(nocc, nvir):
    """Dense spatial coordinates; doubles have only simultaneous pair exchange."""
    if any(type(n) is not int or n < 1 for n in (nocc, nvir)):
        raise ValueError("RCCSD requires nonempty occupied and virtual spaces")
    o = IndexSpace("occupied", "occupied", nocc)
    v = IndexSpace("virtual", "virtual", nvir)
    i, j, a, b = Index("i", o), Index("j", o), Index("a", v), Index("b", v)
    common = {
        "representation": "restricted_spatial",
        "role": "parameter",
        "differentiable": True,
    }
    return (
        TensorSpec((i, a), **common),
        TensorSpec((i, j, a, b), symmetries=(Symmetry((1, 0, 3, 2)),), **common),
    )


def amplitude_layouts(nocc, nvir):
    """Reuse #145 orbit ordering and multiplicity-weighted dense inner product."""
    return tuple(PackedLayout.from_spec(s) for s in amplitude_specs(nocc, nvir))


def build_program(nocc, nvir):
    """Unfactorized reference DAG with every term and homogeneous group exposed."""
    s1, s2 = amplitude_specs(nocc, nvir)
    spaces = {"o": s1.indices[0].space, "v": s1.indices[1].space}
    nodes = {"t1": input_tensor("t1", s1), "t2": input_tensor("t2", s2)}
    for name in ("foo", "fov", "fvv", *BLOCKS):
        axes = name.removeprefix("f")
        permutations = {
            "foo": ((1, 0),),
            "fvv": ((1, 0),),
            "ovov": ((2, 3, 0, 1),),
            "ovvo": ((3, 2, 1, 0),),
            "oovv": ((1, 0, 2, 3), (0, 1, 3, 2)),
            "ovvv": ((0, 1, 3, 2),),
            "ovoo": ((0, 1, 3, 2),),
        }
        nodes[name] = input_tensor(
            name,
            TensorSpec(
                tuple(Index(f"p{k}", spaces[x]) for k, x in enumerate(axes)),
                symmetries=tuple(Symmetry(p) for p in permutations.get(name, ())),
                representation="restricted_spatial",
                role="parameter",
                differentiable=True,
            ),
        )
    groups = defaultdict(list)
    outputs = {}
    for term_id, group, coefficient, equation, operands in TERMS:
        term = einsum(equation, *(nodes[x] for x in operands), coefficient=coefficient)
        outputs[term_id] = term
        groups[group].append(term)
    outputs.update({key: add(*terms) for key, terms in groups.items()})
    outputs["correlation_energy"] = add(
        *(outputs[k] for k in groups if k.startswith("energy_"))
    )
    outputs["singles_residual"] = add(
        *(outputs[k] for k in groups if k.startswith("singles_"))
    )
    return Program(
        outputs,
        provenance={
            "method": "real all-electron conventional RCCSD",
            "slice": "A",
            "inventory_version": VERSION,
            "inventory_hash": canonical_hash(TERMS),
            "source": "PySCF 2.14.0 rccsd + rintermediates; see source_manifest.json",
            "residual": "<Phi_i_alpha^a_alpha|exp(-T) H_N exp(T)|Phi>",
        },
    )
