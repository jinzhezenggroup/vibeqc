"""Audited RCCSD doubles inventory and expanded/shared TensorIR frontends.

Adapted from PySCF 2.14.0 rccsd.update_amps and rintermediates (Apache-2.0,
Copyright 2014-2021 The PySCF Developers). See NOTICE/source_manifest.json.
All Fock diagonals are retained; no denominator or level shift occurs here.
"""

from fractions import Fraction
from itertools import product
from string import ascii_letters

from tools.vibeqc_tensor import (
    Index,
    Program,
    Symmetry,
    TensorSpec,
    add,
    einsum,
    input_tensor,
    optimize,
)
from tools.vibeqc_validation.schema import canonical_hash

from .equations import build_program

# Each definition is an ordered rational sum of (factor, einsum, operands).
# Intermediates follow upstream cc_F*, L*, cc_W*; D01-D10 are residual groups.
DEFINITIONS = {
    "Foo": (
        (1, "ki->ki", ("foo",)),
        (2, "kcld,ilcd->ki", ("ovov", "t2")),
        (-1, "kdlc,ilcd->ki", ("ovov", "t2")),
        (2, "kcld,ic,ld->ki", ("ovov", "t1", "t1")),
        (-1, "kdlc,ic,ld->ki", ("ovov", "t1", "t1")),
    ),
    "Fvv": (
        (1, "ac->ac", ("fvv",)),
        (-2, "kcld,klad->ac", ("ovov", "t2")),
        (1, "kdlc,klad->ac", ("ovov", "t2")),
        (-2, "kcld,ka,ld->ac", ("ovov", "t1", "t1")),
        (1, "kdlc,ka,ld->ac", ("ovov", "t1", "t1")),
    ),
    "Loo": (
        (1, "ki->ki", ("Foo",)),
        (1, "kc,ic->ki", ("fov", "t1")),
        (2, "lcki,lc->ki", ("ovoo", "t1")),
        (-1, "kcli,lc->ki", ("ovoo", "t1")),
    ),
    "Lvv": (
        (1, "ac->ac", ("Fvv",)),
        (-1, "kc,ka->ac", ("fov", "t1")),
        (2, "kdac,kd->ac", ("ovvv", "t1")),
        (-1, "kcad,kd->ac", ("ovvv", "t1")),
    ),
    "Woooo": (
        (1, "kilj->klij", ("oooo",)),
        (1, "lcki,jc->klij", ("ovoo", "t1")),
        (1, "kclj,ic->klij", ("ovoo", "t1")),
        (1, "kcld,ijcd->klij", ("ovov", "t2")),
        (1, "kcld,ic,jd->klij", ("ovov", "t1", "t1")),
    ),
    "Wvvvv": (
        (1, "acbd->abcd", ("vvvv",)),
        (-1, "kdac,kb->abcd", ("ovvv", "t1")),
        (-1, "kcbd,ka->abcd", ("ovvv", "t1")),
    ),
    "Wvoov": (
        (1, "kcad,id->akic", ("ovvv", "t1")),
        (-1, "kcli,la->akic", ("ovoo", "t1")),
        (1, "kcai->akic", ("ovvo",)),
        ("-1/2", "ldkc,ilda->akic", ("ovov", "t2")),
        ("-1/2", "lckd,ilad->akic", ("ovov", "t2")),
        (-1, "ldkc,id,la->akic", ("ovov", "t1", "t1")),
        (1, "ldkc,ilad->akic", ("ovov", "t2")),
    ),
    "Wvovo": (
        (1, "kdac,id->akci", ("ovvv", "t1")),
        (-1, "lcki,la->akci", ("ovoo", "t1")),
        (1, "kiac->akci", ("oovv",)),
        ("-1/2", "lckd,ilda->akci", ("ovov", "t2")),
        (-1, "lckd,id,la->akci", ("ovov", "t1", "t1")),
    ),
    "tau": ((1, "ijab->ijab", ("t2",)), (1, "ia,jb->ijab", ("t1", "t1"))),
    "Xv": (
        (-1, "kibc,ka,jc->ijab", ("oovv", "t1", "t1")),
        (1, "iacb,jc->ijab", ("ovvv", "t1")),
    ),
    "Xo": (
        (1, "kcai,jc,kb->ijab", ("ovvo", "t1", "t1")),
        (1, "iajk,kb->ijab", ("ovoo", "t1")),
    ),
    "Xfvv": ((1, "ac,ijcb->ijab", ("Lvv", "t2")),),
    "Xfoo": ((1, "ki,kjab->ijab", ("Loo", "t2")),),
    "Xring": (
        (2, "akic,kjcb->ijab", ("Wvoov", "t2")),
        (-1, "akci,kjcb->ijab", ("Wvovo", "t2")),
    ),
    "Xexchange": ((1, "akic,kjbc->ijab", ("Wvoov", "t2")),),
    "Xcross": ((1, "bkci,kjac->ijab", ("Wvovo", "t2")),),
    "D01_driving": ((1, "iajb->ijab", ("ovov",)),),
    "D02_singles_v": ((1, "ijab->ijab", ("Xv",)), (1, "jiba->ijab", ("Xv",))),
    "D03_singles_o": ((-1, "ijab->ijab", ("Xo",)), (-1, "jiba->ijab", ("Xo",))),
    "D04_oo_ladder": ((1, "klij,klab->ijab", ("Woooo", "tau")),),
    "D05_vv_ladder": ((1, "abcd,ijcd->ijab", ("Wvvvv", "tau")),),
    "D06_virtual": ((1, "ijab->ijab", ("Xfvv",)), (1, "jiba->ijab", ("Xfvv",))),
    "D07_occupied": ((-1, "ijab->ijab", ("Xfoo",)), (-1, "jiba->ijab", ("Xfoo",))),
    "D08_ring": ((1, "ijab->ijab", ("Xring",)), (1, "jiba->ijab", ("Xring",))),
    "D09_exchange": (
        (-1, "ijab->ijab", ("Xexchange",)),
        (-1, "jiba->ijab", ("Xexchange",)),
    ),
    "D10_cross": ((-1, "ijab->ijab", ("Xcross",)), (-1, "jiba->ijab", ("Xcross",))),
}


def _expand(equation, children, coefficient):
    """Distribute products of sums and alpha-rename every captured dummy index.

    A polynomial term is (factor, output labels, ((input name, labels),...)).
    This only expands algebra; it performs no numerical evaluation.
    """
    left, output = equation.split("->")
    result = []
    for terms in product(*children):
        unused = iter(c for c in ascii_letters if c not in equation)
        operands = []
        factor = Fraction(coefficient)
        for assigned, (scale, child_output, child_operands) in zip(
            left.split(","), terms
        ):
            mapping = dict(zip(child_output, assigned))
            factor *= scale
            for name, labels in child_operands:
                for label in labels:
                    if label not in mapping:
                        mapping[label] = next(unused)
                operands.append((name, "".join(mapping[c] for c in labels)))
        result.append((factor, output, tuple(operands)))
    return result


def build_ccsd_program(nocc, nvir, *, form="shared", diagnostics=True):
    """Complete energy/R1/R2; expanded oracle DAG or shared/CSE CPU DAG.

    R2 is the opposite-spin alpha-beta double determinant projection. This
    fixes its spatial normalization independently of same-spin antisymmetry.
    """
    if form not in ("expanded", "shared", "optimized"):
        raise ValueError("unknown RCCSD equation form")
    singles = build_program(nocc, nvir)
    inputs = {n.attrs["name"]: n for n in singles.live_nodes if n.op == "input"}
    for name, space in (
        ("oooo", inputs["t1"].spec.indices[0].space),
        ("vvvv", inputs["t1"].spec.indices[1].space),
    ):
        inputs[name] = input_tensor(
            name,
            TensorSpec(
                tuple(Index(c, space) for c in "pqrs"),
                role="parameter",
                differentiable=True,
                representation="restricted_spatial",
                symmetries=tuple(
                    Symmetry(p) for p in ((1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1))
                ),
            ),
        )
    nodes = dict(inputs)
    polynomials = {
        k: [
            (
                Fraction(1),
                ascii_letters[: len(n.spec.shape)],
                ((k, ascii_letters[: len(n.spec.shape)]),),
            )
        ]
        for k, n in inputs.items()
    }
    for name, definition in DEFINITIONS.items():
        if form == "expanded":
            terms = [
                t
                for coefficient, equation, args in definition
                for t in _expand(equation, [polynomials[a] for a in args], coefficient)
            ]
            polynomials[name] = terms
            nodes[name] = add(
                *(
                    einsum(
                        ",".join(labels for _, labels in args) + "->" + out,
                        *(inputs[a] for a, _ in args),
                        coefficient=scale,
                    )
                    for scale, out, args in terms
                )
            )
        else:
            nodes[name] = add(
                *(
                    einsum(e, *(nodes[a] for a in args), coefficient=c)
                    for c, e, args in definition
                )
            )
    outputs = (
        dict(singles.outputs)
        if diagnostics
        else {k: singles.outputs[k] for k in ("correlation_energy", "singles_residual")}
    )
    if diagnostics:
        outputs.update({k: nodes[k] for k in DEFINITIONS})
    outputs["doubles_residual"] = add(
        *(nodes[k] for k in DEFINITIONS if k.startswith("D"))
    )
    program = Program(
        outputs,
        provenance={
            "method": "RCCSD",
            "slice": "B",
            "form": form,
            "inventory_version": 1,
            "doubles_inventory_hash": canonical_hash(DEFINITIONS),
            "source": "PySCF 2.14.0 rccsd/rintermediates; source_manifest.json",
            "projector": "<Phi_i_alpha,j_beta^a_alpha,b_beta|exp(-T) H_N exp(T)|Phi>",
        },
    )
    return optimize(program) if form == "optimized" else program
