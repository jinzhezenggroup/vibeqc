"""Reduce auxiliary-dependent intermediates before their RCCSD T2 consumers.

All coefficients, permutations and intermediate definitions come from the
existing shared RCCSD inventory. The expanded one-Q residual in df_equations
remains the independently derived fallback/replay path. This decomposition
changes contraction scheduling, not the fitted Hamiltonian or CCSD equations.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from string import ascii_letters

from generativeqc_compiler.tensor import (
    Index,
    Node,
    Program,
    Symmetry,
    TensorSpec,
    add,
    einsum,
    input_tensor,
    optimize,
    reassociate_einsums,
)

from .doubles import _expand, build_ccsd_program

# These shared intermediates contain virtual-Hamiltonian terms but no more than
# two virtual axes. Their expensive T2 consumers belong after the complete Q sum.
HOISTED_INTERMEDIATES = ("Lvv", "Wvoov", "Wvovo", "Xv")
AUXILIARY_OUTPUTS = (*HOISTED_INTERMEDIATES, "D05_vv_ladder", "singles_residual")
Polynomial = list[tuple[Fraction, str, tuple[tuple[Node, str], ...]]]


@dataclass(frozen=True)
class DFAuxiliaryReductionPrograms:
    """Prepare once per amplitude state, accumulate every Q, then execute core.

    ``prepare`` supplies df_tau to each ``auxiliary`` invocation. Each auxiliary
    output is summed before it becomes a core input. A DIIS state change
    invalidates both preparation and all accumulated intermediates. These
    programs expose physical energy/residuals, not an iterative derivative.
    """

    prepare: Program
    auxiliary: Program
    core: Program


def _word(node: Node) -> str:
    """Recover a contraction spelling from the canonical integer labels."""
    return (
        ",".join(
            "".join(ascii_letters[label] for label in labels)
            for labels in node.attrs["labels"]
        )
        + "->"
        + "".join(ascii_letters[label] for label in node.attrs["output"])
    )


def _bounded(program: Program) -> Program:
    """Choose contraction trees without recreating ovvv/vvvv storage."""
    result = optimize(
        reassociate_einsums(program, max_intermediate_axes={"virtual": 2})
    )
    if any(
        sum(i.space.kind == "virtual" for i in n.spec.indices) > 2
        for n in result.live_nodes
    ):
        raise ValueError(
            "DF auxiliary reduction reconstructed a virtual integral block"
        )
    return Program(
        result.outputs,
        provenance={**result.provenance, "native_execution_order": "dependencies"},
    )


def _virtual_specialization(source: Program) -> dict[Node, Node | None]:
    """Zero retained Hamiltonian blocks in a graph proven Hamiltonian-linear.

    Keep amplitudes, including their shared tau, intact. Multiplication by a
    zero Hamiltonian branch removes the entire contraction; sum coefficients
    survive unchanged. A future nonlinear-Hamiltonian inventory must not
    silently acquire an incorrect single-Q interpretation here.
    """
    selected: dict[Node, Node | None] = {}
    degree: dict[Node, int] = {}
    for node in source.dependency_order:
        if node.op == "input":
            name = node.attrs["name"]
            degree[node] = 0 if name in ("t1", "t2") else 1
            selected[node] = node if name in ("t1", "t2", "ovvv", "vvvv") else None
        elif node.op == "add":
            degrees = {degree[child] for child in node.inputs}
            if len(degrees) != 1:
                raise ValueError("RCCSD inventory mixes Hamiltonian degrees in one sum")
            degree[node] = next(iter(degrees))
            terms: list[Node] = []
            coefficients: list[Fraction] = []
            for child, coefficient in zip(
                node.inputs, node.attrs["coefficients"], strict=True
            ):
                value = selected[child]
                if value is not None:
                    terms.append(value)
                    coefficients.append(Fraction(*coefficient))
            selected[node] = (
                add(*terms, coefficients=tuple(coefficients)) if terms else None
            )
        elif node.op == "einsum":
            degree[node] = sum(degree[child] for child in node.inputs)
            if degree[node] > 1:
                raise ValueError(
                    "DF auxiliary reduction requires Hamiltonian linearity"
                )
            children: list[Node] = []
            for child in node.inputs:
                value = selected[child]
                if value is None:
                    break
                children.append(value)
            selected[node] = (
                Node(node.op, tuple(children), node.spec, node.attributes)
                if len(children) == len(node.inputs)
                else None
            )
        else:
            raise ValueError(f"unsupported shared RCCSD inventory operation: {node.op}")
    return selected


def build_df_auxiliary_reduction_programs(
    nocc: int, nvir: int
) -> DFAuxiliaryReductionPrograms:
    """Derive a bounded prepare/Q-reduction/core pipeline from shared RCCSD.

    The virtual ladder remains inside the Q loop. Its amplitude-only tau is
    prepared once; other shared virtual intermediates are reduced before T2
    contraction. No full virtual integral or amplitude Jacobian is constructed.
    """
    source = build_ccsd_program(nocc, nvir, form="shared", diagnostics=True)
    selected = _virtual_specialization(source)
    inputs = {n.attrs["name"]: n for n in source.live_nodes if n.op == "input"}
    occupied, virtual = (index.space for index in inputs["t1"].spec.indices)
    bov = input_tensor(
        "bov",
        TensorSpec(
            (Index("i", occupied), Index("a", virtual)),
            role="parameter",
            differentiable=True,
            representation="restricted_spatial",
        ),
    )
    bvv = input_tensor(
        "bvv",
        TensorSpec(
            (Index("a", virtual), Index("b", virtual)),
            role="parameter",
            differentiable=True,
            representation="restricted_spatial",
            symmetries=(Symmetry((1, 0)),),
        ),
    )
    tau = selected[source.outputs["tau"]]
    if tau is None:
        raise ValueError("RCCSD inventory lost amplitude-only tau")
    lowered = {tau: input_tensor("df_tau", replace(tau.spec, role="parameter"))}

    def high_virtual(node: Node) -> bool:
        return sum(i.space.kind == "virtual" for i in node.spec.indices) > 2

    def polynomial(node: Node) -> Polynomial:
        """Inline only forbidden high-virtual branches, preserving shared leaves.

        Reuse the inventory's alpha-renaming polynomial expansion. In particular,
        distributing Wvvvv's three terms never distributes the shared tau sum.
        """
        labels = ascii_letters[: len(node.spec.indices)]
        if node.op == "input" and node.attrs["name"] in ("ovvv", "vvvv"):
            first = bov if node.attrs["name"] == "ovvv" else bvv
            return [(Fraction(1), labels, ((first, labels[:2]), (bvv, labels[2:])))]
        if not high_virtual(node):
            return [(Fraction(1), labels, ((lower(node), labels),))]
        if node.op == "add":
            return [
                (Fraction(*coefficient) * scale, output, operands)
                for child, coefficient in zip(
                    node.inputs, node.attrs["coefficients"], strict=True
                )
                for scale, output, operands in polynomial(child)
            ]
        if node.op == "einsum":
            return _expand(
                _word(node),
                [polynomial(child) for child in node.inputs],
                Fraction(*node.attrs["coefficient"]),
            )
        raise ValueError("cannot inline high-virtual RCCSD intermediate")

    def lower(node: Node) -> Node:
        """Substitute factors before choosing bounded contraction trees."""
        if node in lowered:
            return lowered[node]
        if node.op == "input":
            result = node
        elif node.op == "add":
            result = add(
                *(lower(child) for child in node.inputs),
                coefficients=tuple(
                    Fraction(*value) for value in node.attrs["coefficients"]
                ),
            )
        elif node.op == "einsum":
            terms = _expand(
                _word(node),
                [polynomial(child) for child in node.inputs],
                Fraction(*node.attrs["coefficient"]),
            )
            result = add(
                *(
                    einsum(
                        ",".join(labels for _, labels in operands) + "->" + output,
                        *(child for child, _ in operands),
                        coefficient=coefficient,
                    )
                    for coefficient, output, operands in terms
                )
            )
        else:
            raise ValueError(f"unsupported RCCSD factor substitution: {node.op}")
        lowered[node] = result
        return result

    outputs: dict[str, Node] = {}
    for name in AUXILIARY_OUTPUTS:
        value = selected[source.outputs[name]]
        if value is None:
            raise ValueError(f"RCCSD inventory lost virtual contribution to {name}")
        outputs[f"df_{name}"] = lower(value)
    provenance = {
        "source_equation": source.logical_hash,
        "method": "correlation-DF RCCSD auxiliary reduction",
        "hoisted_intermediates": HOISTED_INTERMEDIATES,
        "auxiliary_schedule": "prepare once per T; sum every Q before retained core",
    }
    auxiliary = _bounded(Program(outputs, provenance=provenance))

    base = build_ccsd_program(
        nocc, nvir, form="shared", diagnostics=True, external_virtual_correction=True
    )
    cuts = {
        base.outputs[name]: input_tensor(
            f"df_{name}", replace(base.outputs[name].spec, role="parameter")
        )
        for name in HOISTED_INTERMEDIATES
    }
    remapped: dict[Node, Node] = {}
    for node in base.dependency_order:
        if node.op == "input":
            name = node.attrs["name"]
            external_name = {
                "df_virtual_singles": "df_singles_residual",
                "df_virtual_doubles": "df_D05_vv_ladder",
            }.get(name)
            updated = input_tensor(external_name, node.spec) if external_name else node
        else:
            updated = Node(
                node.op,
                tuple(remapped[child] for child in node.inputs),
                node.spec,
                node.attributes,
            )
        remapped[node] = add(updated, cuts[node]) if node in cuts else updated
    core = _bounded(
        Program(
            {
                name: remapped[base.outputs[name]]
                for name in (
                    "correlation_energy",
                    "singles_residual",
                    "doubles_residual",
                )
            },
            provenance=provenance,
        )
    )
    prepare = _bounded(Program({"df_tau": tau}, provenance=provenance))
    return DFAuxiliaryReductionPrograms(prepare, auxiliary, core)
