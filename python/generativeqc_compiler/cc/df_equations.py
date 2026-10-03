"""Audited DF virtual residuals as differentiable, bounded TensorIR programs.

The conventional inventory is the mathematical owner.  Substitute one
auxiliary slice of B into its ovvv/vvvv terms before contraction planning;
an execution owner sums these programs over the auxiliary index.  No complete
four-virtual tensor, three-virtual tensor, or amplitude Jacobian is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from string import ascii_letters

from generativeqc_compiler.tensor import (
    Index,
    JVPProgram,
    Node,
    Program,
    Symmetry,
    TensorSpec,
    VJPProgram,
    add,
    einsum,
    input_tensor,
    linearize,
    optimize,
    reassociate_einsums,
    transpose_program,
)

from .doubles import build_ccsd_program


@dataclass(frozen=True)
class DFVirtualResponsePrograms:
    """One-Q correction and its matrix-free amplitude/factor derivatives.

    Amplitude actions are accumulated over Q. Factor cotangents belong to the
    selected Q slice. All adjoints use the dense Frobenius metric, including the
    symmetric B_vv and simultaneous-pair T2 projections. Lambda solves, metric
    response, orbital response and nuclear forces remain execution consumers.
    """

    primal: Program
    amplitude_jvp: JVPProgram
    amplitude_vjp: VJPProgram
    factor_vjp: VJPProgram


def build_df_virtual_response_programs(
    nocc: int, nvir: int
) -> DFVirtualResponsePrograms:
    """Differentiate the physical correction, never CC iterations or DIIS."""
    primal = build_df_virtual_correction_program(nocc, nvir)
    outputs = tuple(primal.outputs)
    amplitudes = ("t1", "t2")
    return DFVirtualResponsePrograms(
        primal,
        linearize(primal, amplitudes, outputs=outputs),
        transpose_program(primal, outputs, inputs=amplitudes),
        transpose_program(primal, outputs, inputs=("bov", "bvv")),
    )


def build_df_virtual_correction_program(nocc: int, nvir: int) -> Program:
    """Derive all omitted RCCSD residual terms for one auxiliary factor slice.

    Inputs are ``bov[o,v]``, symmetric ``bvv[v,v]``, and the usual dense spatial
    amplitudes.  Summing outputs over Q gives exactly the two external virtual
    corrections accepted by ``build_ccsd_program``.  The conventional Fock and
    retained smaller integral blocks do not participate in this map.

    Derivation from the expanded inventory preserves every rational coefficient
    and permutation.  Binary contraction reassociation removes the accidental
    high-degree fused loops without reconstructing ovvv/vvvv.  Reverse and
    forward actions can then use the shared TensorIR AD implementation.
    """
    source = build_ccsd_program(nocc, nvir, form="expanded", diagnostics=False)
    inputs = {
        node.attrs["name"]: node for node in source.live_nodes if node.op == "input"
    }
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
            symmetries=(Symmetry((1, 0)),),
            role="parameter",
            differentiable=True,
            representation="restricted_spatial",
        ),
    )
    selected: dict[Node, Node | None] = {}
    for node in source.dependency_order:
        if node.op == "add":
            terms: list[Node] = []
            coefficients: list[Fraction] = []
            for child, coefficient in zip(
                node.inputs, node.attrs["coefficients"], strict=True
            ):
                selected_child = selected[child]
                if selected_child is not None:
                    terms.append(selected_child)
                    coefficients.append(Fraction(*coefficient))
            selected[node] = (
                add(*terms, coefficients=tuple(coefficients)) if terms else None
            )
            continue
        if node.op != "einsum" or not any(
            child.op == "input" and child.attrs["name"] in ("ovvv", "vvvv")
            for child in node.inputs
        ):
            selected[node] = None
            continue
        operands: list[Node] = []
        labels: list[str] = []
        # RCCSD is linear in the Hamiltonian. Two integral operands would
        # require independent auxiliary sums, not a single shared Q slice.
        if (
            sum(child.attrs.get("name") not in ("t1", "t2") for child in node.inputs)
            != 1
        ):
            raise ValueError("DF one-slice correction requires Hamiltonian linearity")
        for child, indices in zip(node.inputs, node.attrs["labels"], strict=True):
            if child.op != "input":
                raise ValueError(
                    "DF substitution requires the expanded RCCSD inventory"
                )
            name = child.attrs["name"]
            word = "".join(ascii_letters[index] for index in indices)
            if name in ("ovvv", "vvvv"):
                operands.extend((bov if name == "ovvv" else bvv, bvv))
                labels.extend((word[:2], word[2:]))
            else:
                operands.append(child)
                labels.append(word)
        output = "".join(ascii_letters[index] for index in node.attrs["output"])
        selected[node] = einsum(
            ",".join(labels) + "->" + output,
            *operands,
            coefficient=Fraction(*node.attrs["coefficient"]),
        )
    outputs: dict[str, Node] = {}
    for target, source_name in (
        ("df_virtual_singles", "singles_residual"),
        ("df_virtual_doubles", "doubles_residual"),
    ):
        selected_output = selected[source.outputs[source_name]]
        if selected_output is None:
            raise ValueError("RCCSD inventory has no DF virtual correction")
        outputs[target] = selected_output
    program = Program(
        outputs,
        provenance={
            "method": "correlation-DF RCCSD virtual residual",
            "source_equation": source.logical_hash,
            "factorization": "g[pqrs] = sum_Q B[Q,pq] B[Q,rs]",
            "auxiliary_schedule": "one slice; caller accumulates every Q exactly once",
            "maximum_intermediate_virtual_axes": 2,
        },
    )
    result = optimize(
        reassociate_einsums(program, max_intermediate_axes={"virtual": 2})
    )
    for node in result.live_nodes:
        if sum(index.space.kind == "virtual" for index in node.spec.indices) > 2:
            raise ValueError(
                "DF virtual schedule reconstructed an unbounded virtual block"
            )
    return result
