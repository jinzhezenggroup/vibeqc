"""Staged density-fitted source algebra in a supplied physical MO frame.

The source owner supplies raw public-AO three-center integrals and the existing
thresholded symmetric metric inverse root. The supplied Hamiltonian and
orbital frame are unchanged. Two orbital contractions precede metric whitening; a
native executor must share their outputs across retained post-HF blocks.
"""

from __future__ import annotations

from dataclasses import dataclass

from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Node,
    Program,
    TensorSpec,
    einsum,
    input_tensor,
)
from generativeqc_compiler.tensor.cuda_gemm import gemm_contract


def build_df_mo_source_program(nbf: int, naux: int) -> Program:
    """Return the three bounded matrix contractions defining A_MO and B_MO.

    The first stage costs N^3 Q, the second N^3 Q and whitening N^2 Q^2 scalar
    summands. Directly expanding both orbital indices instead costs N^4 Q.
    There is no four-index AO/MO integral and no reference-library dependency.
    All pairs have unit weight; packed-pair producers must expand that contract
    before executing this program. The cutoff/eigensystem belongs to the shared
    metric owner, not this linear transform.
    """
    if type(nbf) is not int or type(naux) is not int or min(nbf, naux) <= 0:
        raise ValueError("positive integer orbital and auxiliary extents required")
    ao = IndexSpace("df_source_ao", "ao", nbf)
    mo = IndexSpace("df_source_mo", "orbital", nbf)
    aux = IndexSpace("df_source_aux", "auxiliary", naux)
    mu, nu = (Index(name, ao) for name in ("mu", "nu"))
    p = Index("p", mo)
    P, Q = (Index(name, aux) for name in ("P", "Q"))
    raw = input_tensor(
        "raw_three_center",
        TensorSpec((mu, nu, P), role="parameter", differentiable=True),
    )
    coefficients = input_tensor(
        "coefficients", TensorSpec((mu, p), role="parameter", differentiable=True)
    )
    inverse_root = input_tensor(
        "inverse_root", TensorSpec((P, Q), role="parameter", differentiable=True)
    )
    first = einsum("mnP,nq->mqP", raw, coefficients)
    transformed = einsum("mp,mqP->pqP", coefficients, first)
    whitened = einsum("pqP,PQ->pqQ", transformed, inverse_root)
    return Program(
        {"transformed": transformed, "whitened": whitened},
        provenance={
            "method": "shared post-HF DF source",
            "reference": "unchanged supplied orbital frame",
            "pair_layout": "full public-AO/MO pairs, unit weight, auxiliary contiguous",
            "source_traversals": 1,
            "native_execution_order": "dependencies",
        },
    )


@dataclass(frozen=True)
class _PackedStep:
    """A packed column-major call derived from one audited binary contraction."""

    ta: str
    tb: str
    m: str
    n: str
    k: str
    a: str
    b: str
    output: str

    def call(self) -> str:
        return (
            f"gemm('{self.ta}','{self.tb}',{self.m},{self.n},{self.k},"
            f"{self.a},{self.b},{self.output});"
        )


def _packed_step(
    node: Node, pointers: tuple[str, str], output: str, *, source_row: bool = False
) -> _PackedStep:
    """Lower only direct packed products, failing closed on changed equations.

    For the first projection, the leading source/output AO label is fixed by
    the row loop. No summation is removed. All remaining dimensions, operand
    order and transposes come from the TensorIR GEMM contract. Swapping a
    scalar product's operands is permitted only when it matches output order.
    """
    g = gemm_contract(node)
    if g is None or g.batch_labels or g.coefficient != 1 or g.dtype != "float64":
        raise ValueError("DF source requires an unbatched unit FP64 GEMM")
    fixed: tuple[int, ...] = ()
    if source_row:
        label = g.a_labels[0]
        if label not in g.m_labels or g.output_labels[0] != label:
            raise ValueError("DF source row must fix a leading free AO label")
        fixed = (label,)

    def remaining(labels: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(label for label in labels if label not in fixed)

    a, b, c, m, n, k = map(
        remaining,
        (g.a_labels, g.b_labels, g.output_labels, g.m_labels, g.n_labels, g.k_labels),
    )
    pa, pb = pointers
    if c != m + n:
        a, b, m, n, pa, pb = b, a, n, m, pb, pa
    if c != m + n:
        raise ValueError("DF source contraction needs a non-packed output")

    def transpose(
        actual: tuple[int, ...], rows: tuple[int, ...], cols: tuple[int, ...]
    ) -> str:
        if actual == rows + cols:
            return "N"
        if actual == cols + rows:
            return "T"
        raise ValueError("DF source contraction needs input packing")

    dimensions = {
        label: "q" if index.space.kind == "auxiliary" else "n"
        for operand, labels in zip(node.inputs, node.attrs["labels"], strict=True)
        for label, index in zip(labels, operand.spec.indices, strict=True)
    }

    def extent(labels: tuple[int, ...]) -> str:
        value = "1"
        for label in labels:
            value = (
                dimensions[label]
                if value == "1"
                else f"checked_mul({value},{dimensions[label]})"
            )
        return value

    # Row-major A*B is the transpose of a column-major B*A. Native CPU and
    # cuBLAS callbacks therefore share the same packed column-major contract.
    return _PackedStep(
        transpose(b, k, n),
        transpose(a, m, k),
        extent(n),
        extent(m),
        extent(k),
        pb,
        pa,
        output,
    )


def native_header() -> str:
    """Emit a shape-generic traversal derived from the three TensorIR nodes.

    Native owners supply storage and source/BLAS callbacks. ``first`` becomes
    B only after its projection has been consumed; ``transformed`` remains
    available to response consumers. Input rows can be discarded as soon as
    their first GEMM has consumed them on the owner's execution stream.
    """
    program = build_df_mo_source_program(2, 3)
    nodes = [node for node in program.live_nodes if node.op == "einsum"]
    if len(nodes) != 3:
        raise ValueError("DF source schedule requires three contractions")
    calls = [
        _packed_step(
            nodes[0], ("read(mu)", "coefficients"), "first+mu*row", source_row=True
        ),
        _packed_step(nodes[1], ("coefficients", "first"), "transformed"),
        _packed_step(nodes[2], ("transformed", "inverse_root"), "first"),
    ]
    # Derive semantic summands from the same emitted dimensions, including the
    # source-row loop multiplicity; count terms, not hardware FLOPs.
    work = [f"checked_mul(checked_mul({step.m},{step.n}),{step.k})" for step in calls]
    work[0] = f"checked_mul(n,{work[0]})"
    total_work = f"checked_add(checked_add({work[0]},{work[1]}),{work[2]})"
    return "\n".join(
        [
            "// Generated from generativeqc_compiler.method.df_mo_source; do not edit.",
            "#pragma once",
            "#include <cstddef>",
            '#include "posthf/capacity.hpp"',
            "namespace generativeqc::posthf::generated {",
            f'inline constexpr const char* df_mo_source_equation_hash = "{program.logical_hash}";',
            "struct DFMOSourceWork {",
            "  std::size_t raw_values{}, source_rows{}, gemms{}, contraction_summands{};",
            "};",
            "inline DFMOSourceWork df_mo_source_work(std::size_t n, std::size_t q) {",
            '  if (!n || !q) throw std::invalid_argument("empty DF MO source dimensions");',
            f"  return {{checked_mul(checked_mul(n,n),q),n,checked_add(n,2),{total_work}}};",
            "}",
            "// Read(mu) supplies exactly one public AO row [nu,P], with unit pair weights.",
            "// Gemm(ta,tb,m,n,k,a,b,c) uses packed column-major matrices, alpha=1,beta=0.",
            "// Inputs are row-major C[mu,p], M[P,Q]; outputs are A[p,q,P], B[p,q,Q].",
            "// Buffers must be disjoint and large enough; the owner preflights their bytes.",
            "// Both callbacks may throw; callers must not publish partially built state.",
            "// For asynchronous callbacks, the owner orders row reuse and drains before publication.",
            "template<class Read, class Gemm>",
            "void transform_df_mo_source(std::size_t n, std::size_t q, const double* coefficients,",
            "                            const double* inverse_root, double* first,",
            "                            double* transformed, Read&& read, Gemm&& gemm) {",
            "  (void)df_mo_source_work(n,q);",
            "  const auto row = checked_mul(n,q);",
            "  for (std::size_t mu=0; mu<n; ++mu)",
            "    " + calls[0].call(),
            "  " + calls[1].call(),
            "  " + calls[2].call(),
            "}",
            "}  // namespace generativeqc::posthf::generated",
            "",
        ]
    )
