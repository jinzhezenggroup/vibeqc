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
    optimize,
    transpose_program,
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


def build_df_mo_source_vjp_program(nbf: int, naux: int) -> Program:
    """Differentiate the staged MO/whitening source, holding the root as an input.

    All three inputs remain independent dense coordinates. The metric owner's
    fixed-rank inverse-square-root rule consumes ``bar_inverse_root`` next;
    neither its eigensolver nor the rank decision is differentiated here.
    A physical factor consumer supplies the already pair-projected B cotangent.
    """
    primal = build_df_mo_source_program(nbf, naux)
    result = optimize(
        transpose_program(
            primal,
            ("whitened",),
            inputs=("raw_three_center", "coefficients", "inverse_root"),
        ).program
    )
    return Program(
        result.outputs,
        provenance={**result.provenance, "native_execution_order": "dependencies"},
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

    def call(self, *, accumulate: bool | None = None) -> str:
        """Optionally expose beta=0/1 to response callbacks sharing an output."""
        suffix = "" if accumulate is None else "," + str(accumulate).lower()
        return (
            f"gemm('{self.ta}','{self.tb}',{self.m},{self.n},{self.k},"
            f"{self.a},{self.b},{self.output}{suffix});"
        )


def _packed_step(
    node: Node,
    pointers: tuple[str, str],
    output: str,
    *,
    source_row: bool = False,
    reduction_row: bool = False,
) -> _PackedStep:
    """Lower only direct packed products, failing closed on changed equations.

    For the first projection, the leading source/output AO label is fixed by
    the row loop. No summation is removed. All remaining dimensions, operand
    order and transposes come from the TensorIR GEMM contract. Swapping a
    scalar product's operands is permitted only when it matches output order.
    A response coefficient product may instead fix a common leading reduction
    label; its caller must visit every row and accumulate the complete sum.
    """
    g = gemm_contract(node)
    if g is None or g.batch_labels or g.coefficient != 1 or g.dtype != "float64":
        raise ValueError("DF source requires an unbatched unit FP64 GEMM")
    fixed: tuple[int, ...] = ()
    if source_row and reduction_row:
        raise ValueError("DF source row cannot be both free and reduced")
    if source_row:
        label = g.a_labels[0]
        if label not in g.m_labels or g.output_labels[0] != label:
            raise ValueError("DF source row must fix a leading free AO label")
        fixed = (label,)
    if reduction_row:
        label = g.a_labels[0]
        if label != g.b_labels[0] or label not in g.k_labels:
            raise ValueError("DF response row must fix a common leading reduction")
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
    return (
        "\n".join(
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
        + native_response_header()
    )


def native_response_header() -> str:
    """Emit the AD-derived two-pass reverse traversal without a resident raw A.

    Two N*N*Q scratch tensors suffice: consume the forward first projection
    before replacing it by its adjoint, and consume transformed A in the root
    cotangent before replacing it by bar_A_MO. The second source pass forms
    the remaining coefficient derivative while streaming raw-A cotangents.
    """
    program = build_df_mo_source_vjp_program(2, 3)
    bar_c = program.outputs["bar_coefficients"]
    bar_root = program.outputs["bar_inverse_root"]
    bar_raw = program.outputs["bar_raw_three_center"]
    if bar_c.op != "add" or bar_c.attrs["coefficients"] != ((1, 1), (1, 1)):
        raise ValueError("DF source response requires the two coefficient paths")
    first_c, second_c = bar_c.inputs
    bar_b, transformed = bar_root.inputs
    bar_first, c = bar_raw.inputs
    bar_mo = bar_first.inputs[0]
    transformed_seed, root = bar_mo.inputs
    # Demand the audited shared AD topology rather than guessing contractions
    # from shapes (AO and MO happen to have the same dynamic dimension).
    first = transformed.inputs[1]
    raw = first.inputs[0]
    if not (
        bar_first.inputs == (bar_mo, c)
        and transformed.inputs == (c, first)
        and first.inputs == (raw, c)
        and first_c.inputs == (bar_mo, first)
        and second_c.inputs == (bar_first, raw)
        and transformed_seed is bar_b
        and all(x.op == "input" for x in (raw, c, root, bar_b))
        and tuple(x.attrs["name"] for x in (raw, c, root, bar_b))
        == ("raw_three_center", "coefficients", "inverse_root", "bar_whitened")
    ):
        raise ValueError("unsupported DF source reverse dependency topology")
    steps = [
        _packed_step(
            first, ("read(mu)", "coefficients"), "first+mu*row", source_row=True
        ),
        _packed_step(transformed, ("coefficients", "first"), "transformed"),
        _packed_step(bar_root, ("bar_whitened", "transformed"), "bar_inverse_root"),
        _packed_step(bar_mo, ("bar_whitened", "inverse_root"), "transformed"),
        _packed_step(first_c, ("transformed", "first"), "bar_coefficients"),
        _packed_step(bar_first, ("transformed", "coefficients"), "first"),
        _packed_step(
            second_c,
            ("first+mu*row", "read(mu)"),
            "bar_coefficients",
            reduction_row=True,
        ),
        _packed_step(
            bar_raw,
            ("first+mu*row", "coefficients"),
            "raw_cotangent_row",
            source_row=True,
        ),
    ]
    work = []
    for i, step in enumerate(steps):
        term = f"checked_mul(checked_mul({step.m},{step.n}),{step.k})"
        work.append(f"checked_mul(n,{term})" if i in (0, 6, 7) else term)
    work_expression = "0"
    for term in work:
        work_expression = f"checked_add({work_expression},{term})"
    return "\n".join(
        [
            "// Reverse traversal generated from shared TensorIR AD; do not edit.",
            "namespace generativeqc::posthf::generated {",
            f'inline constexpr const char* df_mo_source_response_equation_hash="{program.logical_hash}";',
            "struct DFMOSourceResponseWork { std::size_t source_rows{}, raw_values{}, output_rows{}, output_values{}, gemms{}, contraction_summands{}, scratch_values{}; };",
            "inline DFMOSourceResponseWork df_mo_source_response_work(std::size_t n,std::size_t q) {",
            '  if(!n||!q)throw std::invalid_argument("empty DF MO source response dimensions");',
            "  const auto row=checked_mul(n,q), values=checked_mul(n,row);",
            f"  return {{checked_mul(2,n),checked_mul(2,values),n,values,checked_add(checked_mul(3,n),5),{work_expression},checked_add(checked_mul(2,values),checked_mul(2,row))}};",
            "}",
            "// All arrays are row-major, auxiliary contiguous, and disjoint.",
            "// Read(mu) supplies [nu,P] twice, from the SAME immutable raw source.",
            "// Consume(mu,row) consumes the raw-A cotangent [nu,P] before row reuse.",
            "// Gemm uses packed column-major matrices, alpha=1 and beta=accumulate?1:0.",
            "// The owner provides first/transformed[N*N*Q], raw_cotangent_row[N*Q],",
            "// plus Read's row[N*Q], and orders every callback on its single stream.",
            "template<class Read,class Consume,class Gemm>",
            "void pullback_df_mo_source(std::size_t n,std::size_t q,const double* coefficients,",
            "    const double* inverse_root,const double* bar_whitened,double* first,",
            "    double* transformed,double* raw_cotangent_row,double* bar_coefficients,",
            "    double* bar_inverse_root,Read&& read,Consume&& consume,Gemm&& gemm) {",
            "  (void)df_mo_source_response_work(n,q); const auto row=checked_mul(n,q);",
            "  for(std::size_t mu=0;mu<n;++mu)",
            "    " + steps[0].call(accumulate=False),
            *("  " + step.call(accumulate=False) for step in steps[1:6]),
            "  for(std::size_t mu=0;mu<n;++mu) {",
            "    " + steps[6].call(accumulate=True),
            "    " + steps[7].call(accumulate=False),
            "    consume(mu,raw_cotangent_row);",
            "  }",
            "}",
            "}  // namespace generativeqc::posthf::generated",
            "",
        ]
    )
