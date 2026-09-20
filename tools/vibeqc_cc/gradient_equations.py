"""Generated raw-Hamiltonian and orbital pullbacks for conventional RCCSD.

Only the primal closed-shell Fock/reference-energy map and orbital basis
transforms are specified here. Their reverse derivatives use the existing
TensorIR AD; no handwritten CC orbital/normal-ordering derivative is added.
"""

from dataclasses import dataclass
from fractions import Fraction

from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    JVPProgram,
    Program,
    Symmetry,
    TensorSpec,
    VJPProgram,
    add,
    constant,
    einsum,
    input_tensor,
    linearize,
    slice_tensor,
    transpose,
    transpose_program,
)

from .lambda_equations import PARAMETERS


@dataclass(frozen=True)
class CCSDHamiltonianPrograms:
    """Dense small-system MO map and its generated h/g/U reverse action.

    At U=I, h/g are raw MO one-/two-electron integrals (chemists' order).
    q(U) is built AFTER transforming h and g by U, so the Fock dependence on
    the rotated occupied density is included. U need not be orthogonal when
    differentiating: the symmetric U derivative fixes the overlap multiplier.
    ``reference_electronic_energy`` excludes nuclear repulsion.
    """

    primal: Program
    pullback: VJPProgram
    weights: Program
    orbital_jvp: JVPProgram
    nocc: int
    nvir: int


def build_hamiltonian_programs(nocc: int, nvir: int) -> CCSDHamiltonianPrograms:
    """Map independent raw h/g/U into all #152 input fields and RHF energy.

    The same full-g input feeds overlapping/permuted q blocks, so the generated
    reverse accumulates every alias exactly once. Inputs h/g have their physical
    symmetries; cotangents use the full dense Frobenius metric, not packed RDM
    factors. This is an explicitly dense validation frontier, not GPU tiling.
    """
    if any(type(n) is not int or n < 1 for n in (nocc, nvir)):
        raise ValueError("CC Hamiltonian requires nonempty occupied/virtual spaces")
    n = nocc + nvir
    if n > 12:
        raise ValueError("dense CC gradient programs support at most 12 MOs")
    space = IndexSpace("complete_mo", "orbital", n)
    idx = tuple(Index(c, space) for c in "pqrs")
    common = {
        "role": "parameter",
        "differentiable": True,
        "representation": "restricted_spatial",
    }
    h = input_tensor("h", TensorSpec(idx[:2], symmetries=(Symmetry((1, 0)),), **common))
    g = input_tensor(
        "g",
        TensorSpec(
            idx,
            symmetries=tuple(
                Symmetry(p) for p in ((1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1))
            ),
            **common,
        ),
    )
    rotation = input_tensor("rotation", TensorSpec(idx[:2], **common))
    density = constant(
        tuple(2 if p == q and p < nocc else 0 for p in range(n) for q in range(n)),
        TensorSpec(idx[:2], role="constant", representation="restricted_spatial"),
    )
    # Staged one-axis transforms avoid a single high-rank einsum intermediate.
    rotated_h = einsum("pv,vq->pq", einsum("up,uv->pv", rotation, h), rotation)
    rotated_g = einsum("up,uvwx->pvwx", rotation, g)
    for equation in ("vq,pvwx->pqwx", "wr,pqwx->pqrx", "xs,pqrx->pqrs"):
        rotated_g = einsum(equation, rotation, rotated_g)
    fock = add(
        rotated_h,
        einsum("pqrs,rs->pq", rotated_g, density),
        einsum("prqs,rs->pq", rotated_g, density, coefficient=Fraction(-1, 2)),
    )
    hf = einsum("pq,pq->", density, add(rotated_h, fock), coefficient=Fraction(1, 2))
    ranges = {"o": (0, nocc), "v": (nocc, n)}
    outputs = {
        name: slice_tensor(
            fock if name.startswith("f") else rotated_g,
            tuple(ranges[c] for c in name.removeprefix("f")),
        )
        for name in PARAMETERS
    }
    outputs.update(reference_electronic_energy=hf, fock=fock)
    primal = Program(
        outputs,
        provenance={
            "model": "restricted conventional raw h/g to F/ERI blocks",
            "occupation": "2 in occupied, 0 in virtual",
            "nocc": nocc,
            "orbital_map": "h(U)=U.T h U; g(U)=U^4 g; build F after rotation",
            "derivative_metric": "dense Frobenius; symmetric h and eightfold chemists g",
        },
    )
    reverse = transpose_program(
        primal,
        (*PARAMETERS, "reference_electronic_energy"),
        inputs=("h", "g", "rotation"),
    )
    gradient = reverse.program.outputs["bar_rotation"]
    stationarity = add(gradient, transpose(gradient, (1, 0)), coefficients=(1, -1))
    overlap = add(
        gradient,
        transpose(gradient, (1, 0)),
        coefficients=(Fraction(-1, 4), Fraction(-1, 4)),
    )
    weights = Program(
        {
            "hcore": reverse.program.outputs["bar_h"],
            "eri": reverse.program.outputs["bar_g"],
            "overlap": overlap,
            "rotation_gradient": gradient,
            "stationarity": stationarity,
            "orbital_rhs": add(
                slice_tensor(stationarity, ((0, nocc), (nocc, n))), coefficients=(-1,)
            ),
        },
        provenance={
            "hamiltonian_pullback": reverse.derivative_hash,
            "overlap_rule": "symmetric metric transport dU=-dS/2",
            "orbital_rule": "K_ia=x_ia; K_ai=-x_ia; A z=-dL/dx",
        },
    )
    orbital_jvp = linearize(primal, ("rotation",), outputs=("fov",))
    return CCSDHamiltonianPrograms(primal, reverse, weights, orbital_jvp, nocc, nvir)


def build_fock_weight_program(nocc: int, nvir: int) -> Program:
    """Generate raw h/g/metric/orbital weights from a full-Fock cotangent.

    This is the upstream response needed when a post-HF model depends directly
    on canonical orbital energies. The caller supplies ``bar_fock`` in the
    complete MO basis, normally with only diagonal entries populated from
    dE/d eps_p. Reverse differentiation reuses the exact raw-Hamiltonian primal
    used by the CCSD gradient chain, so density response and basis rotation are
    not reconstructed by handwritten derivative formulae.

    The returned overlap and occupied-virtual orbital RHS use the same metric
    transport and rotation convention as ``build_hamiltonian_programs``.
    This program alone does not solve the RHF response equations or establish a
    complete nuclear gradient.
    """
    parent = build_hamiltonian_programs(nocc, nvir)
    reverse = transpose_program(
        parent.primal,
        ("fock",),
        inputs=("h", "g", "rotation"),
    )
    gradient = reverse.program.outputs["bar_rotation"]
    stationarity = add(gradient, transpose(gradient, (1, 0)), coefficients=(1, -1))
    overlap = add(
        gradient,
        transpose(gradient, (1, 0)),
        coefficients=(Fraction(-1, 4), Fraction(-1, 4)),
    )
    n = nocc + nvir
    return Program(
        {
            "hcore": reverse.program.outputs["bar_h"],
            "eri": reverse.program.outputs["bar_g"],
            "overlap": overlap,
            "rotation_gradient": gradient,
            "stationarity": stationarity,
            "orbital_rhs": add(
                slice_tensor(stationarity, ((0, nocc), (nocc, n))),
                coefficients=(-1,),
            ),
        },
        provenance={
            "hamiltonian_primal": parent.primal.logical_hash,
            "fock_pullback": reverse.derivative_hash,
            "scope": "direct canonical-orbital-energy response",
            "overlap_rule": "symmetric metric transport dU=-dS/2",
            "orbital_rule": "K_ia=x_ia; K_ai=-x_ia; A z=-dL/dx",
        },
    )


def build_ao_weight_program(n: int) -> Program:
    """Generate staged MO -> AO cotangent transforms (no new AD formula)."""
    if type(n) is not int or not 2 <= n <= 12:
        raise ValueError("dense AO weight transform supports 2 to 12 orbitals")
    mo = IndexSpace("complete_mo", "orbital", n)
    ao = IndexSpace("basis_ao", "ao", n)
    p, q, r, s = (Index(c, mo) for c in "pqrs")
    u = Index("u", ao)
    spec = lambda axes: TensorSpec(
        axes, role="input", representation="restricted_spatial"
    )
    c = input_tensor("coefficients", spec((u, p)))
    h = input_tensor("hcore", spec((p, q)))
    overlap = input_tensor("overlap", spec((p, q)))
    eri = input_tensor("eri", spec((p, q, r, s)))
    two = einsum("up,pqrs->uqrs", c, eri)
    for equation in ("vq,uqrs->uvrs", "wr,uvrs->uvws", "xs,uvws->uvwx"):
        two = einsum(equation, c, two)
    return Program(
        {
            "hcore": einsum("uq,vq->uv", einsum("up,pq->uq", c, h), c),
            "overlap": einsum("uq,vq->uv", einsum("up,pq->uq", c, overlap), c),
            "eri": two,
        },
        provenance={
            "operation": "ordered dense MO cotangents to AO; no symmetry folding"
        },
    )


def build_ao_one_electron_weight_program(n: int) -> Program:
    """Generate only O(N^2) MO -> AO h/overlap cotangent transforms."""
    dense = build_ao_weight_program(n)
    return Program(
        {name: dense.outputs[name] for name in ("hcore", "overlap")},
        provenance={
            "operation": "ordered MO one-electron cotangents to AO",
            "parent": dense.logical_hash,
        },
    )


def build_ao_eri_weight_block_program(
    n: int, shape: tuple[int, int, int, int]
) -> Program:
    """Transform one public-AO shell quartet from a dense MO ERI cotangent.

    Each coefficient input contains only the requested AO rows but all MO
    columns. The output is therefore O(product(shape)); no complete AO N^4
    cotangent is materialized. This is the same ordered C^4 dual transform as
    :func:`build_ao_weight_program`, not a symmetry-folded RDM convention.
    """
    if type(n) is not int or not 2 <= n <= 12:
        raise ValueError("blocked AO weight transform supports 2 to 12 orbitals")
    if (
        not isinstance(shape, tuple)
        or len(shape) != 4
        or any(type(value) is not int or not 1 <= value <= n for value in shape)
    ):
        raise ValueError("AO ERI block shape must contain four positive extents")
    mo = IndexSpace("complete_mo", "orbital", n)
    p, q, r, s = (Index(c, mo) for c in "pqrs")
    axes = tuple(
        IndexSpace(f"basis_ao_block_{slot}_{extent}", "ao", extent)
        for slot, extent in enumerate(shape)
    )
    u, v, w, x = (Index(c, space) for c, space in zip("uvwx", axes, strict=True))
    spec = lambda indices: TensorSpec(
        indices, role="input", representation="restricted_spatial"
    )
    c0 = input_tensor("coefficients_0", spec((u, p)))
    c1 = input_tensor("coefficients_1", spec((v, q)))
    c2 = input_tensor("coefficients_2", spec((w, r)))
    c3 = input_tensor("coefficients_3", spec((x, s)))
    eri = input_tensor("eri", spec((p, q, r, s)))
    value = einsum("up,pqrs->uqrs", c0, eri)
    value = einsum("vq,uqrs->uvrs", c1, value)
    value = einsum("wr,uvrs->uvws", c2, value)
    value = einsum("xs,uvws->uvwx", c3, value)
    return Program(
        {"eri": value},
        provenance={
            "operation": "ordered MO ERI cotangent to one AO shell quartet",
            "block_shape": shape,
        },
    )
