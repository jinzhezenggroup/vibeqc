"""Restricted spatial MP2 tile equations expressed in the shared TensorIR."""

from tools.vibeqc_tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    broadcast,
    divide,
    input_tensor,
    multiply,
    reduce_sum,
)


def energy_program(shape):
    """Consume g[i,j,a,b]=(ia|jb), x[i,j,a,b]=(ib|ja), no spin compression.

    Inputs are rectangular tiles, so x is a separately requested/reordered
    block, not a swap of a and b inside a possibly disjoint tile. All ordered
    ijab tuples participate: OS=g*g/D, SS=g*(g-x)/D, total=OS+SS.
    The denominator is eps_i+eps_j-eps_a-eps_b in Hartree. Only scalar
    energies are outputs; the division intermediate is bounded by this tile.
    """
    if len(shape) != 4 or any(type(n) is not int or n < 1 for n in shape):
        raise ValueError("MP2 tile must have four positive integer dimensions")
    axes = tuple(
        Index(name, IndexSpace(name + "_tile", kind, n))
        for name, kind, n in zip(
            "ijab", ("occupied", "occupied", "virtual", "virtual"), shape
        )
    )

    def tensor(name, indices):
        return input_tensor(
            name, TensorSpec(indices, representation="restricted_spatial", role="input")
        )

    g, x = tensor("g", axes), tensor("x", axes)
    eps = [tensor("e" + name, (axis,)) for name, axis in zip("ijab", axes)]
    d = add(
        *(broadcast(e, axes, (k,)) for k, e in enumerate(eps)),
        coefficients=(1, 1, -1, -1),
    )
    t = divide(g, d)
    os = reduce_sum(multiply(t, g), (0, 1, 2, 3))
    ss = reduce_sum(multiply(t, add(g, x, coefficients=(1, -1))), (0, 1, 2, 3))
    return Program(
        {"opposite_spin": os, "same_spin": ss},
        provenance={
            "issue": "193 A1 (part of A)",
            "reference": "real all-electron canonical closed-shell RHF",
            "integrals": "unscreened conventional chemists ERIs; all ordered ijab",
            "units": "Hartree",
            "amplitudes": "unantisymmetrized restricted spatial; tile temporary only",
        },
    )


def cpu_capacity(program):
    """Conservative numeric capacity including interpreter temporaries.

    The shared interpreter retains logical nodes. Add two maximum-sized
    temporary arrays for primitive arithmetic/finiteness checks and scalar
    publication. Object headers and NumPy/BLAS allocator overhead are excluded.
    """
    sizes = [n.spec.size * n.spec.itemsize for n in program.live_nodes]
    return sum(sizes) + 2 * max(sizes) + 64
