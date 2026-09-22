"""Compiler-owned scalar GFN2 AES2 equations and generated reverse response.

Runtime code owns ragged topology, cache publication, validation, reductions and
transactionality.  This module owns the repeated AES2 scientific arithmetic so
CPU and CUDA consumers share one equation graph.
"""

from __future__ import annotations

from vibeqc_compiler.tensor.ad_program import VJPProgram, transpose_program
from vibeqc_compiler.tensor.ir import (
    Node,
    add,
    constant,
    divide,
    input_tensor,
    multiply,
)
from vibeqc_compiler.tensor.program import Program
from vibeqc_compiler.tensor.types import TensorSpec

GFN2_AES2_RUNTIME_VERSION = "gfn2-aes2-runtime-ir-v1"
GFN2_AES2_MULTIPOLE_KEXP = 4.0
GFN2_AES2_MULTIPOLE_SHIFT = 1.2
GFN2_AES2_MULTIPOLE_RMAX = 5.0
GFN2_AES2_DAMPING_FACTOR = 6.0

DIPOLE_NAMES = ("dipole_x", "dipole_y", "dipole_z")
QUADRUPOLE_NAMES = tuple(f"quadrupole_{index}" for index in range(6))
FIRST_DIPOLE_NAMES = tuple(f"first_dipole_{axis}" for axis in "xyz")
SECOND_DIPOLE_NAMES = tuple(f"second_dipole_{axis}" for axis in "xyz")
FIRST_QUADRUPOLE_NAMES = tuple(f"first_quadrupole_{index}" for index in range(6))
SECOND_QUADRUPOLE_NAMES = tuple(f"second_quadrupole_{index}" for index in range(6))
PAIR_MULTIPOLE_NAMES = (
    "first_charge",
    "second_charge",
    *FIRST_DIPOLE_NAMES,
    *SECOND_DIPOLE_NAMES,
    *FIRST_QUADRUPOLE_NAMES,
    *SECOND_QUADRUPOLE_NAMES,
)
PAIR_GEOMETRY_NAMES = ("dx", "dy", "dz", "kernel3", "kernel5")


def _input(name: str, *, differentiable: bool = False) -> Node:
    return input_tensor(
        name,
        TensorSpec((), role="input", differentiable=differentiable),
    )


def _sum(nodes: tuple[Node, ...] | list[Node]) -> Node:
    if not nodes:
        return constant(0)
    if len(nodes) == 1:
        return nodes[0]
    return add(*nodes)


def build_gfn2_aes2_radius_from_fraction_program() -> Program:
    """Map a stable logistic fraction to the GFN2 CN-dependent multipole radius."""

    base_radius = _input("base_radius")
    fraction = _input("fraction")
    span = add(
        constant(repr(GFN2_AES2_MULTIPOLE_RMAX)),
        base_radius,
        coefficients=(1, -1),
    )
    radius = add(base_radius, multiply(span, fraction))
    one_minus_fraction = add(constant(1), fraction, coefficients=(1, -1))
    cn_derivative = multiply(
        multiply(
            span,
            constant(repr(GFN2_AES2_MULTIPOLE_KEXP)),
        ),
        multiply(fraction, one_minus_fraction),
    )
    return Program(
        {"radius": radius, "cn_derivative": cn_derivative},
        provenance={
            "kind": "gfn2-aes2-radius-from-stable-logistic",
            "version": GFN2_AES2_RUNTIME_VERSION,
            "source": "GFN2 AES2 multipole radius",
        },
    )


def build_gfn2_aes2_kernel_program() -> Program:
    """Build the damped R^-3/R^-5 pair kernels used by AES2."""

    distance = _input("distance", differentiable=True)
    radius = _input("radius", differentiable=True)
    one = constant(1)
    six = constant(repr(GFN2_AES2_DAMPING_FACTOR))
    inverse = divide(one, distance)
    inverse2 = multiply(inverse, inverse)
    inverse3 = multiply(inverse2, inverse)
    inverse5 = multiply(inverse3, inverse2)
    scaled = multiply(radius, inverse)
    scaled2 = multiply(scaled, scaled)
    scaled3 = multiply(scaled2, scaled)
    scaled4 = multiply(scaled2, scaled2)
    kernel3 = divide(inverse3, add(one, multiply(six, scaled3)))
    kernel5 = divide(inverse5, add(one, multiply(six, scaled4)))
    return Program(
        {"kernel3": kernel3, "kernel5": kernel5},
        provenance={
            "kind": "gfn2-aes2-damped-pair-kernels",
            "version": GFN2_AES2_RUNTIME_VERSION,
            "source": "GFN2 AES2 electrostatics",
        },
    )


def build_gfn2_aes2_kernel_vjp_program() -> VJPProgram:
    """Reverse response of the two damped kernels to distance and radius."""

    return transpose_program(
        build_gfn2_aes2_kernel_program(),
        ("kernel3", "kernel5"),
        inputs=("distance", "radius"),
    )


def build_gfn2_aes2_onsite_energy_program() -> Program:
    """One-atom AES2 dipole/quadrupole self energy."""

    dipole_kernel = _input("dipole_kernel")
    quadrupole_kernel = _input("quadrupole_kernel")
    dipoles = {name: _input(name, differentiable=True) for name in DIPOLE_NAMES}
    quadrupoles = {name: _input(name, differentiable=True) for name in QUADRUPOLE_NAMES}
    dipole_norm2 = _sum(
        [multiply(dipoles[name], dipoles[name]) for name in DIPOLE_NAMES]
    )
    weights = (1, 2, 1, 2, 2, 1)
    quadrupole_norm2 = _sum(
        [
            multiply(
                constant(weight),
                multiply(quadrupoles[name], quadrupoles[name]),
            )
            for name, weight in zip(QUADRUPOLE_NAMES, weights, strict=True)
        ]
    )
    energy = add(
        multiply(dipole_kernel, dipole_norm2),
        multiply(quadrupole_kernel, quadrupole_norm2),
    )
    return Program(
        {"energy": energy},
        provenance={
            "kind": "gfn2-aes2-onsite-energy",
            "version": GFN2_AES2_RUNTIME_VERSION,
        },
    )


def build_gfn2_aes2_onsite_potential_program() -> VJPProgram:
    """Generate onsite multipole potentials from the onsite energy graph."""

    return transpose_program(
        build_gfn2_aes2_onsite_energy_program(),
        ("energy",),
        inputs=(*DIPOLE_NAMES, *QUADRUPOLE_NAMES),
    )


def _packed_pair_tensor(
    dx: Node, dy: Node, dz: Node, kernel5: Node
) -> tuple[Node, ...]:
    two = constant(2)
    return (
        multiply(multiply(dx, dx), kernel5),
        multiply(multiply(two, multiply(dx, dy)), kernel5),
        multiply(multiply(dy, dy), kernel5),
        multiply(multiply(two, multiply(dx, dz)), kernel5),
        multiply(multiply(two, multiply(dy, dz)), kernel5),
        multiply(multiply(dz, dz), kernel5),
    )


def build_gfn2_aes2_pair_energy_program() -> Program:
    """AES2 pair energy; all pair potentials and geometry response derive from it."""

    values = {
        name: _input(name, differentiable=True)
        for name in (*PAIR_GEOMETRY_NAMES, *PAIR_MULTIPOLE_NAMES)
    }
    dx, dy, dz = (values[name] for name in ("dx", "dy", "dz"))
    kernel3 = values["kernel3"]
    kernel5 = values["kernel5"]
    first_charge = values["first_charge"]
    second_charge = values["second_charge"]
    first_dipole = tuple(values[name] for name in FIRST_DIPOLE_NAMES)
    second_dipole = tuple(values[name] for name in SECOND_DIPOLE_NAMES)
    first_quadrupole = tuple(values[name] for name in FIRST_QUADRUPOLE_NAMES)
    second_quadrupole = tuple(values[name] for name in SECOND_QUADRUPOLE_NAMES)
    displacement = (dx, dy, dz)

    first_projection = _sum(
        [multiply(displacement[i], first_dipole[i]) for i in range(3)]
    )
    second_projection = _sum(
        [multiply(displacement[i], second_dipole[i]) for i in range(3)]
    )
    dipole_dot = _sum([multiply(first_dipole[i], second_dipole[i]) for i in range(3)])
    charge_dipole_numerator = _sum(
        [
            multiply(
                displacement[i],
                add(
                    multiply(first_charge, second_dipole[i]),
                    multiply(second_charge, first_dipole[i]),
                    coefficients=(1, -1),
                ),
            )
            for i in range(3)
        ]
    )
    distance2 = _sum([multiply(component, component) for component in displacement])
    charge_dipole = multiply(kernel3, charge_dipole_numerator)
    dipole_dipole = multiply(
        kernel5,
        add(
            multiply(distance2, dipole_dot),
            multiply(first_projection, second_projection),
            coefficients=(1, -3),
        ),
    )
    packed = _packed_pair_tensor(dx, dy, dz, kernel5)
    first_packed_dot = _sum(
        [multiply(packed[i], first_quadrupole[i]) for i in range(6)]
    )
    second_packed_dot = _sum(
        [multiply(packed[i], second_quadrupole[i]) for i in range(6)]
    )
    charge_quadrupole = add(
        multiply(first_charge, second_packed_dot),
        multiply(second_charge, first_packed_dot),
    )
    energy = add(charge_dipole, dipole_dipole, charge_quadrupole)
    return Program(
        {"energy": energy},
        provenance={
            "kind": "gfn2-aes2-pair-energy",
            "version": GFN2_AES2_RUNTIME_VERSION,
            "response": "multipole potentials and coordinate VJP use reverse AD",
        },
    )


def build_gfn2_aes2_pair_potential_program() -> VJPProgram:
    """Generate q/d/Q pair potentials as derivatives of the pair energy."""

    return transpose_program(
        build_gfn2_aes2_pair_energy_program(),
        ("energy",),
        inputs=PAIR_MULTIPOLE_NAMES,
    )


def build_gfn2_aes2_pair_geometry_vjp_program() -> VJPProgram:
    """Generate fixed-kernel displacement and kernel adjoints from pair energy."""

    return transpose_program(
        build_gfn2_aes2_pair_energy_program(),
        ("energy",),
        inputs=PAIR_GEOMETRY_NAMES,
    )


def build_gfn2_aes2_pair_vjp_compose_program() -> Program:
    """Compose pair-energy and kernel reverse passes into R/CN adjoints."""

    names = (
        "dx",
        "dy",
        "dz",
        "distance",
        "bar_dx",
        "bar_dy",
        "bar_dz",
        "bar_distance",
        "bar_radius",
        "first_radius_cn_derivative",
        "second_radius_cn_derivative",
    )
    value = {name: _input(name) for name in names}
    radial = divide(value["bar_distance"], value["distance"])
    outputs = {
        "gradient_x": add(value["bar_dx"], multiply(radial, value["dx"])),
        "gradient_y": add(value["bar_dy"], multiply(radial, value["dy"])),
        "gradient_z": add(value["bar_dz"], multiply(radial, value["dz"])),
        "first_cn_adjoint": multiply(
            constant("0.5"),
            multiply(value["bar_radius"], value["first_radius_cn_derivative"]),
        ),
        "second_cn_adjoint": multiply(
            constant("0.5"),
            multiply(value["bar_radius"], value["second_radius_cn_derivative"]),
        ),
    }
    return Program(
        outputs,
        provenance={
            "kind": "gfn2-aes2-pair-vjp-compose",
            "version": GFN2_AES2_RUNTIME_VERSION,
        },
    )
