"""Fixed-state GFN2 Hamiltonian algebra in TensorIR (#505).

This compiler layer deliberately excludes SCC iteration, eigensolvers,
occupations, convergence policy, and mutable history.  It represents the
already-prepared H0/S/D/Q operators and fixed shell/atomic potentials as one
auditable tensor program, including the directed ket-origin multipole
convention used by xTB/tblite and the pinned xTBloom reference.

Rationale: .agents/notes/implemented/numerics/2026-09-20-gfn2-electronic-contract.md
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from vibeqc_compiler.tensor.ad_program import VJPProgram
    from vibeqc_compiler.tensor.ir import Node

from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    einsum,
    indexed_gather,
    input_tensor,
    multiply,
    transpose_program,
)

from .xtb import GFN2_PARAMETER_SET, XtbMethodIR, resolve_xtb_method

GFN2_ELECTRONIC_VERSION = "gfn2-fixed-state-electronic-ir-v1"
GFN2_ELECTRONIC_REFERENCE_REVISION = "2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3"
GFN2_DIPOLE_COMPONENTS = ("x", "y", "z")
GFN2_QUADRUPOLE_COMPONENTS = ("xx", "xy", "yy", "xz", "yz", "zz")


def _offsets(values: Iterable[int], label: str) -> tuple[int, ...]:
    values = tuple(values)
    if (
        len(values) < 2
        or values[0] != 0
        or any(type(value) is not int or value < 0 for value in values)
        or any(left >= right for left, right in pairwise(values))
    ):
        raise ValueError(f"{label} must be a nonempty strictly increasing partition")
    return values


@dataclass(frozen=True)
class Gfn2ElectronicTopology:
    """Immutable ragged AO/shell/atom topology for a fixed batch."""

    system_atom_offsets: tuple[int, ...]
    system_shell_offsets: tuple[int, ...]
    system_orbital_offsets: tuple[int, ...]
    orbital_to_shell: tuple[int, ...]
    orbital_to_atom: tuple[int, ...]

    def __post_init__(self) -> None:
        atoms = _offsets(self.system_atom_offsets, "atom offsets")
        shells = _offsets(self.system_shell_offsets, "shell offsets")
        orbitals = _offsets(self.system_orbital_offsets, "orbital offsets")
        object.__setattr__(self, "system_atom_offsets", atoms)
        object.__setattr__(self, "system_shell_offsets", shells)
        object.__setattr__(self, "system_orbital_offsets", orbitals)
        shell_map = tuple(self.orbital_to_shell)
        atom_map = tuple(self.orbital_to_atom)
        object.__setattr__(self, "orbital_to_shell", shell_map)
        object.__setattr__(self, "orbital_to_atom", atom_map)
        if not len(atoms) == len(shells) == len(orbitals):
            raise ValueError("GFN2 ragged partitions must have one entry per system")
        if len(shell_map) != orbitals[-1] or len(atom_map) != orbitals[-1]:
            raise ValueError("GFN2 orbital ownership maps have the wrong extent")
        shell_atoms: dict[int, int] = {}
        for system in range(self.system_count):
            shell_begin, shell_end = shells[system : system + 2]
            atom_begin, atom_end = atoms[system : system + 2]
            orbital_begin, orbital_end = orbitals[system : system + 2]
            for orbital in range(orbital_begin, orbital_end):
                shell = shell_map[orbital]
                atom = atom_map[orbital]
                if type(shell) is not int or not shell_begin <= shell < shell_end:
                    raise ValueError(
                        "GFN2 orbital-to-shell map crosses a system boundary"
                    )
                if type(atom) is not int or not atom_begin <= atom < atom_end:
                    raise ValueError(
                        "GFN2 orbital-to-atom map crosses a system boundary"
                    )
                if shell_atoms.setdefault(shell, atom) != atom:
                    raise ValueError("GFN2 shell orbitals must share one atom owner")

    @property
    def system_count(self) -> int:
        return len(self.system_orbital_offsets) - 1

    @property
    def atom_count(self) -> int:
        return self.system_atom_offsets[-1]

    @property
    def shell_count(self) -> int:
        return self.system_shell_offsets[-1]

    @property
    def orbital_count(self) -> int:
        return self.system_orbital_offsets[-1]

    @property
    def matrix_offsets(self) -> tuple[int, ...]:
        offsets = [0]
        for begin, end in zip(
            self.system_orbital_offsets,
            self.system_orbital_offsets[1:],
        ):
            orbitals = end - begin
            offsets.append(offsets[-1] + orbitals * orbitals)
        return tuple(offsets)

    @property
    def matrix_count(self) -> int:
        return self.matrix_offsets[-1]

    def _canonical_maps(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        forward = []
        reverse = []
        row_orbital = []
        column_orbital = []
        matrix_offsets = self.matrix_offsets
        for system in range(self.system_count):
            orbital_begin = self.system_orbital_offsets[system]
            orbitals = self.system_orbital_offsets[system + 1] - orbital_begin
            matrix_begin = matrix_offsets[system]
            for local_row in range(orbitals):
                for local_column in range(orbitals):
                    lower = min(local_row, local_column)
                    upper = max(local_row, local_column)
                    forward.append(matrix_begin + lower * orbitals + upper)
                    reverse.append(matrix_begin + upper * orbitals + lower)
                    row_orbital.append(orbital_begin + lower)
                    column_orbital.append(orbital_begin + upper)
        return (
            tuple(forward),
            tuple(reverse),
            tuple(row_orbital),
            tuple(column_orbital),
        )

    @property
    def canonical_forward(self) -> tuple[int, ...]:
        return self._canonical_maps()[0]

    @property
    def canonical_reverse(self) -> tuple[int, ...]:
        return self._canonical_maps()[1]

    @property
    def canonical_row_orbital(self) -> tuple[int, ...]:
        return self._canonical_maps()[2]

    @property
    def canonical_column_orbital(self) -> tuple[int, ...]:
        return self._canonical_maps()[3]

    def to_payload(self) -> dict:
        return {
            "system_atom_offsets": list(self.system_atom_offsets),
            "system_shell_offsets": list(self.system_shell_offsets),
            "system_orbital_offsets": list(self.system_orbital_offsets),
            "orbital_to_shell": list(self.orbital_to_shell),
            "orbital_to_atom": list(self.orbital_to_atom),
        }

    @property
    def identity(self) -> str:
        return canonical_hash(self.to_payload())


def _index(name: str, kind: str, size: int) -> Index:
    return Index(name, IndexSpace(name, kind, size))


def _input(
    name: str, indices: tuple[Index, ...], *, differentiable: bool = False
) -> Node:
    return input_tensor(
        name,
        TensorSpec(
            tuple(indices),
            role="input",
            differentiable=differentiable,
        ),
    )


def _scaled(value: Node, coefficient: str) -> Node:
    return add(value, coefficients=(coefficient,))


def _assemble_channel(
    *,
    h0: Node,
    overlap: Node,
    dipole_integrals: Node,
    quadrupole_integrals: Node,
    shell_potential: Node,
    dipole_potential: Node,
    quadrupole_potential: Node,
    topology: Gfn2ElectronicTopology,
    orbital: Index,
    matrix: Index,
) -> Node:
    forward, reverse, row_orbital, column_orbital = topology._canonical_maps()

    shell_by_orbital = indexed_gather(
        shell_potential,
        0,
        topology.orbital_to_shell,
        orbital,
    )
    row_scalar = indexed_gather(shell_by_orbital, 0, row_orbital, matrix)
    column_scalar = indexed_gather(shell_by_orbital, 0, column_orbital, matrix)
    scalar_sum = add(row_scalar, column_scalar)
    canonical_overlap = indexed_gather(overlap, 0, forward, matrix)
    scalar_shift = _scaled(multiply(canonical_overlap, scalar_sum), "-1/2")

    dipole_by_orbital = indexed_gather(
        dipole_potential,
        0,
        topology.orbital_to_atom,
        orbital,
    )
    dipole_row = indexed_gather(dipole_by_orbital, 0, row_orbital, matrix)
    dipole_column = indexed_gather(dipole_by_orbital, 0, column_orbital, matrix)
    dipole_forward = indexed_gather(dipole_integrals, 1, forward, matrix)
    dipole_reverse = indexed_gather(dipole_integrals, 1, reverse, matrix)
    dipole_forward_shift = einsum(
        "cm,mc->m",
        dipole_forward,
        dipole_column,
        coefficient="-1/2",
    )
    dipole_reverse_shift = einsum(
        "cm,mc->m",
        dipole_reverse,
        dipole_row,
        coefficient="-1/2",
    )

    quadrupole_by_orbital = indexed_gather(
        quadrupole_potential,
        0,
        topology.orbital_to_atom,
        orbital,
    )
    quadrupole_row = indexed_gather(
        quadrupole_by_orbital,
        0,
        row_orbital,
        matrix,
    )
    quadrupole_column = indexed_gather(
        quadrupole_by_orbital,
        0,
        column_orbital,
        matrix,
    )
    quadrupole_forward = indexed_gather(
        quadrupole_integrals,
        1,
        forward,
        matrix,
    )
    quadrupole_reverse = indexed_gather(
        quadrupole_integrals,
        1,
        reverse,
        matrix,
    )
    quadrupole_forward_shift = einsum(
        "cm,mc->m",
        quadrupole_forward,
        quadrupole_column,
        coefficient="-1/2",
    )
    quadrupole_reverse_shift = einsum(
        "cm,mc->m",
        quadrupole_reverse,
        quadrupole_row,
        coefficient="-1/2",
    )

    return add(
        h0,
        scalar_shift,
        dipole_forward_shift,
        dipole_reverse_shift,
        quadrupole_forward_shift,
        quadrupole_reverse_shift,
    )


@dataclass(frozen=True)
class Gfn2ElectronicProgram:
    method: XtbMethodIR
    topology: Gfn2ElectronicTopology
    reference: str
    program: Program
    version: str = GFN2_ELECTRONIC_VERSION

    def __post_init__(self) -> None:
        if self.method.model_flavor != "gfn2":
            raise ValueError("GFN2 electronic program requires model_flavor='gfn2'")
        if self.method.parameter_set.identity != GFN2_PARAMETER_SET.identity:
            raise ValueError(
                "GFN2 electronic program requires the audited parameter set"
            )
        if self.reference not in ("restricted", "unrestricted"):
            raise ValueError(
                "GFN2 electronic reference must be restricted or unrestricted"
            )
        if self.version != GFN2_ELECTRONIC_VERSION:
            raise ValueError("unsupported GFN2 electronic compiler version")

    @property
    def identity(self) -> str:
        return canonical_hash(
            {
                "version": self.version,
                "method": self.method.identity,
                "topology": self.topology.identity,
                "reference": self.reference,
                "equation": self.program.logical_hash,
                "reference_revision": GFN2_ELECTRONIC_REFERENCE_REVISION,
            }
        )

    def integral_vjp(self, output: str | None = None) -> VJPProgram:
        if "nuclear-gradient" not in self.method.requested_products:
            raise ValueError(
                "GFN2 S/D/Q adjoints require the nuclear-gradient compiler product"
            )
        if output is None:
            output = (
                "hamiltonian" if self.reference == "restricted" else "hamiltonian_alpha"
            )
        if output not in self.program.outputs:
            raise ValueError(f"unknown GFN2 Hamiltonian output: {output}")
        return transpose_program(
            self.program,
            [output],
            inputs=("overlap", "dipole_integrals", "quadrupole_integrals"),
        )


def build_gfn2_electronic_program(
    method: str | XtbMethodIR,
    topology: Gfn2ElectronicTopology,
    *,
    reference: str = "restricted",
) -> Gfn2ElectronicProgram:
    """Build fixed-state GFN2 Hamiltonian assembly without an SCC loop."""

    if isinstance(method, str):
        method = resolve_xtb_method(
            method,
            requested_products=("energy", "nuclear-gradient"),
        )
    elif not isinstance(method, XtbMethodIR):
        raise TypeError("method must be a GFN2 catalog name or XtbMethodIR")
    if method.model_flavor != "gfn2":
        raise ValueError("GFN2 electronic lowering requires model_flavor='gfn2'")
    if method.parameter_set.identity != GFN2_PARAMETER_SET.identity:
        raise ValueError("GFN2 electronic lowering requires the audited parameter set")
    if not isinstance(topology, Gfn2ElectronicTopology):
        raise TypeError("topology must be Gfn2ElectronicTopology")
    if reference not in ("restricted", "unrestricted"):
        raise ValueError("GFN2 electronic reference must be restricted or unrestricted")

    shell = _index("shell", "shell", topology.shell_count)
    atom = _index("atom", "atom", topology.atom_count)
    orbital = _index("orbital", "orbital", topology.orbital_count)
    matrix = _index("matrix", "matrix", topology.matrix_count)
    dipole_component = _index(
        "dipole_component",
        "component",
        len(GFN2_DIPOLE_COMPONENTS),
    )
    quadrupole_component = _index(
        "quadrupole_component",
        "component",
        len(GFN2_QUADRUPOLE_COMPONENTS),
    )

    h0 = _input("h0", (matrix,))
    overlap = _input("overlap", (matrix,), differentiable=True)
    dipole_integrals = _input(
        "dipole_integrals",
        (dipole_component, matrix),
        differentiable=True,
    )
    quadrupole_integrals = _input(
        "quadrupole_integrals",
        (quadrupole_component, matrix),
        differentiable=True,
    )
    shell_charge = _input("shell_scalar_potential", (shell,))
    dipole_charge = _input(
        "atomic_dipole_potential",
        (atom, dipole_component),
    )
    quadrupole_charge = _input(
        "atomic_quadrupole_potential",
        (atom, quadrupole_component),
    )

    common = {
        "h0": h0,
        "overlap": overlap,
        "dipole_integrals": dipole_integrals,
        "quadrupole_integrals": quadrupole_integrals,
        "topology": topology,
        "orbital": orbital,
        "matrix": matrix,
    }
    if reference == "restricted":
        outputs = {
            "hamiltonian": _assemble_channel(
                **common,
                shell_potential=shell_charge,
                dipole_potential=dipole_charge,
                quadrupole_potential=quadrupole_charge,
            )
        }
    else:
        shell_magnetization = _input(
            "shell_magnetization_potential",
            (shell,),
        )
        dipole_magnetization = _input(
            "atomic_dipole_magnetization_potential",
            (atom, dipole_component),
        )
        quadrupole_magnetization = _input(
            "atomic_quadrupole_magnetization_potential",
            (atom, quadrupole_component),
        )
        alpha_shell = add(shell_charge, shell_magnetization)
        beta_shell = add(
            shell_charge,
            shell_magnetization,
            coefficients=(1, -1),
        )
        alpha_dipole = add(dipole_charge, dipole_magnetization)
        beta_dipole = add(
            dipole_charge,
            dipole_magnetization,
            coefficients=(1, -1),
        )
        alpha_quadrupole = add(
            quadrupole_charge,
            quadrupole_magnetization,
        )
        beta_quadrupole = add(
            quadrupole_charge,
            quadrupole_magnetization,
            coefficients=(1, -1),
        )
        outputs = {
            "hamiltonian_alpha": _assemble_channel(
                **common,
                shell_potential=alpha_shell,
                dipole_potential=alpha_dipole,
                quadrupole_potential=alpha_quadrupole,
            ),
            "hamiltonian_beta": _assemble_channel(
                **common,
                shell_potential=beta_shell,
                dipole_potential=beta_dipole,
                quadrupole_potential=beta_quadrupole,
            ),
        }

    program = Program(
        outputs,
        provenance={
            "kind": "gfn2-fixed-state-electronic",
            "version": GFN2_ELECTRONIC_VERSION,
            "reference": reference,
            "topology": topology.to_payload(),
            "method_identity": method.identity,
            "reference_revision": GFN2_ELECTRONIC_REFERENCE_REVISION,
            "dipole_components": list(GFN2_DIPOLE_COMPONENTS),
            "quadrupole_components": list(GFN2_QUADRUPOLE_COMPONENTS),
            "directed_multipole_origin": "ket-ao-atom",
        },
    )
    return Gfn2ElectronicProgram(
        method,
        topology,
        reference,
        program,
    )
