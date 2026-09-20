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
from math import isfinite
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
    constant,
    einsum,
    indexed_gather,
    input_tensor,
    multiply,
    scatter_add,
    transpose,
    transpose_program,
)

from .xtb import GFN2_PARAMETER_SET, XtbMethodIR, resolve_xtb_method

GFN2_ELECTRONIC_VERSION = "gfn2-fixed-state-electronic-ir-v1"
GFN2_MIXED_ELECTRONIC_VERSION = "gfn2-fixed-state-mixed-spin-electronic-ir-v1"
GFN2_POPULATION_VERSION = "gfn2-fixed-state-population-ir-v1"
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


@dataclass(frozen=True)
class Gfn2SpinLayout:
    """Immutable per-system restricted/unrestricted channel layout."""

    channels: tuple[int, ...]

    def __post_init__(self) -> None:
        channels = tuple(self.channels)
        if not channels or any(
            type(value) is not int or value not in (1, 2) for value in channels
        ):
            raise ValueError(
                "GFN2 spin channels must contain only one or two channels per system"
            )
        object.__setattr__(self, "channels", channels)

    def validate(self, topology: Gfn2ElectronicTopology) -> None:
        if len(self.channels) != topology.system_count:
            raise ValueError("GFN2 spin layout must have one channel count per system")

    def to_payload(self) -> dict:
        return {"channels": list(self.channels)}

    @property
    def identity(self) -> str:
        return canonical_hash(self.to_payload())


def _shell_to_atom(topology: Gfn2ElectronicTopology) -> tuple[int, ...]:
    owners: list[int | None] = [None] * topology.shell_count
    for shell, atom in zip(
        topology.orbital_to_shell, topology.orbital_to_atom, strict=True
    ):
        owner = owners[shell]
        if owner is None:
            owners[shell] = atom
        elif owner != atom:
            raise ValueError("GFN2 shell orbitals must share one atom owner")
    resolved_owners: list[int] = []
    for owner in owners:
        if owner is None:
            raise ValueError("GFN2 every shell must own at least one orbital")
        resolved_owners.append(owner)
    return tuple(resolved_owners)


def _packed_spin_maps(
    topology: Gfn2ElectronicTopology,
    spin_layout: Gfn2SpinLayout,
) -> tuple[
    tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]
]:
    """Map system-major spin matrices to immutable matrix/shell/atom owners."""

    spin_layout.validate(topology)
    density_to_matrix: list[int] = []
    density_to_shell: list[int] = []
    density_to_atom: list[int] = []
    density_to_system: list[int] = []
    magnetization_weight: list[int] = []
    matrix_offsets = topology.matrix_offsets
    for system, channels in enumerate(spin_layout.channels):
        orbital_begin = topology.system_orbital_offsets[system]
        orbitals = topology.system_orbital_offsets[system + 1] - orbital_begin
        matrix_begin = matrix_offsets[system]
        for channel in range(channels):
            spin_weight = 0 if channels == 1 else (1 if channel == 0 else -1)
            for local_row in range(orbitals):
                for local_column in range(orbitals):
                    orbital = orbital_begin + local_column
                    density_to_matrix.append(
                        matrix_begin + local_row * orbitals + local_column
                    )
                    density_to_shell.append(topology.orbital_to_shell[orbital])
                    density_to_atom.append(topology.orbital_to_atom[orbital])
                    density_to_system.append(system)
                    magnetization_weight.append(spin_weight)
    return (
        tuple(density_to_matrix),
        tuple(density_to_shell),
        tuple(density_to_atom),
        tuple(density_to_system),
        tuple(magnetization_weight),
    )


def _validated_gfn2_method(
    method: str | XtbMethodIR,
    *,
    requested_products: tuple[str, ...],
) -> XtbMethodIR:
    if isinstance(method, str):
        method = resolve_xtb_method(method, requested_products=requested_products)
    elif not isinstance(method, XtbMethodIR):
        raise TypeError("method must be a GFN2 catalog name or XtbMethodIR")
    if method.model_flavor != "gfn2":
        raise ValueError("GFN2 electronic lowering requires model_flavor='gfn2'")
    if method.parameter_set.identity != GFN2_PARAMETER_SET.identity:
        raise ValueError("GFN2 electronic lowering requires the audited parameter set")
    return method


@dataclass(frozen=True)
class Gfn2PopulationProgram:
    """Fixed-density Mulliken q/d/Q contractions and H0 energy bookkeeping."""

    method: XtbMethodIR
    topology: Gfn2ElectronicTopology
    spin_layout: Gfn2SpinLayout
    reference_shell_occupations: tuple[float, ...]
    program: Program
    version: str = GFN2_POPULATION_VERSION

    def __post_init__(self) -> None:
        if self.version != GFN2_POPULATION_VERSION:
            raise ValueError("unsupported GFN2 population compiler version")
        self.spin_layout.validate(self.topology)
        if len(self.reference_shell_occupations) != self.topology.shell_count:
            raise ValueError("GFN2 reference shell occupations have the wrong extent")

    @property
    def identity(self) -> str:
        return canonical_hash(
            {
                "version": self.version,
                "method": self.method.identity,
                "topology": self.topology.identity,
                "spin_layout": self.spin_layout.identity,
                "reference_shell_occupations": [
                    repr(x) for x in self.reference_shell_occupations
                ],
                "equation": self.program.logical_hash,
                "reference_revision": GFN2_ELECTRONIC_REFERENCE_REVISION,
            }
        )


def build_gfn2_population_program(
    method: str | XtbMethodIR,
    topology: Gfn2ElectronicTopology,
    *,
    spin_channels: tuple[int, ...],
    reference_shell_occupations: tuple[float, ...],
) -> Gfn2PopulationProgram:
    """Build fixed-density q/d/Q and P:H0 contractions for a ragged GFN2 batch.

    Density storage is system-major, then alpha/beta channel-major, then row-major
    AO matrices. Charge outputs are spin-summed. Magnetization follows the pinned
    xTBloom/tblite population convention ``raw_alpha - raw_beta``; because the
    raw Mulliken contraction is ``-P:I``, this is ``N_beta - N_alpha`` for overlap
    populations. Restricted systems have exact zero magnetization slices.
    """

    method = _validated_gfn2_method(method, requested_products=("energy",))
    if not isinstance(topology, Gfn2ElectronicTopology):
        raise TypeError("topology must be Gfn2ElectronicTopology")
    spin_layout = Gfn2SpinLayout(spin_channels)
    spin_layout.validate(topology)
    references = tuple(float(value) for value in reference_shell_occupations)
    if len(references) != topology.shell_count or any(
        not isfinite(value) for value in references
    ):
        raise ValueError(
            "GFN2 reference shell occupations must be finite and match shell extent"
        )

    (
        density_to_matrix,
        density_to_shell,
        density_to_atom,
        density_to_system,
        mag_weight,
    ) = _packed_spin_maps(topology, spin_layout)
    shell_to_atom = _shell_to_atom(topology)

    matrix = _index("matrix", "matrix", topology.matrix_count)
    density_element = _index("spin_matrix", "matrix", len(density_to_matrix))
    shell = _index("shell", "shell", topology.shell_count)
    atom = _index("atom", "atom", topology.atom_count)
    system = _index("system", "batch", topology.system_count)
    dipole_component = _index(
        "dipole_component", "component", len(GFN2_DIPOLE_COMPONENTS)
    )
    quadrupole_component = _index(
        "quadrupole_component", "component", len(GFN2_QUADRUPOLE_COMPONENTS)
    )

    density = _input("density", (density_element,), differentiable=True)
    h0 = _input("h0", (matrix,), differentiable=True)
    overlap = _input("overlap", (matrix,), differentiable=True)
    dipole_integrals = _input(
        "dipole_integrals", (dipole_component, matrix), differentiable=True
    )
    quadrupole_integrals = _input(
        "quadrupole_integrals", (quadrupole_component, matrix), differentiable=True
    )
    reference = constant(
        tuple(repr(value) for value in references),
        TensorSpec((shell,), role="constant"),
    )
    magnetization = constant(
        mag_weight,
        TensorSpec((density_element,), role="constant"),
    )

    overlap_by_density = indexed_gather(overlap, 0, density_to_matrix, density_element)
    overlap_charge_elements = einsum(
        "m,m->m", density, overlap_by_density, coefficient="-1"
    )
    weighted_density = multiply(density, magnetization)
    overlap_magnetization_elements = einsum(
        "m,m->m", weighted_density, overlap_by_density, coefficient="-1"
    )
    shell_charge = add(
        scatter_add(overlap_charge_elements, 0, density_to_shell, shell),
        reference,
    )
    shell_magnetization = scatter_add(
        overlap_magnetization_elements, 0, density_to_shell, shell
    )
    atom_charge = scatter_add(shell_charge, 0, shell_to_atom, atom)
    atom_magnetization = scatter_add(shell_magnetization, 0, shell_to_atom, atom)

    dipole_by_density = indexed_gather(
        dipole_integrals, 1, density_to_matrix, density_element
    )
    dipole_charge_elements = einsum(
        "m,cm->cm", density, dipole_by_density, coefficient="-1"
    )
    dipole_magnetization_elements = einsum(
        "m,cm->cm", weighted_density, dipole_by_density, coefficient="-1"
    )
    atomic_dipole = transpose(
        scatter_add(dipole_charge_elements, 1, density_to_atom, atom), (1, 0)
    )
    atomic_dipole_magnetization = transpose(
        scatter_add(dipole_magnetization_elements, 1, density_to_atom, atom), (1, 0)
    )

    quadrupole_by_density = indexed_gather(
        quadrupole_integrals, 1, density_to_matrix, density_element
    )
    quadrupole_charge_elements = einsum(
        "m,cm->cm", density, quadrupole_by_density, coefficient="-1"
    )
    quadrupole_magnetization_elements = einsum(
        "m,cm->cm", weighted_density, quadrupole_by_density, coefficient="-1"
    )
    atomic_quadrupole = transpose(
        scatter_add(quadrupole_charge_elements, 1, density_to_atom, atom), (1, 0)
    )
    atomic_quadrupole_magnetization = transpose(
        scatter_add(quadrupole_magnetization_elements, 1, density_to_atom, atom), (1, 0)
    )

    h0_by_density = indexed_gather(h0, 0, density_to_matrix, density_element)
    core_energy_elements = einsum("m,m->m", density, h0_by_density)
    core_energy = scatter_add(core_energy_elements, 0, density_to_system, system)

    program = Program(
        {
            "shell_charge": shell_charge,
            "shell_magnetization": shell_magnetization,
            "atomic_charge": atom_charge,
            "atomic_magnetization": atom_magnetization,
            "atomic_dipole": atomic_dipole,
            "atomic_dipole_magnetization": atomic_dipole_magnetization,
            "atomic_quadrupole": atomic_quadrupole,
            "atomic_quadrupole_magnetization": atomic_quadrupole_magnetization,
            "core_energy": core_energy,
        },
        provenance={
            "kind": "gfn2-fixed-state-population",
            "version": GFN2_POPULATION_VERSION,
            "topology": topology.to_payload(),
            "spin_layout": spin_layout.to_payload(),
            "method_identity": method.identity,
            "reference_revision": GFN2_ELECTRONIC_REFERENCE_REVISION,
            "density_packing": "system-spin-row-major",
            "multipole_origin": "ket-ao-atom",
            "magnetization": "raw-alpha-minus-raw-beta",
        },
    )
    return Gfn2PopulationProgram(method, topology, spin_layout, references, program)


@dataclass(frozen=True)
class Gfn2MixedElectronicProgram:
    """Mixed restricted/unrestricted fixed-state Hamiltonian in one ragged graph."""

    method: XtbMethodIR
    topology: Gfn2ElectronicTopology
    spin_layout: Gfn2SpinLayout
    program: Program
    version: str = GFN2_MIXED_ELECTRONIC_VERSION

    def __post_init__(self) -> None:
        if self.version != GFN2_MIXED_ELECTRONIC_VERSION:
            raise ValueError("unsupported mixed-spin GFN2 electronic compiler version")
        self.spin_layout.validate(self.topology)

    @property
    def identity(self) -> str:
        return canonical_hash(
            {
                "version": self.version,
                "method": self.method.identity,
                "topology": self.topology.identity,
                "spin_layout": self.spin_layout.identity,
                "equation": self.program.logical_hash,
                "reference_revision": GFN2_ELECTRONIC_REFERENCE_REVISION,
            }
        )

    def integral_vjp(self) -> VJPProgram:
        if "nuclear-gradient" not in self.method.requested_products:
            raise ValueError(
                "GFN2 S/D/Q adjoints require the nuclear-gradient compiler product"
            )
        return transpose_program(
            self.program,
            ["hamiltonian"],
            inputs=("overlap", "dipole_integrals", "quadrupole_integrals"),
        )


def build_gfn2_mixed_electronic_program(
    method: str | XtbMethodIR,
    topology: Gfn2ElectronicTopology,
    *,
    spin_channels: tuple[int, ...],
) -> Gfn2MixedElectronicProgram:
    """Build one system-major spin-packed Hamiltonian for a heterogeneous batch."""

    method = _validated_gfn2_method(
        method, requested_products=("energy", "nuclear-gradient")
    )
    if not isinstance(topology, Gfn2ElectronicTopology):
        raise TypeError("topology must be Gfn2ElectronicTopology")
    spin_layout = Gfn2SpinLayout(spin_channels)
    spin_layout.validate(topology)
    density_to_matrix, _, _, _, spin_weight = _packed_spin_maps(topology, spin_layout)

    shell = _index("shell", "shell", topology.shell_count)
    atom = _index("atom", "atom", topology.atom_count)
    orbital = _index("orbital", "orbital", topology.orbital_count)
    matrix = _index("matrix", "matrix", topology.matrix_count)
    spin_matrix = _index("spin_matrix", "matrix", len(density_to_matrix))
    dipole_component = _index(
        "dipole_component", "component", len(GFN2_DIPOLE_COMPONENTS)
    )
    quadrupole_component = _index(
        "quadrupole_component", "component", len(GFN2_QUADRUPOLE_COMPONENTS)
    )

    h0 = _input("h0", (matrix,))
    overlap = _input("overlap", (matrix,), differentiable=True)
    dipole_integrals = _input(
        "dipole_integrals", (dipole_component, matrix), differentiable=True
    )
    quadrupole_integrals = _input(
        "quadrupole_integrals", (quadrupole_component, matrix), differentiable=True
    )
    shell_charge = _input("shell_scalar_potential", (shell,))
    dipole_charge = _input("atomic_dipole_potential", (atom, dipole_component))
    quadrupole_charge = _input(
        "atomic_quadrupole_potential", (atom, quadrupole_component)
    )
    shell_magnetization = _input("shell_magnetization_potential", (shell,))
    dipole_magnetization = _input(
        "atomic_dipole_magnetization_potential", (atom, dipole_component)
    )
    quadrupole_magnetization = _input(
        "atomic_quadrupole_magnetization_potential", (atom, quadrupole_component)
    )

    common = {
        "overlap": overlap,
        "dipole_integrals": dipole_integrals,
        "quadrupole_integrals": quadrupole_integrals,
        "topology": topology,
        "orbital": orbital,
        "matrix": matrix,
    }
    charge_hamiltonian = _assemble_channel(
        **common,
        h0=h0,
        shell_potential=shell_charge,
        dipole_potential=dipole_charge,
        quadrupole_potential=quadrupole_charge,
    )
    zero_h0 = constant(
        (0,) * topology.matrix_count,
        TensorSpec((matrix,), role="constant"),
    )
    magnetization_shift = _assemble_channel(
        **common,
        h0=zero_h0,
        shell_potential=shell_magnetization,
        dipole_potential=dipole_magnetization,
        quadrupole_potential=quadrupole_magnetization,
    )
    expanded_charge = indexed_gather(
        charge_hamiltonian, 0, density_to_matrix, spin_matrix
    )
    expanded_magnetization = indexed_gather(
        magnetization_shift, 0, density_to_matrix, spin_matrix
    )
    spin_sign = constant(
        spin_weight,
        TensorSpec((spin_matrix,), role="constant"),
    )
    hamiltonian = add(expanded_charge, multiply(expanded_magnetization, spin_sign))
    program = Program(
        {"hamiltonian": hamiltonian},
        provenance={
            "kind": "gfn2-fixed-state-mixed-spin-electronic",
            "version": GFN2_MIXED_ELECTRONIC_VERSION,
            "topology": topology.to_payload(),
            "spin_layout": spin_layout.to_payload(),
            "method_identity": method.identity,
            "reference_revision": GFN2_ELECTRONIC_REFERENCE_REVISION,
            "spin_packing": "system-alpha-beta-row-major",
            "directed_multipole_origin": "ket-ao-atom",
        },
    )
    return Gfn2MixedElectronicProgram(method, topology, spin_layout, program)
