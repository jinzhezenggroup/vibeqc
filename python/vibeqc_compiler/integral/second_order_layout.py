"""Shell-local Hessian indexing and exact two-index translation recovery.

IntegralIR remains the owner of operator and derivative semantics. These
helpers describe small mathematical-center outputs and validation adapters;
they do not imply an executable second-order backend or a molecular Hessian.
"""

from dataclasses import dataclass
from math import sqrt

import numpy as np

from .blocks import TensorLayout
from .ir import DerivativeSpec, OperatorSpec
from .shell_signature import checked_index


@dataclass(frozen=True, slots=True)
class HessianLayout:
    """Ordered shell-center xyz pairs with dense or orthonormal svec storage.

    Coordinate order is ``(centers[0].x, .y, .z, centers[1].x, ...)``.
    Svec traverses the upper triangle by row and multiplies off-diagonal
    entries by sqrt(2). Consequently ordinary packed dot products equal the
    full Frobenius inner product, with no hidden factor for mixed derivatives.
    """

    centers: tuple[int, ...]
    packing: str = "dense"

    def __post_init__(self):
        object.__setattr__(self, "centers", tuple(self.centers))
        if not 1 <= len(self.centers) <= 4 or len(set(self.centers)) != len(
            self.centers
        ):
            raise ValueError(
                "a Hessian layout requires one to four distinct mathematical centers"
            )
        for center in self.centers:
            checked_index(center, "Hessian center")
        if self.packing not in ("dense", "svec"):
            raise ValueError("Hessian packing must be dense or svec")

    @property
    def dimension(self):
        """Number of ordered mathematical-center Cartesian coordinates."""
        return 3 * len(self.centers)

    @property
    def pairs(self):
        """Upper-triangular coordinate pairs in deterministic packed order."""
        return tuple(
            (i, j) for i in range(self.dimension) for j in range(i, self.dimension)
        )

    @property
    def tensor_layout(self):
        """Reuse the common numeric layout and byte-count contract."""
        if self.packing == "svec":
            return TensorLayout(("symmetric_coordinate_pair",), (len(self.pairs),))
        return TensorLayout(
            ("center_row", "xyz_row", "center_column", "xyz_column"),
            (len(self.centers), 3, len(self.centers), 3),
        )

    def encode(self, matrix):
        """Store a finite symmetric coordinate matrix without averaging its entries.

        This bounded diagnostic adapter allows only floating-point roundoff in
        mixed-partial symmetry. Asymmetric data cannot silently become a packed
        symmetric output, even when a caller only reads the upper triangle.
        """
        matrix = _finite_array(matrix, (self.dimension, self.dimension))
        scale = max(1.0, float(np.max(np.abs(matrix))))
        if np.max(np.abs(matrix - matrix.T)) > 64 * np.finfo(float).eps * scale:
            raise ValueError("Hessian matrix is not symmetric within roundoff")
        if self.packing == "dense":
            return matrix.reshape(self.tensor_layout.shape).copy()
        with np.errstate(over="raise", invalid="raise"):
            return np.array(
                [matrix[i, j] * (1 if i == j else sqrt(2)) for i, j in self.pairs]
            )

    def decode(self, values):
        """Recover a coordinate matrix using the same packed inner-product factors."""
        values = _finite_array(values, self.tensor_layout.shape)
        if self.packing == "dense":
            return values.reshape(self.dimension, self.dimension).copy()
        matrix = np.empty((self.dimension, self.dimension))
        for value, (i, j) in zip(values, self.pairs, strict=True):
            matrix[i, j] = matrix[j, i] = value / (1 if i == j else sqrt(2))
        return matrix


@dataclass(frozen=True, slots=True)
class CenterRecovery:
    """Small center recovery matrix R, shared by both Hessian indices.

    The complete Hessian is ``(R kron I3) H_ind (R kron I3).T``. An HVP first
    projects its direction with the transposed recovery matrix, then recovers
    its result with the forward matrix. Recovering only the output index would
    omit dependent-center contributions to the input direction.
    """

    centers: tuple[int, ...]
    independent: tuple[int, ...]
    rows: tuple[tuple[int, ...], ...]

    def __post_init__(self):
        object.__setattr__(self, "centers", tuple(self.centers))
        object.__setattr__(self, "independent", tuple(self.independent))
        object.__setattr__(self, "rows", tuple(tuple(row) for row in self.rows))
        HessianLayout(self.centers)
        HessianLayout(self.independent)
        if (
            not set(self.independent) <= set(self.centers)
            or len(self.centers) - len(self.independent) > 1
        ):
            raise ValueError(
                "one translation invariant can recover at most one requested center"
            )
        expected = tuple(
            tuple(
                -1 if center not in self.independent else int(center == other)
                for other in self.independent
            )
            for center in self.centers
        )
        if self.rows != expected or any(
            type(value) is not int for row in self.rows for value in row
        ):
            raise ValueError(
                "center recovery rows must match the declared independent-center basis"
            )

    @property
    def coordinate_matrix(self):
        """Expand center recovery into ordered xyz coordinates for diagnostics."""
        return np.kron(np.asarray(self.rows, dtype=float), np.eye(3))

    def project_direction(self, direction):
        """Project a fixed shell-center direction before differentiating its scalar."""
        direction = _finite_array(direction, (len(self.centers), 3))
        with np.errstate(over="raise", invalid="raise"):
            return np.asarray(self.rows, dtype=float).T @ direction

    def recover_hessian(self, independent):
        """Recover both raw coordinate indices for a small validation block."""
        independent = _finite_array(independent, (3 * len(self.independent),) * 2)
        matrix = self.coordinate_matrix
        with np.errstate(over="raise", invalid="raise"):
            return matrix @ independent @ matrix.T

    def recover_vector(self, independent):
        """Recover an HVP output after its input direction has been projected."""
        independent = _finite_array(independent, (len(self.independent), 3))
        with np.errstate(over="raise", invalid="raise"):
            return np.asarray(self.rows, dtype=float) @ independent


@dataclass(frozen=True, slots=True)
class SecondAtomMap:
    """Exact shell-center to physical-atom chain rule for small response tiles.

    Atom direction/output rows use sorted distinct physical atom indices.
    Repeated indices remain separate mathematical centers until the input
    direction is expanded and the resulting HVP is scattered. This implements
    A.T H A even when multiple basis slots share one atom.
    """

    centers: tuple[int, ...]
    center_atoms: tuple[int, ...]

    def __post_init__(self):
        object.__setattr__(self, "centers", tuple(self.centers))
        object.__setattr__(self, "center_atoms", tuple(self.center_atoms))
        HessianLayout(self.centers)
        if len(self.centers) != len(self.center_atoms):
            raise ValueError(
                "each requested mathematical center needs one physical atom"
            )
        for atom in self.center_atoms:
            checked_index(atom, "physical atom")

    @property
    def atom_indices(self):
        """Stable compact atom output order, including noncontiguous atom labels."""
        return tuple(sorted(set(self.center_atoms)))

    @property
    def matrix(self):
        """Center-to-atom incidence; its transpose scatters a center response."""
        return np.array(
            [
                [int(atom == other) for other in self.atom_indices]
                for atom in self.center_atoms
            ],
            dtype=float,
        )

    def expand_direction(self, atom_direction):
        """Copy each physical atom displacement to all requested center slots."""
        atom_direction = _finite_array(atom_direction, (len(self.atom_indices), 3))
        return self.matrix @ atom_direction

    def scatter_hvp(self, center_response):
        """Sum center responses only after complete translation recovery."""
        center_response = _finite_array(center_response, (len(self.centers), 3))
        with np.errstate(over="raise", invalid="raise"):
            return self.matrix.T @ center_response

    def scatter_hessian(self, center_hessian):
        """Diagnostic small-block chain rule on both coordinate indices."""
        center_hessian = _finite_array(center_hessian, (3 * len(self.centers),) * 2)
        incidence = np.kron(self.matrix, np.eye(3))
        with np.errstate(over="raise", invalid="raise"):
            return incidence.T @ center_hessian @ incidence


def second_center_recovery(
    operator: OperatorSpec, derivative: DerivativeSpec
) -> CenterRecovery:
    """Read existing exact invariants without calling the first-derivative ABI.

    A partial requested-center set recovers a center only when its complete
    declared invariant is available. Other mathematical coordinates are held
    fixed and remain outside this requested Hessian subblock.
    """
    if derivative.order != 2:
        raise ValueError("second-center recovery requires derivative order two")
    if not set(derivative.invariants) <= set(operator.invariants):
        raise ValueError("second derivative invariants must belong to the operator")
    centers = derivative.requested_centers(operator)
    dependent = tuple(
        c
        for invariant in derivative.invariants
        if (c := invariant.recovered_center(centers, operator.centers)) is not None
    )
    independent = tuple(c for c in centers if c not in dependent)
    rows = tuple(
        tuple(-1 if c in dependent else int(c == i) for i in independent)
        for c in centers
    )
    return CenterRecovery(centers, independent, rows)


def second_coordinate_tiles(centers, *, packing="dense", hvp=False, tile_size=6):
    """Partition shell-local coordinate outputs without allocating a Hessian.

    Native helpers own at most twelve coordinates each. The default six-root
    tile permits recomputation of geometry across tiles to bound scalar CSE
    liveness; callers may choose any explicit size in the validated interval.
    AO component tiling remains an independent consumer selection.
    """
    checked_index(tile_size, "coordinate tile size", minimum=1)
    if tile_size > 12 or type(hvp) is not bool:
        raise ValueError(
            "coordinate tiling requires a boolean HVP flag and at most twelve roots"
        )
    layout = HessianLayout(centers, packing)
    if hvp and packing != "dense":
        raise ValueError("HVP coordinate outputs are not svec Hessian pairs")
    size = layout.dimension if hvp else layout.tensor_layout.storage_elements
    return tuple(
        tuple(range(begin, min(size, begin + tile_size)))
        for begin in range(0, size, tile_size)
    )


def _finite_array(value, shape):
    """Reject complex, nonfinite or mismatched scientific diagnostic buffers."""
    if np.iscomplexobj(value):
        raise ValueError("Hessian buffers must be real")
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"expected a finite Hessian buffer of shape {shape}")
    return array
