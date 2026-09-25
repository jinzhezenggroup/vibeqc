"""Bounded native XC matrix schedules generated from the compact bilinear.

Qualified shared-memory contractions reuse each density/AO load across a tile.
The conservative production fallback remains the already-qualified 16x16 shape.
Potential assembly packs the scalar/spatial coefficients into the dead density
work panels, then evaluates a symmetric cross product. The tiled path also
folds the deterministic three-channel point-total reduction into the potential
launch. No library handle, provider workspace, global allocation, screening or
precision change is needed.
"""

from dataclasses import dataclass
from typing import Any

from vibeqc_compiler.integral.cuda import CudaEmitter

from .xc_bilinear import ao_pair_bilinear


@dataclass(frozen=True, slots=True)
class XcMatrixSchedule:
    """One compiler-owned CUDA matrix tile with explicit resource admission.

    Alternative tiles are compile/tuning candidates only. Production keeps the
    qualified 16x16 schedule unless a caller explicitly selects another legal
    candidate and independently qualifies it.
    """

    tile: int

    def __post_init__(self) -> None:
        if type(self.tile) is not int or self.tile < 1 or self.tile & (self.tile - 1):
            raise ValueError("XC matrix tile must be a positive power of two")
        # All three totals channels need an x lane; larger tiles can exceed
        # CUDA block/shared-memory limits. Direct emitters share this boundary.
        if self.tile not in (8, 16, 32):
            raise ValueError(
                "XC matrix tile must be a supported candidate (8, 16, 32)"
            )

    @property
    def threads(self) -> int:
        return self.tile * self.tile

    @property
    def density_shared_bytes(self) -> int:
        return 2 * self.tile * (self.tile + 1) * 8

    @property
    def potential_shared_bytes(self) -> int:
        # Four padded FP64 panels plus the two shared tile indices.
        return 4 * self.tile * (self.tile + 1) * 8 + 16

    @property
    def shared_bytes(self) -> int:
        return max(self.density_shared_bytes, self.potential_shared_bytes)

    @property
    def additional_workspace_bytes(self) -> int:
        """The tiled candidates reuse already-borrowed AO/work panels."""

        return 0

    def admitted(
        self,
        n: int,
        count: int,
        *,
        spins: int = 1,
        work_jets: int = 1,
        maximum_threads_per_block: int = 1024,
        maximum_shared_bytes: int = 48 * 1024,
        maximum_grid_dimension: int = 65535,
    ) -> bool:
        """Check launch/resource legality without making a profitability claim."""

        for label, value in (
            ("AO count", n),
            ("point count", count),
            ("spin count", spins),
            ("work jet count", work_jets),
            ("maximum threads", maximum_threads_per_block),
            ("maximum shared bytes", maximum_shared_bytes),
            ("maximum grid dimension", maximum_grid_dimension),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        tiles = (n + self.tile - 1) // self.tile
        point_tiles = (count + self.tile - 1) // self.tile
        tile_pairs = tiles * (tiles + 1) // 2
        return (
            n >= self.tile
            and count >= self.tile
            and self.threads <= maximum_threads_per_block
            and self.shared_bytes <= maximum_shared_bytes
            and tiles <= maximum_grid_dimension
            and point_tiles <= maximum_grid_dimension
            and tile_pairs <= maximum_grid_dimension
            and spins * work_jets <= maximum_grid_dimension
        )


XC_MATRIX_SCHEDULE_CANDIDATES = tuple(XcMatrixSchedule(tile) for tile in (8, 16, 32))
DEFAULT_XC_MATRIX_SCHEDULE = XcMatrixSchedule(16)


def qualified_xc_matrix_schedules(
    n: int,
    count: int,
    *,
    spins: int = 1,
    work_jets: int = 1,
    maximum_threads_per_block: int = 1024,
    maximum_shared_bytes: int = 48 * 1024,
    maximum_grid_dimension: int = 65535,
) -> tuple[XcMatrixSchedule, ...]:
    """Return resource-legal candidates; ordering is not a performance ranking."""

    return tuple(
        schedule
        for schedule in XC_MATRIX_SCHEDULE_CANDIDATES
        if schedule.admitted(
            n,
            count,
            spins=spins,
            work_jets=work_jets,
            maximum_threads_per_block=maximum_threads_per_block,
            maximum_shared_bytes=maximum_shared_bytes,
            maximum_grid_dimension=maximum_grid_dimension,
        )
    )


def select_xc_matrix_schedule(
    n: int,
    count: int,
    *,
    preferred_tile: int | None = None,
    spins: int = 1,
    work_jets: int = 1,
    maximum_threads_per_block: int = 1024,
    maximum_shared_bytes: int = 48 * 1024,
    maximum_grid_dimension: int = 65535,
) -> XcMatrixSchedule | None:
    """Resolve an explicit qualified candidate or preserve the 16x16 fallback.

    With no profile/preference, this deliberately does *not* promote a new tile
    from a static heuristic. If the qualified 16x16 schedule is unavailable,
    callers retain the scalar fallback. Autotuning may request 8x8 or 32x32
    explicitly and benchmark it under the same scientific contract.
    """

    candidates = qualified_xc_matrix_schedules(
        n,
        count,
        spins=spins,
        work_jets=work_jets,
        maximum_threads_per_block=maximum_threads_per_block,
        maximum_shared_bytes=maximum_shared_bytes,
        maximum_grid_dimension=maximum_grid_dimension,
    )
    if preferred_tile is not None:
        if type(preferred_tile) is not int or preferred_tile < 1:
            raise ValueError("preferred XC matrix tile must be a positive integer")
        for schedule in candidates:
            if schedule.tile == preferred_tile:
                return schedule
        raise ValueError(f"XC matrix tile {preferred_tile} is not qualified")
    return (
        DEFAULT_XC_MATRIX_SCHEDULE if DEFAULT_XC_MATRIX_SCHEDULE in candidates else None
    )


def compact_panel_program(family: str) -> tuple:
    """Factor the symmetric bilinear into A*W^T + W*A^T.

    The value panel absorbs both gradient legs. The scalar and diagonal kinetic
    terms get a one-half because symmetric assembly visits each twice. Derive
    their coefficients from the canonical bilinear instead of restating them.
    """
    graph, x, y, _coefficients, bilinear = ao_pair_bilinear(family)

    def entry(row: int, column: int) -> Any:
        return graph.differentiate(graph.differentiate(bilinear, x[row]), y[column])

    value = entry(0, 0) * x[0] / 2
    for j in range(1, len(x)):
        value += entry(j, 0) * x[j]
    roots = [value]
    if family == "mgga":
        roots.extend(entry(j, j) * x[j] / 2 for j in range(1, 4))
    return graph, tuple(roots)


def _emit_panels() -> str:
    lines = [
        r"""
__global__ void compact_potential_panels(const double* ao, const double* coefficients,
    const double* weights, I n, I count, I spins, I terms, I work_jets, double* work, int* error) {
  const I panel = count*n;
  for (I i = I(blockIdx.x)*blockDim.x+threadIdx.x; i < spins*panel;
       i += I(blockDim.x)*gridDim.x) {
    const I spin = i/panel, index = i%panel, point = index/n;
"""
    ]
    for family, terms in (("lda", 1), ("gga", 4), ("mgga", 5)):
        graph, roots = compact_panel_program(family)
        bindings = {f"x{j}": f"ao[{j}*panel+index]" for j in range(4)}
        bindings.update(
            {f"c{j}": f"coefficients[(spin*terms+{j})*count+point]" for j in range(5)}
        )
        emitter = CudaEmitter(graph, bindings)
        emitter.emit(roots)
        lines.append(f"    if (terms == {terms}) {{")
        lines.extend("    " + line for line in emitter.lines)
        lines.extend(
            f"      work[(spin*work_jets+{j})*panel+index] = finite(weights[point]*({emitter.reference(root)}), error, 3);"
            for j, root in enumerate(roots)
        )
        lines.append("    }")
    lines.append("  }\n}\n")
    return "\n".join(lines)


_TILED_TEMPLATE = r"""
// Padding avoids bank conflicts when the AO lane reads a transposed panel.
// Partial tiles still participate in both barriers and load exact zero padding.
__global__ void tiled_density_product(const double* density, const double* ao, I n, I count,
                                      I work_jets, double* work, int* error) {
  __shared__ double d[@TILE@][@PAD@], a[@TILE@][@PAD@];
  const I x = threadIdx.x, y = threadIdx.y;
  const I mu = I(blockIdx.x)*@TILE@+x, point = I(blockIdx.y)*@TILE@+y;
  const I spin = blockIdx.z/work_jets, jet = blockIdx.z%work_jets;
  const I panel = count*n;
  const double* source = ao+jet*panel;
  const double* matrix = density+spin*n*n;
  double value = 0.0;
  for (I begin = 0; begin < n; begin += @TILE@) {
    const I row = I(blockIdx.x)*@TILE@+y, col = begin+x;
    d[y][x] = row < n && col < n ? 0.5*matrix[row*n+col]+0.5*matrix[col*n+row] : 0.0;
    a[y][x] = point < count && col < n ? source[point*n+col] : 0.0;
    __syncthreads();
    for (I k = 0; k < @TILE@; ++k) value += d[x][k]*a[y][k];
    __syncthreads();
  }
  if (mu < n && point < count)
    work[(spin*work_jets+jet)*panel+point*n+mu] = finite(value, error, 1);
}

// One triangle is authoritative, including on diagonal and partial blocks.
// A compact linear block domain enumerates only tile_mu <= tile_nu instead of
// launching the unused lower half of a square grid. One lane decodes the tile
// pair; all @THREADS@ lanes then execute the unchanged symmetric contraction. The
// first block of spin zero also performs the historical serial-per-channel
// point-total reduction after its matrix work, preserving the exact arithmetic
// order while avoiding a separate kernel launch for every point tile. The first
// point tile owns output initialization, so admitted tiled execution needs no
// matrix-sized or totals memset before every XC evaluation.
__global__ void tiled_potential(const double* ao, const double* work, I n, I count,
                                I work_jets, const double* point_totals, double* potential,
                                double* totals, bool accumulate, int* error) {
  __shared__ I tile_mu, tile_nu;
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    const I pair = blockIdx.x;
    I low = 0, high = (n+@TILE_MINUS_ONE@)/@TILE@;
    while (low+1 < high) {
      const I mid = (low+high)/2;
      if (mid*(mid+1)/2 <= pair)
        low = mid;
      else
        high = mid;
    }
    tile_nu = low;
    tile_mu = pair-low*(low+1)/2;
  }
  __syncthreads();
  __shared__ double am[@TILE@][@PAD@], an[@TILE@][@PAD@], wm[@TILE@][@PAD@], wn[@TILE@][@PAD@];
  const I x = threadIdx.x, y = threadIdx.y;
  const I mu = tile_mu*@TILE@+x, nu = tile_nu*@TILE@+y;
  const I nu_load = tile_nu*@TILE@+x, spin = blockIdx.z, panel = count*n;
  double value = 0.0;
  for (I jet = 0; jet < work_jets; ++jet) {
    const double* a = ao+jet*panel;
    const double* w = work+(spin*work_jets+jet)*panel;
    for (I begin = 0; begin < count; begin += @TILE@) {
      const I p = begin+y;
      am[y][x] = p < count && mu < n ? a[p*n+mu] : 0.0;
      an[y][x] = p < count && nu_load < n ? a[p*n+nu_load] : 0.0;
      wm[y][x] = p < count && mu < n ? w[p*n+mu] : 0.0;
      wn[y][x] = p < count && nu_load < n ? w[p*n+nu_load] : 0.0;
      __syncthreads();
      for (I k = 0; k < @TILE@; ++k) value += am[k][x]*wn[k][y]+wm[k][x]*an[k][y];
      __syncthreads();
    }
  }
  if (mu < n && nu < n && mu <= nu) {
    const I index = (spin*n+mu)*n+nu;
    const double prior = accumulate ? potential[index] : 0.0;
    value = finite(prior+value, error, 3);
    potential[index] = value;
    potential[(spin*n+nu)*n+mu] = value;
  }
  if (blockIdx.x == 0 && blockIdx.z == 0 && threadIdx.y == 0 && threadIdx.x < 3) {
    const I channel = threadIdx.x;
    double sum = 0.0;
    for (I p = 0; p < count; ++p) sum += point_totals[channel*count+p];
    totals[channel] = finite((accumulate ? totals[channel] : 0.0)+sum,error,3);
  }
}

// The compiler owns schedule admission; native only supplies borrowed buffers.
// Tiny shapes and dimensions outside the two-dimensional launch domain retain
// the bounded grid-stride scalar schedule. Both cover the same scientific work.
inline bool tiled_xc_admitted(I n, I count, I spins, I work_jets) {
  if (n < @TILE@ || count < @TILE@ || spins < 1 || work_jets < 1 ||
      work_jets > 65535 || spins > 65535/work_jets) return false;
  // Avoid overflow before rejecting an out-of-domain launch shape.
  const I tiles = 1+(n-1)/@TILE@;
  const I point_tiles = 1+(count-1)/@TILE@;
  if (tiles > 65535 || point_tiles > 65535) return false;
  const I tile_pairs = tiles*(tiles+1)/2;
  return tile_pairs <= 65535;
}
inline void scheduled_density_product(cudaStream_t stream, const double* density,
    const double* ao, I n, I count, I spins, I work_jets, double* work, int* error) {
  if (tiled_xc_admitted(n, count, spins, work_jets)) {
    tiled_density_product<<<dim3((n+@TILE_MINUS_ONE@)/@TILE@, (count+@TILE_MINUS_ONE@)/@TILE@, spins*work_jets),
                            dim3(@TILE@,@TILE@), 0, stream>>>(density, ao, n, count, work_jets, work, error);
  } else {
    density_product<<<vibeqc_tensor::blocks(spins*work_jets*count*n,128),128,0,stream>>>(
        density,ao,n,count,spins,work_jets,work,error);
  }
}
inline void scheduled_potential(cudaStream_t stream, const double* ao,
    const double* coefficients, const double* weights, I n, I count, I spins,
    I terms, I work_jets, double* work, const double* point_totals,
    double* potential, double* totals, bool accumulate, int* error) {
  if (tiled_xc_admitted(n, count, spins, work_jets)) {
    compact_potential_panels<<<vibeqc_tensor::blocks(spins*count*n,128),128,0,stream>>>(
        ao,coefficients,weights,n,count,spins,terms,work_jets,work,error);
    vibeqc_tensor::cuda_check(cudaGetLastError());
    const I tiles = (n+@TILE_MINUS_ONE@)/@TILE@, tile_pairs = tiles*(tiles+1)/2;
    tiled_potential<<<dim3(tile_pairs,1,spins),dim3(@TILE@,@TILE@),0,stream>>>(
        ao,work,n,count,work_jets,point_totals,potential,totals,accumulate,error);
  } else {
    if (!accumulate) {
      vibeqc_tensor::cuda_check(cudaMemsetAsync(potential,0,spins*n*n*sizeof(double),stream));
      vibeqc_tensor::cuda_check(cudaMemsetAsync(totals,0,3*sizeof(double),stream));
    }
    assemble_potential<<<vibeqc_tensor::blocks(spins*n*n,128),128,0,stream>>>(
        ao,coefficients,weights,n,count,spins,terms,potential,error);
    vibeqc_tensor::cuda_check(cudaGetLastError());
    accumulate_totals<<<1,32,0,stream>>>(point_totals,count,totals,error);
  }
}
"""


def _emit_tiled(schedule: XcMatrixSchedule) -> str:
    if not isinstance(schedule, XcMatrixSchedule):
        raise TypeError("XC matrix emission requires XcMatrixSchedule")
    replacements = {
        "@TILE@": str(schedule.tile),
        "@PAD@": str(schedule.tile + 1),
        "@TILE_MINUS_ONE@": str(schedule.tile - 1),
        "@THREADS@": str(schedule.threads),
    }
    source = _TILED_TEMPLATE
    for token, value in replacements.items():
        source = source.replace(token, value)
    return source


def emit_native_xc_matrix_schedule(
    schedule: XcMatrixSchedule = DEFAULT_XC_MATRIX_SCHEDULE,
) -> str:
    """Emit compact graph lowering for one explicitly qualified tile candidate."""
    return (
        "\nnamespace vibeqc::dft::cuda_xc_detail {\nnamespace {\n"
        + _emit_panels()
        + _emit_tiled(schedule)
        + "\n} // namespace\n} // namespace vibeqc::dft::cuda_xc_detail\n"
    )
