"""Bounded native XC matrix schedules generated from the compact bilinear.

A 16x16 shared-memory contraction reuses each density/AO load across a tile.
Potential assembly packs the scalar/spatial coefficients into the dead density
work panels, then evaluates a symmetric cross product. The tiled path also
folds the deterministic three-channel point-total reduction into the potential
launch. No library handle, provider workspace, global allocation, screening or
precision change is needed.
"""

from typing import Any

from vibeqc_compiler.integral.cuda import CudaEmitter

from .xc_bilinear import ao_pair_bilinear


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


_TILED = r"""
// Padding avoids bank conflicts when the AO lane reads a transposed panel.
// Partial tiles still participate in both barriers and load exact zero padding.
__global__ void tiled_density_product(const double* density, const double* ao, I n, I count,
                                      I work_jets, double* work, int* error) {
  __shared__ double d[16][17], a[16][17];
  const I x = threadIdx.x, y = threadIdx.y;
  const I mu = I(blockIdx.x)*16+x, point = I(blockIdx.y)*16+y;
  const I spin = blockIdx.z/work_jets, jet = blockIdx.z%work_jets;
  const I panel = count*n;
  const double* source = ao+jet*panel;
  const double* matrix = density+spin*n*n;
  double value = 0.0;
  for (I begin = 0; begin < n; begin += 16) {
    const I row = I(blockIdx.x)*16+y, col = begin+x;
    d[y][x] = row < n && col < n ? 0.5*matrix[row*n+col]+0.5*matrix[col*n+row] : 0.0;
    a[y][x] = point < count && col < n ? source[point*n+col] : 0.0;
    __syncthreads();
    for (I k = 0; k < 16; ++k) value += d[x][k]*a[y][k];
    __syncthreads();
  }
  if (mu < n && point < count)
    work[(spin*work_jets+jet)*panel+point*n+mu] = finite(value, error, 1);
}

// One triangle is authoritative, including on diagonal and partial blocks.
// A compact linear block domain enumerates only tile_mu <= tile_nu instead of
// launching the unused lower half of a square grid. One lane decodes the tile
// pair; all 256 lanes then execute the unchanged symmetric contraction. The
// first block of spin zero also performs the historical serial-per-channel
// point-total reduction after its matrix work, preserving the exact arithmetic
// order while avoiding a separate kernel launch for every point tile.
__global__ void tiled_potential(const double* ao, const double* work, I n, I count,
                                I work_jets, const double* point_totals, double* potential,
                                double* totals, int* error) {
  __shared__ I tile_mu, tile_nu;
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    const I pair = blockIdx.x;
    I low = 0, high = (n+15)/16;
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
  __shared__ double am[16][17], an[16][17], wm[16][17], wn[16][17];
  const I x = threadIdx.x, y = threadIdx.y;
  const I mu = tile_mu*16+x, nu = tile_nu*16+y;
  const I nu_load = tile_nu*16+x, spin = blockIdx.z, panel = count*n;
  double value = 0.0;
  for (I jet = 0; jet < work_jets; ++jet) {
    const double* a = ao+jet*panel;
    const double* w = work+(spin*work_jets+jet)*panel;
    for (I begin = 0; begin < count; begin += 16) {
      const I p = begin+y;
      am[y][x] = p < count && mu < n ? a[p*n+mu] : 0.0;
      an[y][x] = p < count && nu_load < n ? a[p*n+nu_load] : 0.0;
      wm[y][x] = p < count && mu < n ? w[p*n+mu] : 0.0;
      wn[y][x] = p < count && nu_load < n ? w[p*n+nu_load] : 0.0;
      __syncthreads();
      for (I k = 0; k < 16; ++k) value += am[k][x]*wn[k][y]+wm[k][x]*an[k][y];
      __syncthreads();
    }
  }
  if (mu < n && nu < n && mu <= nu) {
    const I index = (spin*n+mu)*n+nu;
    value = finite(potential[index]+value, error, 3);
    potential[index] = value;
    potential[(spin*n+nu)*n+mu] = value;
  }
  if (blockIdx.x == 0 && blockIdx.z == 0 && threadIdx.y == 0 && threadIdx.x < 3) {
    const I channel = threadIdx.x;
    double sum = 0.0;
    for (I p = 0; p < count; ++p) sum += point_totals[channel*count+p];
    totals[channel] = finite(totals[channel]+sum,error,3);
  }
}

// The compiler owns schedule admission; native only supplies borrowed buffers.
// Tiny shapes and dimensions outside the two-dimensional launch domain retain
// the bounded grid-stride scalar schedule. Both cover the same scientific work.
inline bool tiled_xc_admitted(I n, I count) {
  return n >= 16 && count >= 16 && (n+15)/16 <= 65535 && (count+15)/16 <= 65535;
}
inline void scheduled_density_product(cudaStream_t stream, const double* density,
    const double* ao, I n, I count, I spins, I work_jets, double* work, int* error) {
  if (tiled_xc_admitted(n, count)) {
    tiled_density_product<<<dim3((n+15)/16, (count+15)/16, spins*work_jets),
                            dim3(16,16), 0, stream>>>(density, ao, n, count, work_jets, work, error);
  } else {
    density_product<<<vibeqc_tensor::blocks(spins*work_jets*count*n,128),128,0,stream>>>(
        density,ao,n,count,spins,work_jets,work,error);
  }
}
inline void scheduled_potential(cudaStream_t stream, const double* ao,
    const double* coefficients, const double* weights, I n, I count, I spins,
    I terms, I work_jets, double* work, const double* point_totals,
    double* potential, double* totals, int* error) {
  if (tiled_xc_admitted(n, count)) {
    compact_potential_panels<<<vibeqc_tensor::blocks(spins*count*n,128),128,0,stream>>>(
        ao,coefficients,weights,n,count,spins,terms,work_jets,work,error);
    vibeqc_tensor::cuda_check(cudaGetLastError());
    const I tiles = (n+15)/16, tile_pairs = tiles*(tiles+1)/2;
    tiled_potential<<<dim3(tile_pairs,1,spins),dim3(16,16),0,stream>>>(
        ao,work,n,count,work_jets,point_totals,potential,totals,error);
  } else {
    assemble_potential<<<vibeqc_tensor::blocks(spins*n*n,128),128,0,stream>>>(
        ao,coefficients,weights,n,count,spins,terms,potential,error);
    vibeqc_tensor::cuda_check(cudaGetLastError());
    accumulate_totals<<<1,32,0,stream>>>(point_totals,count,totals,error);
  }
}
"""


def emit_native_xc_matrix_schedule() -> str:
    """Emit compact graph lowering and an allocation-free shared-memory schedule."""
    return (
        "\nnamespace vibeqc::dft::cuda_xc_detail {\nnamespace {\n"
        + _emit_panels()
        + _TILED
        + "\n} // namespace\n} // namespace vibeqc::dft::cuda_xc_detail\n"
    )
