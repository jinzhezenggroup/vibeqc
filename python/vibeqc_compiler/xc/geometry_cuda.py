"""CUDA AO translation pullback and Becke local partials from shared graphs."""

from vibeqc_compiler.dft.ao import jet_indices
from vibeqc_compiler.integral.scalar_c import ScalarCEmitter

from .coefficients import jet_pullback_program
from .grid_native import emit_grid_partials


def emit_geometry_cuda(*, pbe, iterations=3):
    """Lower AO bilinear AD; the caller supplies exact SCF point coefficients."""
    if type(pbe) is not bool:
        raise TypeError("geometry lowering requires a boolean PBE flag")
    program = jet_pullback_program("gga" if pbe else "lda")
    variables = {
        **{f"c{j}": f"c[{j}]" for j in range(len(program.roots))},
        **{f"{leg}{j}": f"w[{j}]" for leg in "xy" for j in range(len(program.roots))},
    }
    emitter = ScalarCEmitter(program.graph, variables)
    emitter.emit(program.roots)
    domain = jet_indices(1 if pbe else 0)
    lookup = jet_indices(2 if pbe else 1)
    shifts = []
    for index in domain:
        row = []
        for k in range(3):
            shifted = list(index)
            shifted[k] += 1
            row.append(lookup.index(tuple(shifted)))
        shifts.append("{" + ",".join(map(str, row)) + "}")
    return "\n".join(
        [
            '#include "dft/grid_response_adjoint.hpp"',
            emit_grid_partials(iterations, device=True),
            f"constexpr bool stationary_pbe = {'true' if pbe else 'false'};",
            f"constexpr unsigned stationary_jets = {len(domain)};",
            f"__device__ __constant__ unsigned stationary_shift[{len(domain)}][3] = {{{','.join(shifts)}}};",
            "__device__ void ao_pullback(const double* c, const double* w, double* out) {",
            *emitter.lines,
            *(
                f"out[{j}] = {emitter.reference(r)};"
                for j, r in enumerate(program.roots)
            ),
            "}",
            "",
        ]
    )
