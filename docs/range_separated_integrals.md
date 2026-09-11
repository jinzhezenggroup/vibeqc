# Range-separated four-center integrals

The integral compiler distinguishes ordinary `1/r`, long-range
`erf(omega*r)/r`, and short-range `erfc(omega*r)/r` operators. `omega` is a
finite, nonnegative parameter in inverse Bohr. At zero omega, long-range
integrals vanish and short-range integrals become full Coulomb integrals.
Ordinary Coulomb requests require omega zero.

`CoulombKernel` and `four_center_eri_operator` construct the mathematical
identity. Range requests use IntegralIR schema version 2, including the exact
normalized FP64 parameter. Ordinary requests retain schema version 1 and
their existing generated-source identity. Changing either the family or
omega changes the range request's IR and emitted source.

The existing external-weight Hermite DAG generates a scalar and twelve
shell-center derivatives. Omega and Gaussian exponents are held fixed during
nuclear differentiation. Independent shell slots remain separate even when
two slots belong to one physical atom; the consumer accumulates them after
differentiation. Weights are fixed cotangents, including any explicitly
requested exchange coefficient or sign. Their electronic response belongs
to the caller.

The modified moments obey the same chain rule as ordinary Boys moments:

```text
b = omega / hypot(omega, sqrt(rho))
LR_n(T) = integral from 0 to b of u^(2n) exp(-T*u^2) du
SR_n(T) = integral from b to 1 of u^(2n) exp(-T*u^2) du
d moment_n / dT = -moment_(n+1)
```

The native evaluator uses positive interval quadrature through order 13,
which includes the additional derivative moment for public f/f/f/f shells.
Short-range evaluation uses a rational expression for the interval width
and never subtracts full and long-range values. Factored decay and powers
resolve narrow intervals and large Boys arguments. The independent Python
validation oracle uses adaptive SciPy quadrature; SciPy is not a generation
or execution dependency.

`emit_weighted_eri_primitive_header` binds that evaluator to the existing
generated CPU/CUDA scalar arithmetic. Its input weights follow the explicit
component subset order, so a selected f/f/f/f component does not require a
full shell weight array. Subsets contain at most 64 components. Primitive
contraction and normalized public-basis pullbacks remain consumer operations.
These callables are experimental and unscreened. The shared primitive stream
emits range-tagged 224-byte v2 records containing the exact omega; ordinary
Coulomb streams retain their 208-byte format. The leading family tag makes
legacy native execution reject range records before dispatch. A v2 executor
validates the stride, version, family and compiled omega together. Legacy
HF emitters continue to reject range-separated requests.

## Explicit prepared execution

`weighted_eri_execute.compile_weighted_eri` compiles a range request with an
explicit `CppCompilerAdapter` or `CudaCompilerAdapter`. It reuses the shared
native artifact cache and verifies the source, compiler, header and binary
identities. The new CPU cache schema is version 2; existing CUDA cache keys
retain their version 1 identity. Installed wheels bundle all required native
headers and can compile and execute without a checkout or reference engine.

`PreparedWeightedEri` retains bounded record staging and compact result
buffers. CUDA uses the same arena, stream, device checks and metrics owner as
TensorIR and DFT. `contract` accepts streams from
`prepare_weighted_eri_stream`, including signed exchange cotangents prepared
with the existing `eri_weights` interfaces. It returns detached, read-only
rows containing a scalar and twelve ordered shell-center derivatives.
`weighted_eri_response` applies the request's center selection or physical-atom
scatter. Callers fold distinct AO permutation weights explicitly; the stream
adds no symmetry factor and never screens a record.

For example, after constructing normalized primitives, Bohr coordinates and
a weighted stream with the existing block adapter:

```python
from pathlib import Path

from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.integral.weighted_eri_execute import (
    PreparedWeightedEri,
    compile_weighted_eri,
)

artifact = compile_weighted_eri(
    stream.request.integral, CppCompilerAdapter(Path("c++")), Path("local-cache")
)
with PreparedWeightedEri(artifact, record_capacity=64, tile_capacity=3) as plan:
    weighted = plan.contract(stream)
    raw = plan.raw(primitives, centers, (0, 1, 2), projections=projections)
```

`raw` consumes flattened public-basis component indices and runs unit
cotangents through that same provider. Raw rows have unit sign and coefficient,
independent of the weighted request's factors. A spherical signature requires
explicit normalized Cartesian-to-public projection matrices. Every nonzero
Cartesian term of a selected public component must be present in the compiled
subset; incomplete coverage is rejected. Large classes therefore require
explicit caller-selected subsets, and a public component needing more than
64 Cartesian terms cannot use this first raw adapter. No dense molecular
four-index derivative tensor is created.

Operator family, exact omega, angular components and backend are immutable
program identities. Changing omega requires another compiled artifact and
prepared plan. Geometry, primitive coefficients and weights can change between
calls. Each stream is validated before dispatch; each native record repeats
the radial/ABI check. Invalid or nonfinite results leave caller output
unpublished, and valid replay remains possible after a failed call. Calls,
diagnostics and close operations on one Python plan serialize with its lock.

The shared resource planner preflights numeric host/device capacities. Its
request includes native publication storage, upload staging, chunk results and
one detached result. `raw` reserves an additional bounded stream workspace
and one nested row. Caller streams, Python/compiler object metadata, native
call stacks and CUDA context are explicitly excluded; stream preparation has
its own budget. Device ordinal and allocation-size ABI limits are checked
before native allocation. Returned diagnostics retain resource identities,
stream identities, chunk counts and optional device timings.

Capability queries use `cpu_range_weighted_eri` and
`cuda_range_weighted_eri`, with `component_indices` for an explicit bounded
subset. These names describe eligibility for the experimental prepared
provider and its raw method. The existing `cuda` and `cuda_weighted_eri`
capabilities retain their full-Coulomb boundary.

## Reproducible validation

Install `.[test,reference-test]` to run the independent reference suites.
CI selects this extra, pinning PySCF 2.14.0. Ordinary runtime installation
does not depend on PySCF or SciPy; reference tests skip when they are absent.

Native CPU/CUDA tests compare LR and SR separately against Libcint values
and all four center derivatives for psss, dpsp, fsss, and selected f/f/f/f
components. Moment tests cover zero and enormous omega, small and enormous
Boys arguments, translation, and multiple finite-difference steps. No RSH
functional, range-separated DF, Hessian, omega derivative, or performance
promotion follows from this capability. Prepared tests additionally cover
contracted Cartesian/spherical raw values, all eight quartet permutations,
exchange/orbit weights, partial chunks, multiple tiles, capacity/identity
failures, failed-call isolation and replay. CPU ASan/UBSan and scheduled CUDA
memcheck exercise the generated bounded native lifecycle.

`tools/validate_range_eri.py` retains independent raw and weighted LR/SR
reference values, all center derivatives, translation checks, three
finite-difference steps, LR+SR identities, shared resource plans, compiler
resources and synchronous timing samples. For CUDA, run it with an explicit
finite Slurm allocation and the desired compiler on `PATH`:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:10:00 python tools/validate_range_eri.py --backend cuda \
  --architecture sm_120 --output .artifacts/range-cuda-run
```

Use `--backend cpu` for CPU evidence and `--publish <new-directory>` to select
a compact bundle with the existing validation/publication schema. Publication
requires committed measured source. Timing samples are diagnostic; no
production schedule or speedup is promoted without the separate performance
gates.
