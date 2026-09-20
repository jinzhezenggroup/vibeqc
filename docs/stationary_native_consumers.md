# Compiled stationary CPU consumers

The internal `complete_rks_gradient_diagnostic` has two explicit execution
routes. `reference` remains the default validated diagnostic; `native` replaces
its three scalar/TensorIR interpreter consumers without changing the complete
LDA/PBE RKS energy definition, stationary-state lease or signed source inventory.
Neither route enables public `Calculator` forces.

The separate [CUDA diagnostic](stationary_cuda_diagnostic.md) consumes a live
native CUDA snapshot and executes all seven sources on device with explicit
host export/orchestration. The CPU selectors and their boundaries below remain
unchanged.

```python
result = complete_rks_gradient_diagnostic(
    state,
    basis,
    cache=".cache/stationary-cpu",
    execution="native",
)
print(result.execution)
# compiled-cpu-consumers/python-numpy-orchestration-v1
```

Construct `state` from the actual live native CPU batch as documented in
[KS diagnostics](ks_diagnostics.md#native-cpu-stationary-gradient-diagnostic).
The same direct, all-electron, integer-occupation FP64 LDA/PBE RKS and s/p basis
domain applies. An unknown selector, unsupported spin/backend/ingredient,
stale state or failed provider raises rather than selecting a hidden fallback.

## Ownership and supported lowering

`tensor.cpu.NativeTensorProgram` consumes ordinary immutable TensorIR, not a
method/source-name switch. The initial FP64 subset is input, constant, add,
multiply, reduce and einsum; other operations, declared tensor symmetries and
other dtypes fail at generation. It reuses existing CUDA scalar-index helpers
without modifying CUDA emission. Input shape/dtype/finite checks, retained
numeric byte and scalar-work bounds precede native execution. Every generated
SSA output is checked; native outputs are copied from staging only on complete
success. Noncontiguous inputs are admitted with explicit staging.

`xc.native.NativeContractionProgram` supplies the existing compiler-generated
local AO-jet pullback. The authoritative point coefficients still come from
the exact native SCF-domain model through the live snapshot; the diagnostic
interior functional is not substituted. NumPy density-feature calculations,
BLAS matrix products and atom-map reductions remain host-side operations.

`xc.grid_native.NativeGridContraction` lowers the existing scalar
norm/ratio/log/Becke primal and partial graphs with `ScalarCEmitter`. The
bounded native traversal composes normalized-product adjoints and maps
point/nuclear motion to physical atoms. For `P` points and `A` atoms it makes
two `A*(A-1)/2` pair passes per point, plus one center-validation pass per call.
It retains atom-sized scratch rather than a point-by-coordinate Jacobian.
Scientific pair polynomials and local derivatives are not independently
reimplemented for PBE or another functional.

Seeds are the exact native atomic quadrature measure times the unweighted XC
energy density. Products are normalized in the log domain. Zero-factor counts
retain a nonzero derivative when exactly one rounded-zero factor contributes;
two zero factors kill that first derivative. Saturated/clipped branches and
collision rejection remain explicit. No force projection masks omissions.

## Resources, identity and failure

Tensor execution defaults to an 8 MiB logical numeric bound and 100 million
scalar-work units per program. Grid execution defaults to an 8 MiB component
bound and 100 million pair visits per call, counting center validation.
Checked arithmetic rejects impossible dimensions before dereferencing pointers
or allocating shape-dependent storage. Late arithmetic/provider failure leaves
raw output buffers unchanged. Empty point tiles and single-center domains have
explicit zero-result handling inside their admitted geometry contract.

These are component bounds, not a shared reservation covering SCF, snapshots,
compiler processes, mapped libraries, Python overhead and all simultaneously
live arrays. Such global resource admission and public capability registration
remain separate work. Python primitive enumeration, center scatter and NumPy
operations also remain; this is not a claim of an all-C++ molecular driver.

Generated sources are atomically published through a common source cache.
An existing corrupted source fails instead of being silently overwritten.
The shared artifact cache binds compiler, target, flags, source and complete
project-header contents. Runtime allocation/traversal templates are included
in compiler package assets. Source generation is tested with public runtime,
PySCF/CuPy/Torch imports, compiler processes and native loading forbidden.

## Validation and measurement

The maintained tests compare every signed source and total against independent
PySCF/Libxc/Libcint analytic references, and all nine water Cartesian derivatives
against three-step fully rebuilt/reconverged finite differences for LDA and PBE.
The native path also runs with scalar-array and TensorIR interpreter entrypoints
blocked. Dedicated native tests cover high-precision Decimal references,
rounded-zero products, translations/permutations/partial tiles, invalid
inputs/dtypes, tiny budgets, size overflow and transactional late failure.

```sh
PYTHONPATH=.:python VIBEQC_LIBRARY="$PWD/build/libvibeqc.so" \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CXX=c++ \
python -m pytest -q tests/python/test_tensor_cpu.py \
  tests/python/test_grid_native.py tests/python/test_dft_complete_cpu.py
```

For complete prepared CPU SCF + snapshot + AO owner + gradient measurements:

```sh
PYTHONPATH=.:python VIBEQC_LIBRARY="$PWD/build/libvibeqc.so" \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CXX=c++ \
python tools/benchmark_stationary_consumers.py --repeats 5 \
  --cache .cache/stationary-consumers --output /tmp/stationary-consumers.json
```

Pin an available CPU core externally when collecting latency evidence. The
runner warms both routes, alternates their order, uses independent prepared
states, checks equal SCF iteration counts and matched energy/gradient outputs,
and separates fixed from changed geometries. Initial warmup times are retained
but are **not** fair cold-compile comparisons because the routes share some
compiled primitives. No general system-size or GPU speedup follows from these
small CPU diagnostic workloads.

See the [decision record](../.agents/notes/implemented/architecture/2026-09-19-compiled-stationary-consumers.md)
for rationale, qualification evidence and the remaining public/native boundary.
