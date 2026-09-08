# HF snapshots and integral providers (CG10)

The internal `tools.vibeqc_posthf` interface supplies validated reference states,
explicitly indexed MO integrals, and an MP2 validation bridge. It does not
register an MP2 or CC method in the public calculator. Source checkouts expose
this development interface with `PYTHONPATH=.:python`; PySCF is needed only to
regenerate independent test fixtures.

## Reference ownership and validity

`ReferenceSnapshot` owns immutable FP64 host copies of S, h, F, C, orbital
energies and occupations. Their storage is backed by immutable bytes, so
turning a NumPy write flag back on cannot mutate a cached reference. A snapshot
retains geometry/basis identities, generation ID, electron count, HF energy,
physical SCF residual, representation, screening/precision controls, Hamiltonian
identity, HF backend and device ordinal. The original HF arena can be released.

Only canonical, real, closed-shell RHF is supported. Validation checks S
conditioning, CᵀSC=I, CᵀFC=diag(epsilon), FC=SC epsilon, SCF convergence, sorted
orbitals and ordered 2/0 spatial occupations. ROHF/UHF, complex orbitals,
linear-dependence-reduced spaces, missing virtual orbitals and frozen masks
fail explicitly. All electrons participate. The frozen-mask field represents
future intent and never silently enables frozen-core calculations.

The reference identity hashes all arrays and metadata, including C and the
generation ID. Recomputing on the same topology creates fresh state. Geometry,
basis, occupation, metric or algorithm changes cannot reuse a provider built
for another snapshot. Individual failed native HF exports raise before a
snapshot is returned; independent neighboring source/provider objects survive.

`export_rhf(source)` runs VibeQC's existing native RHF solver, exports its
owned density, rebuilds the physical Fock with the specified Hamiltonian, and
canonicalizes on the CPU. It checks both the original SCF commutator and the
density/Fock change after canonicalization. This initial convenience exporter
is limited to 12 AOs. The existing CPU HF oracle's dense tensors and HF forces
are part of that separate HF setup, outside subsequent provider budgets.
CUDA HF export explicitly stages the density and canonicalizes on the host.
Production providers can consume a validated supplied snapshot directly.

## Index dictionary

| Object | Exact definition and axis order |
| --- | --- |
| AO/MO ERI | Chemists' `(pq|rs)`, `g[p,q,r,s]`; spatial orbitals |
| MO coefficients | `C[mu,p]`; occupied columns first, then virtual columns |
| Occupied/virtual slots | Explicit tuples of global MO columns; `o=(0,...,nocc-1)`, `v=(nocc,...,nmo-1)` |
| MP2 `ovov` block | `g[i,a,j,b]=(ia|jb)`; a shorthand is expanded into four explicit slots |
| Restricted t2 | `t[i,j,a,b]=(ia|jb)/(epsilon_i+epsilon_j-epsilon_a-epsilon_b)` |
| Restricted symmetry | `t[i,j,a,b]=t[j,i,b,a]`; neither occupied nor virtual axes are independently antisymmetric |
| DF B | `B[Q,p,q]=sum_(mu,nu,P) C[mu,p] C[nu,q] A[mu,nu,P] M^(-1/2)[P,Q]` |

`MOBlock` stores the four explicit slot tuples. All occupied/virtual block
combinations needed by a later CC implementation are available if their full
requested output fits the budget, including `oooo`, `ooov`, `oovv`, `ovov`,
`ovvv` and `vvvv`. Arbitrarily ordered unique subsets and empty selections are
supported. No hidden pair compression or symmetry weight is applied.
`ovov_to_ijab` is the explicit transpose `(0,2,1,3)` for the amplitude numerator.

For a complete 2-AO/2-MO tensor the transformation is

```python
one = np.einsum('uvwx,up->pvwx', ao, C, optimize=False)
two = np.einsum('pvwx,vq->pqwx', one, C, optimize=False)
three = np.einsum('pqwx,wr->pqrx', two, C, optimize=False)
mo = np.einsum('pqrx,xs->pqrs', three, C, optimize=False)
```

`test_complete_tiny_transform_explicit_eight_loops` independently checks every
entry against the eight nested summations. `dense_ao_to_mo` guards all dense
oracle inputs at 12 AOs/MOs; streaming providers do not call it.

## Sources, transforms and placement

`NativeSource` owns normalized basis/geometry data copied from native system
handles. It adapts the existing independent contracted Hermite integral
implementation to values-only shell-component tiles. It allocates neither
molecular AO N⁴ tensors nor nuclear derivative jets. Cartesian and
real-spherical bases through f use the existing normalization and expansions.
Unsupported angular momenta fail before evaluation. CG02 `BlockRequest`,
`ShellTile`, `RawBlock`, center bindings and `BlockResponse` carry exact
shell identities, offsets, layouts and explicit status. Partial component
tiles, padding, signs and empty global slices are checked.

The conventional CPU provider contracts one AO axis at a time with NumPy
matrix products. The CUDA provider uploads selected coefficient panels once,
retains a private stream, cuBLAS handle, reusable panels and complete requested
MO output, and performs all four transforms plus accumulation on the device.
The AO source is CPU-evaluated and explicitly host-staged. A cache hit retains
the device block; `device_pointer` is borrowed with the owning object, and
`to_host()` explicitly downloads a detached copy. Ordinary streams are used.
No CUDA graph or entirely device-resident integral-generation claim is made.

The pinned cache is default: repeated residual requests do not repeat a full
AO-to-MO transform. An exhausted budget fails before evaluation. Optional LRU
policy reports evictions and recomputation. `clear()`/`close()` invalidate
borrowed CUDA block objects; the source has its own independent lifetime.
Calls on one provider are serialized. Distinct providers own distinct buffers,
handles and streams. Handle preparation/destruction reuses CG09's allocation
measurement lock; executions can remain independent.

## Budgets and costs

`ConventionalProvider.plan(block)` reports a checked capacity before reading
integrals or allocating a GPU plan. It counts the reference/source numeric
state, selected C panels, four-coordinate raw response copies, two rotating
stage panels, requested outputs, host staging/publication/download capacity,
and retained cached outputs. The source has a separate conservative 8 MiB
through-f recurrence allowance. CuBLAS plans add a 4 MiB explicit workspace
and CG09's checked 96 MiB retained-provider allowance. Native construction
verifies the exact arena size and provider allocation delta; failure releases
partial state. Native cuBLAS dimensions are bounded to int32.

These are numeric-buffer capacity bounds, not portable total-process/VRAM
bounds. Python/C++ object headers, allocator rounding, BLAS host workspaces,
CUDA context/module/stack storage and caller-retained detached exports are
outside scope. Multiple simultaneously live providers each have their own
budget; callers must sum them for a fleet. Timings separate setup, source
work, transformations, device transfers, explicit download and cache reuse.

## Density fitting

`MetricFactor.from_source` reuses the SCF factorization policy. It records the
auxiliary basis identity, actual metric hash, relative/absolute threshold,
effective rank and **square symmetric thresholded inverse square root**.
Directions with eigenvalue <= threshold × largest eigenvalue are zeroed; the
auxiliary dimension stays uncompressed. Geometry/metric changes invalidate the
DF Hamiltonian identity. The metric requires bounded O(naux²) setup storage.

`DFProvider.three_index(p,q,auxiliary_begin=...,auxiliary_count=...)` constructs
only the requested B tile. `get(block)` streams auxiliary tiles, reconstructs
the requested four-index MO output and pins it for reuse. No full AO N⁴ tensor
or full-molecule B tensor is required. Whitening and MO transformation are
currently CPU operations.

`CudaDFSource` replaces only the raw A/M source with CG05's generated Rys
implementation. It retains native source metadata and a bounded device tile,
while explicitly staging the metric and raw tiles to the CPU DF provider.
A source selecting a different backend is rejected. The existing source
reports its setup capacity after preparation; a separate source-budget check
releases an oversized source before exposing it. This setup check is distinct
from preflight transformation budgets. GPU generation and D2H transfer times
are measured separately. This mixed pipeline is not a GPU DF transformation.

## MP2 bridge and independent references

The restricted bridge evaluates
`E_corr=sum_ijab t_ijab [2(ia|jb)-(ib|ja)]`. A separate small spin-orbital oracle
sums `1/4 |<IJ||AB>|²/D` with explicit spin deltas. Small or nonnegative
canonical denominators fail with the problematic indices; they are never
clamped. CUDA MO blocks are explicitly downloaded for this CPU bridge.

`tools/generate_posthf_references.py` pins PySCF 2.14.0 and records original
geometry/basis coefficients, exact AO normalization, orbital phases, settings,
licenses, versions and array hashes. Committed H2/H2O/LiH/HeH+ fixtures include
an actual f shell. Every MO element, amplitude, exchange factor and MP2 energy
is independently checked. DF fixtures use a separately reconstructed **same DF
Hamiltonian** in PySCF. Conventional-versus-DF fitting differences are reported
separately and do not loosen layout or MP2 gates. A second generation checks
identical array hashes. CI consumes these fixtures without importing PySCF.

```bash
PYTHONPATH=.:python VIBEQC_LIBRARY=$PWD/build/cpu/libvibeqc.so \
  python -m pytest tests/python/test_posthf_reference.py \
  tests/python/test_posthf_providers.py -q
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:10:00 \
  env PYTHONPATH=.:python VIBEQC_LIBRARY=$PWD/build/cuda/libvibeqc.so \
  VIBEQC_POSTHF_CUDA_TEST=1 python -m pytest tests/python/test_posthf_cuda.py -q
```

The CUDA transform runtime is compiled explicitly with `compile_cuda`, using
the shared finite NVCC process-tree adapter and a hash-verified local cache.
The generated DF source also requires the native library built with CUDA.

The [CG10 evidence archive](../benchmarks/results/posthf-147/README.md) records
the clean scientific revision, independent numerical gates, sanitizer logs,
capacity bounds and raw cold/reuse timings. CUDA is slower on these small
host-source fixtures; this interface validation makes no performance promotion.
