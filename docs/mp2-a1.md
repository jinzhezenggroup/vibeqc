# Canonical MP2 A1 internal prototype (historical development boundary)

Part of #193, acceptance A1 within original A. This document describes the
earlier internal consumer retained for regression/reference work. The current
native/public implementation and its placement contract are documented in
[mp2.md](mp2.md). This historical prototype is not the acceptance target or
proof that the public native route has passed its outstanding gates.

## Scientific contract

Reuse the validated immutable CG10 `ReferenceSnapshot`: real FP64, canonical
closed-shell RHF, all electrons, occupied columns first with spatial 2/0
occupations, supported Cartesian/real-spherical all-electron bases, and the
unscreened conventional Hamiltonian. No RI, open-shell, frozen-core, ECP or
analytic derivatives are enabled. The energy consumer never calls PySCF.

For every ordered spatial tuple `(i,j,a,b)`, define
`g=(ia|jb)`, `x=(ib|ja)`, `D=eps_i+eps_j-eps_a-eps_b < 0`:

```text
E_OS   = sum g*g/D
E_SS   = sum g*(g-x)/D       # alpha-alpha plus beta-beta
E_corr = E_OS + E_SS        # sum g*(2*g-x)/D
E_MP2  = E_RHF + E_corr
```

There is no additional 1/2 or 1/4 in these restricted full-domain sums. The
independent spin-orbital test uses `1/4 sum |<IJ||AB>|^2/D` and splits occupied
spins explicitly. Restricted t2 would obey `(i,j,a,b) <-> (j,i,b,a)`, not separate
occupied/virtual antisymmetry. No complete t2 output is allocated.

Each direct block has CG10 slots `(i,a,j,b)` and is transposed `(0,2,1,3)`.
The exchange request has slots `(i,b,j,a)` and is transposed `(0,2,3,1)` to
the same local ijab shape. This separate request is necessary for disjoint or
unequal virtual tiles; swapping axes inside the direct tile is incorrect.

## Execution and memory boundary

`tools.vibeqc_mp2.PreparedMP2Energy` creates its own CG10 conventional provider,
borrows the caller-owned source/reference, and queries CG10/CG09 plans before
any integral reads or device preparation. It clears only its own block cache.
Provider source passes repeat per energy block, which bounds storage but is
not an optimized transformation schedule. There is no speedup claim.

The default CPU path uses the unchanged TensorIR reference interpreter. An
explicit `energy_backend="cuda"` uses CG09 to compile and execute each complete
tile equation natively. It requires an explicit compiler/cache and never falls
back. `integral_backend="cuda"` reuses CG10's cuBLAS transforms and requires its
compiled transform artifact. The CPU AO value source is unchanged. GPU MO
outputs are downloaded and reuploaded into the current host-input tensor ABI.
The molecular tile loop and compensated scalar fold remain Python/CPU. These
are disclosed development paths, not a completed public native method.

The single internal numeric budget composes the provider's peak, detached tile
feeds, temporary arithmetic/validation capacity and the interpreter or CG09
tensor peak. Only one tensor shape's runtime remains live, including at shape
transitions. It includes the existing provider's retained CUDA allowance where
applicable. It is not a measured RSS/free-VRAM bound: Python metadata, allocator
rounding, BLAS host internals and CUDA context/module/stack are outside scope.
Caller-owned previous outputs and external HF setup require separate accounting.
The current `export_rhf` remains a 12-AO bridge whose CPU HF setup may allocate
dense integrals and forces; this consumer does not change or conceal that fact.

Denominator extrema are checked in constant storage before integral reads;
near-zero/nonnegative/nonfinite denominators are errors, never regularized.
Invalid references are rejected by CG10. Source identity changes/closure,
nonfinite inputs/intermediates, infeasible budgets and backend errors are
explicit. Execution states are `prepared/running/ready/failed/closed`; any
failed execution clears `last_result` rather than publishing partial energy.
Force or amplitude properties raise `NotImplementedError` before reading
integrals. Neither HF forces nor a zero placeholder is returned as an MP2 force.

## Reproduction and evidence

From this worktree, with its own Python environment and `PYTHONPATH=.:python`:

```bash
python -m pytest tests/python/test_mp2_energy.py -q
python tests/mp2_energy_evidence.py --output build/mp2-a1/numerical-evidence.json
```

The evidence runner deliberately uses the **dense test-only AO fixture source**
to exercise the unchanged streamed CG10 transform. H2, H2O, LiH and HeH+ with an
actual f shell reuse hash-checked PySCF 2.14.0 fixtures from #147, including its
original exporter, geometry, basis coefficients, versions and licenses.
Independent explicit spin sums check OS and SS at `atol=1e-11, rtol=1e-10`;
correlation energies use the existing `1e-9 Eh` gate. PySCF is not installed or
used as a production backend. CPU fixture success establishes neither native
HF setup nor device execution. The runner emits the shared #138 evidence schema.

Inside a separately authorized, allocated Slurm GPU job, use a CUDA-enabled
native library and an isolated build/cache; this command does not submit a job:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.:python \
VIBEQC_LIBRARY="$PWD/build/cuda/libvibeqc.so" \
VIBEQC_MP2_CUDA_TEST=1 VIBEQC_NVCC=/path/to/nvcc VIBEQC_MP2_ARCH=sm_120 \
python -m pytest tests/python/test_mp2_cuda.py -q \
  --basetemp=build/mp2-a1/gpu-tmp --junitxml=build/mp2-a1/gpu.xml
```

Repeat under Compute Sanitizer before device acceptance; retain hashes,
toolchain/device identities, measured memory and explicit transfer records.
The current opt-in tests cover supplied-orbital native integral/transform/tile
energy parity and replay; they are not yet public HF-to-MP2 acceptance tests.

The handoff lists exact SHA anchors, environment, commands, logs, full baseline
comparison and remaining native/device gates. D owns independent review.
Recommended three review areas: (1) OS/SS and exchange/denominator semantics;
(2) numeric lifetime/budget composition and invalidation/failure publication;
(3) truthful public/native/device capability boundaries and next shared patch.
