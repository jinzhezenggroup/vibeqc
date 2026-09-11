# Generated density-fitting response

CUDA DF-HF gradients use generated two-/three-center derivatives and contract
external response weights before downloading the gradient. The former
coordinate-wise CUDA force implementations and `VIBEQC_DF_DERIVATIVES` selector
are removed. CPU DF calculations and independent CPU/libcint validation remain
available. The separate `VIBEQC_ONE_ELECTRON_DERIVATIVES` selector still controls
overlap, kinetic, and nuclear-attraction response and has its own promotion gate.

## Independent response contract

For raw three-center integrals `A[mu,nu,P]` and the auxiliary Coulomb metric
`M[P,Q]`, the generic consumer evaluates

\[
 g_{ax}=\sum_{\mu\nu P}\bar A_{\mu\nu P}
          \frac{\partial A_{\mu\nu P}}{\partial R_{ax}}
       +\sum_{PQ}\bar M_{PQ}\frac{\partial M_{PQ}}{\partial R_{ax}}.
\]

The C API `vibeqc_system_df_gradient_cuda` accepts full row-major weights,
including arbitrary nonsymmetric matrices. The auxiliary index is contiguous
in A: `((mu*nbf+nu)*naux+P)`. Each dense element contributes once; there are no
implicit triangular factors. A null weight pointer denotes zero for that
operator, with the declared full dimension still required. The returned
quantity is a **positive energy gradient**, without one-electron, overlap
Pulay, or nuclear-repulsion terms. HF force assembly negates the total once.
The interface does not require a density matrix or an HF energy expression.

Orbital and auxiliary systems share physical atom coordinates, but their
shells can have different owners, orderings, and Cartesian/spherical
representations. Both auxiliary centers move in dM. The three mathematical
centers of dA accumulate on their actual atoms, including shared-atom cases.
The standalone API changes its output only on success and restores the
calling thread's CUDA device after stream cleanup.

## Mathematical generation

`python/vibeqc_compiler/integral/df_derivatives.py` differentiates the physical value DAG
introduced by the generated DF value implementation. The interpreter covers
the prefactor, Gaussian decay, shifts, and Boys argument; differentiating a
Boys node uses `dF_m(T)/dT = -F_(m+1)(T)`.

The CUDA emitter rewrites the same Gaussian axis-moment DAG as coefficients
in `u=t^2`. Products of the three axis polynomials integrate against shared
Boys moments F0 through F10. Center derivatives use the Gaussian identity
`d_A g_a = 2*alpha*g_(a+1) - a*g_(a-1)`; A/B responses share primitive and
axis-polynomial intermediates, and C follows from translation. This avoids
differentiating fitted quadrature roots and avoids runtime AD objects. The
mixed-axis covariance is `(.5/p) * (p/(p+q)) * u`.

Public shells remain s/p/d/f. Raising one internal orbital power to four is
an implementation requirement of differentiation and does not expose a new
public shell family. A metric primitive uses its two real auxiliary centers;
no normalized dummy basis contributes to its derivative.

Bulk coordinate tensors, bounded metric/three-center source tiles, and the
externally weighted consumer instantiate the same generated derivative policy.
`src/runtime/cuda_gaussian_products.cuh` owns rank-generic traversal of normalized
primitive products and sparse Cartesian AO terms. Its policy supplies the
scientific evaluator and accumulated channels; the traversal supplies cyclic
lane ownership, physical atom projection, and final gradient scatter. DF values
instantiate this same runtime with a separately generated value policy.
Adding a supported angular class therefore does not require another native
formula-specific contraction loop. Coordinate consumers project the contracted
center channels, while weighted consumers scatter them after contraction.

The shared generated scalar headers use internal device linkage so independent
CUDA translation units can include them safely. Value and derivative policies
have separate headers; a derivative-only consumer does not register unused Rys
value tables. CPU and libcint definitions remain independent numerical oracles.

`--derivatives` selects these programs in `tools/generate_df_kernels.py` and
`tools/validate_df_values.py`. Raw all-class fixtures include coincident
centers, center translation, independent libcint derivatives, spherical
projection, and multiple finite-difference steps.

## HF reverse chain

For a symmetric density D, write `rho_P = sum_ij D_ij A_ijP` and
`Q_PQ = A_P : (D^T A_Q D)`. A contribution to the two-electron energy has form

\[
 E=\tfrac12 c_J\rho^T M^+\rho-c_K Q:M^+.
\]

The bounded host adapter in `src/scf/df_response_weights.cpp` forms

\[
 \bar{M^+}_{PQ}=\tfrac12 c_J\rho_P\rho_Q-c_K Q_{PQ},\qquad
 \bar A_{ijP}=c_J D_{ij}(M^+\rho)_P
              -2c_K\sum_Q(M^+)_{PQ}(D^T A_Q D)_{ij}.
\]

RHF uses `(c_J,c_K)=(1,1/4)`. UHF adds total-density `(1,0)` and the two
spin-density `(0,1/2)` contributions. The adapter retains only a bounded
auxiliary block of raw AO matrices and response weights. Its two-dimensional
strided callback submits that whole block, avoiding a complete transpose into
dense A-weight storage. The same range mapping preserves row position when an
upload splits a block inside an AO row. Exposing auxiliary and AO-pair work in
one launch avoids underfilling the GPU with one small kernel per auxiliary.
Callback uploads finish before these host buffers are reused.
Partial auxiliary blocks deliberately reread Q slices and recompute exchange
responses. Retaining all such responses can exceed the same hard budget;
secondary blocking or an explicitly charged optional cache remains future
performance work. The generated adapter applies the metric reverse map once
per HF response. Raw coordinate derivatives remain available through the generated
integral API; complete CUDA HF response no longer materializes those tensors.

The metric weight is the self-adjoint spectral Frechet response of the
truncated pseudoinverse applied to `bar_(M+)`. In an eigenbasis its divided
difference is `(f(lambda_i)-f(lambda_j))/(lambda_i-lambda_j)`, with
`f(lambda)=1/lambda` in the retained subspace and zero in the discarded
subspace. Retained/retained pairs use `-1/(lambda_i*lambda_j)`; discarded pairs
use zero. Mixed pairs preserve finite discarded eigenvalues. For example,
retained eigenvalue 1 and discarded eigenvalue .2 give a mixed coefficient
+1.25. Replacing the discarded eigenvalue with zero incorrectly gives +1.
This includes motion between retained and discarded subspaces
without differentiating an arbitrary eigenvector gauge.

An eigenvalue within `128*epsilon*largest_eigenvalue` of the requested cutoff
is diagnosed as an unresolved rank crossing. An inverse whose retained mask
disagrees with that cutoff is rejected. The legacy helper can infer a mask
when given threshold zero, but only callers supplying the actual positive
threshold can diagnose distance from their cutoff; HF supplies that threshold.

## Boundaries, memory, and replay

The iterative CUDA DF J/K, density, and eigensolver path remains intact.
Generated force finalization replaces complete coordinate-indexed dA/dM
storage with a small device gradient and bounded weight tiles. `thread`
(the retained default mapping name) uses a generated 32-thread block schedule:
four lanes split one element's primitive products, and eight elements progress
per warp. Generic subgroup reduction combines center channels before one lane
scatters them to physical atoms. Zero weights and ragged tile tails preserve
complete participating subgroups. `serial` is an explicit deterministic traversal selected with
`VIBEQC_DF_DERIVATIVE_MAPPING=serial`.

The standalone `maximum_bytes` limit bounds owned numeric host staging and
owned device allocations separately. `maximum_tile_elements` can further
bound the uploaded weight tile. Caller weights/system data and opaque CUDA
storage are outside those counters. Only `3*Natom` doubles return to the
standalone caller.

HF retains an explicit host boundary: density-derived weights and the metric
Frechet map execute on the host. Resident raw values are borrowed from the
existing preparation. A source-backed plan generates requested value slices
on its owning CUDA stream and downloads them for weight construction. The
same stream uploads the weights and launches the fused derivative consumer.
No derivative tensor crosses D2H. Transfer and synchronization counters include
these value downloads and weight uploads. This is not a fully device-resident
force workflow, including when the generated one-electron path is selected.

For a positive DF working-set request, the J/K planner receives half the
request and the generated response receives half; UHF also charges its total
spin-density temporary against the response portion. Without an explicit
request, the response cap is 128 MiB. A resource ledger, when requested,
enforces total owned device allocations across the complete execution.
Prepared response state and its CUDA plan are invalidated together when the
mapping, effective budget, or metric cutoff changes; geometry invalidation follows
the existing fixed-topology plan contract. Energy-only caches do not require a
bound response. Generated response failures propagate through CUDA force assembly. Python singlepoint errors
retain the native scientific diagnostic before destroying its context.

## Completed shared-traversal retirement

The final [DF ownership bundle](../benchmarks/results/cuda-ownership/df/README.md)
binds the complete five-sample, 18-case comparison to the source and original
optimized objects that include the old response's removal. It retains all four
energy-plus-force geometry phases and the separate energy-only numerical and
2% non-regression gates. Independent CPU/libcint/PySCF comparisons, full native
and Python integration tests and actual-library sanitizer results accompany it.
Historical opt-in timings below describe their recorded source only.

## Historical validation and reproduction

The archive below describes the earlier opt-in implementation at its recorded
revisions. Reproduce selector comparisons from those checkouts; the current tree
cannot select the deleted coordinate-wise implementation. These historical
measurements alone do not establish promotion of the current shared traversal.

The raw RTX 5090 archive at
`benchmarks/results/df-derivatives-rtx5090/` contains 162 fixtures and 67,679
primitive records. The maximum absolute error over all Cartesian and spherical
blocks is **3.63e-13**. The emitted header is 715,626 bytes, the raw object is
1,663,096 bytes, and isolated compilation takes 5.277 seconds. The raw kernel
uses 244 registers, a 664-byte stack, and no reported spill stores/loads. These
isolated measurements describe raw derivatives, not complete HF performance.

The numerical gates cover arbitrary nonsymmetric weights, mixed basis
representations, auxiliary shell permutation, shared and independent centers,
two constrained budgets, partial tiles, deterministic repeats, and allocation
rejection. CPU HF weight tests contract against independent raw derivative
tensors before the corresponding CUDA HF tests. Complete HF checks compare
both RHF and UHF with the previous response and independent PySCF, including
energy finite differences, auxiliary-only atom motion, replay, and rank
crossing diagnostics.

Initial complete integration on Slurm job 9104 passes all 13 native CUDA tests and 116
Python GPU tests: 27 DF response, 29 one-electron response, 41 checkpoint,
14 resource-budget and 5 basis-projection cases. The matching CPU build passes
13 native tests and 156 Python tests. No enabled tier skips tests. Exact
source and binary hashes, logs and XML reports are recorded under
`benchmarks/results/df-derivatives-rtx5090/integration/`.
After review fixes, Slurm job 9107 passes 13 native CUDA tests and 118 Python
GPU tests, including both accepted selector aliases in 16 resource tests.
The 131 affected DF/XC/CPKS tests also pass under Python 3.11 with coverage.
These final reports and exact identities are in `review-integration/`; the
generated CUDA derivative header is unchanged by the iterative DAG-clone fix.

Reproduction scripts resolve their checkout relative to their own location.
Set `PYTHON` and optionally `CUDA_HOME`, `NSYS`, `CXX`, and `VIBEQC_LIBRARY` for
the local environment. Each hardware script creates a fresh directory under
`/tmp`, named by Slurm job and script, and writes that run's source/binary
provenance there. Set `OUTPUT_DIR` to choose a different new directory;
existing directories are rejected so historical records cannot be overwritten.
The profile runner also regenerates its summary from that run's trace databases.
Run every hardware script through a finite allocation:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:20:00 bash benchmarks/results/df-derivatives-rtx5090/gpu-validation.sh
```

`gpu-benchmarks.sh` compares interleaved complete cold, unchanged, and changed
geometry endpoints. `gpu-profiles.sh` captures comparable warm replays with
Nsight Systems. Derivative generation and contraction are fused in one kernel;
its measured time is reported jointly, without inventing separate stages.
`gpu-components.sh` compares the full raw, previous source-streamed, and new
weighted HF responses and records scoped bytes and explicit transfers. Its
process host high-water includes an allocated CPU oracle, as the field name
states; it cannot establish the host peak of a tensor-free SCF workflow.

## RTX 5090 endpoint evidence

Five interleaved repeats hold generated one-electron response constant and
change only the DF response selector. Median complete energy-plus-force
latencies are milliseconds, shown as reference / generated:

| Workload | DF budget | Cold | Unchanged geometry | Changed geometry |
| --- | ---: | ---: | ---: | ---: |
| sp8, batch 3 | default resident | 63.698 / 61.481 | 5.514 / 4.574 | 13.969 / 12.315 |
| sp8, batch 3 | 1 MiB source | 211.774 / 133.992 | 114.406 / 36.458 | 153.786 / 78.905 |
| sp8, batch 3 | 4 MiB source | 212.245 / 134.084 | 114.272 / 36.345 | 154.158 / 78.278 |
| sdf18, batch 1 | 4 MiB source | 261.069 / 131.993 | 157.897 / 39.144 | 229.232 / 113.130 |
| sp8, batch 1, eight primitives/shell | 1 MiB source | 16948.300 / 8004.180 | 11313.560 / 2366.852 | 15783.557 / 7144.788 |

All paired energy differences are zero and maximum force differences are
below 1.80e-13 Hartree/Bohr across all five reports. The larger workload exceeds
the current 16-AO whole-HF resource-inventory capability; its endpoint runs
without that ledger. Separate component counters still cover its generated
response allocations.

Component tests use identical arbitrary symmetric densities and Hamiltonians,
with auxiliary weight caps 3/7 for sp8 and 5/11 for sdf18. Every cap leaves a
final partial block. Source plans use auxiliary tiles of three and AO-pair
tiles of one row for both consumers. These deliberately small tiles are
different from endpoint planner choices; their large streamed speedups must
not be interpreted as complete-HF speedups. Resident reference timings begin
with full raw derivative tensors already available. For RHF, generated
resident contraction takes 0.727 ms versus 0.638 ms on sp8 and 7.074 ms versus
2.086 ms on sdf18. Generated response staging is bounded: sp8 uses 7,568 /
11,536 host bytes at 16/64 KiB, while sdf18 uses 43,872 / 74,976 at 64/128 KiB.
The largest component error is 3.71e-13. The reference component has no scoped
allocation counters; its `null` resource field does not mean zero storage.

The four Nsight traces each capture five warm calls of three systems. At a
1 MiB request, generated response reduces complete-call D2H from 1,576,950 to
241,350 bytes, while H2D rises from 79,360 to 161,920 bytes and explicit stream
synchronizations rise from 545 to 605. Its fused derivative/contraction kernels
take 4.176 ms total; raw value generation remains the dominant kernel at
116.804 ms. In resident mode, H2D falls from 557,940 to 148,980 bytes, D2H
remains 40,830 bytes, and stream synchronizations rise from 75 to 210. The
resident fused kernel total is 2.300 ms. These are complete captured windows,
not counts attributed solely to the DF consumer.

Separate fresh processes measure complete-HF host resident high-water without
a CPU integral oracle. The six sp8 measurements range from 1079.770 to
1084.625 MiB, including interpreter, driver and library startup. Candidate and
reference differ by at most 1.875 MiB; these runs establish no host-peak win.
Both report owned device peaks of 628,324 bytes in resident mode and 617,130
bytes in source mode. For sdf18 the process peaks are 1071.918 / 1071.922 MiB;
whole-HF device-ledger data is explicitly unavailable at that AO dimension.

`provenance.json` and the compressed scientific-source patch preserve the
measured implementation independently of subsequent integration fixes.
Those fixes align automatic resource planning with the J/K/response budget
split, charge UHF spin staging, preserve structured resource exceptions, and
invalidate retained plans when the metric cutoff changes. Their validation
is archived separately and does not overwrite performance provenance.

At these archived revisions the generated path remained opt-in. The artificial
small workloads supported integration, but did not establish the broader
real-molecule promotion gate. Direct-HF and one-electron derivatives have separate
promotion requirements.
