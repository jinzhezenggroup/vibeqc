# Decision: bound and omit three-center force blocks before generation

Status: implemented
Date: 2026-10-04

## Problem

The 96-atom occupied DF optimization in PR #1719 still spends about 1.95 s
evaluating three-center derivatives in a 4.57 s complete warm energy-and-force
endpoint. Most interfragment AO products are negligible. Computing their
derivatives and then multiplying by tiny response weights wastes work, even
when no full derivative tensor is stored.

## Decision

The compiler emits a geometry-only radial Gaussian Coulomb-norm envelope.
The runtime prepares an orbital shell-pair derivative table and auxiliary
value vector once per force call. Each existing shell subgroup folds actual
Cartesian response weights, bounds the entire shell contribution, and returns
before primitive recurrences when its allocated force budget permits omission.

The envelope follows from an optimized near/far split of the Coulomb potential,
Gaussian-product radial polynomial moments, and the Coulomb triangle inequality.
The derivative envelope bounds the sum of both orbital-center norms. Translation
then covers the auxiliary center and every shared-atom combination. Signed
contractions, Cartesian normalization and spherical expansions enter through
absolute normalized primitive coefficients and folded weights.

The absolute force-component allowance is distributed over the complete ordered
public AO/auxiliary domain. Symmetric/packed consumers overcount rather than
underfund it, and auxiliary shells clipped across panels receive only their
in-panel share. The default is `1e-10 Eh/Bohr` only within the established
sm_120 automatic shell domain. Other consumers stay strict unless explicitly
qualified through the diagnostic control. The old SSS primitive screen remains
independently off; explicitly enabling both adds their two omission budgets.

Allocate norm metadata only after the pre-existing response tile and scratch
choices. Capacity or optional allocation failure retains strict work. Do not
shrink panels to accommodate screening. Geometry changes always rebuild norms;
their lifetime is the existing force arena and stream.

## Evidence

The final same-binary RTX 5090 A/B reduces five-repeat warm median from
4.564127 s to 3.314961 s (27.4%), and moved warm from 4.584961 s to 3.335904 s.
Cold/moved iterations remain 24/13; all warm calls retain one iteration.
All complete calls pass unchanged independent `1e-8 Eh` and `1e-7 Eh/Bohr`
gates. Maximum off/on force difference is 4.23e-13 Eh/Bohr. This A/B in
Slurm job 12192 is distinct from the final six-size figure campaign (12189).

The separate work replay executes 94,730,671 of 342,802,304 primitive products,
skipping 248,071,633 (72.37%). It skips 67,578,393 of 101,861,760 active shell
tasks (66.34%). All 58 auxiliary panels remain. New norm storage is 1,190,400
bytes; its observed preparation is 2.840 ms. Three-center derivative GPU time
drops from 1,955.433 ms to 701.513 ms. These component/counter observations are
diagnostic and excluded from clean medians.

Host tests compile the emitted helper and compare all 64 s/p/d/f triples with
independent libcint Coulomb self-norms, values and weighted derivatives, including
signed contractions and shared-atom sums. Molecular tests exercise both public
representations, RHF/UHF, full/symmetric/packed weights, finite differences,
batch geometry replacement, clipped panels and strict budget fallback. Retained
qualification, sanitizer receipts and final figure data are linked below.

The final binary passes all 30 new molecular GPU cases both locally and on n2
(RTX PRO 6000 Blackwell, Slurm job 2180). Independent-device memcheck and
synccheck include orbital f shells; racecheck covers the spherical packed
def2-SVP case. All three report zero errors/hazards. These n2 runs qualify
numerics and execution, and do not contribute to the RTX 5090 performance plot.

The expanded local regression has 165 passes and 17 failures. Every failed
case reproduces on the clean `7d1152038` baseline binary with the same leading
assertion. They concern signature counters, BLAS response-route attribution,
scratch/residency expectations and old insufficient-budget assumptions; none
is an energy/force gate failure. The retained validation record lists each
case and both assertions. This suite is not classified as passing.

The final 99-atom attempt rejects before SCF under the existing admission
policy. Clean baseline and both screening arms fail identically. This is a
retained resource boundary; the older PR qualification's successful 99-atom
result is not a success of this newer source baseline. The six-size figure
contains only fully accepted endpoints.

## Rejected and deferred alternatives

- Raw integral magnitude alone is not a force bound. Screen with derivative
  envelopes and the current adjoint weights.
- Screening separate auxiliary entries before metric whitening does not preserve
  sparsity: `M^(-1/2)` mixes those directions. Whole AO-pair rows do survive it.
  A CPU-only exact pair census finds many tiny rows in the benchmark, but this
  is opportunity evidence, not an SCF/force certificate.
- A future forward-row screen could use the Coulomb projection inequality
  `||B_mu,nu|| <= sqrt((mu nu|mu nu))`, avoiding a naive metric-condition bound.
  It still needs consistent forward, derivative, fallback and geometry policies.
  This change intentionally leaves forward `A`, fitted `B`, their allocation,
  metric derivatives and SCF iterations unchanged.
- Fragmenting retained lower AO rows into many tiny whitening GEMMs is unlikely
  to be profitable. A possible follow-up batches retained pair indices, whitens
  through disjoint existing scratch, and scatters to the packed owner. It is
  not implemented or qualified here.

## Numerical and measurement limits

### CI portability follow-up

The first published screening head failed CuMetal compilation because its
host/device envelope called host-only `std` math overloads. A shared emitted
host/device math binding now selects the CUDA global overloads for the device
pass and standard C++ overloads for the host pass. The envelope equations,
screening budget and runtime admission/fallback code are unchanged. The
inventory test fixture also now materializes every registered source, including
sources outside the control-scanning list. A negative host-only-name probe and
both emitted name-lookup branches are exercised on the host; this does not
substitute for real NVIDIA/CuMetal compilation in CI.

The frozen measured-source hashes and GPU receipts above retain their original
identities. The portability correction is not a new GPU execution or performance
measurement. In particular, the retained one-test `integration-budget-test.log`
summary does not identify a complete native CUDA cache-budget fixture run or
record its statuses/plan state, so it does not independently clear that earlier
runtime verification item.

The analytic envelope has FP64 headroom and conservative tiny/invalid-value
handling; it is not interval arithmetic. Tests do not establish universal
floating-point certification. Screening error is separate from the existing
SCF and auxiliary-metric approximation errors.

An initial Cartesian cc-pVDZ-JKFIT dimer fixture crossed the native metric cutoff
(260/262 directions versus an untruncated Cholesky oracle); its unscreened energy
already differed by 1.83e-8 Eh. The Cartesian molecular matrix instead uses
full-rank def2-SVP auxiliaries, preserving the independent gates. A distant
water/OH UHF fixture failed unscreened SCF convergence; water plus atomic H
provides a converged open-shell control. Neither failure is a screened success.

Local disk exhaustion interrupted two diagnostic attempts. Final builds used
an executable temporary-memory build directory, preserving the CMake path and
ccache use. Node n4 was excluded after CUDA initialization failed with a
driver/library mismatch. Failed attempts remain in ignored local artifacts.

## References

- [Current screening contract](../../../../docs/developer/df_tuning.md)
- [Retained qualification](../../../../benchmarks/results/df-source-screening-20261004/README.md)
- PR #1719, whose reviewed changes are preserved in the baseline merge
  `7d11520383a01dce7b0997f16d745cdb95de8efd`.
