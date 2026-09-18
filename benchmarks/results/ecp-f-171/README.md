# Scalar ECP orbital-f qualification

This slice extends the scalar ECP orbital basis through f in Cartesian and
real-spherical layouts. Nonlocal projectors remain s/p/d, local labels through
f, radial powers 0..4 and complete methods direct RHF/UHF. It makes no broad
heavy-element, additional-method or speedup claim. Refs #171; depends on #425.

## Source and environment

Baseline: `d0f6b127a2e9e0049994c5f14e4748441d71fc43` (PR #425). The measured
candidate is this public Git archive plus the 12 exact source/test files in
the remote `source-overlay.tar.gz`, not a claimed clean Git checkout. The full
source manifests verify every archive file, and CPU/CUDA candidate manifests
must match. Documentation and ownership records do not alter measured code.
The reviewed `source.patch` reconstructs the 12-file change from that baseline;
`source-match.json` binds it to the publishing source commit after LF normalization.
Archive compression can differ across Git versions; per-file source hashes
and the archive's base-commit PAX record bind the actual scientific baseline.

RTX 4090, CUDA 12.9, sm_89, Release, AOT shells disabled, one OpenMP/BLAS thread.
Compute Sanitizer uses CUDA 12.8. `summary.json` records exact hardware,
toolchain, source/library hashes, numerical errors, all timing samples, work
counts and planned/observed resources. `raw-evidence.manifest.json` binds the
full diagnostic archive retained in the user's original qb-ilm workspace under
`evidence-171/f-results/raw-evidence.zip` and downloaded locally. Routine logs,
retries and generated products remain outside Git under the evidence-retention
policy. Reproduction scripts, source reconstruction, accepted samples, test
outcomes and kernel resource records are retained here; reproduction does not
depend on access to the diagnostic archive. The initial failed native build
is retained in that archive; it
used a nonexistent CPU enum, subsequently corrected to `CPU_REFERENCE`.
An initial baseline finished measurement/provenance but its mutable driver
encountered a trailing parse error; the queued candidate correctly refused
that nonzero prerequisite. Both failures and the completed measurements are
retained. Baseline and candidate qualification subsequently use `f-run-v2.sh`
as an immutable script; no scientific acceptance gate is weakened.

## Gates and scope

Independent host tests cover every Cartesian component through f, analytic
Gaussian derivatives, gamma-function normalization and invalid dispatch aliases.
C API tests cover f acceptance/g rejection on both ECP and all-electron atoms,
Cartesian/spherical layouts, rejection of f projectors and the grid wrapper's
shape/weight/append contract. Existing s/p/d numerical and failure tests remain.

Contracted f shells on Na and H exercise mixed centers and signed coefficients.
Libcint matrices are normalized with the overlap diagonal for Cartesian f.
All-center finite differences use steps 2e-4 and 7e-5 bohr, with arbitrary
nonsymmetric AO weights. A separate ECP center without an orbital basis tests
f orbitals against d projectors. Complete RHF/UHF energies and forces use PySCF
2.14.0 for both representations and backends; CUDA adds complete energy finite
differences. Planned-budget execution checks allocation bounds and prepared
geometry replay. Sanitizer checks both native error recovery and actual f raw
matrices/derivatives in both representations.

Absolute gates: raw matrices 2e-9 Eh, grid-refinement derivatives 2e-8, raw
finite-difference derivatives 3e-7 (relative 3e-6), complete energies 2e-8 Eh,
complete forces 2e-6 Eh/bohr. The 160/32/64 and 224/44/88 quadratures are
empirical convergence checks, not a universal error estimate.

Four matched f endpoint cases per backend cover Cartesian/spherical RHF/UHF
with f on Na; CUDA also measures a larger Cartesian RHF case with f on both
atoms. Each has a warmup, then one CPU or three CUDA complete synchronous
`singlepoint` timing samples. All CPU cases and the 16-AO spherical CUDA cases
also run an explicitly budgeted prepared calculation. CUDA inventory v1 still
rejects more than 16 public AOs: the 19/29-AO Cartesian CUDA endpoints retain
numerical/timing evidence and the explicit rejection diagnostic, with no
claimed budget estimate. The larger two-f-center CPU replay remains explicitly opt-in; its initial
incomplete run is retained below rather than repeated in endpoint timing.
These small samples are diagnostics, not a CPU/GPU speed comparison. Records include actual shells,
primitives, AO/pair counts, iterations, energies, forces and resource diagnostics.
The former baseline cannot execute f, so it supplies only an existing-domain
regression comparison; its timings must not be reported as f speedups.

The original two-f-center CPU prepared replay was still computing after a
30-minute suite run; it and its duplicate in the CUDA-linked suite were
interrupted, with logs and the exact earlier overlay retained. This is incomplete
large-CPU qualification, not a numerical failure or a passing performance gate.
The default CPU/CUDA replay uses f on the all-electron H atom (16 public AOs),
complementing the independent complete-HF checks with f on Na. CUDA also tests
explicit rejection of budget estimation above 16 public AOs; raw
matrices/all-center derivatives retain both centers on both backends. Set
`VIBEQC_ECP_LARGE_CPU_TEST=1` to reproduce the expensive original CPU replay.

The first CPU endpoint driver expected the backend label `cpu`; the actual
public result uses `cpu_reference`. Its failed log and exact script are
retained alongside the successful rerun. The first CUDA Python suite had
79 passes and one failure because its budget fixture exceeded the existing
16-AO inventory. After correcting that fixture and adding the explicit
unsupported-domain check, `f-resume.sh` reruns the CUDA-selected tests and the
changed CPU replay test, then completes benchmarks and sanitizer checks.
The failed full run is retained and is not relabeled as a passing full rerun.

## Results

CPU CTest passed 31/31 and CUDA ECP CTest passed 3/3. The CPU Python suite
passed 61 tests with 19 expected GPU skips. Following the budget correction,
all 19 CUDA-selected tests passed and the changed CPU replay passed separately.
Both native error recovery and the two f matrix/derivative cases completed
Compute Sanitizer with zero errors. Final local checks passed 66 tests,
174 compiler modules, 217 shared SCF modules and the 182-file CUDA inventory.

Across the four CPU and five CUDA f endpoints, maximum raw-matrix error was
3.47e-12 Eh, complete-energy error 8.89e-15 Eh and complete-force error
5.11e-10 Eh/bohr. CPU complete calls took 85.4–86.0 s in the single retained
sample per case. CUDA medians were 1.53/1.30 s for 19-AO Cartesian RHF/UHF,
9.41/9.37 s for 16-AO spherical RHF/UHF, and 1.91 s for 29-AO Cartesian RHF.
The slower spherical route is explicitly retained; these are distinct layouts,
not a matched representation speed comparison or a schedule promotion.

On the existing s/p/d domain, matched baseline/candidate complete RHF medians
were 989.10/993.60 ms (+0.45%) and UHF 983.36/989.19 ms (+0.59%). Full-engine
component endpoint medians changed by -0.17% to +1.18%. Three samples per case
show no material regression in this bounded run; they do not establish a
general performance guarantee. Full samples and numerical gates are in `summary.json`.

## Reproduction

1. Obtain baseline `d0f6b127a2e9e0049994c5f14e4748441d71fc43` with `git archive`
   as `f-baseline.tar.gz`, extract it, and apply `source.patch` in a separate
   candidate tree. Verify the 12 normalized file hashes in `source-match.json`;
   package those paths as `f-changes.tar.gz` and `f-formatted.tar.gz`.
2. Use the scripts in `reproduction/`. Adapt the workspace prefix in
   `f-prepare.sh`, `f-run-v2.sh`,
   `f-provenance.py` and `f-collect.py`. Keep the flags, grids and test commands.
3. Run preparation, then `f-run-v2.sh cpu` and `f-run-v2.sh baseline`. After baseline
   succeeds, run `f-run-v2.sh cuda`; GPU timing must remain sequential. The
   original script and its initial orchestration failures are archived too;
   never edit a script while a running shell may still read it.
4. Candidate CUDA execution explicitly regenerates the header, touches native
   inputs and reconfigures CMake so archive mtimes cannot hide changed code or
   newly registered tests. The endpoint script takes its source and exact
   `VIBEQC_LIBRARY` explicitly.
5. `f-collect.py` requires successful exit markers, matching complete candidate
   source manifests, all four CPU/five CUDA endpoint cases and exact library
   identities before archiving. `f-resume.sh` records the actual focused rerun;
   it expects the completed initial builds. On a clean full rerun, preserve the
   new full-suite totals rather than manufacturing the historical focused logs.
   The collector's focused-log checks describe this campaign's exact sequence.
   Retain failures and diagnostic follow-ups outside Git.

## Ownership

This extension reuses the common scalar DAG and existing molecular spherical
expansion. Maintained scientific CUDA and runtime LOC each change by +0/-0;
no native production path is removed. The independent CPU oracle/fallback,
including its own quadrature and normalization, remains separate from generated
production arithmetic. Generated header growth is recorded independently and
does not offset handwritten code. Other generated-family measurements remain
preserved in the current ownership report.
The ECP header grows from 21,910 to 34,335 bytes (+12,425), with 1,196 nonblank
noncomment generated lines. Its SHA-256 is
`1393e2f010da000407100a12d2cc8c0da650163f5fb4c54d4b36892507247334`.
The final branch also integrates upstream `9af9e08`; its unrelated DF-generated
measurements are retained when reconciling the report, whose aggregate is
27,541,945 bytes. The qualification snapshot above remains explicit rather than
being relabeled as a clean build of the later merge commit.
All five ECP kernels retain their register, stack, shared-memory and local-memory
counts. The AO kernel's second constant-memory bank grows from 184 to 248 bytes;
it retains 56 registers, 64 stack bytes and zero reported local-memory bytes.

See the [current contract](../../../docs/ecp.md) and
[decision](../../../.agents/notes/implemented/numerics/2026-09-17-ecp-orbital-f.md).
