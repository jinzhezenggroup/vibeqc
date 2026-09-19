# Decision: rank TensorIR candidates before full qualification

Status: implemented
Date: 2026-09-19

## Problem

The resource-aware search in #517 already prunes illegal, equivalent and
resource-infeasible plans before complete endpoint timing. Every surviving
candidate still received the full all-fixture campaign, so extending the search
could spend most of its remaining budget on uncompetitive schedules.

## Decision

Keep the existing tuner, compiled-artifact cache, numerical tolerances and #459
promotion profiles. Add an immutable `TensorScreeningPolicy` controlling a
representative-fixture ranking stage and a bounded number of full-qualification
finalists. Defaults are fixture zero, five A/B pairs and three finalists.

Rank by the worst median baseline/candidate ratio over the requested existing
fixtures. Break ties by generation order. Do not grant performance qualification,
apply a promotion threshold or emit a performance guard from screening. Unknown
or numerically invalid candidates still fail closed.

Bypass screening when the precompile candidate budget fits the finalist limit,
or when screening plus finalist qualification would not reduce planned A/B
pairs. Record the decision and its counts. This is a sample-work heuristic,
not a model of elapsed time: compilation, startup, profiling and candidate
reopening are explicitly excluded. Resource failures can reduce the actual
candidate count after this planning decision.

Close each screening context before opening another. Reopen at most one
finalist beside the baseline, using the existing compiled artifact. Fresh
warmup and fresh paired measurements on **all** original fixtures remain
mandatory. Finalist count bounds qualification attempts; failed finalists do
not authorize unlimited backfilling. The existing deadline is checked before
preparation/measurement/profiling work and preserves the baseline on failure.

Schema-v3 selection evidence binds the screening policy, active decision and
representative indices. Mathematical identity and artifact-key recipes are not
redefined; ordinary compiler source identity still invalidates cached outputs.
Raw screening and qualification samples stay separate. Non-finite clock samples
fail qualification and are retained as JSON null plus an explicit description,
rather than causing negative-evidence serialization itself to fail.

## Rejected alternatives

- Promoting the best short probe: cannot establish all-fixture correctness or
  performance and would bypass existing statistical/noise gates.
- Reusing screening samples as final evidence: would mix selection samples with
  qualification. Every finalist receives a fresh campaign instead.
- Keeping every finalist resident: unnecessarily multiplies device arenas and
  weakens bounded-lifetime behavior.
- Screening every search unconditionally: can increase work for small searches
  or a single fixture. The explicit bypass and `screening=None` retain the full
  unfiltered route.
- Declaring a lower pair count to be a timing win: setup and compilation can
  dominate. A matched wall-time comparison is separate evidence.

## Invariants

No lowered equations, CUDA scientific kernels, precision or scientific gates
change. Promotion still requires CPU and unfused-GPU parity, complete PTXAS
resource gates, all original fixtures, the shared noise gate and the bootstrap
gate. Unmeasured layouts/devices do not gain #459 eligibility. CPU interpretation
is used as a tuning oracle only, not a production execution fallback.

## Evidence

- Focused host tests: 73 passed, including 33 new cases for policy validation,
  immutable indices, ordering, ties, all-fixture numerical/performance failures,
  noise, deadline/resource rejection, invalid timing retention, distinct
  evidence identities and bounded context lifetime.
- Broader TensorIR/specialization/compiler-structure regression: 507 passed,
  112 skipped (explicit GPU/native capability gates).
- Combined statement/branch coverage of the two modified scheduling modules is
  95% (search 96%, tuner 93%) in the focused suite; this is module coverage, not
  a claim that every new branch is covered.
- Two new real CUDA contract tests passed on an allocated RTX 5090 using CUDA
  12.9/sm_120. Each compiled three candidates, screened all three and fully
  qualified one using fresh measurements on three input layouts/value fixtures.
  Both correctly retained the baseline after performance rejection.
- Observed maximum absolute error was 0 for elementwise arithmetic and
  4.440892098500626e-16 for the contraction case. These are tensor fixtures, not
  force/Hessian or complete molecular-method validation.
- Each real case records 30 A/B pairs versus 45 requested by the equivalent
  unfiltered campaign. No matched overall tuning-time speedup is claimed.
- Compact source-pinned evidence and all scalar timing samples are retained in
  `benchmarks/results/tensor-screening-508/qualification.json`.

The subsequent broad GPU regression completed on retry in finite Slurm job
10083: **113 passed, 1 skipped** in 523.05 s, with pytest exit status 0. It covers
`test_tensor_cuda_execution.py`, `test_tensor_layout_cuda.py`,
`test_tensor_cuda_fp32.py` and `test_tensor_transcendentals_cuda.py`; the one skip
requires the separately opted-in CUDA-linked HF library. The earlier allocation
attempt was rejected while the node was busy and is not counted as a test run.
All applicable pre-commit hooks passed. Final host regression again returned
507 passed, 112 skipped. Native full-SCF/DFT/CC endpoint qualification and a
matched tuning-wall-time campaign remain outside the results above.

## Consequences and revisit conditions

The shortlist may discard a schedule that would win on other fixtures, so this
is bounded heuristic search, not an optimum guarantee. Use multiple representative
indices or disable screening when examining that risk. Revisit scheduling of
screening rounds when richer measured workloads justify adaptive allocation or
racing. Do not weaken final promotion gates to obtain a winner.

This supersedes only the missing-shortlist boundary in the earlier
[resource-search note](2026-09-19-tensor-schedule-search.md). Resource calibration,
additional executable schedule dimensions and broader method consumers remain
under #508; this slice does not close it. #459 continues to own shared guards.

Agent: ChatGPT
Model: GPT-6 Astra Pro
