# Resource-aware TensorIR CUDA scheduler completion

Status: implemented (#508 completion)
Date: 2026-09-20

## Problem

The first #508 slice established structured schedule enumeration, static pruning,
PTXAS resource gates and complete-endpoint promotion. The representative-screening
follow-up reduced measurement cost. The remaining gaps were executable CUDA
schedule dimensions beyond the original layout/fusion/tile switches, a calibrated
resource/compile ledger, explicit traffic accounting and allocated-GPU evidence
that distinct workloads can qualify distinct schedules without weakening fallback.

## Decision

Keep the existing TensorIR plan, artifact cache, #459 implementation-profile
records, paired measurement runner and fresh full-fixture qualification. Extend
that one scheduler rather than introducing another tuning database or dispatch
layer.

Three schedule dimensions now change emitted CUDA while preserving the default
kernel body: generic kernels may process 1/2/4 elements per thread, scalar
reduce/einsum loops may request 1/2/4-way compiler unrolling, and packed-GEMM
pack/scatter kernels may process 1/2/4 elements per thread. Effective execution
identity includes each axis only where it can affect the plan, so irrelevant values are still removed before compilation. The bounded default tile axes now include 32/64/128/256/512. The default structured
search examines a deterministic 256-candidate prefix and ranks ready plans by
semantic traffic, register pressure, source size and generation order before
spending its 12-candidate compilation budget.

Static admission remains conservative. Its register estimate now incorporates
bounded work-per-thread, unroll and staging pressure. After compilation, each
candidate records the static estimate beside PTXAS registers, stack, spills,
shared memory, optional reported local memory and the resulting resident-block
upper bound. Generated-source bytes remain a compile-cost proxy, but the evidence
now calibrates that proxy against the compiler-reported duration and separately
retains cache/load wall time. These calibration records are audit evidence; they
never bypass the existing hard PTXAS or endpoint gates.

Plan schema 3 adds a semantic-traffic ledger. H2D/D2H copies and pack/scatter
conversion bytes are exact for the declared plan; the logical-tensor component is
the existing coarse planner traffic model. Successful executions repeat these
plan-bound values in endpoint metrics. The record explicitly excludes hardware
cache/DRAM transactions and cuBLAS internal workspace/traffic, so it must not be
read as a hardware-counter measurement.

## Validation

CPU/compiler validation from the isolated checkout:

```text
PYTHONPATH=.:python python -m pytest -q tests/python/test_tensor*.py \
  tests/python/test_specialization.py tests/python/test_compiler_structure.py \
  tests/python/test_local_profiles.py \
  -k 'not test_native_source_identity_and_probe_abi_match_checkout'

561 passed, 113 skipped, 1 deselected
```

The deselected check requires a matching built native `libvibeqc`, which is not
present in the isolated worktree. It had been run once without deselection and
failed only on the missing native library. `tools/check_compiler_structure.py`
checked 226 compiler modules with zero dependency errors. Ruff formatting/lint,
`git diff --check`, and the focused CUDA-resource/codegen tests also passed.

On an allocated RTX 5090 (sm_120, 170 SMs, CUDA 12.9 toolchain), a dedicated real
CUDA test compiled and executed all three new dimensions: `elements_per_thread=4`,
`reduction_unroll=4`, and forced packed-GEMM `staging_width=4`. Each matched
the TensorIR interpreter and the endpoint traffic record matched the executed plan.

Two clean CC-fragment tuning runs used `tools/tensor_cuda_examples.py --mode tune`,
seed 146, eight paired repeats and scales 0.001/1/100. For
`virtual_residual`, both (o=3,v=7) and (o=5,v=17) rejected every finalist and
retained the baseline. For `denominator_update`, the smaller bucket qualified
views+fusion at 128 threads with median endpoint speedups
1.0370 / 1.0290 / 1.0304; the larger bucket qualified views+fusion at 256 threads
with 1.0382 / 1.0267 / 1.0386. All supplied fixtures passed the unchanged
numerical, shared-noise and bootstrap gates. The qualified kernels reported
42 registers and zero stack/spills/shared memory versus a 46-register static
estimate.

A separate packed-GEMM workload used `ik,kj->ji` with M=384, N=448, K=512,
deterministic seed 508 and three stable input scales. An explicit qualification
first showed that 256x256x256 tiles materially outperform the 128 baseline and
that staging widths 2 and 4 compile, execute and qualify without becoming the
winner. After adding 256 to the default tile axes, the ordinary 128-candidate
search independently selected a different schedule: views+fusion with
tile_n=256 at 128 threads. It pruned 105 candidates before compilation, compiled
12, fully qualified 3 finalists and accepted 1. Median complete-endpoint speedups
were 1.3184 / 1.2921 / 1.2795. PTXAS reported 34 registers, zero stack/spills/
shared memory versus a 26-register static estimate.The raw local evidence hashes are retained here to make accidental evidence
substitution detectable:

```text
d2d39460ae02919c32e5f79165a8b5e4f7d0ae5ade79960e28c6b65c9197bd5
  denominator_update-o3-v7-tuning.json
4acc47d70795c068667aef96c1cfbe999e31573ce6d83ab9de1c0be7cafad634
  virtual_residual-o3-v7-tuning.json
2d1c61350f95e469cfc6d41f6e63872307f9041ff13a60f08446d5db84f16d1b
  denominator_update-o5-v17-tuning.json
263cf82abb2f50bfd4ab8b2d8312ec45c38256be549fea7a0eea2f8dfe3ed08f
  virtual_residual-o5-v17-tuning.json
72d3fb1ba32a23968fc0d163193a7ec3dfbc0270cdd714d120e1e9a2fa0daa43
  packed-GEMM default-search evidence.json
0f9604a100774d694d3287346a2f5205947c6c31d8cd30d2049add0ec30e6250
  packed-GEMM explicit-schedule evidence.json
```

These results satisfy the scheduler acceptance boundary: distinct workloads
qualify distinct schedules only when complete endpoint evidence supports them,
while other workloads retain baseline fallback.

A final current-code qualification after widening the default search to 256
candidates and adding the static compile shortlist produced two stronger,
qualitatively different winners on the same RTX 5090/CUDA 12.9 environment.
A layout-sensitive 1024x1024 contraction with an internal transposed result
selected producer layouts with 256 threads and two generic elements per thread;
two independent fixtures measured 2.9215x and 2.9097x complete-endpoint median
speedups, with bootstrap lower bounds 2.7421x and 2.7559x and both shared noise
gates passing. A named-transpose 2048x256 by 256x2048 packing-sensitive
contraction used the ordinary default search with no manual schedule list:
256 generated, 167 statically pruned, 12 admitted to compilation, 3 finalists,
and 3 accepted. It selected an effective 512x512x128 panel schedule and measured
2.3144x complete-endpoint median speedup with a 2.2454x bootstrap lower bound;
the shared noise gate passed. Compact retained evidence, including raw winner
samples, PTXAS/compile calibration, endpoint traffic metrics, guarded promotion
profiles and negative campaigns, is archived under
`benchmarks/results/tensor-resource-scheduler-508/`.

## Full-method boundary

Two attempts to collect an additional RCCSD-residual performance sample were
aborted when independent direct-GPU Python processes appeared on the same physical
RTX 5090 after the Slurm jobs had started. Only the #508 jobs were cancelled; the
other processes were not touched. Those interrupted timing runs are excluded from
all performance claims above. RCCSD numerical correctness remains covered by the
existing TensorIR/CC validation suite, but no contaminated timing is used as a
promotion argument.

This does not weaken the scheduler acceptance: #508 changes the compiler/autotune
mechanism, and the accepted evidence above includes two qualitatively different
TensorIR endpoint families plus conservative baseline fallback. Broader method-
level production benchmarking can continue independently without changing the
promotion rules or scheduler implementation.

## References

#508; #517; #533; #459; #146; #149;
`docs/tensor_cuda.md`;
`2026-09-19-tensor-schedule-search.md`;
`2026-09-19-tensor-screening-shortlist.md`.

Agent: ChatGPT
Model: GPT-5.6 Sol
