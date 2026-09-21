# #309 lazy-core ablation, first #308 optimization increment

> **Checkout retention (2026-09-21):** `raw-evidence.zip` was moved out of the normal checkout. Exact bytes remain in Git revision `d8f64a93fe0dfebd889fd0ba1fadbd5ad7d840e5` and are checksum-bound by [the checkout-trim manifest](../retention-2026-09-21/migration.json). Restore locally with:
>
> ```bash
> python tools/restore_retained_evidence.py benchmarks/results/issue309-lazy-core/raw-evidence.zip \
>   --manifest benchmarks/results/retention-2026-09-21/migration.json \
>   --output .artifacts/issue309-lazy-core/raw-evidence.zip
> ```
> Restored archives belong under ignored `.artifacts/`; do not recommit them.

For commands below that previously unpacked the checkout directly, pass the
restored archive explicitly:

```bash
python -m tools.unpack_evidence benchmarks/results/issue309-lazy-core \
  --archive .artifacts/issue309-lazy-core/raw-evidence.zip \
  --output .artifacts/issue309-lazy-core-unpacked
```

Valid supplied density now skips core-Hamiltonian diagonalization. Initial
orbital outputs are optional and cleared before validation. Cold RHF/UHF,
UHF frontier mixing, density symmetry/electron traces and empty-spin handling
retain their existing conventions. Consumers needing warm core orbitals use
an explicit request. Iterative HF/KS consumers overwrite their orbital frame
before reading it; finalization/export construct their own physical frames.

This is the lazy-guess slice of #309. Overlap caching, its lifecycle/resource
accounting, #310 production solvers and #311 final-state reuse remain open.

## Measured first domain

96-AO RHF spherical def2-SVP, batch 1, 1 GiB DF allowance on Slurm's RTX 5090.
Five interleaved eager/lazy pairs per workload use one binary, one prepared
owner and a fixed post-cold density. The eager diagnostic control restores the
discarded core frame without changing the supplied density or SCF controls.

| Clean endpoint | Eager median (s) | Lazy median (s) | Eager / lazy |
| --- | ---: | ---: | ---: |
| Cold energy, with preparation and destruction | 0.443548 | 0.444007 | 0.999 |
| Warm energy | 0.233418 | 0.160044 | 1.458 |
| Changed-geometry energy | 0.409679 | 0.333200 | 1.230 |
| Warm energy plus complete force | 0.781138 | 0.711037 | 1.099 |
| Changed-geometry energy plus force | 0.964042 | 0.885154 | 1.089 |

The same five-pair protocol was then run from the clean lazy-core commit for
two additional domains. Every cold/warm/changed sample and traced solve count
is retained under `larger/`; these clean warm endpoint medians are:

| Domain | Endpoint | Eager (s) | Lazy (s) | Eager / lazy |
| --- | --- | ---: | ---: | ---: |
| 96 AO, batch 4 | energy | 0.913925 | 0.621068 | 1.472 |
| 96 AO, batch 4 | energy plus force | 2.533787 | 2.233660 | 1.134 |
| 192 AO, batch 1 | energy | 3.243773 | 2.201214 | 1.474 |
| 192 AO, batch 1 | energy plus force | 6.391965 | 5.326289 | 1.200 |

All additional iteration/retry branches match, and the energy/force gates
pass. Their exact clean source/library identities are in each collection;
the iteration guard executes in the runner for these additional domains.

In the first 96-AO batch-1 domain, both selections have identical
iteration/retry branches: 34 cold, 2 warm and 22 changed-geometry iterations. Energy and full-force endpoints match the
separately prepared same-model cold references at 1e-9 Eh / 1e-8 Eh/Bohr.
The changed geometry is restored before each sample. These cache/replay checks
supplement existing independent scientific tests; they do not claim external
engine parity or completion of #308's full matrix.

The separate traced pairs prove one eager warm core solve versus zero lazy
warm core solves, while both cold selections still solve once. Overlap and
final Fock reference solves remain. Raw traces include their actual leaf
invocations; a missing preparation scope or ineffective mode flag fails the
traced ablation gate. Profiled timing is not used for the table.

Warm/changed improvements exceed the existing timing assessor's noise floor.
Cold work has no meaningful gain, so the aggregate assessment requiring every
workload to improve reports `not-run`. This record does not relabel that as
an all-workload performance promotion.

## Validation and provenance

25 CPU native tests, including an analytically known nonidentity-overlap
initial-density fixture, actual solver counts, invalid inputs, explicit frame
requests, UHF cold mixing/empty beta and failure/retry. The CPU Python suites
passed 64 tests (16 optional GPU skips) and 109 additional density/reference
checks. Benchmark protocol tests passed 99 cases, including negative gates
for changed iteration/retry branches. GPU Python passed 39 tests covering
actual RHF/UHF eager/lazy counts, occupied/dense replay, resources and output
selection. Both GPU count cases passed memcheck with zero errors/leaks.
Pre-commit passed. Every GPU run used a finite Slurm allocation.

All samples preserve source revision, dirty patch, actual library hash,
inputs, setup/destruction cost and per-item convergence/energy/force records.
The first-domain measured checkout is P0 commit `07b83d2` plus
`measured-source.patch`; the additional domains use clean commit `b362dbe`.
The embedded library/source identity was independently checked and is retained
in `summary.json`. The later iteration-branch guard was checked against every
retained sample; the exact earlier measured runner remains reconstructible.
`raw-evidence.zip` preserves the original JSON/JSONL collections, source patch
and `files.json` byte for byte. The adjacent standard manifest records the
archive size/SHA-256, each member hash and measured source identities. All
14 members were restored and compared with their originals before removing
the expanded copies. No profiler database or routine log is published here.

## Reproduction

Verify the archive or restore it into a fresh directory using the repository's
standard verifier (the README and compact summary stay directly reviewable):

```bash
python -m tools.unpack_evidence benchmarks/results/issue309-lazy-core \
  --archive .artifacts/issue309-lazy-core/raw-evidence.zip \
  --output /tmp/issue309-lazy-core-evidence
```

Paths such as `larger/` and `measured-source.patch` below refer to restored
archive members. Original measurements, distinct domains and hashes are unchanged.

Build the measured source as Release with CUDA 12.9.1, architecture 120 and
AOT shells disabled, following the [P0 build recipe](../issue308-host-baseline/README.md).
Use a fresh output directory outside the checkout and preserve Slurm's device
visibility:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:20:00 \
  env PYTHONPATH=python OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python benchmarks/issue206_df_matrix.py --run --host-workloads \
  --eager-core-ablation --case water-tetramer-def2-svp-spherical --batch 1 \
  --library build/cuda/libvibeqc.so --memory-budget-bytes 1073741824 \
  --energy-only --repeats 5 --output-dir /tmp/issue309-energy
```

Omit `--energy-only` for complete forces. Use a separate invocation with
`--host-trace-dir /tmp/issue309-traces` and a new output directory to check
actual calls. 192-AO batch 4, 384 AOs, constrained-memory, other spin/representation
domains and independent matched #206 acceptance remain required.
