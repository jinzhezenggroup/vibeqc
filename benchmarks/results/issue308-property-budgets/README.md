# Property/provider-aware DF preparation

Slurm job 9548 used the rebuilt CUDA 12.9.1 sm_120 library with native source
identity `bce05e237e7a423cce63b0004445c5da1465b244a178f3c6cacfb6426dac6d7e`
and SHA-256 `8100f0df403a7b139a0baa89af197076f09c947be8d4d4089d95defc99dd3499`.
The measured dirty native source patch, base commit, exact input identity and
build configuration are retained in `summary.json` and `evidence.zip`.

The 768-AO / 800-Cartesian-AO / 96-atom input is the independently qualified
PySCF fixture from [the stage ledger](../issue308-stage-ledger/README.md).
These probes isolate **preparation**, with a 1-GiB DF subbudget. They do not run
SCF or full 768-AO forces, and are not global-resource or clean timing acceptance.

| Request/provider | Preparation result | Seconds | Process host peak RSS | AO derivative values | Nuclear derivatives |
| --- | --- | ---: | ---: | ---: | ---: |
| Energy only | accepted | 0.6575 | 458.4 MiB | 0 | 0 |
| Forces, tensor one-electron | rejected as OOM | 0.000112 | 276.4 MiB | not allocated | not allocated |
| Forces, generated one-electron | accepted | 2.5552 | 458.4 MiB | 0 | 288 |

Both complete returned overlap matrices agree with independent PySCF to
9.9921e-16. The tensor-force rejection retains the real Cartesian/public
derivative-copy requirement (about 5.67 GB for those arrays alone). Accepting
generated preparation does not establish that its later force live set fits.
Process RSS includes library/runtime state; it is not the numeric preparation
bound. No GPU-memory peak was sampled for this job.

All 96 GPU regressions passed, including per-item chunk rejection, both
representations, zero-budget compatibility, source/global-ledger limits,
RHF/UHF, final physical states and force transitions. A separate 96-AO tetramer
test uses 64 MiB for energy→force→energy: energy selects resident value storage,
forces replan to streamed storage, and both tensor/generated one-electron
providers match every PySCF force component at 1e-8 absolute and energy at 1e-9.
The global inventory still admits only its existing supported shape range.
Its energy/force envelope and native allocation rejection remain enforced.

The first budget implementation exposed a cached-plan error in job 9547:
positive-budget fleets discard host preparation data but retain the device
plan, so a host-cache check missed the property change. That candidate is
explicitly rejected. Its frozen library and two exact native source overrides
are retained. The final implementation records the value allowance on each
device plan and rebuilds it before replacement preparation when it changes.

To reproduce, restore the archive into a new directory, apply its native patch
to the recorded base commit, and use the retained CMake configuration. Compile
`reproduction/prepare-large.cpp` with `-std=c++20 -O3 -Isrc -Iinclude`, linking
the rebuilt `libvibeqc` and setting its runtime library path. Prepare the
768-AO input through `benchmarks.issue308_stage_probe --prepare` and the retained
independent checkpoint. Adapt the absolute workspace paths in
`reproduction/run-validation.sh`, then run it with finite
`srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:06:00`.
Preserve Slurm's device visibility. Every selected archive member was restored
byte-for-byte; full transient matrices and routine logs remain outside Git.

Streamed-K raw reuse, retained-B/bounded-scratch storage, SCF orchestration,
the complete 768-AO analytic-force endpoint and the #206 timing matrix remain
outstanding. This preparation evidence does not close their acceptance gates.
