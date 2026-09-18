# AOT direct confirmation and stock DF operator observations

> **Historical supporting data:** bulky reports from this campaign remain in
> existing Git history, with [checksum-verified snapshot recovery](../retention-checkout/README.md).
> The summary below and compact records remain here. Restore the complete
> snapshot before running historical scripts or verifying its original
> manifests; those manifests describe the original snapshot, not this reduced
> checkout. No measurements, rejected cases or acceptance thresholds changed.

An AOT-enabled release resolves the original 192-AO direct execution failures.
Both 192-AO batch sizes pass all seven original energy/full-force comparisons.
The 96-AO strict force checks still fail, so the complete direct matrix remains
failed and #206 remains open. No scientific source or acceptance setting changed.

This follows the failures in [#421](https://github.com/jinzhezenggroup/vibeqc/pull/421)
and their [#422 diagnosis](https://github.com/jinzhezenggroup/vibeqc/pull/422).
Old failed samples remain preserved; this campaign has a separate binary identity.

## Direct capability and unchanged acceptance

Both builds have source identity
`ceafaa3df32bf05154e7532fb962c19b35c5d919603a9651b94274dcf17d0356`.
The earlier library disabled AOT shells, reported `generic_cuda`, and failed
initial 192-AO direct Fock construction with CUDA 801. The new Release build
enables AOT shells, reports `sm_120`, and has SHA-256
`7aed5066ca6de0ac8c6e700ab8571a7518f9ac88192b95b4517b2556ec24457f`.
Its selected CMake options and build resource observations are in
[build/configuration.json](build/configuration.json). The observed 33:29.46 wall
time and 3,566,456 KiB peak RSS are not a controlled clean-build comparison.

Slurm job 9872 uses the original four cases, seven interleaved repeats, FP64,
energy tolerance `1e-12`, density tolerance `1e-10`, screening `1e-14`, and the
original fixture-specific reference controls and error gates. Each engine starts
from its own frozen post-cold density, with untimed priming. Local compilation
and profiling had finished before clean timing began.

| AO / batch | Native median (s) | Stock median (s) | Maximum paired force error (Eh/Bohr) | Force gate | All seven pairs |
| --- | ---: | ---: | ---: | ---: | --- |
| 96 / 1 | 0.199980 | 1.691940 | 5.726133e-11 | 3e-11 | FAIL |
| 96 / 4 | 0.455265 | 6.750864 | 7.727527e-11 | 3e-11 | FAIL |
| 192 / 1 | 0.489359 | 2.232075 | 1.464315e-10 | 5e-10 | PASS |
| 192 / 4 | 1.635698 | 8.203617 | 2.033624e-10 | 5e-10 | PASS |

All energy pairs pass their original gates; the largest difference is
`6.593837e-12 Eh`. Native warm updates are one per item; stock branches differ.
These are ordinary converged-latency observations. They do not establish
fixed-work kernel speedups, and failed 96-AO timings remain unqualified.

`summary.json` checks the maximum error across every retained pair. The original
runner's `gate` object reports the last pair; those original bytes are preserved
alongside the full arrays and all-pair verdict. Its historical timing note also
mentions CUDA DF for these direct runs: this is a report-label defect, not an
integral-generation/contraction measurement. The direct numbers above are
complete warm endpoints, without such a component attribution.

The saved 96-AO geometries exactly match the earlier tighter CPU oracles in
[#422](https://github.com/jinzhezenggroup/vibeqc/pull/422). Comparing these new
samples to those oracles gives the following maximum force errors:

| Batch | Native vs CPU | Stock vs CPU |
| --- | ---: | ---: |
| 1 | 3.741173e-11 | 1.985634e-11 |
| 4 | 3.944635e-11 | 4.075407e-11 |

The unchanged AOT binary also completes the native test suite in Slurm job 9873:
47 passed, one opt-in MP2 allocation-status test skipped. That test was then
explicitly enabled and passed in a separate finite Slurm allocation. Both
records are retained in `validation/`; the skipped result was not rewritten.

## Separate intrusive stock DF profiles

Jobs 9870 and 9871 profile the already retained 384/768-AO stock complete-force
endpoints, using the same orbital and auxiliary def2-SVP basis. They do not add
clean timing samples. Each observes two logical J/K calls, one eigensystem, one
CDERI rebuild and seven raw three-center blocks. CSV exports, detailed operation
records, thread-local inclusive/exclusive scopes and accounting scripts remain.

| AO | H2D bytes | D2H bytes | D2D bytes | Device synchronizations | Profile force difference from first clean stock sample |
| --- | ---: | ---: | ---: | ---: | ---: |
| 384 | 440,844 | 1,215,723 | 164,912 | 134 | 2.220446e-13 |
| 768 | 832,636 | 4,785,863 | 369,408 | 152 | 8.135714e-13 |

Both energy differences are zero. All device synchronizations are accounted
for by the deliberate wrapper fences; they are not clean production counts.
Thread-local exclusive scopes subtract synchronous children, but concurrent
threads and CPU/GPU activity can overlap. Their sums do not form endpoint time.
These are CUPTI transfer bytes, not measured hardware DRAM traffic.

The 384-AO profile follows the first clean sample's iteration branch. At 768 AO
the profile has one update, whereas the first clean sample has three; the seven
original clean counts are `[3, 1, 1, 5, 1, 1, 1]`. The mismatch is explicit in
`stock-768/account.json`; no fixed-work attribution or pooling is performed.
Even matching iteration counts alone would not prove identical operator inputs.

Sampled host/device memory includes Nsight and wrapper overhead. It remains a
diagnostic observation, not unprofiled memory acceptance. The earlier large DF
regressions remain unchanged; these profiles do not establish superiority.

## Reproduction and remaining work

`manifest.json` binds 40 records, including full direct sample arrays, capability
journals, selected build options, stock CSV/JSON exports and exact runner source.
Workstation paths preserve original provenance; adapt paths to a fresh checkout
without changing inputs or controls. Use the measured revision and comparator
bytes identified by `direct/campaign.json`. Large native libraries, profiler
databases and routine logs remain transient and are identified by their hashes.
Every real GPU execution requires a finite Slurm `main` / `gpu:5090:1` allocation.

The direct 96-AO failures still need a numerical fix and fresh qualification.
Strict DF 96-AO batch-4 and changed-192 reference failures also remain, as do
missing cold/changed/warm-energy, smaller/UHF, unequal-auxiliary, constrained-memory
and unprofiled resource coverage. This evidence advances the parent audit but
does not complete #206 or justify a general GPU4PySCF superiority claim.
