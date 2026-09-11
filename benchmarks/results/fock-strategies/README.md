# Independent Fock provider production comparison

The architecture change preserves the measured exact/DF HF endpoints. All
energies and raw J/K matrices are identical before and after; complete-force
differences remain below 1e-14 Hartree/bohr. The accepted publications have
**numerical scope**. Timing review below supports non-regression on this machine;
no faster scientific schedule, tuner selection or speedup is promoted.

Baseline: `3da58410bb02a903ea6341a7caac0afc9314355b`.
Candidate: `913fb20a25aab87363b19e9d788580230493b688`.
Both source trees were clean. Both libraries use native Release builds,
CUDA 12.9.1, sm_120, AOT on, FAST_COMPILE off, and GCC 11.4.0. Measurements
ran on the AMD EPYC 7K62 and NVIDIA RTX 5090 under finite Slurm job 9137,
with the assigned device visibility preserved.
Published library paths use baseline/candidate checkout placeholders; exact
source/native identities and compiler locations remain recorded. Subsequent
review follow-ups validate unused C API controls and clarify threading and
test contracts; the measurements above identify their exact earlier revision.

| Complete endpoints | CPU | CUDA |
| --- | ---: | ---: |
| Cases / samples per case | 40 / 7 | 40 / 7 |
| Median candidate/baseline ratio | 1.007665 | 1.002777 |
| Ratio range | 0.965264–1.028171 | 0.977385–1.021853 |
| Maximum energy difference (Hartree) | 0 | 0 |
| Maximum force difference (Hartree/bohr) | 3.775e-15 | 9.034e-15 |

Each backend covers H2/water, restricted/unrestricted spin, exact/DF,
singlepoint, prepared warm replay, changed geometry, fresh four-item batch
plans and warm four-item batch plans. Library/kernel warmup precedes timing;
these are not process cold-start measurements. Full workers run sequentially
per revision, so the results do not establish a statistically significant
speedup. Existing plan allocation and schedule diagnostics are retained;
complete endpoint peak memory and matched total compilation cost were not
measured. Production speedup gates therefore remain explicitly not run.

## Raw dispatch and the resolved CPU DF outlier

The final 16-AO CPU DF contraction changes from 82.372 to 82.692 microseconds
(ratio 1.003879). The 16-AO exact contraction changes from 60.047 to 60.743
microseconds. Tiny two-AO calls expose about 0.154 microseconds of additional
exact dispatch and 0.039 microseconds for DF; these fractions of a microsecond
do not materially affect the measured complete endpoints. CUDA raw DF ratios
are 1.0142 (two AOs) and 1.0292 (16 AOs). Baseline CUDA has no independent raw
exact API, so direct CUDA comparisons use its complete fused HF endpoints.

At pre-fix revision `6fc010e`, 16-AO CPU DF took about 107 microseconds.
Calling that revision's raw service without the new wrapper retained the
slowdown; the wrapper cost was about 0.2 microseconds. The private exchange
function had identical instructions except relocated targets/constants, but
its address shifted by 16 bytes modulo 32. Interposing the same source with
32-/64-byte alignment restored 81–82 microseconds; 16-byte alignment retained
107–110 microseconds. The final source applies a local optional 32-byte
alignment hint. Arithmetic and reduction order are unchanged.
[Layout measurements](cpu/layout-diagnosis.json) retain all timing samples and
matrix identities from these controlled experiments.

The initial small H2 UHF DF warm-batch outlier did not persist in six
alternating-order baseline/candidate cycles (nine samples each): median ratio
0.999811, range 0.990932–1.011094. See
[focused samples](cpu/focused-repeats.json). Every repeat also passed its
energy/force and raw-matrix comparison.

## Reproduction and retained bytes

Use the production build configuration above in separate checkouts, then run
`tools/benchmark_fock_strategies.py` with the two paths under a finite
`--time=00:35:00` Slurm allocation, as recorded in each publication manifest.
The complete comparison uses seven samples per endpoint and nine samples per
fixed-density shape/provider. All GPU commands must keep Slurm's device
visibility. The CPU bundle also contains [a focused reproducer](cpu/reproduce_focused.py)
and [a layout reproducer](cpu/reproduce_layout.py), each with explicit input
and output paths; use `--help` for their arguments.

[CPU publication](cpu/publication.json) and [CUDA publication](cuda/publication.json)
were selected using `tools/vibeqc_validation/publication.py`. Each has a
passing shared validation envelope, declared tolerances, exact source/native
identities, toolchain/hardware provenance, compact summaries and complete
paired timing/numerical samples. Transient logs, build products and retry
worker directories remain outside Git. No external archive is needed to
reproduce the decision.
