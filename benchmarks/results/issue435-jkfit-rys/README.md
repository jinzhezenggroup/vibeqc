# Practical JKFIT auxiliary-f derivative qualification (#435)

> Full historical raw records are recoverable from the [verified Git snapshot](../retention-df-integration/README.md). Restore it before replaying the original raw-record commands. Compact summaries and all production-manifest references remain in this checkout.

## Change and scope

The compiler now generates cooperative Rys/compact derivatives for auxiliary-f
`003`, `103`, `113`, `203`, and `213`. Their quadrature orders are 3, 3, 4, 4, 4.
The existing strict DF quadrature deliberately does not admit the five-root
`223` class; that class retains the exact polynomial fallback. No basis shells
are dropped. The preceding 18 s/p/d manifest mappings are unchanged.

This builds on the workload-based derivative admission in #486. Consumer
admission and mathematical capability are separate: removing the old equal-size
or water-histogram guard does not create missing f-shell mathematics. The
combined version uses the general workload policy and the promoted 23-class
sm_120 manifest without new molecule, AO/rank, or marketing-name fingerprints.

The first campaign below isolates **only the five new f lowerings**, with a
fixed explicit shell/packet consumer. Its `auto` arm means the old qualified
18-class profile inside a single candidate binary, not an unrestricted old
whole-default path. Do not confuse this with the separate final default-path
cross-binary qualification.

## Final automatic-path qualification

The combined build is now independently qualified in its ordinary default
configuration, without `WEIGHTED_EXECUTION`, `DERIVATIVE_PAIRS`, packet, schedule,
response or occupied-exchange overrides. All three arms use `SHELL_POLICY=auto`.
The old arm already contains #480; it is **not** the older multi-second baseline
from the opening issue description.

Seven round-major interleaved triples per case use fresh processes and identical
frozen density blobs. Each invocation independently reconstructs its exact CPU
PySCF reference and performs a complete untimed prime before one clean sample.
All 42 clean measurements finish before six separate diagnostic invocations.

| Orbital/auxiliary | Before general admission (post-#480) | #486 admission only | #486 + this PR | Combined reduction |
| --- | ---: | ---: | ---: | ---: |
| 96/464 | 400.069681 ms | 116.220613 ms | 86.454980 ms | 78.39% (4.63x) |
| 192/928 | 341.615192 ms | 303.464744 ms | 248.064558 ms | 27.38% (1.38x) |

The incremental f-kernel reduction versus #486 is 25.61% / 18.26%. All seven
combined/baseline and combined/parent pairs improve. Every arm satisfies the
predeclared relative median-absolute-deviation threshold of 5%; the observed
maximum is below 0.60%. All raw measurements are retained, without outlier
removal. Warm SCF updates are matched at two / three, and traced J/K/eigensolve/
physical-final-state calls match across all three arms. Density SHA-256 and the
actual loaded native source identity are verified per invocation, not inferred
from a file path or the benchmark driver's current Git commit.

The largest error across the 42 clean samples is 3.184e-12 Eh and
1.920e-12 Eh/Bohr, within the unchanged 3e-11 / 3e-11 gates. The final trace
selects angular grouping for 96/464, signature packets for 192/928, Rys for all
five newly qualified f classes, and polynomial for `223`. The class work agrees
with host reconstruction and the preceding admitted path; no screening or
scientific work is removed. A postprocessor fix handles the angular-only
`p0_0_0` signature sentinel as a group of real primitive lengths, not literal
zero primitives. For sparse angular groups, host-task costs give explicit tight
bounds; full groups and concrete signatures retain exact primitive counts.
This reporting fix does not change the measured CUDA library.

`default-96.json` and `default-192.json` retain all 21 clean result documents per
case plus three clearly separated diagnostic documents. `default-summary.json`
records paired ratios, branch counts, numerical maxima and the operator ledger;
`default-campaign.json` records the actual invocation ordering, return codes,
source bases and input hashes. `final-ledger-96.json` and
`final-ledger-192.json` contain the fully reduced and independently reconstructed
class/signature work. Only repeated diagnostic `_work_` counter fields are
omitted from endpoint documents, with the exact predicate and raw hashes
recorded. No clean result field is removed.

The combined Release library SHA-256 is
`05ba2fd4c7e5996dc2405928a3c3b939d85e0594644c14a70e71bca823e639e5`;
its actual native identity is
`b028248591d664fd4aff1751b90c951e03f89bf1fb3da90b5bfcdd44327e3a23`.
Reconstruct `4168e93978d3df5f686bda6ac67e6e3a229f3fa8` plus `final.patch`;
`final-build-identity.json` binds each maintained and generated source file.
The preceding arms' exact source bases, patches, hashes and native identities
are in `default-summary.json`; their patches are retained by the existing
`issue445-df-rys-admission/candidate.patch` and
`issue445-df-work-admission/candidate.patch` campaigns. The benchmark
`git_head` identifies the driver checkout, not an older compiled binary.

The final, unmodified compiled implementation passes **40 GPU regression tests**
covering the parent automatic-admission suite and the complete practical suite
(`validation.json`, Slurm job 10023). It is ready for review together with #486;
this evidence does not imply that either PR has already merged.

## Same-binary causal experiment

RTX 5090; CUDA 12.9.1 / nvcc 12.9.86; GNU C++ 11.4; Release `-O3 -DNDEBUG`;
sm_120; FP64. Fast compile and the unrelated Direct four-center AOT bundle are
off. The generated DF kernels are enabled. The actual explicit bases are the
existing `issue206-practical-auxiliary/identity/cc-pvdz.json` and
`cc-pvdz-jkfit.json`; hashes and full basis metadata are retained per result.

Each cell has seven interleaved clean samples per arm, identical frozen density,
an untimed complete prime, and strict completion through the forces. All clean
samples precede the intrusive counter pass. There is no outlier deletion.

| Orbital/auxiliary | Common consumer | Old f polynomial | New f Rys | Reduction |
| --- | --- | ---: | ---: | ---: |
| 96/464 | angular groups | 113.216481 ms | 83.515086 ms | 26.23% |
| 96/464 | signature packets | 112.259995 ms | 81.722711 ms | 27.20% |
| 192/928 | angular groups | 321.959374 ms | 262.473826 ms | 18.48% |
| 192/928 | signature packets | 282.434179 ms | 228.233652 ms | 19.19% |

`practical-<aos>-off.json` and `practical-<aos>-packet.json` retain every clean
sample, full energy/force arrays, independent errors, SCF updates/residuals,
metric ranks/resources and source/library identities. Both arms execute two
warm SCF updates at 96/464 and three at 192/928. Traced J/K/eigensolve/final-state
call counts also match; `summary.json` retains the ledger rather than estimating
operator speed by dividing wall time by the number of SCF iterations.

Fresh CPU PySCF density-fitting references use the exact same explicit bases,
`conv_tol=1e-13`, `conv_tol_grad=1e-12`, `direct_scf_tol=1e-14` and full
auxiliary-basis response. The unchanged practical gates are **3e-11 Eh** and
**3e-11 Eh/Bohr**, including initialization and untimed prime calls. The largest
errors in the four clean campaigns are 2.274e-12 Eh and 1.558e-12 Eh/Bohr.
Metric effective ranks remain 464 and 928. No cutoff, convergence rule,
precision, response formula, final-state criterion or screening is weakened.

## Scientific work and per-class attribution

The four `ledger-<aos>-<policy>.json` files validate the actual class/signature
counters against a fresh host reconstruction with distinct orbital/auxiliary
shell domains, including partial panels. The two arms preserve:

| Shape | Shell visits | Active primitive products |
| --- | ---: | ---: |
| 96/464 | 202272 | 1464752 |
| 192/928 | 1606320 | 11528520 |

Cartesian component products and public weight loads also match. The candidate
trace observes all five f classes selecting Rys and `223` staying polynomial.
The ledger counts shared recurrence work once per active primitive, then checks
the remaining active-component work; it no longer applies an equal-basis domain
assumption or mistakes a shared cache for repeated component recurrence.

Detailed `SHELL_WORK` counters are deliberately intrusive. Their wall times are
not performance results. `light-components-96.json` and
`light-components-192.json` instead retain separate ordinary component tracing
without those detailed work atomics. All five f classes improve in this pass;
for example `003` changes from 15.95 to 4.53 ms at 96/464, while the unchanged
`223` is 4.15 ms in both arms. CUDA event scopes can include stream idle gaps;
these are not Nsight hardware-only kernel durations, and they must not be added
to parent inclusive scopes or pooled with clean endpoint samples.

## Controls and validation

Five interleaved pairs at equal 96/192/384/768 AO give candidate/control ratios
1.00592, 0.99715, 0.99949 and 1.00012 respectively. These are neutral controls,
not new speed claims. The old independent 1e-9 Eh / 1e-8 Eh/Bohr gates remain;
the exact values, references, full outputs and branch counts are in
`equal-96.json`, `equal-192.json`, `equal-384.json`, and `equal-768.json`.

`validation.json` records 231 independent host Rys tests, the related generation
and ledger suites, and 32 CUDA practical tests. Host suites overlap other
invocations and are not added up as a unique-test total. CUDA tests cover
RHF/UHF, cold/warm/moved geometry, packed/dense response, budget limits and
actual f-class execution. A bounded practical OH packet test passes memcheck,
racecheck and synccheck with no errors or hazards. Those sanitizer runs repeat
one test and are not counted as three additional independent fixtures.

## Reproduce the candidate A/B

Reconstruct base `8853004023e0af3b74d23f3bb92a84efbe83d2a5` plus
`candidate.patch`, then build Release for sm_120 with generated DF and the
unrelated Direct AOT bundle disabled. `candidate-manifest.json` is the exact
23-class unqualified campaign profile with the old 18-class qualified baseline.
The first campaign library SHA-256 is
`3f9e2027dff3505a48228b246d88cdb3d9a91c1d239ee4ddeaad180faacab6ca`;
its native source identity is
`f5fcacb551b44c9ec8e57cea55575c303b16a4d617a48251c7ffd23dfff4b25d`.
The patch hash is recorded in `summary.json` and every endpoint result.

Inside an allocated GPU job, run (change `96` to `192` for the larger fixture):

```bash
export PYTHONPATH=python:.
export VIBEQC_LIBRARY=/absolute/path/to/candidate/libvibeqc.so
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export VIBEQC_DF_WEIGHTED_EXECUTION=shell VIBEQC_DF_DERIVATIVE_PAIRS=symmetric
export VIBEQC_DF_SHELL_SCHEDULE=compact VIBEQC_DF_PRIMITIVE_BUCKETS=packet
BASIS=benchmarks/results/issue206-practical-auxiliary/identity
python -m benchmarks.df_policy_endpoint --aos 96 --repeats 7 \
  --control VIBEQC_DF_SHELL_POLICY --policies auto candidate \
  --cpu-reference --orbital-basis-file "$BASIS/cc-pvdz.json" \
  --auxiliary-basis-file "$BASIS/cc-pvdz-jkfit.json" \
  --source-patch path/to/candidate.patch --components-after --shell-work \
  --output .artifacts/practical-96-packet.json
```

Use `VIBEQC_DF_PRIMITIVE_BUCKETS=off` for the angular-group control. The current
promoted production profile does not contain the old campaign baseline, so
`auto` versus `candidate` on the final binary is not that historical A/B.

## Retention and boundaries

All clean samples, numerical outputs and identities remain unrounded. Large
JSON values use compact whitespace and exact equality was checked on retained
fields. Repeated detailed `_work_` counter keys are omitted from endpoint
component aggregates; validated packet class/signature work is retained once in
the four work ledgers. `retention.json` names the exact omitted field predicate
and original/retained hashes and byte counts. Logs, checkpoints, binaries and
raw profiler streams are not committed. No file exceeds the 1-MiB guard.

This evidence does not claim fresh GPU4PySCF timing, a cold/changed-geometry
throughput win, other-device performance, screening (#437), response fusion
(#419), or completion of every remaining packed-layout selector. The five-root
fallback is explicit rather than silently dropping an actual JKFIT shell.

Agent: ChatGPT
Model: GPT-6 Astra Pro
