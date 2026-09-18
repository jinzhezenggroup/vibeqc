# DF consumer and packet admission by work

This is a bounded follow-up to #480 for #444/#445 and the practical-basis
admission problem in #435. It replaces bridge consumer/packet fingerprints,
not the separate packed-response layout selector, derivative screening or
response-factor fusion.

## Final automatic endpoints

RTX 5090 / sm_120, FP64, Release CUDA 12.9.1, complete warm energy plus forces.
Each value is the median of seven clean replays. Baseline and candidate restore
**byte-identical frozen density arrays** for that case and use the same SCF
update count. Both include a complete untimed prime before each measured replay.

| Workload / AO and auxiliary dimensions | Baseline auto, ms | Final auto, ms | Interpretation |
| --- | ---: | ---: | --- |
| Water tetramer, equal 96/96 | 85.866060 | 35.590552 | 58.55% lower time |
| Water tetramer, cc-pVDZ/JKFIT 96/464 | 398.945602 | 106.001749 | 73.43% lower time |
| Water octamer, cc-pVDZ/JKFIT 192/928 | 322.230717 | 282.027114 | 12.48% lower time |
| Non-water ammonia trimer, equal 87/87 | 57.630727 | 30.113173 | 47.75% lower time |
| Non-water ammonia tetramer, equal 116/116 | 110.360510 | 41.777427 | 62.14% lower time |
| Single water, equal 24/24 | 6.881327 | 6.891730 | Generic fallback; neutral |
| Single ammonia, equal 29/29 | 9.005535 | 9.027680 | Generic fallback; neutral |
| OH UHF, equal 19/19 | 9.146294 | 9.110447 | Generic fallback; neutral |
| Ammonia dimer, equal 58/58 | 22.889712 | 22.920473 | Conservative generic fallback; neutral |

**Comparison boundary:** the baseline and final binaries were measured in
separate pinned campaigns, not seven cross-binary interleaved pairs. Within
each binary, the `auto`, explicit shell and explicit packet controls are
interleaved in alternating order. The final campaign uses an exclusive finite
Slurm allocation, after compilation and numerical testing. The separate
intrusive ledger follows every clean sample collection. No cold/setup,
changed-geometry throughput or external GPU4PySCF timing win is claimed.
An additional cross-binary allocation started just before its cancellation.
Its two completed pairs and interrupted next initialization are retained in
`cross-binary-partial.json`, but do not meet the planned seven-pair gate and
are not counted as promotion evidence.

The baseline binary also provides seven fully interleaved causal controls:
96/96 auto versus explicit shell is 85.866/35.889 ms; 96/464 is
398.946/103.385 ms; 192/928 auto versus explicit packets is
322.231/282.736 ms. Thus the admission mechanism is measured within one binary,
not inferred solely by subtracting the two campaigns. The final patched auto
route is independently validated against these existing explicit consumers.

The final 192/928 same-binary control compares auto/packet with explicit
angular-only shell execution: 282.027/282.218 versus 321.445 ms. This is an
interleaved attribution result at unchanged shell/primitive work. At 96/464,
shell admission explains the dominant change; packet grouping is not necessary
for the admitted default.

The automatic 192/384/768 equal-basis controls have seven-sample medians
81.582348 / 334.412588 / 2140.545552 ms. All take three updates. The latter two
are consistent with the retained #480 controls, not new large-case speedups.
The 192 control adds packet admission on top of #480's Rys lowering.

## Numerical and work contract

Practical cases preserve the existing **3e-11 Hartree and 3e-11 Hartree/Bohr**
independent energy/force gates. Equal-basis cases preserve 1e-9 / 1e-8 gates.
Every cold, prime, clean and diagnostic result is checked against a fresh
independent PySCF full gradient with auxiliary-basis response enabled. All
completed results pass; no tolerance, auxiliary function, metric direction,
precision mode or physical-force-state check is removed.

The practical basis snapshots are the existing canonical cc-pVDZ and
cc-pVDZ-JKFIT files, including auxiliary f functions. Non-water holdouts use
29-AO ammonia replicas separated by 8 Bohr and solve the interacting full
cluster, not a sum of isolated fragment energies. The 58-AO case remains on the
conservative generic side even though its explicit shell trial wins modestly.

At 192/928, shell and packet diagnostics retain 1,606,320 shell triples,
11,528,520 primitive products, 82,060,208 Cartesian component products and
34,209,792 public weights. Signature grouping changes execution grouping, not
the derivative equations or those work counts. At 96/464, the admitted shell
path records 202,272 shell triples, 1,464,752 primitive products, 10,411,644
Cartesian products and 4,276,224 public weights. Generic-to-shell admission
changes symmetry/reuse work, so it is not described as identical kernel work.
J/K, response, state and transfer counters remain in the diagnostic records.

## Implementation and rejected alternative

The CPU-testable sm_120 profile admits shell execution at estimated public work
`NAO^2*Naux >= 2^18`. Packets additionally require ordered primitive work
`P_orbital^2*P_auxiliary >= 2^22`, heterogeneous contraction lengths within an
angular class, one density term, spherical bases and the existing compatible
shell/pair schedule. Correctness/source/metric/resource guards remain separate.
No molecular histogram or equal-auxiliary condition enters those two rules.

The first candidate added about 1 ms to small endpoints by querying the entire
CUDA property record. A 200-call probe measured 1.047210155 ms per full query
versus 0.000112265 ms for the two required compute-capability attributes. The
shared `runtime/cuda_architecture.hpp` helper now reads only those attributes
in the bridge and per-panel lowering lookup. It does not cache device ordinals,
change streams or suppress runtime failures. The old product-name query survives
only inside the separately retained 768/rank-160 packed-layout eligibility.

`rejected-full-query.json` preserves all clean v1 timings and numerical errors;
the small-system regression is not erased. `device_query.cpp` reproduces the
micro-diagnosis, not an endpoint-speedup claim.

## Validation

- 83 CPU policy, benchmark-contract and ownership tests pass.
- 46 distinct real-GPU regressions pass, including genuine auto cold/warm/moved
  geometry, 96/192 AO batch four, practical auxiliary f shells, UHF, existing
  panel/group/packet lowering and bounded-budget checks.
- The first CUDA invocation has 44 passes and one loader failure: a standalone
  executable linked the renamed frozen library's `libvibeqc.so.0` SONAME without
  its search-path alias. Supplying the alias fixes that environment issue.
  The final auto/selector invocation passes all nine tests, including the added
  192-AO batch-four case. `validation.json` preserves both receipts and the
  unique-test count; no production source changed for the loader repair.
- CUDA Release compilation and the shared SCF/compiler dependency checks pass.
  The generated mathematical kernels and DF manifest are unchanged.

## Provenance and reproduction

Measured source base: `8853004023e0af3b74d23f3bb92a84efbe83d2a5` (#480).
`candidate.patch` reconstructs the exact compiled follow-up. `summary.json`
records hashes of every affected source and both binaries. Baseline library:
`14da088e24c8ad9707659c889785e3643ba8b006b88e04645476e5274c108659`.
Final library:
`2f72792115ab26d4d909471677867991b612ebb4ecd8039a516a1610ca35ada8`.

Both builds use CUDA 12.9.1, GNU C++ 11.4, sm_120, `-O3 -DNDEBUG`, no fast-compile
option and the generated DF kernels. The unrelated Direct four-center AOT
bundle is disabled in both. The final library grows by 7,448 bytes; no new
scientific kernel family is generated. The CUDA ownership inventory counts
three extra lines in conservatively classified scientific host glue; this is
not a scientific-code retirement claim.

Use the current `benchmarks.df_admission_probe` with a Python environment
containing PySCF and the benchmark dependencies. Set `VIBEQC_LIBRARY`,
`PYTHONPATH=python:.`, the CUDA runtime path, and OMP/OpenBLAS/MKL thread counts
to eight. Preserve Slurm's device visibility. For example:

```bash
srun --exclusive --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --time=00:15:00 \
  python -m benchmarks.df_admission_probe \
  --case water-tetramer-def2-svp-spherical --variants auto shell packet --repeats 7 \
  --orbital-basis-file benchmarks/results/issue206-practical-auxiliary/identity/cc-pvdz.json \
  --auxiliary-basis-file benchmarks/results/issue206-practical-auxiliary/identity/cc-pvdz-jkfit.json \
  --source-patch path/to/measured-source.patch --output .artifacts/practical96.json
```

The probe saves a checkpoint beside the transient output. Use
`--warm-checkpoint-in` with the same checkpoint in both binaries, restoring only
a compatible density and retaining its hash. `--copies 3` with the ammonia case
reproduces the non-water trimer. `--no-diagnostics` supports a clean-only run;
it was used for the additional partial cross-binary run, which is not labelled
as a completed seven-pair campaign.
The existing `benchmarks.df_policy_endpoint` reproduces the 192/384/768 equal
controls with the prior fixed checkpoints and independent retained references.

## Retention and remaining scope

All clean timings and complete energy/force vectors are retained without
rounding. Identical basis and metric metadata are deduplicated losslessly;
`metric_indices` addresses each arm's `metric_records`. Untimed primes retain
convergence, timing and maximum errors instead of repeated full vectors.
Selected lowerings, resource maxima and semantic counters remain; each record
names exact omitted trace detail and the original record's hash and byte count.
Logs, checkpoints, profiler archives and binaries are not committed.

The broader #435 practical per-class/external matrix, #437 screening, #419
factorized consumer fusion, remaining packed-layout admission under #444/#445,
and compiler-wide specialization #459 remain separate. This report does not
claim those milestones or extrapolate the sm_120 profile to another device.

Agent: ChatGPT
Model: GPT-6 Astra Pro
