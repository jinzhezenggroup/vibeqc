# Checked RHF density seeds and retained final exchange (#399)

A supplied low-rank PSD density now seeds RI-K through `D=L L^T`, using the
existing compact GPU eigensolver and occupied exchange with weight one. The
separate final physical Fock build uses retained canonical C with weight two
only when the supplied density, final token, device generation and solver
status all match. Both still execute ordinary strict SCF/force gates.

Automatic selection is limited to RTX 5090, singleton resident RHF,
`nbf=naux=768`, `nocc=160`, full tiles and existing occupied storage. Unsupported,
indefinite, excessive-rank or inaccurate factors keep dense K. No persistent GPU
buffer or full three-index tensor is added.

## Matched clean comparisons

Every entry has five alternating warm samples per arm and exactly three SCF
updates in every sample. Each policy receives an untimed prime, and all arms at
a given size use the identical frozen density. Timings include the complete
strict endpoint; diagnostics below are separate.

| Change | AOs | Energy only, before → after (s) | Energy + force, before → after (s) |
| --- | ---: | ---: | ---: |
| Both automatic selectors versus dense seed/final | 768 | 3.461292 → 2.192251 | 5.330060 → 4.067887 |
| Seed, dense → factor; final dense | 384 | 0.379769 → 0.349297 | 0.853629 → 0.820646 |
| Seed, dense → factor; final dense | 768 | 3.445536 → 2.818769 | 5.316988 → 4.689939 |
| Final, dense → retained; seed factor | 384 | 0.349290 → 0.311753 | 0.807683 → 0.777450 |
| Final, dense → retained; seed factor | 768 | 2.820675 → 2.177101 | 4.685339 → 4.045592 |

The coupled final default reduces 768 AO energy+force time by **23.7%** (1.31×)
and energy-only time by **36.7%** (1.58×). The other rows isolate each saving.

The 384 comparisons explicitly force occupied SCF in both arms. They do not
compare against the normal 384 dense-SCF default, and do not promote 384.
The isolated comparisons use frozen v1; v2 enables the exact-domain automatic
selectors, propagates CUDA errors from final-token queries, and is measured
again with both baseline/default controls coupled in each interleaved arm.

There is no cross-engine speed claim. GPU4PySCF supplies independent numerical
references, validated against the exact geometry, basis fingerprints, method,
representation and scientific settings before numerical comparisons. These are
checkpoint warm replays; initialization includes a complete untimed force call
and is recorded separately. It is not a cold-solve benchmark.

## Numerical and work contract

RHF, spherical def2-SVP, the same orbital/auxiliary basis, FP64, batch one,
metric cutoff `1e-10`, screening `1e-12`, energy convergence `1e-12`, density
convergence `1e-10`, 100-update limit. Independent complete energy/force gates
remain `1e-9` Ha and `1e-8` Ha/bohr. No tolerances were loosened.

| Direct seed check | 384 | 768 |
| --- | ---: | ---: |
| Retained rank | 80 | 160 |
| Reconstruction maximum | 3.109e-15 | 2.665e-15 |
| Reconstruction RMS | 7.086e-17 | 6.245e-17 |
| K maximum | 6.573e-14 | 1.870e-13 |
| K RMS | 7.089e-16 | 2.098e-15 |
| Complete Fock maximum | 3.286e-14 | 9.348e-14 |
| Complete Fock RMS | 3.545e-16 | 1.049e-15 |

The two K matrices use the identical input D; shared H/J make the complete
Fock difference exactly half the K difference. The intrusive verification
restores candidate K before executing the remaining real endpoint. Seed
reconstruction checks every matrix entry, including both triangles. Negative
spectrum below `-1e-13`, rank above capacity, discarded spectral Frobenius norm
above `1e-12`, maximum error above `1e-12`, or RMS above `1e-13` rejects promotion.

At 768, seed-only diagnostics remove one dense `ri_k`, add one eigensystem and
one occupied projection/Gram, and preserve two canonical occupied updates.
Factorization is about 16.4 ms; complete seed iteration is 924 → 276 ms.
The baseline's two dense K calls become one after seed factoring and zero when
retained final K is enabled. Final dense K is specifically the physical callback
in `finalize_density_fitting_rhf` → `select_cuda_df_final_state` / `select_final_state`.
Its traced host caller is `final_state_dense_jk`; the replacement is
`final_state_retained_jk`, which still recomputes physical J/K.

The 768 final reuse gates observe commutator maximum about `1.414e-12`, density
RMS `2.948e-18` and idempotency maximum `2.842e-14` in the seed validation run.
Full journal records and final automatic-mode observations remain in the evidence.
No canonical generation is assigned to the algebraic seed factor.

## Transfers and memory

Separate Nsight CUDA-activity windows, in capture order:

| AOs / policy | H2D bytes | D2H bytes | D2D bytes | Stream synchronizations | Sampled device peak bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 768 / dense seed + final | 28,416,308 | 28,320,179 | 0 | 12 | 21,304,967,168 |
| 768 / factor seed, dense final | 28,416,308 | 28,326,363 | 4,718,592 | 15 | 21,304,967,168 |
| 768 / both automatic | 28,416,308 | 28,326,391 | 4,718,592 | 16 | 21,304,967,168 |
| 384 / dense seed + final | 7,130,292 | 5,902,579 | 0 | 8 | 8,573,157,376 |
| 384 / factor seed + retained final | 7,130,292 | 5,905,699 | 1,179,648 | 11 | 8,573,157,376 |

These include library-internal calls. The seed helper itself adds two explicit
stream drains and 6164/3092 D2H bytes at 768/384, plus a device copy of D. The
additional observed 768 solver transfers/synchronization are charged above.
Event synchronization counts are retained separately in the profile API tables;
they include the existing compact solver's internal work.

Every warm capture's sampled high-water is unchanged between arms. These profiles
share a process across sizes and include opaque CUDA/library caches, so the 384
absolute value also reflects prior 768 initialization. The overall sampled
initialization/warm high-water is 24,584,912,896 bytes. No allocator-exact peak or
achieved hardware occupancy is claimed. Unprofiled 768 resident readings are
21,252,538,368 bytes in both seed arms.

The final runtime passes all **82** selected real-GPU regression cases, the
native density/final-identity fixtures and memcheck with zero errors. Endpoint,
reference, ownership and evidence-tool CPU checks pass **65** cases; the earlier
checkpoint/resource suite passes 61 with two explicit CUDA skips. Validation
records bind these counts to the exact logs.

## Evidence and reproduction

- `timings.json` contains every raw clean sample and median; `measurements/`
  retains all energy/force arrays and convergence records, with repeated basis
  metadata interned once per run and original JSON hashes preserved.
- `components/` contains separate intrusive traces and executed counters.
  Nested GPU/host regions are inclusive and must not be added together.
- `journals.json` retains the SCF/final-state journal prefix before force response,
  with original complete-journal hashes and row counts.
- `profiles/` contains reduced Nsight captures, transfer/API counts and sampled
  memory. Sampled process high-water is a lower bound, including opaque runtime
  storage; it is not an exact allocation peak. No hardware occupancy is inferred.
- `validation.json`, `builds.json` and `ownership.json` bind tests and physical
  CUDA changes to their logs and artifacts. Routine logs, binaries, profiler
  databases and checkpoint files are kept outside Git.

V1 base: `1c3f2ab8d6df6f06bee526b89a807511ef726301`.
V2 base: `575b83418bb87db9b666f41a34b94302d7ea7718`; intervening ECP and historical
DF evidence changes do not alter this all-electron DF workload. Runtime source
patches, native/library identities and runner hashes accompany the measurements.

The 768 density hash is
`09b2516e07b8483b47ac9607b9c30abc3485d2dfe7da92ca8f541017614738d9`;
384 is `90c4beddd055f1ffe2c600298ef060dc6ad286cbc17d04a909d0698006177e47`.
Checkpoints were produced by the retained issues388–391 ablation workflow and
remain in its local artifact directory; reproduction scripts record their exact
paths and reference files. Use the recorded checkpoint, not a new post-cold D,
for fixed-work comparisons.

Build with the production `cuda-release-sm120` preset (fast compile off), CUDA
12.9.1, and one host numerical thread. Freeze the library and source patch
alongside a `libvibeqc.so.0 → libvibeqc.so` symlink. Run every GPU command inside
a finite `srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1` allocation,
preserving Slurm's device visibility. Exact drivers and reducers are in
`reproduction/`; the [Agent Note](../../../.agents/notes/implemented/performance/2026-09-16-density-exchange-seed.md)
explains ownership, fallbacks and rejected alternatives.
