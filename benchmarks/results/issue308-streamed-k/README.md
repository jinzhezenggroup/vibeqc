# Shared streamed-K source-work policy (#308)

The independently qualified fixed-D probes complete J, dense K and occupied K
at 192, 384 and 768 AOs. Dense and occupied K use the same row/raw-P/output-Q
policy and preserve the complete DF metric model. The tight reported 768-AO
layout still times out; its interrupted journal is retained.

| AO / explicit layout | Executed rows / Q | Raw tensor passes per K | Repeated J (s) | Repeated dense K (s) | Repeated occupied K (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| 192 / reported 8192 × 128 | 192 / 28 | 7 | 1.4437 | 4.9944 | 4.9922 |
| 192 / shape-selected 36864 × 94 | 192 / 94 | 3 | 1.4409 | 2.1589 | 2.1659 |
| 384 / shape-selected 147456 × 209 | 384 / 209 | 2 | 15.9731 | 16.0305 | 15.9956 |
| 768 / reported 8192 × 128 | 256 / 5 | 462 predicted; incomplete | — | 120-second timeout | — |
| 768 / shape-selected 589824 × 437 | 768 / 437 | 2 | 207.8483 | 209.0239 | 208.3984 |

These are diagnostic measurements with progress/component tracing and process
memory sampling. Each complete run includes a first and repeated J/dense-K/
occupied-K call using the same prepared plan and independent converged D.
They are not cold SCF, full-force, clean performance, or GPU4PySCF parity results.
The pass count is logical source work, not a timing multiplier.

The shape-selected layouts came from 128-MiB, 1-GiB and 8-GiB **value-plan
queries with an assumed 1-MiB fixed reservation**, respectively. The native
probe accepts explicit tiles and records actual source-inclusive plan diagnostics;
it does not enforce a public budget request. Whole-HF global-resource acceptance
is not established by these runs. The existing qualified global shape range is
unchanged.

At 768 AOs, every J component agrees with PySCF within 6.502e-12; dense and
occupied K agree within 9.806e-13 and 9.895e-13. Their mutual maximum error is
1.599e-14. Setup took 0.2301 s because this plan regenerates values during J/K.
Sampled GPU process memory peaked at 8,530 MiB; the source-inclusive value-plan diagnostic peak was 8,549,069,993 bytes. Neither is a whole-HF global-resource acceptance. Each completed J, dense K and occupied K submits two full raw tensor passes.
At the reported small allocation, execution now reuses each raw P panel across
five Q outputs. Its 462-pass full traversal remains prohibitively expensive and
is not validated by its 120-second partial probe. Retained B with independently
bounded contraction scratch remains the next storage slice.

The implementation is `bd05c51`, based on `dfd062b`. Slurm job 9549 passed
74 Python GPU tests plus native DF and capture-recovery suites. Those tests
include independent 96-AO J/K checks and exact executed source-work counts for
both K routes, including the final Q=1 tail. Host shape/resource and exhaustive
small scheduling checks, structure checks and pre-commit passed. All real GPU
work ran under finite `srun` allocations on `main` with `gpu:5090:1` and the
scheduler-provided visibility; the diagnostic probe job is 9550.

The measured library SHA-256 is
`851360b13b56f9611b9b358cc3c2bf9fa81b89a294d4acbabb9c123bbecc04f9`;
native source identity is
`a9d959ed2d6aad6070028acdf7270d8a863d2053d99d8387ed728bbc7a404ca0`.
The frozen library and exact probe remain local under
`.artifacts/issue308-frozen-bd05c51/`. The archive retains completed and
interrupted journals, source/library/build/input identities, memory samples,
the exact reconstruction patch, runners, and the first 24 rows of D/J/both K
arrays. All components were compared before recording their maximum errors;
complete transient output arrays remain local with their hashes in the summary.
The independent 768-AO checkpoint is permanently retained in
`../issue308-large-diagnostics/reference.zip`.

Every member of `evidence.zip` was restored and compared byte for byte. Its exact
hash is listed in `summary.json` and the evidence-policy exception. Check out the
recorded source, reconstruct the independent fixtures with the archived driver,
and run `reproduction/run-probes.sh` inside its finite allocation to reproduce.
Keep trace/memory instrumentation separate from the clean #206 timing matrix.
