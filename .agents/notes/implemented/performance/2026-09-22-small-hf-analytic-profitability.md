# Small-HF analytical profitability model

Issue: #597

## Decision

The legacy AO boundary is not treated as a CUDA semantic. The current small-HF path couples three independent choices around 16/17 AOs: cached full-ERI Fock construction, the native one-thread-per-output matrix product, and the small native eigensolver. This change starts separating those choices.

For matrix products, the runtime derives exact work from the workload:
- native FLOPs: `2 * matrix_count * nbf^3`;
- native semantic traffic: `matrix_count * nbf^2 * (16 * nbf + 8)` bytes for the present untiled FP64 kernel;
- tiled-provider traffic lower bound: `24 * matrix_count * nbf^2` bytes;
- native blocks: `ceil(matrix_count * nbf^2 / 32)`;
- native waves: blocks divided by the target-legal resident one-warp-block capacity.

A complete target/profile calibration provides only launch latency, effective FP64 rate, and effective byte rate for the native and cuBLAS routes. The resolver compares
`launch + max(FLOPs / rate, bytes / bandwidth)`.
If calibration is absent or incomplete, the previously qualified 17-AO crossover is retained only as an explicit compatibility fallback.

For cached ERIs the model derives, without benchmarking:
- ERI elements: `system_count * nbf^4`;
- ERI storage: `8 * system_count * nbf^4` bytes;
- cached-Fock AO-quartet contractions: `fock_state_count * nbf^4`.

The persistent-ERI route is not dynamically switched by this slice. A meaningful comparison also needs screened Direct-J/K shell-quartet work plus expected SCF reuse. More importantly, topology/layout construction currently consumes the 16-AO compatibility boundary before runtime profitability resolution. Changing that selector alone would make topology/cache ownership inconsistent.

## Rejected alternatives

- Derive 16/17 directly from peak device FLOPs or memory bandwidth: the workloads are too small and dispatch/scheduling overhead dominates.
- Keep one shared `small HF = 16` concept: eigensolver, matrix-product and ERI profitability have different cost functions.
- Benchmark every molecule/AO count: exact structural work is already available and should be derived; calibration should measure only a small set of device constants.
- Dynamically change persistent-ERI routing before topology consumes the same policy: this would risk plan/layout mismatches.

## Validation intent

Native policy tests pin the exact 16-AO structural counts, preserve 16/17 behavior with no calibration, and use synthetic calibration to prove that the matrix-product crossover can move below or above 17. No endpoint speedup claim is made by this structural slice.

Agent: ChatGPT
Model: GPT-5.6 Sol
