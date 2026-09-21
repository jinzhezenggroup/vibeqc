# Direct-HF psss force retirement evidence (#356)

This bundle records the decision to retire the parallel handwritten psss
weighted-force formula after #717 introduced the force-only generated expression.
The production scheduler remains native; only the scientific derivative algebra
moves to the compiler-owned expression.

Measured source before retirement: `41be6f8d8db7b74336810cf3e27c5b33595bbc52`.
Same-binary A/B library SHA-256:
`35d77e571598182caeedd2170fe21982d235d2989dc5d1666d23a50b06794564`.
Retirement candidate: `84a469068dddac8551a750d63de01ed1c849b873`.

All GPU measurements used finite Slurm `main` allocations on the RTX 5090,
CUDA 12.9 / sm_120 Release path, with one CPU BLAS/OpenMP thread. A ratio above
one below means the generated formula is faster: reference time / generated time.

## RHF def2-SVP matrix

Five-repeat complete energy+force checks covered batch 1/3 and fixed, resident,
and bounded-paged scheduling. Every numerical comparison passed the existing
validator gates.

| batch | schedule | cold | warm | changed | changed warm |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | fixed | 0.9754 | 0.9933 | 1.0020 | 0.9870 |
| 1 | resident | 1.0067 | 1.0233 | 1.0017 | 0.9992 |
| 1 | paged | 0.9860 | 1.0049 | 0.9988 | 1.0053 |
| 3 | fixed | 1.0067 | 1.0065 | 0.9943 | 0.9979 |
| 3 | resident | 1.0053 | 1.0008 | 0.9941 | 1.0092 |
| 3 | paged | 1.0077 | 0.9974 | 0.9967 | 0.9991 |

The apparent 2.46% cold regression in the first non-interleaved fixed run was
retested with an eight-repeat reference/generated ABBA campaign. The interleaved
result is **1.2656 cold**, **1.0159 warm**, **0.9985 changed**, and **0.9992
changed-warm**. The original cold result is therefore treated as run-order drift,
not a production regression.

## UHF holdout

The generic validator's separated multi-fragment UHF fixture was rejected for
promotion evidence because its reference route itself failed SCF convergence.
The replacement holdout is the established
`oh-def2-svp-spherical-uhf` case: OH doublet, 19 real spherical AOs.

| batch | schedule | cold | warm | changed | changed warm |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | fixed | 1.1958 | 1.0212 | 1.0039 | 1.0040 |
| 1 | resident | 0.9983 | 1.0028 | 1.0052 | 1.0015 |
| 1 | paged | 1.0037 | 1.0032 | 0.9914 | 1.0005 |
| 4 | fixed | 0.9984 | 0.9790 | 1.0008 | 1.0109 |
| 4 | resident | 1.0015 | 1.0009 | 1.0040 | 1.0136 |
| 4 | paged | 0.9992 | 1.0003 | 0.9918 | 0.9999 |

The first batch-4 fixed warm median was 2.15% slower and therefore received a
longer 12-repeat ABBA rerun (Slurm job 10609). That rerun measured **1.1751
cold**, **1.0119 warm**, **0.9963 changed**, and **0.9945 changed-warm**.
Maximum observed reference/generated UHF differences were on the order of
`4.3e-14 Eh` in energy and `1.2e-14 Eh/bohr` in force.

## Decision

The revised force-only generated psss mathematics has no reproducible complete
endpoint regression beyond the 2% structural-retirement ceiling. Both initial
outliers reverse or disappear under interleaved/longer sampling. RHF/UHF,
batching, fixed/resident/paged scheduling, and changed geometry retain numerical
parity.

Retire the handwritten weighted psss derivative formula and
`VIBEQC_PSSS_WEIGHTED` selector. Keep the qualified native schedulers,
screening, density contraction, resident primitive reuse, normalization,
physical-center scatter, and independent scalar/oracle validation.

Agent: ChatGPT
Model: GPT-5.6 Sol
