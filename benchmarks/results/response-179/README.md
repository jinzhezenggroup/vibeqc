# Issue #179 response-solver evidence

This directory records partial shared RHF/semilocal-CPKS response-solver
infrastructure evidence for `#179`; it is not full #179 acceptance.  The
remaining acceptance items are a shared UHF response action and the native
converged RKS/UKS CPKS endpoint owned by `#162`.  The shared production path
recorded here is the CPU native shell-tile matrix-free RHF J/K operator.  The
CUDA record is generated only inside a Slurm GPU allocation; the optional CUDA
MO-block explicit control is not substituted for the matrix-free result.

The JSON records below predate the final review fixes to scalar/multi-RHS
workspace reservations and bounded recycle replacement. Their timings and
workspace figures remain historical measurements. Current solves reserve the
shared RHS, all retained results, recycle projection/replacement and numeric
solver temporaries; successful reported peaks fit the requested budget.
The CUDA DF backend also verifies the source/metric/threshold identity before
creating its plan. Reproduction commands generate results for the current code.

## CPU numerical evidence

Command:

```bash
VIBEQC_LIBRARY=$PWD/build/libvibeqc.so \
PYTHONPATH=$PWD/python:$PWD \
/group/software/deepmd-kit-3.1.1/bin/python tools/response_examples.py \
  --case water --device cpu --rhs-count 4 \
  --output benchmarks/results/response-179/cpu.json
```

Observed from `cpu.json`:

| quantity | result |
| --- | --- |
| sequential / blocked / recycled converged | true / true / true |
| maximum true residual | `6.17e-14` |
| maximum solution error vs explicit solve | `1.20e-14` |
| JVP/VJP dot identity relative error | `3.73e-16` |
| finite-rotation JVP error | `1.88e-9` |
| operator actions (4 RHS) | 80 sequential / 26 blocked / 84 recycled |
| solve time (4 RHS) | 57.7 s sequential / 18.7 s blocked / 60.4 s recycled |

The small water case does not amortize recycling: recycled solves used more
operator actions than the sequential baseline.  The sequential path remains
the default, and recycling is retained as an explicitly measured option.

## CPU gates

```text
python -m pytest tests/python -q
1200 passed, 189 skipped

ctest --test-dir build --output-on-failure
12/12 passed
```

The response-specific suite includes explicit tiny-matrix oracles, finite
orbital rotations, multiple RHS orderings, rank-deficient RHS blocks,
deliberate nonconvergence, singular/workspace failures, stale-subspace
rejection, near-degenerate-reference diagnostics, and CPKS unsupported-mode
fail-closed tests.

## CUDA evidence

The real-device tier is `tests/python/test_response_cuda.py`.  It is skipped
unless `VIBEQC_RESPONSE_CUDA_TEST=1` and `SLURM_JOB_ID` are present.  The
recorded run uses the streamed native CUDA DF J/K plan (`CudaDFJKBackend`) and
compares it with the independent explicit MO response matrix from `DFProvider`.

The same `water`, 4-RHS evidence run was executed through Slurm with
`--partition=main --gres=gpu:5090:1`:

| quantity | result |
| --- | --- |
| sequential / blocked / recycled converged | true / true / true |
| maximum true residual | `4.71e-14` |
| maximum solution error vs explicit solve | `6.66e-14` |
| JVP/VJP dot identity relative error | `0.0` |
| finite-rotation JVP error | `8.17e-10` |
| operator actions (4 RHS) | 80 sequential / 26 blocked / 84 recycled |
| solve time (4 RHS) | 2.126 s sequential / 0.690 s blocked / 2.232 s recycled |
| CUDA DF source generation / transfer | 1.786 s / 0.0414 s |
| backend device-resident / peak bytes | 594,440 B / 1,170,588 B |

This is a DF Hamiltonian record, not a direct four-center speedup comparison
against the CPU row.  The CPU row uses the conventional unscreened shell-tile
source; the CUDA row uses the declared DF metric and its matching reference.
See `gpu.json` for per-RHS iterations and all raw timings.

## CPKS boundary

`FixedDensityXCDerivativeKernel` is checked against central finite differences
of the fixed-density XC potential.  `CPKSResponseOperator` is wired to the
same GMRES/recycling layer and tested with a synthetic converged KS reference.
Exact exchange/RSH and nonzero tau derivatives fail closed.  A native
converged RKS/UKS endpoint remains owned by `#162`; this directory does not
claim that endpoint or a production CPKS method.
