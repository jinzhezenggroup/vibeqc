# Practical auxiliary bases: both strict force gates fail

The first cc-pVDZ/cc-pVDZ-JKFIT comparison completes on CUDA with unequal
orbital/auxiliary dimensions, but both strict force gates fail. An independent
tighter CPU DF solve localizes the dominant error to the native forces. This
keeps #206 open and supplies no qualified performance win.

## Exact model and protocol

`compare_gpu4pyscf_batch.py` now accepts `--orbital-basis-file` and
`--auxiliary-basis-file`. Both engines consume the same canonical local
snapshots, including every shell and general-contraction column. Representation
mismatches, ECP/alchemical metadata and unsupported CUDA shells are rejected
before GPU packages are imported. Existing fixture defaults and numerical gates
are unchanged. An auxiliary override requires CUDA DF.

The geometry/spin selectors retain their original def2-SVP case names, while
the explicit orbital input in this experiment is **cc-pVDZ**. The result's
`basis_overrides` contains the complete canonical snapshots and provenance;
its description labels the override. The auxiliary input is **cc-pVDZ-JKFIT**.
Data come from unmodified PySCF 2.14.0 packaged records, whose package metadata
declares Apache-2.0. The canonical input files are retained in `identity/`.

The selected cases are water tetramer RHF (96 orbital / 464 auxiliary AOs) and
OH UHF (19 / 93), spherical throughout. Their auxiliary shells reach f. The
common O/N def2-SVP-JKFIT and def2-universal-JFIT records instead contain g
shells, beyond the present CUDA limit; no shells were truncated to admit them.

Slurm 9904 ran seven interleaved, synchronized warm pairs per case under a finite
30-minute RTX 5090 allocation, preserving scheduler visibility. Measured
benchmark source: `67b35f3a0a236de39ebbdb2876908a3c2da78c6c`. The native library
is the unchanged qualified #428 production binary, SHA-256
`ea39fae62486240a98021726d970ec80c965e8f4a687056e59636a7179faf5be`, native identity
`06fa1f9d7108f86edfaa4ba9285df58c20a1d191b3c20785a2500cad1798730d`.
Its build metadata and generated-source hashes are retained in `identity/`;
the complete build qualification is in
[`../issue418-cooperative-rys/`](../issue418-cooperative-rys/README.md).

Separate stock audits verify cuTENSOR, explicit physical basis data and full
factor ranks. Native and stock ranks agree at 464 and 93; the CPU metric
condition numbers are about 5.435e6 and 1.696e6. Stock keeps its normal Cholesky
policy. Versions are GPU4PySCF 1.8.1, PySCF 2.14.0, CuPy 14.2.0, cuTENSOR 2.3.1
and NumPy 2.5.3. No active local build or profiler overlaps timing.

Native controls remain dense value storage, automatic shell/exchange/projection
policies, device final validation and disabled force screening. The equal-size
384/768 cooperative automatic admission boundary does not include these cases.
Each engine restores its own fixed post-cold density outside timing, after
equivalent untimed priming; cross-engine density byte identity is not asserted.

Predeclared energy/density convergence is 1e-12 / 1e-10; stock gradient tolerance
is 1e-10 and screening is 1e-14. **Both energy and full-force maximum errors must
be <=3e-11 in every pair.** These new mathematical models cannot replace the
earlier failed equal-basis cells or waive any original acceptance.

The ordinary-latency criterion was declared before launch: both relative MADs
<=3%, median native time reduction >=2%, and a paired bootstrap 95% lower bound
above 2%, using 10,000 resamples and NumPy `default_rng(206)`. Numeric failure
forbids performance admission regardless of timing. The audit retains this
calculation and every convergence branch; no fixed-work claim follows.

## Retained first campaign

Times below are ordinary complete energy-and-force warm medians in seconds.
All 14 pairs are retained, with no failed point rerun for a passing outcome.
The controller exits 1; both comparator cells exit 2 after writing results.

| Model | Native | Stock | Maximum energy error (Eh) | Maximum force error (Eh/Bohr) | Strict gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Water RHF 96/464 | 0.605891 | 0.271146 | 1.251e-12 | **1.757e-9** | fail |
| OH UHF 19/93 | 0.017855 | 0.202947 | 1.762e-12 | **1.304e-10** | fail |

Native uses two iterations and stock one in every pair, for both cases. No
common iteration branch exists. The OH timing is unqualified despite its lower
ordinary latency; the water endpoint is slower than stock.

## Independent force diagnosis

A separate CPU PySCF DF calculation uses the exact same explicit basis inputs,
full auxiliary response and tighter convergence (energy 1e-13, gradient 1e-12).
It converges with final orbital-gradient norms 3.445e-13 (water) and 6.526e-13
(OH). This CPU calculation is diagnostic; it does not replace paired acceptance
or any measured timing.

| Model | Native max force error vs CPU | Stock max force error vs CPU |
| --- | ---: | ---: |
| Water RHF 96/464 | 1.752e-9 | 1.741e-11 |
| OH UHF 19/93 | 1.237e-10 | 6.666e-12 |

The larger discrepancy is on the native side. Matching rank and energy do not
establish force accuracy. The current evidence does not yet distinguish native
state consistency, response-weight conditioning, and integral/derivative error;
that separation is the next numerical diagnostic, before any promotion claim.

Slurm 9905 then runs a separate, untimed native diagnostic from one frozen
post-cold density per case. It tests the existing host-final-validation,
shell-contraction, BLAS-response and occupied-response controls, then returns
to the automatic control. None removes the discrepancy:

| Selected diagnostic control | Water max force error vs CPU | OH max force error vs CPU |
| --- | ---: | ---: |
| auto | 1.752e-9 | 1.237e-10 |
| host final validation | 1.752e-9 | 1.237e-10 |
| shell contraction | 1.752e-9 | 1.237e-10 |
| BLAS response | 1.668e-9 | 2.372e-10 |
| shell + BLAS | 1.668e-9 | 2.372e-10 |
| occupied response | 2.185e-9 | 3.607e-10 |
| return to auto | 1.752e-9 | 1.237e-10 |

All 14 diagnostic force arrays and their traces are retained separately. They
do not replace clean samples or support an endpoint timing claim. The final
automatic replay reproduces the original error. Response-algebra changes alter
the discrepancy, but this alone does not identify its mathematical source.

## Verification and reproduction

All 44 benchmark tests pass with the frozen native library under finite Slurm.
An initial ECP test fixture error and the initial missing-library test failure
are documented in `validation/`; the missing-library diagnostic is retained.
The initial host suite had 43 passing tests and 1 failure because the fresh
checkout had no native library; all 44 passed with the frozen library. The
hash-pinned `initial-validation-observations.json` preserves the original
observation text, including its spacing, without changing any reported result.
No pinned GPU Python environment was changed to install test dependencies.

`manifest.json` maps 41 lossless retained records to their original hashes.
This includes all 28 clean timed calls, all paired arrays, progress records,
provider/rank audits, exact controls, CPU reference forces and launch scripts.
The CPU-only audit recomputes every paired error, rank/identity check, median,
branch, predeclared statistic and CPU-reference error:

```bash
python benchmarks/results/issue206-practical-auxiliary/reproduction/analyze.py \
  benchmarks/results/issue206-practical-auxiliary/raw /tmp/auxiliary-summary.json
cmp benchmarks/results/issue206-practical-auxiliary/summary.json /tmp/auxiliary-summary.json
```

NumPy is required for the predeclared bootstrap. Optimized Python is rejected
because it would strip the audit assertions. Gzip records restore exact bytes.
Restore the controller, qualifier and basis files to their original paths in
the measured checkout, use the matching frozen library, then reproduce with:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:30:00 /tmp/vibeqc-308-gpu4pyscf-env/bin/python \
  .artifacts/issue206-auxiliary/run-practical-matrix.py
```

The driver refuses an existing result directory. Future experiments must retain
this failed campaign. Repeated cold/changed geometry, full resource/transfer
accounting and the prior original DF/direct failures remain open requirements.
