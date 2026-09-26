> **Checkout retention note (2026-09-25):** bulky raw campaign members were moved out of the normal checkout and remain byte-for-byte recoverable from existing Git history. See [the bulk retention manifest](../retention-2026-09-25/bulk.manifest.json) and use `tools/restore_retained_evidence.py` before following links or reproduction steps that require archived members.

# Practical-auxiliary force error: response arithmetic diagnosis

The retained final native densities reproduce accurate forces when used in an
independent CPU assembly. For OH, CUDA derivative consumers also agree with
libcint when given identical explicit weights. A separate call to the unchanged
production response implementation then reproduces a force error using exact
independent integrals. Rearranging the full-rank response arithmetic removes
that error in a CPU diagnostic with the **same CUDA metric eigensystem**.

This identifies response arithmetic as a concrete numerical target. It changes
no production algorithm, gate, clean sample or performance admission and keeps
#206 open. The original practical auxiliary campaign remains failed; see
[`../issue206-practical-auxiliary/`](../issue206-practical-auxiliary/README.md).

## Final-state checks

Slurm 9906 captures one final native output density per case. Cold solve,
freeze post-cold updates, prime, then enable updates immediately before the
target replay: the exported density is the output, not the retained seed.
Both captures use the exact cc-pVDZ/cc-pVDZ-JKFIT inputs from #430 and the
unchanged final #428 library, SHA-256
`ea39fae62486240a98021726d970ec80c965e8f4a687056e59636a7179faf5be`.

Independent electron traces, idempotency, physical Fock energies, commutators
and canonical density reconstruction check the public AO convention. Maximum
commutators are 1.29e-11 (OH) and 7.37e-12 (water); physical energy discrepancies
are 8.67e-13 and 4.32e-12. Consistent Pulay weights are `Ds @ Fs @ Ds` for each
UHF spin and `D @ F @ D / 2` for RHF. PySCF's automatic orbital tagging is
disabled so its exchange response derives factors from the fixed native D.

| Model | Native force vs tight CPU oracle | CPU force at native D vs oracle |
| --- | ---: | ---: |
| OH UHF 19/93 | 1.23724e-10 | 1.90559e-12 |
| Water RHF 96/464 | 1.75216e-9 | 1.62004e-12 |

The initial diagnostic's RHF CPU adapter incorrectly accepted only keyword
arguments and failed after both native densities had been saved. Its script,
traceback, report and arrays remain intact. The corrected CPU-only script
reuses those exact arrays; it performs **no native replay**. It also checks OH
again without changing its retained result.

## Derivatives at fixed explicit weights

Slurm 9907 uses only the saved OH density. Independent libcint A/M, full moving
orbital/auxiliary derivatives and a full-rank Cholesky solve define the UHF
response weights. Their complete CPU force agrees with the independent PySCF
fixed-density assembly. The public AO transformations and expanded general
contractions are checked against PySCF A/M and one-electron values.

The same explicit weights are passed to the standalone CUDA derivative APIs.
Both supported schedules are retained; neither is selected as a repair.

| Fixed-weight contribution | Maximum CUDA/libcint difference across both schedules |
| --- | ---: |
| One electron, including Pulay | 2.221e-14 |
| Three center | 1.066e-14 |
| Coulomb metric | 1.310e-14 |
| Assembled complete force | 3.376e-14 |

These results separate the tested derivative consumers from response-weight
generation. They do not by themselves establish derivative accuracy for water
or for every representation and contraction.

## Response arithmetic with identical metric factors

Slurm 9908 calls the **existing function in the frozen library** through a
small separately compiled diagnostic bridge. It supplies independent OH A/M,
the saved native density, and the production CUDA planner's own eigensystem
and inverse square root. The bridge performs no SCF, uses the dense panel
response with normal BLAS metric dots and scalar density products, and copies
all A/M adjoints and factors for independent contraction. This is a response
isolation experiment, not a byte-for-byte replay of the endpoint's native
integral values. Its error need not equal the original endpoint error.

For full rank, define `B_P = sum_Q A_Q (M^-1)_QP`, `c_P = Dtotal:B_P` and
`U_s,P = Ds B_P Ds`. The stable diagnostic constructs:

```text
bar_A_P  = Dtotal c_P - sum_s U_s,P
bar_M_PQ = -c_P c_Q / 2 + sum_s (U_s,P:B_Q) / 2
```

The existing route instead forms the adjoint of the inverse from raw A
quadratic products, rotates that matrix into the metric eigenbasis, multiplies
by `-1/(lambda_i lambda_j)` and rotates back. The metric condition number is
1.696e6. Small errors in the raw quadratic products and rotations can therefore
be amplified in weak metric directions.

| Response diagnostic | Three-center force difference | Metric force difference | Combined difference vs Cholesky weights |
| --- | ---: | ---: | ---: |
| Unchanged CUDA response | 3.925e-11 | 2.458e-10 | **2.850e-10** |
| CPU applies inverse to factors first, same CUDA eigensystem | 3.410e-11 | 3.411e-11 | **6.545e-16** |
| CPU raw quadratic products and reverse map in FP64 | 3.410e-11 | 9.444e-11 | 6.034e-11 |
| Rounded FP64 products, extended reverse map only | 3.410e-11 | 2.966e-11 | 6.376e-11 |
| Products and reverse map both extended | 3.410e-11 | 3.329e-11 | 8.163e-13 |

The CPU spectral rows keep the stable three-center weights fixed and vary only
the metric adjoint. All use the exact saved FP64 CUDA eigenvectors/values.
Extended arithmetic is the host's 63-bit-mantissa `longdouble`; it is a
diagnostic, not a proposed precision or backend contract. All 93 directions
remain present. The opposing component changes in the stable row illustrate
why only the complete force establishes its consistency.

## Reproduction and limits

`manifest.json` records 39 losslessly retained logical reports, arrays, traces
and original scripts, including the failed adapter. The 2.1-MiB response NPZ is
expanded into exact NPY member payloads, each hash checked, to respect the
repository's per-file size policy. Eight semantic member names contain only
three unique byte sequences; five duplicate stored copies are therefore mapped
to canonical retained blobs. Every original member name, decoded SHA-256 and
size remains a separate manifest record, so reconstruction is lossless while
Git stores each identical payload once. The original ZIP container and compiled
bridge remain in local artifacts; their SHA-256 identities are recorded. No
numerical array content is rounded, filtered or dropped. Original scripts are
gzipped to preserve the exact bytes executed, including the failed version.

The CPU-only audit checks every retained hash and recomputes the displayed
force errors directly from arrays:

```bash
python benchmarks/results/issue206-response-diagnosis/analyze.py
```

It requires NumPy on a host with the recorded 63-bit `longdouble` mantissa and
refuses optimized Python. Restore the original paths from the manifest, plus
the explicit bases and campaign manifest from #430, to execute a diagnostic
again. The scripts refuse existing result directories. All GPU scripts must
run through finite Slurm on `main` with `--gres=gpu:5090:1`; the original jobs
used `--nodes=1 --ntasks=1 --time=00:10:00`, one CPU math thread, `PYTHONPATH=python:.`
and the CUDA 12.9.1 library directory. They preserve Slurm device visibility.

The bridge was compiled with GCC 11.4.0, `-std=c++20 -O2 -shared -fPIC`, includes
`src`, `include` and the CUDA 12.9.1 include directory, and links `libvibeqc`,
`cudart` and `cublas` from the recorded frozen library and CUDA directories.
The frozen production library itself was never rebuilt or modified.

A production correction still needs bounded workspace/dataflow, independent
water verification, full original force gates, and rank-deficient/rank-crossing
coverage. The raw spectral response includes retained/discarded subspace
motion: **the full-rank formula above must not replace that response for
rank-deficient metrics**. No clean timing or performance conclusion is present
in this diagnostic bundle.
