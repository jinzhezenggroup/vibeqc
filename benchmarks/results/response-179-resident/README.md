# Exact-CUDA resident multi-RHS response qualification

This compact record qualifies bounded RHF response execution for #179 and its
#180 block HVP consumer. Both modes use conventional unscreened FP64 direct
CUDA J/K, identical converged fixture references, four identical RHS columns
(one dependent column), solver thresholds, and final solution publication with
`collect_basis=False`. No CPU-vs-DF ratio is used.

The original measured source is `cda824e2389ae2a44b4822029d56e0f790ba708d` (clean). The loaded native
binary SHA-256 is `bdb5ebdc583371c4b7e7575630d03597921443ea9af16b88f9b26077f84d47a1`. `evidence.json` records the
compiled native source identity, actual selected generic CUDA profile, device,
driver/toolkit, thread settings, all individual samples and memory/counter scopes.
The build uses Release, CUDA 12.9.1, sm_120 and AOT shells disabled.

## Numerical acceptance

All 54 response samples (H2/LiH/water, three strategies, two vector engines,
three repeats) passed independently. The matrix uses committed Libcint AO
integrals transformed into the fixture's MO frame, separately from native J/K.
Maximum relative residual: `4.271820e-14`;
maximum absolute solution error: `9.769963e-14`.
Gates are 1e-9 relative residual and 3e-9 absolute solution error for every
sample. All six complete H2 HVP endpoints pass a 1e-9 maximum-error gate against
the native dense Hessian assembly. That assembly shares the #179 solver, so
these six historical errors establish parity, not independent consumer
qualification. No timing sample is discarded by its accuracy. See the external
consumer gate below for independent qualification.

## Measured endpoints and work

Times below are medians of repeats 1 and 2 (repeat 0 is also retained). Bytes
and action counts are the first sample; every repeat's raw counts are in JSON.
Host/resident columns compare vector execution, while J/K runs on CUDA in both.

| Case | Strategy | Host solve s | Resident solve s | Actions H/R | H2D bytes H/R | D2H bytes H/R |
| --- | --- | ---: | ---: | --- | --- | --- |
| h2 | sequential | 0.015050 | 0.014772 | 8 / 8 | 256 / 32 | 512 / 256 |
| h2 | blocked | 0.016508 | 0.016040 | 9 / 9 | 288 / 32 | 576 / 332 |
| h2 | recycled | 0.021003 | 0.020478 | 12 / 11 | 384 / 32 | 768 / 420 |
| lih | sequential | 1.135620 | 1.134343 | 48 / 48 | 13824 / 256 | 27648 / 2400 |
| lih | blocked | 0.568179 | 0.569663 | 24 / 24 | 6912 / 256 | 13824 / 1808 |
| lih | recycled | 1.210033 | 1.216819 | 52 / 51 | 14976 / 256 | 29952 / 5260 |
| water | sequential | 2.319602 | 2.316502 | 80 / 80 | 31360 / 320 | 62720 / 5152 |
| water | blocked | 0.870444 | 0.870021 | 30 / 30 | 11760 / 320 | 23520 / 2448 |
| water | recycled | 2.407799 | 2.416411 | 84 / 83 | 32928 / 320 | 65856 / 8172 |

Host payloads/synchronizations are derived from observed successful restricted
J/K calls: one FP64 AO density upload, two AO matrix downloads and one stream
fence per action, as implemented in `src/scf/cuda/direct_jk.cpp`. Resident
counts are raw native diagnostic deltas, including scalar reductions and
4-byte action status. They exclude provider setup/teardown; resident setup
counters are recorded separately. They are API-level accounting, not profiler
measurements of total PCIe traffic or hidden library synchronizations.

The host recycled path checks its explicit zero initial guess on the first RHS;
the resident path recognizes an empty bound space, so its recorded action count
can be one lower. This work difference is exposed, not assumed to be speedup.
Reduced long-vector movement does not establish lower complete endpoint time.
The samples do not support a blanket resident or recycling speedup claim.

The complete-solve timer includes RHS validation, internal rank preparation,
all iteration, final solution publication, automatic recycle teardown and
assembly of the returned solution matrix. The JSON separately retains plan
setup, final owner teardown, and their sum with the first solve. The cold field
is that declared plan-level sum, not a process-start benchmark. Oracle/fixture
loading and RHS creation are outside the timed response endpoint. Warm repeats
reuse the same owner but begin with an empty automatic recycle space; reuse
occurs within each multi-RHS solve. Persistent cross-call recycling is covered
by lifecycle/numerical tests, not inferred from these warm timings.

Operator time measures engine application only. Orthogonalization includes
basis/projection/range factorization; recycling covers retained projection and
replacement. These are disjoint partial components, not a full decomposition:
residual vector arithmetic, small least squares, validation and publication
remain in complete wall time. Small SVD/least-squares and convergence are host
controlled. Native retained J/K/response allocations and the separate
conservative logical solver reservation are recorded; CUDA context/library
memory and provider preparation transients are outside this numeric scope.

## Complete consumer

The native H2 state and native dense Hessian assembly are prepared before timing.
Each endpoint builds its own CUDA response provider, prepares three directional
RHS, solves, reconstructs the response, evaluates the declared CPU first/second
integral consumers and publishes complete HVPs. These are mixed host/device
endpoints. A single endpoint sample per combination is numerical/composition
and cost evidence, not a robust performance ranking.

| Strategy | Host HVP s | Resident HVP s | Native assembly parity error |
| --- | ---: | ---: | ---: |
| sequential | 6.391809 | 4.971216 | 3.686e-16 |
| blocked | 4.962943 | 4.972025 | 3.686e-16 |
| recycled | 4.999662 | 4.994462 | 3.686e-16 |

All requested budgets and modeled phase peaks are retained in the consumer
records. Resident consumer counters include resident setup, unlike response
sample deltas. Independent tests additionally cover capacity rejection before
work, changed references, failed replacement and exceptional lease cleanup.

## Independent consumer gate

The current runner additionally computes PySCF's analytic RHF Hessian using the
same exact shell primitives, Cartesian geometry in Bohr, charge and
multiplicity. PySCF independently reconverges SCF and evaluates all integral
and CPHF derivatives; neither native dense assembly nor the shared response
solver supplies its result. Every host/resident and sequential/blocked/recycled
complete HVP must meet a 1e-9 maximum absolute error gate. The original native
assembly check remains a separate 1e-9 parity gate.

`test_hessian_block_resident_cuda.py` checks all three resident strategies
against both references and forbids the native Hessian/shared solver while
constructing the external oracle. The original `evidence.json` is retained
unchanged: its consumer `maximum_error` and historical scope text refer to
native parity. New schema v2 records explicitly identify the external oracle
and its version, retain directions and actual/reference HVPs, and report native
parity in `native_dense_maximum_error`.

The supplemental `consumer-oracle-evidence.json` records clean source
`b326568699acbeea9153ad62c91e845f49af4518` and PySCF 2.14.0. The matching
native binary SHA-256 is
`b34f4435647b90e9ffc3adb6ef7e2f98eeb95ba6d6ff1a481bb2da6eaaf79757`.
Slurm job 1136 passed all six complete HVP endpoints, with maximum independent
error `7.993606e-15` and native parity error `3.686287e-16`. It also passed
103/103 response/consumer/Krylov tests without skips (613.83 s), including
native CUDA RKS/UKS and UHF exact/DF integration, plus 78/78 independent CUDA
point directions. The fresh Release build includes merged #672 and current
master; native sources/bindings are identical to the build's `3be8635c` snapshot.
Production response solver/consumer code remains unchanged from the original
measured `cda824e2` commit. This supplemental numerical campaign does not replace
the original matched response timing or establish a new performance ranking.

| Strategy | Host HVP s | Resident HVP s | Independent PySCF error |
| --- | ---: | ---: | ---: |
| sequential | 6.537851 | 4.998863 | 7.993606e-15 |
| blocked | 4.999966 | 4.964074 | 7.993606e-15 |
| recycled | 4.984317 | 4.979464 | 7.993606e-15 |

## Reproduction and limits

Build the native library in Release with CUDA enabled, sm_120 and
`VIBEQC_ENABLE_AOT_SHELLS=OFF`. Use one coherent CUDA runtime installation and
preserve the Slurm-assigned device visibility. From the repository root:

```bash
export PYTHONPATH=python:.
export VIBEQC_LIBRARY="$PWD/build-cuda/libvibeqc.so"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:15:00 python tools/response_resident_benchmark.py \
  --repeats 3 --consumer --output .artifacts/benchmarks/resident/evidence.json
```

PySCF is required only for this optional qualification command. To repeat the
six consumer gates without rerunning the unchanged response campaign, use
`--consumer-only` in place of `--repeats 3 --consumer`.

Acceptance is limited to the bounded tools domain. No default/auto-selection
policy changes. Larger systems, changed-geometry performance and constrained
budget performance are required before promotion; they are not established by
this three-fixture campaign. The old `response-179` CPU/DF records remain
historical and are not a speedup baseline for this exact-Hamiltonian comparison.
