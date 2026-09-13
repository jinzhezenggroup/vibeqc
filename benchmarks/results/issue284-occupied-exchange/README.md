# Occupied-factor CUDA RI-K (#284)

Dense remains the default. `VIBEQC_DF_EXCHANGE=occupied` enables generation-linked occupied factors in the same RI-K model. J and the complete analytic-force model retain their existing consumers. The fixed-K and complete endpoint measurements below cover distinct costs; a faster isolated K does not establish a full-SCF speedup.

Primary runtime: `9cb50386a07d38bfee655775a8897ddcb862ac31`; library SHA-256 `4f1c18e5dbd8d620e16efb0e33f4a9097f716caed0b78378548570edc82ba1c7`. All primary tables use the corrected frozen runtime. Earlier measurements, the resource-review correction, the detected dense regression and diagnostic failures are retained separately in the archive.

Timed molecules are RHF water clusters with spherical def2-SVP, Naux = NAO, and occupied ranks 20/40/80. UHF has dedicated correctness and lifecycle coverage; these tables do not establish large-UHF, TZ/QZ or larger-system crossover behavior.

## Fixed-density K

Five dense/occupied pairs per domain, including density/factor upload and K readback. Offline orbital construction is outside the call. Setup, factor rank, explicit transfer sizes, full arrays and component traces are retained in `summary.json` and the raw archive.

| Domain | AO / aux / rank | Dense (s) | Occupied (s) | Dense / occupied |
|---|---:|---:|---:|---:|
| 96ao-q0-r0 (resident) | 96 / 96 / 20 | 0.000426356 | 0.000188621 | 2.260x |
| 192ao-q0-r0 (resident) | 192 / 192 / 40 | 0.003555087 | 0.001282115 | 2.773x |
| 384ao-q0-r0 (resident) | 384 / 384 / 80 | 0.052842846 | 0.016365438 | 3.229x |
| 96ao-q24-r96 (panels) | 96 / 96 / 20 | 0.299959550 | 0.299709040 | 1.001x |
| 192ao-q48-r192 (panels) | 192 / 192 / 40 | 2.877421652 | 2.874663523 | 1.001x |
| 96ao-q4-r16 (tight) | 96 / 96 / 20 | 47.191274453 | 47.103099317 | 1.002x |

Maximum independently checked K error: 1.24345e-14. Resident and constrained schedules introduce no additional three-center tensor solely for occupied K. Dense/occupied generated-value counts agree in every controlled component trace. Tight row traversal still pays for bounded regeneration.

## Complete warm endpoints on one frozen density

Five dense/occupied pairs per endpoint, alternating their order. Both policies replay one immutable post-cold dense snapshot. Policy changes rebuild the reserved native value/SCF plan; untimed priming and its cost are retained. Both policy-specific plan ledgers are recorded and independently checked for the exact occupied reservation increment.

| AO | Batch | Endpoint | Dense (s) | Occupied (s) | Dense / occupied |
|---:|---:|---|---:|---:|---:|
| 96 | 1 | energy | 0.232573 | 0.232636 | 1.000x |
| 96 | 1 | energy + force | 0.782026 | 0.786107 | 0.995x |
| 96 | 4 | energy | 0.909687 | 0.907801 | 1.002x |
| 96 | 4 | energy + force | 2.529137 | 2.519918 | 1.004x |
| 192 | 1 | energy | 3.236569 | 3.230123 | 1.002x |
| 192 | 1 | energy + force | 6.351016 | 6.341866 | 1.001x |
| 192 | 4 | energy | 12.984557 | 12.962095 | 1.002x |
| 192 | 4 | energy + force | 24.961760 | 24.938037 | 1.001x |
| 384 | 1 | energy | 39.956980 | 39.801454 | 1.004x |
| 384 | 1 | energy + force | 65.433242 | 65.352637 | 1.001x |

All 50 endpoint pairs pass the energy/force gates. Matching dense/occupied iteration branches: 10/10 domains. Executed 96/192/384-AO SCF provenance traces verify the dense seed and subsequent occupied iterations; capture records alone are not treated as executed timings.

## #206 cross-engine matrix

All 16 cases and 80 raw VibeQC/GPU4PySCF warm pairs pass independent numerical checks, covering both policies, energy/force, 96/192 AOs and batch 1/4. Maximum energy error is 2.80806e-11 Eh, and maximum force error is 9.00003e-11 Eh/bohr. Cross-engine warm densities and iteration branches can differ, so this matrix does not establish an iteration-matched speed comparison.

Full per-case timing, iteration arrays and energy/force errors are in `summary.json`. The fastest measured RI-K execution should be used in #246 crossover studies for its matching domain; complete endpoint results determine the default policy.

## Dense compatibility and resource boundaries

Only occupied plans reserve two extra AO matrices and generation controls. Dense native minimum/residency thresholds and common-ledger admission remain unchanged. Native SCF rejects unreserved factor allocation. Ordinary prepared batches replan on a policy change; global `ResourceBudget` identities require a newly prepared batch. Fixed-density factors borrow existing tile scratch independently of the SCF reservation.

The 8-AO dense minimum remains 25,170 bytes; full host/generated thresholds remain 45,394/41,298 bytes. Occupied adds 1,036 bytes in this fixture. Native and common boundary regressions, RHF/UHF policy transitions and rank-zero beta pass.

The initial implementation regressed the 192-AO dense energy endpoint by about 33%. Native CPU sampling located the unchanged reference eigensolver, and normalized disassembly found identical 631-instruction sequences shifted by 16 bytes. A 32-byte alignment preserves its arithmetic and restores old/corrected medians to 3.234540/3.229307 s under one Slurm allocation, ten warm samples per library, with identical iteration branches. This correction is included in all primary tables above.

## Validation and retention

`summary.json` records the exact validation counts and limitations. CPU/CUDA Release builds, host/native/resource/SCF tests, occupied reruns, memcheck (zero errors) and hooks pass. One existing CUDA MP2 status tier and six host-only optional GPU cases are explicitly skipped.

Value-plan diagnostics and declared DF budgets are capacity accounting, not measured global GPU peaks; opaque library retention is separate. Trace timings are diagnostic, and graph-capture records describe construction. The original input-preparation log predates a text-only symbol-to-integer-Z correction; input metadata/manifests hash the actual GPU inputs.

The archive includes all raw numerical arrays, repeats, setup/priming costs, commands, source/library identities, reconstruction patches, CMake cache, validation logs, earlier failures and CPU sampling/disassembly. Binaries are omitted. Every member is hash-verified and restored byte-for-byte.

```bash
python tools/unpack_evidence.py benchmarks/results/issue284-occupied-exchange --output /tmp/issue284-evidence
python /tmp/issue284-evidence/audit-occupied-evidence.py --directory /tmp/issue284-evidence/df-occupied-corrected
python /tmp/issue284-evidence/audit-occupied-baseline.py /tmp/issue284-evidence/df-occupied-corrected/df-occupied-baseline-check
```

A representative real-GPU rerun, using the recorded CUDA/GPU4PySCF environment and a Release library:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:10:00 \
  env PYTHONPATH="$PWD/python:$PWD" VIBEQC_LIBRARY="$PWD/build/cuda/libvibeqc.so" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python benchmarks/compare_df_exchange.py --case water-octamer-s4-def2-svp-spherical \
  --batch 1 --repeats 5 --memory-budget-bytes 1073741824 --energy-only --output /tmp/df284.json
```

Preserve Slurm-assigned device visibility and avoid concurrent local compilation during measurements. See [equations and lifecycle](../../../docs/df_occupied_cuda.md).
