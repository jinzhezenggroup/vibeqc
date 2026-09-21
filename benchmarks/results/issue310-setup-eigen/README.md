# #310 cold setup provider qualification

> **Checkout retention (2026-09-21):** `raw-evidence.zip` was moved out of the normal checkout. Exact bytes remain in Git revision `d8f64a93fe0dfebd889fd0ba1fadbd5ad7d840e5` and are checksum-bound by [the checkout-trim manifest](../retention-2026-09-21/migration.json). Restore locally with:
>
> ```bash
> python tools/restore_retained_evidence.py benchmarks/results/issue310-setup-eigen/raw-evidence.zip \
>   --manifest benchmarks/results/retention-2026-09-21/migration.json \
>   --output .artifacts/issue310-setup-eigen/raw-evidence.zip
> ```
> Restored archives belong under ignored `.artifacts/`; do not recommit them.

For the standard raw-evidence bundle, verify or unpack it against the retained
member manifest with:

```bash
python -m tools.unpack_evidence benchmarks/results/issue310-setup-eigen \
  --archive .artifacts/issue310-setup-eigen/raw-evidence.zip \
  --output .artifacts/issue310-setup-eigen-unpacked
```

CUDA DF RHF/UHF single/bucket setup borrows the ordinary FP64 provider already
used by finalization. Symmetric X, the exact strict `<1e-10` cutoff, core-guess
UHF mixing and lazy/cache semantics remain. Post-plan invalid setup frames are
isolated by original source index before one bounded survivor-plan rebuild.

## Separate setup substitution

The reference selection restores actual overlap/core reference solves; the
candidate uses ordinary Xsyevd. Both retain the same preparation work and device
finalization. Five interleaved pairs, RHF spherical def2-SVP, matching auxiliary
basis, 1 GiB DF allowance, RTX 5090. Only clean synchronized timings appear here.

| AO / batch | Endpoint | Workload | Reference (s) | Device (s) |
| --- | --- | --- | ---: | ---: |
| 96 / 1 | energy | cold-start | 0.376359 | 0.240964 |
| 96 / 1 | energy | changed-geometry | 0.262349 | 0.189424 |
| 96 / 1 | force | cold-start | 0.915639 | 0.749351 |
| 96 / 1 | force | changed-geometry | 0.780553 | 0.676098 |
| 96 / 4 | energy | cold-start | 1.145480 | 0.587889 |
| 96 / 4 | energy | changed-geometry | 0.558990 | 0.479662 |
| 96 / 4 | force | cold-start | 2.638683 | 2.035503 |
| 96 / 4 | force | changed-geometry | 2.067578 | 1.950280 |
| 192 / 1 | energy | cold-start | 3.316630 | 1.191293 |
| 192 / 1 | energy | changed-geometry | 2.151617 | 1.026122 |
| 192 / 1 | force | cold-start | 6.440933 | 4.271843 |
| 192 / 1 | force | changed-geometry | 5.266539 | 4.099336 |

All iteration/retry branches match. Cold items perform one overlap and one core
solve on the selected provider. Changed geometry replaces only the last item's
X and skips its warm core frame. Unchanged warm items perform zero setup solves
on both sides; their retained timings are a protocol control. Final physical
Fock solves remain one device call per RHF item in both selections. Separate
intrusive host traces gate these actual leaves by reason and cannot enter clean
timing. Energy/full-force replay gates pass at 1e-9 Eh and 1e-8 Eh/Bohr against
separately prepared cold endpoints. This is not external-engine parity.

26 CPU native and 179 initial protocol checks pass, followed by 84 focused
protocol checks after the count/tooling update. Five GPU native suites and
145 GPU Python checks pass, including actual cutoff boundaries, corrupted
cached overlap/neighbor isolation/recovery, RHF/UHF Cartesian/spherical,
batch 1/4, empty beta and complete forces. Memcheck reports zero errors and
zero leaked bytes. All real-device tests and probes use finite Slurm jobs.
The first pre-build protocol attempt's missing-library errors are retained
separately from scientific qualification and their resolution is recorded.

A separate fresh-calculation CUDA component probe retains actual setup/final
operations and transfers for 96/192 AOs. Each cold RHF endpoint executes three
ordinary eigensolves. Event intervals include host-induced idle time; graph
capture records are not executed kernel timings. Intrusive cold component
records do not provide warm speedup percentages.

The compiled scientific identity and exact binary hash are checked. Original
results/manifests, trace texts, validation logs and reproduction scripts are
hashed, compressed and restored byte for byte before publication.

```bash
python -m tools.unpack_evidence benchmarks/results/issue310-setup-eigen \
  --archive .artifacts/issue310-setup-eigen/raw-evidence.zip \
  --output /tmp/issue310-setup-evidence
```

Rebuild the manifest source with CUDA 12.9.1, Release, architecture 120 and AOT
shells disabled. The archived `reproduction.sh` records all nine #206 commands;
adjust paths and use fresh outputs, then execute through finite Slurm with
`--partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:30:00`.
The independent host-iterative Fock API, #311 final-state retention/correction,
remaining larger/constrained timing and #206/#308 acceptance stay open.
