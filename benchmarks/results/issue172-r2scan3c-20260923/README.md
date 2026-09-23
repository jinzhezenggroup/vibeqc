# Issue #172: bounded r2SCAN-3c H100 acceptance snapshot

This snapshot qualifies all-electron CUDA s/p/d forces for the listed small
systems. It does not establish force support for every H-Ar molecule, ECP d
shells, or an end-to-end GPU-resident correction stack. Linear OH UKS remains
branch-sensitive on changed-versus-fresh replay, and its failed result is
retained here. Issue #172 was still open when this snapshot was recorded.

## Exact execution identity

- Production source: LF Git tree `0e6350acbf193cde487c6772c2f783baf9cd86f2`,
  archived at SHA-256 `535e5ce826a6f32caf099aa3c77cc145f8106cc9d799383c91a6f0ad30dd963e`.
  It is based on `e128dd927f32e3f7f193e78902d5ba191c585f83`; later test
  and benchmark files were supplied as separately hashed overrides, without
  changing the production Python/C++ source or native binary.
- Native `libvibeqc.so.0.1.0` SHA-256:
  `9d8931fca4004b4945280b405e96733f95b0da211bd6a74ddff43511e2bfefca`.
  Packaged r2SCAN RKS/UKS stationary AOT library hashes are
  `1a61502e8a6ab85cb2a7cc33d79d597e760a7daac31fb64e1fa34818db1d5f6b` and
  `b5853e7231cfe06a2f6ee5f7ee21c964c3857f2528957f367546cad6b55b4a3b`.
- H100 80 GB, `sm_90`, CUDA compiler 12.9.86, image
  `pytorch:25.06-py3:25.06`, one GPU / 20 CPU / 200 GiB. PySCF 2.14.0 and
  DFT-D4 4.2.0 supplied independent references. The canonical gCP parameter
  JSON hash was `c1cede24b2527a2b688981b651d91da7206d8da1a223c3f217a86800c47eded2`.
  Default production grid and `E=1e-12 / D=1e-10` were used; the correction
  backend reported `cuda+cpu`, including a CPU gCP component.
- The final test override files were SHA-256
  `0ea84e3d217046f5ac14566674e04f8dd4caaf2750e9d069c5d3ab593ff337cf`
  (`test_dft_complete_cpu.py`) and
  `0aeac3282222c6a891c8ccbd78a2ef3d7a8925999eb9fca19faf727cfe2a5262`
  (`test_r2scan3c_execution.py`). The versioned
  [bounded endpoint runner](../../issue172_r2scan3c_endpoint.py) was
  `ae9ccdf149c829a3e9056ac6f4716a68d388ca959b92a9593e0d96ec1ed3d3ce`.

## Numerical and resource gates

Job `issue-0172-final-h100-0e6350ac-20260923` finished **SUCCEEDED / exit 0**:
60 device-free tests passed, then ten opt-in real-device cases passed in
770.43 seconds. Independent totals used PySCF analytic r2SCAN gradients on
the identical explicit grid plus DFT-D4 and the separate gCP reference.
The gates stayed at `2e-9 Eh` total energy and `1e-7 Eh/Bohr` maximum force.

| System | AOs | PySCF initial guess | Energy error / Eh | Max force error / Eh/Bohr |
| --- | ---: | --- | ---: | ---: |
| H2O RKS | 34 | default | 7.11e-14 | 1.21e-11 |
| H2 dimer RKS | 20 | default | 1.07e-14 | 1.57e-11 |
| OH UKS | 29 | native final density | 7.11e-14 | 5.60e-14 |
| H2O+ UKS | 34 | default | 1.42e-13 | 1.21e-11 |

The OH PySCF gradient is independently evaluated after convergence, but its
initial density selects the native stationary basin. PySCF default, atomic,
and second-order starts did not converge under the same strict controls.
H2O+ supplies an open-shell d-shell reference from PySCF's own default guess.
Water's two finite-difference steps differed by `7.01e-9 Eh/Bohr`; the
smallest-step analytic error was `2.01e-9 Eh/Bohr`. HF/water ragged changed
geometry replay matched fresh energies and forces exactly in the tested case;
a malformed H2 batch item did not poison a charged H3+ neighbor.

The water force consumed **12,134,769 ordered primitive records** through
2,568,000 task descriptors and 628 batches, within its 16,000,000-record cap.
The additional-device peak bound was 110,044,688 bytes, below the
536,870,912-byte limit. Its direct stationary diagnostic took 39.04 seconds
on this run; that is a component timing, not a complete E+F endpoint.

## Complete energy-plus-force endpoint

Job `issue-0172-endpoint-bounded-h100-0e6350ac-20260923` finished
**SUCCEEDED / exit 0**. Each entry below is one complete public calculation
with r2SCAN, D4 and gCP; warm is the median of three retained raw samples.
Cold includes first-execution setup, while batch preparation is recorded
separately. No speedup or scaling claim is inferred from this one-run matrix.

| System | Cold / s | Warm median / s | Changed / s | Fresh / s | Changed/fresh max force error / Eh/Bohr |
| --- | ---: | ---: | ---: | ---: | ---: |
| H2 RKS | 11.477 | 0.602 | 0.979 | 1.310 | 1.67e-12 |
| H3+ RKS | 2.373 | 1.315 | 1.974 | 2.410 | 4.52e-12 |
| H2 dimer RKS | 4.073 | 2.777 | 3.713 | 4.104 | 1.02e-11 |
| H2O RKS | 120.322 | 43.110 | 49.445 | 52.886 | 3.82e-11 |
| H2O+ UKS | 55.193 | 44.241 | 51.881 | 55.942 | 6.72e-11 |

All five passed `2e-9 Eh / 1e-7 Eh/Bohr` changed-versus-fresh gates and
unchanged-geometry replay gates. The d-shell endpoint cost is material even
for water; the measured 12.1 million primitive records and 205 MB of source
host-to-device traffic explain why a bounded memory footprint is not a low-work
claim. These results do not compare against another implementation or justify
performance promotion.

## Retained failure and boundary

The first endpoint job, `issue-0172-endpoint-h100-0e6350ac-20260923`,
failed its OH UKS changed/fresh force gate: energy differed by `2.61e-10 Eh`
and maximum force by `4.57e-7 Eh/Bohr`. Its cold and three warm results agreed
to about `5e-14`. A second pinned run reproduced the same values. The
`issue-0172-oh-density-h100-0e6350ac-20260923` diagnostic found alpha/beta
density-matrix maximum differences of `0.0225 / 0.2148` at the same displaced
geometry; each solution had a physical residual near `4e-12`, while fresh
replay changed its density by only about `1e-11`. They are distinct stationary
spin-density branches. Their ordering and general OH branch policy are not
qualified here. This failed case was excluded **explicitly** from the five-row
passing matrix, not relabeled as a pass.

`endpoint-bounded-h100.json` preserves every timing, energy, force, iteration,
residual and correction backend; `endpoint-h100-matrix.json` and
`endpoint-ohdiag-h100.json` preserve both failed OH runs.
`oh-density-h100.json` holds the branch comparison. `evidence-final-h100/`
contains the ten numerical/resource receipts. The ignored raw bundle at
`/inspire/qb-ilm/project/chemicalreaction/czxs25220150/experiments/vibeqc/issue-0172-closure/0e6350ac/issue172-h100-evidence-20260923.tar.gz`
retains those JSON files plus test and runner logs, including failed oracle
attempts; its SHA-256 is
`6f2edd2e9e74c7f8882575a10ee4cf50bee18d84c2fc2247c1a668efda431843`.
