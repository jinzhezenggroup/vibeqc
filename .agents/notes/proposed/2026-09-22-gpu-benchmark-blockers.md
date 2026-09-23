# Decision: pause GPU benchmark publication and qualify the measured bottlenecks

Status: proposed (implementation changes not made; diagnostic evidence retained)
Date: 2026-09-22

## Problem and scope

The requested GPU-only README comparison covers direct/DF HF and DFT through
96 atoms, plus small CCSD(T) compositions. The user paused benchmarking after
abnormal DFT latency and explicitly included the earlier 900-second HF-DF
96-atom failure. Preserve the draft measurements, but do not treat incomplete
or reference-only series as completed native qualification.

Measurements use master `572fbd6d4cfc3dd0db03fb0d1abaac4add0b9f12`, full Release
O3 sm120 AOT, native library SHA256
`78d01e579cff2ad2832e1b18effce007f27ecb1ac9bdc3e2806fea2268bd54e6`.
Freshly fetched master `817c481637834b91c0a819f7c958cbd7f88db4b3` has identical
relevant KS, direct-J/K, DF planner/source/J/K and calculator capability files.
This is a source comparison, not a performance measurement of a rebuilt newer
master. The workspace and measured binary were not switched during diagnosis.

Hardware: RTX 5090, 33,666,498,560 reported device bytes, CUDA 12.9 runtime,
AMD EPYC 7K62 host; OMP/OpenBLAS/MKL threads set to eight. Every real-GPU
diagnostic ran through finite Slurm `main --gres=gpu:5090:1` allocations with
assigned device visibility preserved. No CPU performance series was added.

## Findings and proposed priorities

### Native direct DFT: optimize Coulomb dispatch first

Nsight Systems 2025.1.3 captured one fixed post-cold warm PBE energy solve.
Both sizes used two iterations / two Fock builds, spherical def2-SVP and
unchanged default quadrature/tolerances.

| Atoms / AOs / points | Clean warm median | Profiled endpoint | J kernel sum | J share |
| --- | ---: | ---: | ---: | ---: |
| 3 / 24 / 82,944 | 0.806 s | 0.840 s | 0.6523 s, 2 launches | 78.5% |
| 6 / 48 / 165,888 | 5.740 s | 5.767 s | 5.3389 s, 2 launches | 92.8% |

`CudaKsPlan::enqueue_one` calls `enqueue_cuda_direct_jk_device`, which launches
`independent_jk_kernel`. Each ordered output AO pair traverses ordered density
pairs and evaluates the generic contracted integral after screening. It does
not dispatch HF's optimized generated shell-quartet Fock schedules. Candidate
quartet visits per build grow from 331,776 to 5,308,416; these are traversal
bounds, not measured unscreened integral counts.

Warm transport deltas are 208 scalar D2H bytes, zero density H2D and matrix D2H,
and two synchronization checkpoints. API wait times overlap GPU work and cannot
be added as independent overhead. Compiler replay/synchronization work alone
cannot explain away the measured J kernel cost.

At six atoms, XC potential assembly totals 0.1524 s and XC point evaluation
0.1338 s, each over 1,296 launches. Grid tiling/fusion remains a secondary
opportunity under #168. Prioritize typed pure-J output lowering into existing
generated integral tasks. Preserve generic/nonsymmetric compatibility; never
substitute HF's combined `J-0.5K` result for a pure-J consumer.

Tracking: [#1077](https://github.com/jinzhezenggroup/vibeqc/issues/1077).

### HF-DF 96 atoms: streamed source work already dominates before forces

The 900-second original whole-point timeout did not identify a phase. New
40-second Nsight and 45-second progress diagnostics localize an earlier SCF
bottleneck. Python `prepare_batch` returns in approximately 0.027 s; substantial
setup is lazy inside `execute`. One-electron preparation completes in 2.474 s,
DF setup in 0.445 s (including 0.411 s metric factorization).

The actual workload is 768 AOs, 3,712 auxiliary functions, 160 occupied orbitals,
batch one, cc-pVDZ-JKFIT, energy plus analytic forces. It is distinct from
equal-NAO/Naux workloads in #439.

| Automatic planning observation | Bytes or dimensions |
| --- | ---: |
| Total budget | 21,421,977,600 B |
| Value allowance | 13,685,173,124 B |
| Response allowance | 7,736,804,476 B |
| Single full dense B tensor | 17,515,413,504 B |
| Executed AO-pair tile / auxiliary tile | 589,824 / 580 |
| Storage / SCF exchange provider | streamed / dense |

The first streamed J is still running at the short cancellation. Two full raw
panels finish in 14.431 and 14.564 seconds; no completed SCF iteration or force
response appears. The independent Nsight capture shows two completed raw-source
kernel launches totaling 29.035 s, 90.3% of completed GPU kernel time. The kernel
is named `build_cuda_df_transformed_tile_kernel`, but the raw path invokes it
with `apply_metric_transform=false`; its name must not be misread as evidence
of repeated per-output whitening inside those two launches.

J uses two full raw-source passes. The captured K schedule has one AO row block
and seven auxiliary blocks, hence seven source passes. These are structural
counts, not seven completed K passes in the short trace. Together J/K request
19,704,840,192 public source values per Fock build, before contractions or forces.
This provides a concrete finite-work explanation for the severe cliff; it does
not prove the full 900-second run's final convergence trajectory.

Compiler/resource work should model source regeneration, representation costs
and phase lifetimes. Existing packed storage is opt-in, not unimplemented. Its
raw and whitened tensors together cost approximately 17.54 GB before scratch,
already more than this value allowance. Simply enabling packed storage is not
a demonstrated fix. Retain bounded fallback, exact metric/Frechet response and
independent energy/force gates when changing storage or budget lifetimes.

[PR #1076](https://github.com/jinzhezenggroup/vibeqc/pull/1076), head
`4b4503362f688dae3977c9f6a7f820184c5433e6`, was open when checked. It makes BLAS
the production **response algebra** default in `df_gradient_bridge.cu`, removing
resource-dependent scalar fallback. It is related but changes a later phase;
the measured streamed SCF J/K bottleneck is outside its diff. Do not create a
duplicate scalar-response fix or claim this PR resolves the entire timeout
without a complete endpoint measurement.

Tracking: [#1078](https://github.com/jinzhezenggroup/vibeqc/issues/1078), linked to
#1076, #206 and #409.

### Reference correctness and remaining capability gaps

GPU4PySCF CUDA12 1.8.1 skips zero-AO blocks by default, while its RKS density
assembly indexes a full-grid array by yielded point count and subsequently uses
all original weights. On the common atom/radial-ordered grid this shifts density
offsets and leaves an uninitialized tail. Six-atom PBE returned a nonconverged
energy near -1.719e11 Eh. Strict order alone failed in the zero-AO scaling kernel.
The benchmark-local strict-order/zero-row adapter preserves offsets and exactly
zero contributions; the fixed PBE comparison agrees within 3.0412e-12 Eh over
cold, priming and repeats. Installed dependency code was not modified. Broader
functional/empty-block regression qualification remains required:
[#1079](https://github.com/jinzhezenggroup/vibeqc/issues/1079).

Native DFT density fitting is explicitly rejected by `Calculator` with
`DFT supports conventional Coulomb only`. This is a capability gap, not slow
native DF-DFT. Pure semilocal DF J needs a consumer-specific plan, not all HF
K scratch. Track common-provider integration separately:
[#1080](https://github.com/jinzhezenggroup/vibeqc/issues/1080).

The 120-second r2SCAN/96-atom timeout belongs to GPU4PySCF reference qualification.
It covers startup/cold/priming/repeats, not warm latency. A bounded three-cycle
probe finishes deliberately unconverged in 14.981 s on 2,359,296 points;
effective-potential calls take 3.872, 3.321, 3.274 and 3.259 s. Gradient norms
are 3.667, 5.225 and 0.114. This demonstrates early finite progress, not complete
convergence or absence of later stagnation. Track cycle-level diagnosis before
raising any timeout: [#1081](https://github.com/jinzhezenggroup/vibeqc/issues/1081).

The reference environment is PySCF 2.14.0 / CuPy 14.2.0 and reports GPU4PySCF's
CuPy contraction fallback rather than cuTENSOR. Record this for performance
interpretation. Cross-engine iteration counts differ; comparisons are latency
observations, not equal-work speedups.

## Evidence and reproduction

Exact diagnostic scripts, immutable raw traces, statistics and `summary.json`
are retained locally in `.artifacts/readme-benchmarks-20260922/diagnosis/`.
Issue bodies contain reproduction settings and bounded commands. Raw captures
are ignored local artifacts, not published release assets. The progress trace
adds synchronization for phase completion, so its durations are diagnostic;
clean endpoint timing remains separate.

| Evidence | SHA256 |
| --- | --- |
| `pbe-3.nsys-rep` | `1132c5e64a3f0abb02602a5685755ff9f6cad8afb7eb83e0e9e5442d6e0aaf7e` |
| `pbe-6.nsys-rep` | `cef8ce1442d3430a3b8735cc30246c83cdcc81b5492f9cf36f2776a7d69084f9` |
| `hf-df-96.nsys-rep` | `cbf83c0584e745e4d3d061569c6ed14fc44975df99908682b62931c18ea2e984` |
| `hf-df-96-native.jsonl` | `27785561eb94979647b485b2a0caeacdea71d949e8b79765e9341bf533c1fc51` |
| `r2scan-reference-96.log` | `f94cbdbc4cfd4b235e10184d48881c4daba5cfd232fab12dbbc26450b7219bf5` |

## Rejected shortcuts and reopening gate

Do not resume large benchmark matrices, merely raise timeouts, attribute GPU
compute to host synchronization, relax scientific tolerances, silently switch
direct to DF, or claim memory savings as endpoint speedup. #1076 and the existing
compiler infrastructure should be reused where applicable, with phase evidence
separating their effects.

Resume publication after independently qualified small controls improve, staged
early-abort scaling establishes 96-atom completion, and all cold/priming/repeat
records pass convergence and numerical gates. Production fixes are tracked in
the issues above; this investigation did not implement or merge them.
