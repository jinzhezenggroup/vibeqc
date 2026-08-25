# Published GPU benchmark artifacts

These JSON files retain raw synchronized timing samples, accuracy differences,
the clean source commit, package versions, and CUDA device metadata. The two
engines use the same case-specific AO representation, SCF tolerances,
energy-plus-gradient workload, and allocated RTX 5090. QCE submits one
homogeneous fixed-topology batch; GPU4PySCF 1.8.1 retains one warm object per
system and invokes its single-system interface sequentially inside the same
batch timing boundary.

## Realistic 96-AO acceptance status

| Artifact | Batch | QCE warm median | GPU4PySCF warm median | Scoped speedup | Max energy error | Max force error | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| [`WATER27 tetramer`](rtx5090-ccdd250-water-tetramer-def2-svp-spherical-b1.json) | 1 | 1319.720 ms | 1071.889 ms | 0.812x | 1.36e-12 Eh | 9.68e-12 Eh/bohr | speed fails |
| [`WATER27 tetramer`](rtx5090-ccdd250-water-tetramer-def2-svp-spherical-b4.json) | 4 | 5059.831 ms | 4287.969 ms | 0.847x | 5.63e-12 Eh | 1.08e-11 Eh/bohr | speed fails |

Both clean artifacts come from commit
`ccdd250247da0af98fee2a5b921e7b3cf99f7f0a` on 2026-08-25. Every QCE and
GPU4PySCF system converged, and both points pass the explicit `3e-11 Eh` and
`3e-11 Eh/bohr` accuracy limits. They fail only the required `1.0x` minimum
speedup, so the 96-AO milestone remains open. All three synchronized warm
samples are retained because both engines show material timing variation at
this size. Relative to clean `35af36c`, hoisting pair-coefficient gradients
reduces the observed QCE medians by 14.7% at batch 1 and 10.9% at batch 4.
The separately executed GPU4PySCF medians also shift downward by 22.4% and
22.1%, so scoped speedup falls to `0.812x`/`0.847x` despite the absolute QCE
improvement. The final QCE convergence states use two iterations at batch 1
and `[2, 3, 2, 2]` at batch 4. Direct-J/K atomic reductions can alter warm SCF
trajectories for both engines, and the current harness does not yet preserve
iteration counts for every timed repeat. These endpoint medians must therefore
be read together with all retained raw samples; neither required speed point
has passed yet.

## Realistic 192-AO correctness status

| Artifact | Batch | QCE warm | GPU4PySCF warm | Informational speedup | Max energy error | Max force error | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| [`WATER27 S4 octamer`](rtx5090-a4db5f3-water-octamer-s4-def2-svp-spherical-b1.json) | 1 | 13233.879 ms | 2245.000 ms | 0.170x | 2.16e-12 Eh | 5.60e-11 Eh/bohr | passes |
| [`WATER27 S4 octamer`](rtx5090-a4db5f3-water-octamer-s4-def2-svp-spherical-b4.json) | 4 | 72096.300 ms | 7812.312 ms | 0.108x | 8.98e-12 Eh | 2.36e-10 Eh/bohr | passes |

These clean commit-`a4db5f3` artifacts use one synchronized warm replay per
point because of the direct-J/K cost. Both satisfy the current `1e-10 Eh` and
`5e-10 Eh/bohr` correctness limits. The recorded warm times remain sensitive
to the number of SCF iterations reached after nondeterministic direct-J/K
atomic reductions; the batch-1 artifact records two iterations and batch 4
records 2/7/3/5 across its four systems. Their speedups are retained for
transparency but are not acceptance criteria until complete DF J/K lands.

## 96-AO warm component profile

One CUDA-profiler-range capture per engine isolates a warm batch-1
energy-plus-force execution after the cold plan/object has already completed.
The exact acceptance workload and tolerances are unchanged.

| Component | QCE kernel time | GPU4PySCF kernel time | QCE minus GPU4PySCF |
| --- | ---: | ---: | ---: |
| Direct J/K | 491.499 ms | 501.613 ms | -10.114 ms |
| Two-electron force | 385.819 ms | 311.187 ms | +74.631 ms |
| Eigensolver | 31.288 ms | 5.823 ms | +25.465 ms |
| One-electron force | 22.041 ms | 2.519 ms | +19.522 ms |
| QCE one-electron values and Schwarz | 0.000 ms | n/a | n/a |

QCE records 56 kernel launches and a 1.555 s kernel span; GPU4PySCF records
4,419 launches and a 1.515 s kernel span. Launch count is therefore not the
primary explanation for the gap. Direct J/K kernel time remains comparable.
Hoisting pair-coefficient gradients out of the high-order Cartesian product
reduces order 4/5/6 from 67.703/64.266/44.045 ms to
58.539/50.458/30.582 ms. The complete force pass improves by 9.0%. Order 2 is
now the largest isolated gap at 88.488 ms versus 57.084 ms; order 5 follows at
50.458 ms versus 26.380 ms. Orders 6--8 together cost 57.230 ms versus
GPU4PySCF's 41.315 ms fallback. GPU4PySCF executes common classes through
generated Rys `ip1` kernels with cooperative workers and shared primitive-pair
intermediates. The next primary target is therefore generated/cooperative
order-2 contraction and primitive-pair reuse, rather than more direct-J/K
tuning. See the
[`ccdd250` component artifact](rtx5090-ccdd250-water-tetramer-warm-component-profile.json)
for the full per-order breakdown and capture metadata. Its QCE values come
from clean `ccdd250`; the unchanged GPU4PySCF values come from the matching
clean 1.8.1 capture previously published with `c28b1e9`.

## Current exact shell-class results

| Artifact | Batch | QCE warm median | GPU4PySCF warm median | Scoped speedup | Max energy error | Max force error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [`sdf18-direct`](rtx5090-8300dff-sdf18-direct-b16.json) | 16 | 67.709 ms | 2499.369 ms | 36.91x | 6.35e-14 Eh | 4.87e-14 Eh/bohr |
| [`water-def2-svp`](rtx5090-8300dff-water-def2-svp-b8.json) | 8 | 1202.903 ms | 7300.320 ms | 6.07x | 1.72e-12 Eh | 5.29e-13 Eh/bohr |
| [`water-def2-svp-spherical`](rtx5090-320ead9-water-def2-svp-spherical-b8.json) | 8 | 2313.694 ms | 7331.882 ms | 3.17x | 1.78e-12 Eh | 4.24e-13 Eh/bohr |
| [`water-def2-tzvp`](rtx5090-e303ca4-water-def2-tzvp-b4.json) | 4 | 2069.985 ms | 12084.960 ms | 5.84x | 1.21e-12 Eh | 9.24e-13 Eh/bohr |
| [`water-def2-tzvp-spherical`](rtx5090-1575a46-water-def2-tzvp-spherical-b4.json) | 4 | 911.532 ms | 12099.769 ms | 13.27x | 1.01e-12 Eh | 8.81e-13 Eh/bohr |
| [`oh-def2-svp-uhf`](rtx5090-6d3b9ec-oh-def2-svp-uhf-b8.json) | 8 | 571.419 ms | 7244.466 ms | 12.68x | 1.24e-12 Eh | 2.11e-13 Eh/bohr |
| [`oh-def2-svp-spherical-uhf`](rtx5090-f44fdf7-oh-def2-svp-spherical-uhf-b8.json) | 8 | 1369.543 ms | 7304.363 ms | 5.33x | 1.09e-12 Eh | 4.13e-11 Eh/bohr |

The RHF artifacts were recorded from clean commit
`8300dff71090c8ef705532f964c58f14a7e4b0cb`; the named-basis UHF artifact was
recorded after adding its workload at clean commit
`6d3b9eccd812421098d73c80ca704d13dbbd4884`. The real-spherical artifact was
recorded from clean commit
`320ead906eb0b0e3335aa1b9e2893f066dd02eee`. The real-spherical UHF artifact
was recorded after the open-shell cold-guess fix at clean commit
`f44fdf7d6d84e92ef7405e09643a69f78c627e52`. The current 43-AO def2-TZVP
artifact was recorded after the closed order-4 recurrence at clean commit
`1575a4693821a0c41f5f52b1aabeb4ab81c2329c`. All were measured on 2026-08-24.
The matching Cartesian def2-TZVP artifact was recorded from clean commit
`e303ca4aae9930dafd9f52bde813a030953a09cc`.
They establish performance only for these exact homogeneous batch workloads;
they are not a claim of broad QCE leadership.

## Prior closed order-3 baseline

| Artifact | Batch | QCE warm median | GPU4PySCF warm median | Scoped speedup |
| --- | ---: | ---: | ---: | ---: |
| [`water-def2-tzvp-spherical`](rtx5090-89d9e90-water-def2-tzvp-spherical-b4.json) | 4 | 963.050 ms | 12123.476 ms | 12.59x |

The clean `89d9e902c77141f8a8ff2dd0c1357c6202ef1c57` baseline used closed
recurrences through total order three but retained generic order-4 Hermite and
Coulomb workspaces. Keeping it beside the closed order-4 artifact makes the
5.3% QCE warm-time reduction independently auditable.

## Prior closed order-2 baseline

| Artifact | Batch | QCE warm median | GPU4PySCF warm median | Scoped speedup |
| --- | ---: | ---: | ---: | ---: |
| [`water-def2-tzvp-spherical`](rtx5090-1041d86-water-def2-tzvp-spherical-b4.json) | 4 | 1027.326 ms | 12158.679 ms | 11.84x |

The clean `1041d865c0f995214baec917f8f3c725a25ac903` baseline used closed
recurrences through total order two but retained generic order-3 Hermite and
Coulomb workspaces. Keeping it beside the closed order-3 artifact makes the
6.3% QCE warm-time reduction independently auditable.

## Prior closed order-1 baseline

| Artifact | Batch | QCE warm median | GPU4PySCF warm median | Scoped speedup |
| --- | ---: | ---: | ---: | ---: |
| [`water-def2-tzvp-spherical`](rtx5090-f260de0-water-def2-tzvp-spherical-b4.json) | 4 | 1090.931 ms | 12137.277 ms | 11.13x |

The clean `f260de081a009f391537f6697b672aa9e98b6a87` baseline used the closed
`(p s | s s)` recurrence but retained generic total-order-2 Hermite and
Coulomb workspaces. Keeping it beside the closed order-2 artifact makes the
5.8% QCE warm-time reduction independently auditable.

## Prior generic order-1 baseline

| Artifact | Batch | QCE warm median | GPU4PySCF warm median | Scoped speedup |
| --- | ---: | ---: | ---: | ---: |
| [`water-def2-tzvp-spherical`](rtx5090-f644916-water-def2-tzvp-spherical-b4.json) | 4 | 1137.845 ms | 12106.988 ms | 10.64x |

The clean `f644916918a946e24f869145244db16296d54628` baseline used the generic
Hermite coefficient arrays for `(p s | s s)`. Keeping it beside the closed
order-1 artifact makes the 4.1% QCE warm-time reduction independently
auditable.

## Prior 256-thread direct-tile baseline

| Artifact | Batch | QCE warm median | GPU4PySCF warm median | Scoped speedup |
| --- | ---: | ---: | ---: | ---: |
| [`water-def2-tzvp-spherical`](rtx5090-b4a13fa-water-def2-tzvp-spherical-b4.json) | 4 | 1561.725 ms | 12136.266 ms | 7.77x |

The clean `b4a13fab29828429a92f224c1fe3ec98ddf27677` baseline launched one
256-thread block per compact descriptor. Keeping it beside the virtual
one-warp artifact makes the 27.1% QCE warm-time reduction independently
auditable.

## Prior sparse spherical contraction baseline

| Artifact | Batch | QCE warm median | GPU4PySCF warm median | Scoped speedup |
| --- | ---: | ---: | ---: | ---: |
| [`water-def2-tzvp-spherical`](rtx5090-a057e82-water-def2-tzvp-spherical-b4.json) | 4 | 4303.961 ms | 12124.321 ms | 2.82x |

The clean `a057e82af019912818e26f91d0dfba1d9861e316` baseline contracted sparse
real-spherical expansion terms inside every direct quartet. Keeping it beside
the `b4a13fa` Cartesian-source artifact makes the 63.7% QCE warm-time reduction
independently auditable.

## Prior spherical force baseline

| Artifact | Batch | QCE warm median | GPU4PySCF warm median | Scoped speedup |
| --- | ---: | ---: | ---: | ---: |
| [`water-def2-tzvp-spherical`](rtx5090-40cef2f-water-def2-tzvp-spherical-b4.json) | 4 | 5644.454 ms | 12095.489 ms | 2.14x |

The clean `40cef2f1f06d812b401993e1da2f3dceb8b3167a` baseline used one scalar
forward recurrence per Cartesian axis. Keeping it alongside the `a057e82`
artifact makes the 23.8% three-component force-propagation reduction
independently auditable.

## Prior angular-order baseline

| Artifact | Batch | QCE warm median | GPU4PySCF warm median | Scoped speedup |
| --- | ---: | ---: | ---: | ---: |
| [`sdf18-direct`](rtx5090-b042488-sdf18-direct-b16.json) | 16 | 78.346 ms | 2503.287 ms | 31.95x |
| [`water-def2-svp`](rtx5090-b042488-water-def2-svp-b8.json) | 8 | 1536.982 ms | 7340.873 ms | 4.78x |

The baseline artifacts came from clean commit
`b04248846e1d5a4eafb481121cfca6b6160bd10f`. Keeping both generations makes
the exact shell-class improvement auditable from raw samples.

Reproduce them from the repository root after building the CUDA library:

```bash
export PATH=/group/software/cuda-12.9.1/bin:$PATH
export LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64:/group/software/cuda-12.9.1/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export CUPY_CACHE_DIR=/tmp/cupy-sm120-$SLURM_JOB_ID
export PYTHONPATH=$PWD/python
export QCE_LIBRARY=$PWD/build/libqce.so.0.1.0

uv run --isolated \
  --with cupy-cuda12x==14.2.0 \
  --with gpu4pyscf-cuda12x==1.8.1 \
  --with pyscf==2.14.0 \
  python benchmarks/compare_gpu4pyscf_batch.py \
  --case sdf18-direct --batch 16 --repeats 5 \
  --minimum-speedup 30 \
  --maximum-energy-error 3e-12 \
  --maximum-force-error 3e-12 \
  --output result.json
```

Use `--case water-def2-svp --batch 8` or
`--case oh-def2-svp-uhf --batch 8` for the named-basis workloads, with
conservative minimum speedups of 5x and 10x respectively. The standard pure
basis workload uses `--case water-def2-svp-spherical --batch 8`, a 2.5x
minimum-speedup gate, and a `3e-11 Eh/bohr` force gate to cover parallel
reduction-order variation. Its UHF counterpart uses
`--case oh-def2-svp-spherical-uhf --batch 8`, a 4x gate, and a
`3e-9 Eh/bohr` force gate that covers the arbitrary orientation of the
degenerate pi hole; the recorded maximum error is `4.13e-11 Eh/bohr`. The
larger pure workload uses `--case water-def2-tzvp-spherical --batch 4`, a 10x
gate, and `3e-12`/`3e-11` energy/force gates. Its Cartesian counterpart uses
`--case water-def2-tzvp --batch 4` and a 5x minimum-speedup gate. The harness
records all gate thresholds and failures in the JSON before exiting with
status 2 on a failed gate. On a Slurm cluster, run the command inside an
allocation that owns exactly one GPU.
