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
| [`WATER27 tetramer`](rtx5090-2c0ee1a-water-tetramer-def2-svp-spherical-b1.json) | 1 | 1492.812 ms | 1372.687 ms | 0.920x | 3.13e-12 Eh | 2.00e-12 Eh/bohr | speed fails |
| [`WATER27 tetramer`](rtx5090-2c0ee1a-water-tetramer-def2-svp-spherical-b4.json) | 4 | 5447.831 ms | 5208.659 ms | 0.956x | 3.92e-12 Eh | 5.60e-12 Eh/bohr | speed fails |

Both clean artifacts come from commit
`2c0ee1aad974dbb9821124aec2d8aef7c00e066a` on 2026-08-25. Every QCE and
GPU4PySCF system converged, and both points pass the explicit `3e-11 Eh` and
`3e-11 Eh/bohr` accuracy limits. They fail only the required `1.0x` minimum
speedup, so the 96-AO milestone remains open. All three synchronized warm
samples are retained because both engines show material timing variation at
this size. The batch-1 GPU4PySCF samples descend from 1.707 s through 1.373 s
to 1.077 s, while the batch-4 samples descend from 6.465 s through 5.209 s to
4.292 s. QCE also changes SCF iteration count across repeated warm executions;
the current harness records only the final repeat's two iterations at batch 1
and `[2, 2, 2, 2]` at batch 4. These endpoint medians therefore establish only
that accuracy passes and both speed gates remain open. The component profile
below is the stronger evidence for the local order-2 optimization.

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
| Direct J/K | 487.953 ms | 501.613 ms | -13.659 ms |
| Two-electron force | 354.977 ms | 311.187 ms | +43.790 ms |
| Eigensolver | 31.257 ms | 5.823 ms | +25.435 ms |
| One-electron force | 22.065 ms | 2.519 ms | +19.546 ms |
| QCE one-electron values and Schwarz | 0.000 ms | n/a | n/a |

QCE records 56 kernel launches and a 1.523 s kernel span; GPU4PySCF records
4,419 launches and a 1.515 s kernel span. Launch count is therefore not the
primary explanation for the gap. Direct J/K kernel time remains comparable.
Order 2 now stores only one center's pair-coefficient gradients, evaluates
three full centers, and restores the fourth from translation. Its kernel falls
from 188 registers and 864 stack bytes to 169 registers and 624 stack bytes,
and from 88.488 ms to 65.231 ms. Three additional isolated captures bracket it
at 64.708--65.402 ms.

Orders 4--6 now generate each subset coefficient and its first-center
gradient in one matching traversal instead of materializing the full pair
expansion and revisiting every matching for three Cartesian derivatives.
Their resources fall from `215/226/255` registers and
`1472/2064/3376` stack bytes to `207/218/224` registers and
`880/992/1216` bytes. Five captures bracket order 4 at 54.125--55.164 ms,
order 5 at 47.453--47.572 ms, and order 6 at 28.988--29.164 ms. The clean
complete force pass reaches 354.977 ms, 2.7% below `2c0ee1a`.

The largest remaining isolated force gap is order 5 at 47.572 ms versus
26.380 ms. Order 1 follows at +17.900 ms, while orders 6--8 together cost
55.355 ms versus GPU4PySCF's 41.315 ms fallback. Outside the two-electron
force, eigensolver and one-electron-force gaps are +25.435 and +19.546 ms.
GPU4PySCF executes common classes through generated Rys `ip1` kernels with
cooperative workers and shared primitive-pair intermediates. The next primary
targets are therefore eigensolver, order-5 cooperative contraction,
one-electron force, and order 1 rather than direct-J/K tuning. See the
[`93f6eee` component artifact](rtx5090-93f6eee-water-tetramer-warm-component-profile.json)
for the full per-order breakdown and capture metadata. Its QCE values come
from clean `93f6eee`; the unchanged GPU4PySCF values come from the matching
clean 1.8.1 capture previously published with `c28b1e9`.

The closed order-zero/one primitive derivative now evaluates only three
independent Gaussian centers and reconstructs the fourth from translational
invariance. Order one falls from 222 to 176 registers at the same 208-byte
stack footprint; order zero falls from 194 to 192 registers. Five
two-iteration captures span 52.835--53.849 ms for order one and
28.720--29.430 ms for order zero. The clean complete force pass reaches
353.017 ms. Formal clean endpoint runs pass both accuracy limits but still
miss parity at `0.917x` for batch 1 and `0.913x` for batch 4; the latter has a
3/3/2/2 final SCF-iteration split and is not an iteration-matched speed
comparison. See the
[`962f0bd` component artifact](rtx5090-962f0bd-water-tetramer-warm-component-profile.json)
for resource usage, repeated component samples, and raw endpoint timings.

Commit `32af069` replaces the realistic-size cyclic Jacobi path with FP64
`cusolverDnXsyevBatched`. Device and host workspaces are fixed during setup,
the device arena is also bound as the cuBLAS workspace, and the provider stays
inside the device-launch/tail-launch SCF Graph. The superseded cyclic kernel
was removed. A standalone Graph probe verified computed eigenvalues and a
successful device-tail relaunch at all four acceptance sizes:

| AO count | Batch | Xsyev Graph time | Device workspace | Host workspace |
| ---: | ---: | ---: | ---: | ---: |
| 96 | 1 | 2.744 ms | 1,335,960 B | 0 B |
| 96 | 4 | 2.513 ms | 5,341,920 B | 0 B |
| 192 | 1 | 4.170 ms | 2,589,336 B | 0 B |
| 192 | 4 | 5.316 ms | 10,355,424 B | 0 B |

On the production 96-AO batch-1 capture, cuSOLVER provider kernels sum to
2.612 ms, down from the previous QCE cyclic kernel's 31.226 ms and below the
matching GPU4PySCF eigensolver's 5.823 ms. The eigensolver gap is therefore
closed. The formal end-to-end matrix still misses the 96-AO parity gate:

| AO count | Batch | QCE warm | GPU4PySCF warm | Speedup | Max dE | Max dF | Gate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 96 | 1 | 1397.063 ms | 1351.642 ms | 0.967x | 1.76e-12 Eh | 2.01e-12 Eh/bohr | accuracy pass, speed fail |
| 96 | 4 | 5803.771 ms | 5421.947 ms | 0.934x | 3.47e-12 Eh | 2.82e-12 Eh/bohr | accuracy pass, speed fail |
| 192 | 1 | 13475.169 ms | 2218.889 ms | 0.165x | 1.93e-12 Eh | 5.30e-11 Eh/bohr | accuracy pass; speed deferred |
| 192 | 4 | 66719.807 ms | 7763.699 ms | 0.116x | 8.87e-12 Eh | 2.36e-10 Eh/bohr | accuracy pass; speed deferred |

QCE-only repeated samples give 1.397 s at batch 1 (seven repeats) and
5.605 s at batch 4 (five repeats). Remaining 96-AO work is now dominated by
the two-electron and one-electron force paths rather than diagonalization.
The concise raw summary is
[`rtx5090-32af069-xsyev-provider-gates.json`](rtx5090-32af069-xsyev-provider-gates.json).

A separate 192-AO batch-1 capture explains the large-size endpoint. Its
2-iteration warm execution takes 11.320 s: the device-tail SCF Graph occupies
4.179 s, the two post-SCF direct-J/K rebuilds sum to 4.045 s, and the
two-electron force costs 2.976 s. One-electron force is 53.6 ms and Xsyev is
only 3.90 ms. Relative to the 96-AO capture, visible direct J/K and
two-electron force grow by 8.25x and 8.43x respectively when AO count doubles.
The formal 192-AO batch-4 point is additionally branch-sensitive: its systems
take 8/3/4/2 iterations, versus 3 iterations at batch 1. This is why complete
DF J/K remains the prerequisite for activating the 192-AO speed gate.

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
