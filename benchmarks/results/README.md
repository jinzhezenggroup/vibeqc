# Published GPU benchmark artifacts

These JSON files retain raw synchronized timing samples, accuracy differences,
the clean source commit, package versions, and CUDA device metadata. The two
engines use the same case-specific AO representation, SCF tolerances,
energy-plus-gradient workload, and allocated RTX 5090. QCE submits one
homogeneous fixed-topology batch; GPU4PySCF 1.8.1 retains one warm object per
system and invokes its single-system interface sequentially inside the same
batch timing boundary.

## Current exact shell-class results

| Artifact | Batch | QCE warm median | GPU4PySCF warm median | Scoped speedup | Max energy error | Max force error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [`sdf18-direct`](rtx5090-8300dff-sdf18-direct-b16.json) | 16 | 67.709 ms | 2499.369 ms | 36.91x | 6.35e-14 Eh | 4.87e-14 Eh/bohr |
| [`water-def2-svp`](rtx5090-8300dff-water-def2-svp-b8.json) | 8 | 1202.903 ms | 7300.320 ms | 6.07x | 1.72e-12 Eh | 5.29e-13 Eh/bohr |
| [`water-def2-svp-spherical`](rtx5090-320ead9-water-def2-svp-spherical-b8.json) | 8 | 2313.694 ms | 7331.882 ms | 3.17x | 1.78e-12 Eh | 4.24e-13 Eh/bohr |
| [`water-def2-tzvp-spherical`](rtx5090-40cef2f-water-def2-tzvp-spherical-b4.json) | 4 | 5644.454 ms | 12095.489 ms | 2.14x | 8.53e-13 Eh | 9.42e-13 Eh/bohr |
| [`oh-def2-svp-uhf`](rtx5090-6d3b9ec-oh-def2-svp-uhf-b8.json) | 8 | 571.419 ms | 7244.466 ms | 12.68x | 1.24e-12 Eh | 2.11e-13 Eh/bohr |
| [`oh-def2-svp-spherical-uhf`](rtx5090-f44fdf7-oh-def2-svp-spherical-uhf-b8.json) | 8 | 1369.543 ms | 7304.363 ms | 5.33x | 1.09e-12 Eh | 4.13e-11 Eh/bohr |

The RHF artifacts were recorded from clean commit
`8300dff71090c8ef705532f964c58f14a7e4b0cb`; the named-basis UHF artifact was
recorded after adding its workload at clean commit
`6d3b9eccd812421098d73c80ca704d13dbbd4884`. The real-spherical artifact was
recorded from clean commit
`320ead906eb0b0e3335aa1b9e2893f066dd02eee`. The real-spherical UHF artifact
was recorded after the open-shell cold-guess fix at clean commit
`f44fdf7d6d84e92ef7405e09643a69f78c627e52`. The 43-AO def2-TZVP artifact
was recorded after the Graph-native eigensolver and ERI force-center reduction
at clean commit `40cef2f1f06d812b401993e1da2f3dceb8b3167a`. All were measured on
2026-08-24.
They establish performance only for these exact homogeneous batch workloads;
they are not a claim of broad QCE leadership.

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
larger pure workload uses `--case water-def2-tzvp-spherical --batch 4`, a 1.8x
gate, and `3e-12`/`3e-11` energy/force gates. The harness records all gate
thresholds and failures in the JSON before exiting with status 2 on a failed
gate. On a Slurm cluster, run the command inside an allocation that owns
exactly one GPU.
