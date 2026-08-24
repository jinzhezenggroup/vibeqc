# Published GPU benchmark artifacts

These JSON files retain raw synchronized timing samples, accuracy differences,
the clean source commit, package versions, and CUDA device metadata. The two
engines use the same Cartesian basis, SCF tolerances, energy-plus-gradient
workload, and allocated RTX 5090. QCE submits one homogeneous fixed-topology
batch; GPU4PySCF 1.8.1 retains one warm object per system and invokes its
single-system interface sequentially inside the same batch timing boundary.

| Artifact | Batch | QCE warm median | GPU4PySCF warm median | Scoped speedup | Max energy error | Max force error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [`sdf18-direct`](rtx5090-b042488-sdf18-direct-b16.json) | 16 | 78.346 ms | 2503.287 ms | 31.95x | 7.59e-14 Eh | 4.91e-14 Eh/bohr |
| [`water-def2-svp`](rtx5090-b042488-water-def2-svp-b8.json) | 8 | 1536.982 ms | 7340.873 ms | 4.78x | 1.65e-12 Eh | 5.36e-13 Eh/bohr |

Both artifacts were recorded from clean commit `b04248846e1d5a4eafb481121cfca6b6160bd10f`
on 2026-08-24. They establish performance only for these exact homogeneous
batch workloads; they are not a claim of broad GPU4PySCF leadership.

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
  --case sdf18-direct --batch 16 --repeats 5 --output result.json
```

Use `--case water-def2-svp --batch 8` for the second workload. On a Slurm
cluster, run the command inside an allocation that owns exactly one GPU.
