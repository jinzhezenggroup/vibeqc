# Issue #174 slice-E precision cost matrix

This directory records the slice-E measurement of the **existing** precision
policy: complete solves at identical numerical controls for the `fp64` and
`auto` policies. It selects no policy, promotes nothing, and is not acceptance
of `auto` on any domain — it is the error/cost evidence those decisions need.

The measurement harness lives in `benchmarks/issue174_slice_e_matrix.py` with
its hardware-free protocol regressions in
`tests/python/test_issue174_slice_e_matrix.py`.

## Artifact

`raw-evidence.zip` (SHA-256 `5965ee285b48bfd6...`) holds the two original JSON
artifacts; `raw-evidence.manifest.json` records the archive digest, every member
byte count and digest, and the source commit.

```bash
python -m tools.unpack_evidence benchmarks/results/issue174-slice-e --output /tmp/slice-e
```

Members:

| member | bytes | contents |
|---|---|---|
| `rtx5090-67212f2-issue174-slice-e-matrix.json` | 1,717,043 | 96 + 192 AO, batch sizes 1 and 4 |
| `rtx5090-67212f2-issue174-slice-e-matrix-384.json` | 193,042 | 384 AO, repeats 3, no batch section |

## Controls

`energy_tolerance` `1e-9`, `1e-7`, `1e-6`, `1e-5`; `density_tolerance` `1e-8`;
`screening_tolerance` `1e-12`; `force_target` `1e-6 Eh/bohr`; five repeats
(three for 384 AO); batch sizes 1 and 4; both policies at identical settings;
error measured against a separately computed stricter FP64 reference
(`energy_tolerance` `1e-13`, `density_tolerance` `1e-11`,
`screening_tolerance` `1e-14`). Every recorded `singlepoint` sample is a cold
solve, because a `Calculator` call carries no persisted validated warm state;
the batch section supplies the warm states through a prepared batch.

## Observed results

Complete single solves — `auto` never selected the mixed route:

| case | samples per configuration | `auto` mixed activations | auto/fp64 median wall ratio |
|---|---|---|---|
| water tetramer, 96 AO | 5 | 0 | 0.999 – 1.003 |
| water octamer, 192 AO | 5 | 0 | 0.997 – 1.001 |
| water hexadecamer, 384 AO | 3 | 0 | 0.995 – 1.004 |

Maximum observable error against the strict reference over these records:
`6.6e-12 Eh` and `1.8e-11 Eh` energy at 96/192 AO and 384 AO, `2.6e-8` and
`3.2e-8 Eh/bohr` maximum force component; `vibeqc.accuracy` reports
`observed_met` for all 160 assessed records in the first artifact.

Ragged batches — the mixed route is reachable only from a validated warm state:

| case | size | mode | cold execute (median) | warm execute (median) | items on the mixed route |
|---|---|---|---|---|---|
| tetramer, 96 AO | 1 | `fp64` | 0.863 s | 0.133 s | 0 / 60 |
| tetramer, 96 AO | 1 | `auto` | 0.865 s | 0.300 s | 40 / 60 |
| tetramer, 96 AO | 4 | `fp64` | 2.152 s | 0.358 s | 0 / 240 |
| tetramer, 96 AO | 4 | `auto` | 2.142 s | 0.690 s | 160 / 240 |
| octamer, 192 AO | 1 | `fp64` | 2.373 s | 0.283 s | 0 / 60 |
| octamer, 192 AO | 1 | `auto` | 2.373 s | 0.283 s | 0 / 60 |
| octamer, 192 AO | 4 | `fp64` | 8.094 s | 0.912 s | 0 / 240 |
| octamer, 192 AO | 4 | `auto` | 8.097 s | 0.912 s | 0 / 240 |

Two boundaries are visible here rather than hidden: cold items are never
admitted (the cold column is pure FP64 on both policies), and the 192-AO
octamer never activates the mixed route at all — consistent with a topology
that leaves the exact-arena route, whose per-item census the budget-aware
policy refuses to guess.

## Source and environment identity

- Source: commit `67212f2` of the VibeQC worktree that produced the artifacts
  (the same commit is recorded inside each artifact).
- Device: NVIDIA GeForce RTX 5090, compute capability 12.0, 33.7 GB, 170
  multiprocessors; driver 12080, runtime 12040 as reported by the ctypes CUDA
  facade recorded in the artifact (`cupy` is not installed on this host).
- Python 3.12.12 (DeepMD-kit environment); CUDA toolkit 12.9.1 for the measured
  library build (`sm_120`); executed through Slurm (`--partition=main
  --gres=gpu:5090:1`).

## Reproduction

```bash
ssh n5
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=01:30:00 bash -lc '
  cd ~/vibeqc-issue174-b
  export CUDA_HOME=/group/software/cuda-12.9.1
  export PYTHONPATH=python:.
  export VIBEQC_LIBRARY=$PWD/build/cuda-check/libvibeqc.so
  /group/software/deepmd-kit-3.1.1/bin/python benchmarks/issue174_slice_e_matrix.py \
      --output benchmarks/results/<name>.json'
```

Add `--include-large --repeats 3 --skip-batch` for the 384-AO artifact, and
`--include-bounded-streaming` for the 768-AO case (not run here).

## Limitations

- The provenance reports an item's effective bits but not **why** it stayed
  FP64, so "no validated warm state" and "census-free topology" cannot be
  separated from the artifact alone.
- The 768-AO bounded-streaming case is not included.
- Every endpoint evaluates forces; there is no separate energy-only timing.
- The artifact records the device and Python environment but not the digest of
  the measured library, which is identified only by the source commit and the
  build configuration above.
