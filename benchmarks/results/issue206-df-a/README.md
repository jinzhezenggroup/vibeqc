# Issue #206 Slice A: matched DF matrix

This artifact records the first matched VibeQC DF versus GPU4PySCF DF matrix on
one RTX 5090. The source commit is `074e4584dc4763898918530288dd05471f09fb68`
and the run used Slurm job `1097`, CUDA 12.9, `def2-SVP` spherical orbitals
and the same `def2-SVP` auxiliary basis on both engines.

The four required endpoints all converged with five interleaved warm samples
per engine. The benchmark uses fixed engine-local post-cold density snapshots,
`energy_tolerance=1e-12`, `density_tolerance=1e-10`, reference gradient
tolerance `1e-10`, and screening tolerance `1e-12`.

| AO | Batch | VibeQC DF warm median | GPU4PySCF DF warm median | VibeQC/GPU4PySCF | Max dE | Max dF |
|---:|---:|---:|---:|---:|---:|---:|
| 96 | 1 | 818.6 ms | 252.0 ms | 3.25x | 1.42e-12 Eh | 2.90e-11 Eh/bohr |
| 96 | 4 | 3274.5 ms | 1009.8 ms | 3.24x | 7.84e-12 Eh | 6.01e-11 Eh/bohr |
| 192 | 1 | 10778.2 ms | 329.0 ms | 32.76x | 1.89e-11 Eh | 1.01e-10 Eh/bohr |
| 192 | 4 | 43203.7 ms | 1316.5 ms | 32.80x | 2.40e-11 Eh | 9.85e-11 Eh/bohr |

The numerical parity observations are retained, but no performance gate is
claimed: SCF iteration branches differ (VibeQC 2--3 iterations versus
GPU4PySCF 1), so the iteration-matched speed statistic is unavailable. The
component ledger is therefore intentionally incomplete. The current endpoint
API exposes DF metric diagnostics but not a complete split of raw metric and
three-center generation, factorization, RI-J/RI-K, response, transfers and
synchronization. Those values remain `null` until a synchronized component
capture is available; the large endpoint gap is a follow-up tuning target.

Reproduce the matrix from a clean worktree with the local Python/CUDA environment:

```bash
ISSUE206_PYTHON=/path/to/python CUDA_HOME=/path/to/cuda \
sbatch run_issue206_df.slurm
```

`ISSUE206_PYTHON` defaults to `python3`, `CUDA_HOME` defaults to the site
CUDA 12.9 path, `ISSUE206_REPEATS` defaults to 5, and
`ISSUE206_OUTPUT_DIR` defaults to `.artifacts/issue206-df-a` under the
checkout. Submit from the checkout, or set `ISSUE206_ROOT` to its absolute
path when submitting from another directory. Slurm executes a spool copy of
the script, so the runner uses `SLURM_SUBMIT_DIR` instead of that copy's location. The
archived manifest retains the absolute paths and environment used for job 1097.

The archive contains the exact manifest, four endpoint JSON files, and their
stdout/stderr logs. The manifest records the Slurm job, CUDA visibility, source
identity, command lines, and return codes. SHA-256 and member digests are in
`raw-evidence.manifest.json`.

Verify and restore the original measurements without running GPU work:

```bash
python -m tools.unpack_evidence benchmarks/results/issue206-df-a --output /tmp/issue206-df-a
```
