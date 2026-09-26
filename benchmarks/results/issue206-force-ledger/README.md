# Issue 206 DF force-response ledger

This diagnostic complements the matched DF matrix in
`benchmarks/results/issue206-df-a/`. It runs fresh single-system VibeQC CUDA
DF calculations with `properties=("energy",)` and
`properties=("energy", "forces")` and records the wall-time increment. The two
runs are deliberately not a warm-solve comparison; the ledger isolates the
cost that must be optimized in the energy-plus-force endpoint.

The refreshed version-2 run used Slurm job `9310` on node3 with one RTX 5090
and CUDA 12.9. The clean benchmark checkout was
`f441d65bea2a5b0c4b066a0196e743e5d8c4012b`. Its explicitly selected native
library was built from `a2522d794fd3acedbf81f8145cfe436aab23b1e0` with Release,
CUDA architecture 120 and AOT shells disabled. `ledger.json` records the exact
binary path and SHA-256 independently of the benchmark checkout.

This supersedes the original version-1 ledger, whose native binary identity
was not captured. The two runs do not establish a before/after speedup.

| workload | energy-only | energy + force | force increment |
| --- | ---: | ---: | ---: |
| water tetramer, 96 AO | 0.618 s | 1.066 s | 0.448 s |
| water octamer, 192 AO | 5.524 s | 12.444 s | 6.919 s |

Both paired runs converged with the same iteration count per workload (34 and
39). The ledger is diagnostic evidence, not a performance gate or a claim of
warm endpoint parity. Raw values and provenance are retained in `ledger.json`.

The version-2 runner requires both solves to converge with identical iteration
counts and energies agreeing within `1e-10` Hartree. It checks the requested
force outputs and finite energies, forces and timings before publishing an
increment. Invalid pairs fail without writing a new ledger.

Select the exact native binary explicitly. Each new ledger records its resolved
path and SHA-256 alongside checkout revision, dirty state and patch hash, and
rejects source or binary changes during timing:

```sh
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:10:00 \
  env PYTHONPATH=python:. python benchmarks/issue206_df_force_probe.py \
  --library /absolute/path/to/libvibeqc.so --output .artifacts/force-ledger.json
```
