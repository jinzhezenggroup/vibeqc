# Decision: distinguish endpoint deadlines from complete benchmark jobs

Status: implemented
Date: 2026-09-23

## Problem and decision

The original 96-atom reference r²SCAN point hit a 120-second whole-process
limit. Its cold-start banner did not establish a per-iteration hang or a
convergence defect. Preserve a JSON checkpoint before each native/reference
solve, and append endpoint/cycle progress to a separate JSONL journal.

An independent spawned watchdog kills only its owning benchmark process when
one solve exceeds its finite deadline, retaining the stopped result and prior
successful samples. A Python signal handler alone cannot interrupt a blocked
native call. Slurm's finite allocation remains the outer bound. Cold failure
stops warm timing; tolerances and the 100-cycle ceiling are unchanged.

Reference SCF stage synchronization is explicit opt-in `--trace-scf`. Mark
such samples as diagnostic rather than publishing their times as clean
benchmark results. The frozen post-cold density is reused for priming/repeats.

## Evidence

Slurm 11249, RTX 5090, GPU4PySCF 1.8.1, direct r²SCAN/def2-SVP spherical,
96 atoms/768 AOs and the unchanged 2,359,296-point explicit grid:

- Cold: 53.328 s, 15 cycles, converged.
- Priming: 16.641 s, 4 cycles, converged.
- Two repeats: 16.701 and 16.635 s, 4 cycles each, converged.
- Energy: -2442.90659171757 Eh cold; repeat differences below 4.5e-11 Eh.
- Final orbital gradient: 9.93e-9 cold, 8.68e-10 warm; required threshold 1e-8.
- Ordinary effective-potential stages take about 3.1–4.0 s, with no long stall.

Thus the short-run evidence does not reproduce a reference SCF bug. The sum of
cold, priming, repeats and setup is distinct from a single endpoint deadline.
Do not label the earlier whole-point timeout as warm latency or solver failure.
Host regressions verify actual watchdog termination, preservation of prior
results, and disarming on success and exceptions.

Machine-local evidence: `.artifacts/gpu-blocker-fixes/r2scan96-trace.{json,log}`
and its `.progress.jsonl` journal. See #1081. Benchmark publication remains
paused while the native blockers are resolved.
