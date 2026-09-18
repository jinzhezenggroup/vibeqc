# Stable SCF performance benchmarking

This note defines the policy for SCF performance data when repeated warm starts can follow different convergence branches.

## Problem

A fixed post-cold density does not guarantee a fixed amount of SCF work. Near a convergence threshold, tiny floating-point reduction differences can change whether the first replay is accepted as converged. A repeated benchmark can therefore produce branches such as `1/7/4/4/1` iterations from the same stored density. Mixing those timings into one median measures a mixture of different workloads rather than backend performance.

## Required reporting

Every SCF timing artifact must retain, per repeat:

- the exact frozen warm-start density policy;
- SCF iteration count;
- final energy change and backend-native convergence residuals;
- total time and separately measured SCF / force stages when available.

The summary must also publish an iteration-branch histogram for each engine.
The checker requires nonempty per-system records and nonnegative integer iteration
counts without coercion. At least two warm observations and explicit successful
convergence of every item are necessary for a stability verdict. Missing legacy
convergence flags or a single observation remain diagnostic-only. A passing
stability check does not replace independent energy/force correctness gates.

## Performance claims

1. A normal-convergence run remains the user-facing end-to-end benchmark. It is valid for correctness and real-world latency, but a cross-engine performance ratio is **inconclusive** when the engines do not share a stable iteration branch.
2. Do not fall back from a missing iteration-matched comparison to an ordinary median for a performance gate.
3. A headline cross-engine SCF ratio requires either:
   - a shared iteration branch with enough repeated samples, or
   - a separate fixed-work benchmark in which both engines execute the same declared number of SCF/Fock updates.
4. Fixed-work measurements are performance diagnostics only. They do not replace normal-convergence numerical or force gates.
5. When an engine shows more than one iteration branch across repeated fixed-dm0 replays, mark the normal-convergence performance result as branch-unstable and publish the raw branch-conditioned samples instead of pooling them.

## Fixed-work follow-up

Add a benchmark mode that executes the same declared number of SCF/Fock updates from a frozen engine-local density on both backends. Record the exact work count and residuals. Keep force timing separate unless the force calculation is evaluated from a scientifically valid converged state.

Agent: ChatGPT
Model: GPT-5.6 Sol
