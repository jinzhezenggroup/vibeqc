# Decision: reduce DF auxiliary intermediates before T2 contraction

Status: implemented
Date: 2026-10-03

## Problem

The original bounded one-Q virtual action avoided ovvv/vvvv storage but repeated
expensive T2 consumers for every auxiliary factor. Complete 230-AO ethane CCSD
needed 688.284 s on RTX PRO 6000. Its bounded storage did not imply bounded work.

## Decision

Derive the virtual contributions to Lvv, Wvoov, Wvovo and Xv from the existing
shared RCCSD inventory. Sum each over all Q before its retained-core consumer.
Prepare amplitude-only tau once per amplitude state; keep the virtual ladder
inside Q. Prove Hamiltonian linearity before specializing the inventory, reuse
its polynomial expansion, and reject any materialized node with more than two
virtual axes. Do not introduce new scientific coefficients.

The native owner compares complete semantic contraction work and admits all
numeric storage before allocation. It retries the original bounded schedule if
the faster candidate exceeds the budget. Explicit disabling remains available.
Preparation scratch survives all Q slices, and DIIS invalidates every prepared
and accumulated value. One CUDA arithmetic flag spans prepare, Q and core.
Convergence still uses the original expanded virtual/core replay.

## Evidence

Random finite-amplitude graphs agree with determinant-space residuals, separately
written NumPy virtual equations and the expanded inventory. Auxiliary rotations
leave the full core invariant. CPU/CUDA complete solves cover ordinary,
occupied-rich and low-Q shapes, pinned molecules, exact memory admission,
one-byte-short refusal, memory fallback and prepare/Q/core overflow. Two complete
unfiltered CUDA memchecks report zero errors (Slurm 12122).

The runtime-shape schedule charges 1,037,733,074,263 scalar contraction summands
per primary ethane evaluation versus 4,501,198,326,133 previously; the formal
leading CCSD scaling remains unchanged. This count includes preparation, all
488 Q slices and the core, but excludes elementwise arithmetic. It is not a
hardware FLOP estimate. Per-Q generated operations decrease from 54 to 44 while
accumulation kernels increase from two to six.

On the identical canonical 230-AO ethane input, the complete native
supplied-Hamiltonian solve takes 262.391 s (Slurm 2160), versus 688.284 s in the
previous qualification. Both use 20 iterations, 38 primary evaluations and one
expanded replay. The independent PySCF energy error is 1.45e-12 Eh; maximum
T1/T2 errors are 8.20e-11/5.61e-12. Admitted numeric capacity increases from
2,730,069,584 to 2,920,354,384 bytes, with a tested tight-budget fallback.
The solver timer includes iterations, replay and final amplitude copy but starts
after initial owner allocation/upload. Full process wall time, including those,
binary input and amplitude output, is 690.17 s before and 264.29 s after. These
are observations on RTX PRO 6000, not a complete molecular CCSD(T)/force endpoint
speedup or a repeated timing distribution.

The [retained qualification](../../../../benchmarks/results/df-cc-auxiliary-reduction-20261003/publication.json)
contains the frozen patch, complete small-case arrays, unfiltered sanitizer logs,
large-case amplitude checksums and all acceptance statistics. Large raw outputs
remain ignored locally. Three owned historical artifacts were compressed
losslessly, with decoded identities retained, to stay within the unchanged
64 MiB aggregate evidence limit.

## Rejected alternatives and next boundaries

Reassociating each expanded Q slice independently cannot share its T2 consumers
across Q. Materializing ovvv/vvvv to recover dense consumers would reintroduce
the virtual storage problem. Reusing sums after DIIS changes amplitudes would
change the equations and invalidate convergence.

This work reduces recomputation. The complementary directions in
[#1763](https://github.com/jinzhezenggroup/generativeqc/issues/1763),
[#1764](https://github.com/jinzhezenggroup/generativeqc/issues/1764) and
[#1765](https://github.com/jinzhezenggroup/generativeqc/issues/1765) require
separate evidence: compiler-owned fusion, component-specific mixed precision,
and explicit asynchronous buffer ownership. Current factors are resident, so
this Q loop has no per-slice host copy to overlap. Future source/triples tiles
may justify a two-buffer pipeline. Keep FP64 publication/residual checks and
strict-FP64/single-buffer/unfused fallbacks when qualifying those directions.

Native molecular source, standard (T), Lambda/orbital/factor/metric response and
forces remain separate required owners. No public DF Calculator registration or
force-domain expansion is implied by this schedule.


## Additional complete 264-AO qualification

The same frozen CUDA binary completes benzene/cc-pVTZ with cc-pVTZ-RI
(o/v/Q=21/243/666), n2 Slurm 2162, exit 0. It converges in 23 iterations,
44 primary evaluations and one original expanded replay. Correlation energy
-1.0707950360409848 Eh differs from the independent same-Hamiltonian oracle
by 4.381384144380718e-12 Eh. Every amplitude passes: T1/T2 maximum differences
5.7356795094692936e-11 / 2.165781443075332e-11; expanded residual maxima
1.9587846969426614e-13 / 4.988405521988426e-13.

The solver counter reports 3696.084602716 s; process wall is 1:01:45 at the
time tool's one-second reported precision. Admission/device bytes are
17,381,701,856 / 15,769,752,320. Semantic contraction summands, including replay,
are 570,074,868,504,288. No converged previous-schedule benzene run is available,
so the earlier one-update comparison cannot establish a full-run speedup.
This extends supplied-Hamiltonian CCSD qualification, not molecular CCSD(T)
or force endpoint acceptance. The compressed qualification retains unrounded
statistics, output/amplitude hashes, source input/binary identities and logs;
the full 609,136,761-byte amplitude output remains in ignored local artifacts.
