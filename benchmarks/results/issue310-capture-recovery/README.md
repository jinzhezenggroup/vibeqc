# CUDA DF capture recovery and 96-atom diagnostics

This slice qualifies the capture-recovery fix in #324 and preserves every
failed or terminated large-case attempt. It does **not** contain a converged
96-atom energy or complete force result. The case is 32 waters / 96 atoms /
768 spherical AOs with def2-SVP orbital and auxiliary bases: a synthetic 2x2
array of translated WATER27 S4 octamers, not an optimized 32-mer.

The 1-GiB run failed a conservative host preparation-budget check. At 4 GiB,
768-AO overlap/core device solves passed, but XsyevBatched synchronized inside
SCF graph capture (error 900). Ending capture left error 901 in CUDA's last-error
slot. Ordinary generated J/K launches then observed that stale status. This is
the same class of provider/Graph distinction documented for Direct in #49;
ordinary cuSOLVER eligibility does not imply Graph compatibility.

The repaired owner clears only expected capture errors after capture ends,
remembers rejection, and retains ordinary device execution. Slurm 9521 passed
three native tests, 45 RHF/UHF CUDA Python regressions and zero-error/zero-leak
memcheck on the final-snapshot native test. A separate 768-by-768 tridiagonal
analytic fixture uses the same XsyevBatched owner before capture and twice
after rejection. Both post-capture solves take about 19 ms and match the
analytic spectrum within 1.33227e-15. Those are solver-only diagnostic times.

The repaired 4-GiB molecular run still did not complete cold energy within
about 9m50s. Two bounded follow-ups preserve GPU utilization and native stacks:
the successful 30/60-second samples show transformed DF tile generation and
streamed Coulomb submission, while the GPU stays near 100% utilization. These
samples do not show CPU reference diagonalization. Shape-only planning keeps
the same small streamed tiles for declared budgets through 24 GiB; a separate
resident-mode endpoint run is pending and excluded from this archive.

The original failed memcheck records six intentional/causal CUDA API errors
and zero leaked bytes; it is not a passing sanitizer qualification. Terminated
runs retain their original incomplete result files, with final dispositions
in the manifest. Profiling and concurrent host work prevent clean timing claims.
No matched GPU4PySCF, constrained-memory performance, or full #308 closure is
claimed. The archive was restored and every member compared byte-for-byte.
