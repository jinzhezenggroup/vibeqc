# Device final-frame snapshot qualification

Implementation source: `256649d22ce01d6fa17ead296d5076cb00540b9b`. The manifest pins the matching scientific
source identity, CUDA library, snapshot test executable and CPU library hashes.

The final Slurm job 9491 passes six GPU native suites, 145 GPU Python checks and
snapshot memcheck with zero errors/leaks. CPU qualification passes 27 native
suites and 31 Python resource/ownership checks (one CUDA-only skip). All hooks
pass. The retained Slurm record identifies `main`, one RTX 5090 and a finite
20-minute allocation.

The tests cover RHF/UHF batch one/four, analytic nonidentity metrics, early
inactive neighbors, poisoned scratch, distinct spin storage, stale identities,
changed exchange policy, failed/nonconverged replay, corrupt device generations
and solver info, epoch saturation and recovery. The complete existing DF
energy/force and resource regression suite is included.

`qualification.zip` stores 13 original logs/commands in 6,988 bytes.
Every member was restored and compared byte-for-byte. The two initial resource
expectation failures and their focused resolution are retained separately from
the final complete pass; their initial binary was not separately pinned.

This slice retains candidates and budgets their storage. It does not yet route
production finalizers through verified selection or claim endpoint savings.
