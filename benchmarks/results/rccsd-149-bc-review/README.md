# RCCSD GPU solver review validation

Clean source `b7dcace8263b5a12b5aa9639c47679317fb71cae`, CUDA 12.9.86,
sm_120, RTX 5090. GPU execution used finite Slurm allocations on `main` with
`--gres=gpu:5090:1`, preserving scheduler device visibility.

All five pinned molecular fixtures converged with energy errors below
6e-13 Eh, independently expanded R1/R2 below 1e-10, and successful CPU
replay. Fresh native RHF/integral preparation followed by the CUDA energy
facade also passed for H2, He, H2O, NH3 and CH4; a failed middle batch item
left both neighbors converged. The complete CC test selection passed
**201 tests**, with **8 explicit opt-in skips**. The focused GPU
solver tests include real overflow, iteration-limit failure, replay, early
budget rejection and partial-preparation cleanup.

The compile-only runner compiled both plans for all five fixtures without
allocating a CUDA execution context or claiming numerical acceptance.
`summary.json` retains the numerical maxima; the archive retains every
result, replay state, source/toolchain manifest, test output and reproduction
script. Verify it with `python -m tools.unpack_evidence
benchmarks/results/rccsd-149-bc-review`.

This qualifies correctness of the experimental host-controlled facade only.
It does not complete #149 B/C: the resident iteration and native registry/API
acceptance requirements remain open.
Amplitudes, residuals and integrals still cross the host/device boundary;
there is no resident-loop, performance, force or native public-ABI claim.
