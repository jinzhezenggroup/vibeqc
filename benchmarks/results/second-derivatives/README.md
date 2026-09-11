# Experimental partial-Hessian and HVP shell endpoints

These bundles accept numerical correctness of selected CPU/CUDA second
integral derivatives at `9f120deb8825a8272d3fbdb5c79fae4980fbd064`.
They cover fixed external weights and directions. Electronic response,
molecular Hessian assembly and production/performance promotion are outside
this acceptance.

Each backend covers nine cases: S/T/V pd, attraction fs, ERI psss/dpsp/fsss,
coincident dpsp, and one coordinate/component f/f/f/f tile. Every case uses
three signed primitive contributions. Raw Hessian tiles, weighted Hessian
tiles and HVPs use identical inputs and independent Libcint analytic blocks
through PySCF 2.14. First-gradient central differences use steps 1e-3, 3e-4
and 1e-4 Bohr. All HVP coordinates are retained except for the explicitly
minimal f/f/f/f tile.

| Backend | Maximum analytic absolute error | Numerical result |
| --- | ---: | --- |
| CPU | 1.1991e-14 | Pass |
| RTX 5090 CUDA | 1.1991e-14 | Pass |

Analytic gates use absolute tolerance 5e-11 and relative tolerance 2e-11.
Translation uses 5e-12 times `max(1, max(abs(H)))`. Final finite-difference
error must be below 2e-7 times that scale and below either 3% of the initial
error or the 5e-11 scale-aware roundoff floor. The quantitative gates and
actual/reference arrays remain in `evidence.json` and `samples.json`.

Five warm samples, cold execution, preparation and compiler durations are
retained for each raw/weighted/HVP provider. These diagnostic timings exclude
reference construction and public weight preparation. CPU and CUDA runs
overlapped on the host; these data do not establish comparative performance.
`artifacts.json` records source/binary identities and compiler resource
reports. `resources.json` records shared numeric memory plans, checked against
native allocation during preparation. Caller inputs, Python/compiler
metadata, call stacks and CUDA context are explicit exclusions; process-wide
peak memory was not measured.

Reproduce from the recorded revision with the argv in each `publication.json`,
NumPy/PySCF installed, and the recorded compiler on PATH. CUDA execution uses
the finite Slurm allocation and preserves assigned device visibility. Both
manifests and every attachment checksum are validated by the shared
publication API. No compiled binaries or transient logs are required.

Separate automated tests cover mathematical mixed-partial symmetry and
two-index translation recovery, arbitrary rotations, shell permutations,
contracted Cartesian/spherical transformations, packed factors, atom chain
rules, incomplete-input rejection, bounded partial/empty chunks and replay
after failure. CPU ASan/UBSan with leak detection and CUDA memcheck passed
for attraction HVP, dpsp HVP and the f/f/f/f raw tile; all CUDA probes reported
zero errors and zero leaked allocations.
