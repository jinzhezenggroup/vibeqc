# GFN2 generated geometry evidence for #504

This directory records completion evidence for the geometry-only GFN2 compiler
slice: coordination number (CN), nuclear repulsion energy, and generated
coordinate VJPs.

## Scope

The generated path uses one fixed ragged TensorIR graph with a flattened atom
population, canonical undirected pair ownership, and a strict system-atom
partition. The performance case repeats the pinned 24-atom xtb repulsion fixture
32 times: 768 atoms and 8,832 retained pairs at the sharp 25-bohr cutoff.

The native comparison is pinned to xTBloom
`2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3`, the same revision recorded in
compiler provenance. The repulsion benchmark uses the same 24-atom fixture.
The xTBloom CN benchmark uses its compact synthetic geometry with the same
batch size, atoms/system, cutoff, and 8,832-pair work count. This is a
term-level replacement gate, not a complete SCC comparison.

All GPU runs used `srun --partition=main --gres=gpu:5090:1 --time=<finite>`.
The device was an NVIDIA GeForce RTX 5090, driver 580.95.05. Both projects used
the CUDA 12.9 cohort; the pinned xTBloom loader needs CUDA 12.9 ahead of the
system CUDA 12.4 runtime.

## Numerical and execution gates

Focused CPU qualification: `11 passed, 2 skipped`. It covers pinned independent
CN/repulsion values, generated VJPs, three finite-difference step sizes,
system/atom permutations, cross-system coordinate overlap, cross-system pair
rejection, and stale-topology rejection.

The compiler structure check reports `248 compiler modules; 0 dependency errors`.
The ragged real-device test executes primal, CN VJP, and repulsion-energy VJP
through TensorIR -> CUDA and reports `1 passed, 12 deselected`.

## Performance and compiler cost

The generated workload used five warmups and twenty unprofiled samples. The two
artifacts total 3,388,749 bytes of CUDA source, 4,949,560 bytes of shared
objects, and 12.0441 seconds of NVCC compilation. Maximum reported register
usage is 50 with no spill bytes.

Generated primal plus repulsion VJP sums to 2.44858 ms/batch in unprofiled
`device_ms`; complete `PreparedCuda.execute` endpoint medians sum to
523.85249 ms/batch. The complete endpoint deliberately includes Python
validation/layout staging, transfers, native execution, error checks, and
detached output allocation.

Pinned xTBloom measures 0.18152 ms/batch for CN cache reuse and 0.184 ms/batch
for repulsion energy+force, or 0.36552 ms/batch resident term timing. The
numeric ratio is about 6.70x, but this is **not a pure kernel ratio**: VibeQC
`device_ms` and xTBloom CUDA-event resident timings have different endpoint
boundaries. A section-profiled run was diagnostic only because it inserts
synchronization and is not valid for optimization selection.

Decision: the compiler slice proves scientific expressiveness and generated
derivatives, but the current generated kernels should **not** replace xTBloom's
handwritten production pairwise kernels.

Exact measurements are in `rtx5090.json`. The xTBloom CN command was
`xtbloom_cuda_pairlist_benchmark --atoms 24 --batch-sizes 32 --topology compact
--cutoff 25 --warmups 5 --samples 50`. For repulsion, only the benchmark
fixture batch changed from 512 to 32; production kernel sources were unchanged.

Agent: ChatGPT
Model: GPT-5.6 Sol
