# CUDA DF setup eigensystems

CUDA DF single and bucket RHF/UHF setup uses the prepared ordinary FP64
provider for required overlap and core-Hamiltonian eigenframes. The same
adapter serves finalization. CPU/DFT and the independent host-iterative Fock
API retain their existing reference behavior.

The initial-guess layer accepts a synchronous borrowed operation and retains
no backend object or callback. An overlap cache miss decomposes S, keeps the
historical strict `< 1e-10` singularity test, and forms the full symmetric
`U diag(lambda^-1/2) U^T`. It checks `X^T S X = I` at the existing 1e-8 metric
gate before publishing X. Compatible warm cache hits invoke no operation.
Cold UHF uses one core frame for both spins and preserves the existing beta
frontier rotation. Supplied D continues to skip the unused core solve.

A bucket validates malformed warm input before allocating a shared plan, then
constructs and checks each initial frame using its assigned plan slot. A setup
failure is recorded against the original source index. The failed fixed-stride
plan is freed before one bounded rebuild for surviving items; good detached
X/D and per-source caches remain valid. No failed initial frame is submitted
to SCF. A reduced prepared-data cache is cleared so the next call reconstructs
the complete source map. Tensor storage is released only after this process.
The existing serialized workspace allowance already covers these operations;
there is no additional provider handle or retained per-item workspace.

The private `VIBEQC_DF_REFERENCE_SETUP_EIGEN=1` control restores actual reference
setup decompositions independently of `VIBEQC_DF_REFERENCE_FINAL_EIGEN` and the
lazy-core/overlap-rebuild controls. Host ledgers report device calls by reason,
so a cold overlap/core call cannot be counted as a final solve. Preparation
work-count gates sum reference and device leaves; provider comparison checks
each side separately. Production must not silently retry a rejected device
setup through the reference eigensolver.

The #206 matrix runner exposes `--host-workloads --setup-eigen-ablation` for
five-pair comparisons with frozen warm seeds and matching iteration/retry
branches. It rejects ambient provider controls and mixed ablations. Use
`--host-trace-dir` in a separate intrusive run to verify actual setup and final
calls; those recorded intervals are excluded from clean endpoint timing.

Native tests cover callback forwarding, analytic cold densities/UHF mixing,
warm bypass, cutoff boundaries, failed replacement, actual device cutoff
handling, and a corrupted overlap in a prepared bucket followed by recovery.
Molecular tests compare setup providers for RHF/UHF, Cartesian/spherical,
batch 1/4, cold/warm/changed geometries and complete forces. Provider ablation
timing, the remaining independent host driver, and #311 final-state retention
are separate increments; this document makes no new performance claim.
