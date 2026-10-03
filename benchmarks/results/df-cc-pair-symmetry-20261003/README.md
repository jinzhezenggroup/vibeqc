# Native DF pair symmetry qualification

This evidence covers one physical factor projection before sector selection.
See [the decision note](../../../.agents/notes/implemented/numerics/2026-10-03-df-cc-source-pair-symmetry.md)
for the conditioning diagnosis, independent-oracle limitations and added storage.

Reproduce the small checks with `tests/python/test_df_cc_source_blocks.py`,
`test_df_cc_source_program.py`, `test_df_cc_native_solver.py`,
`test_posthf_prepared_source_lifetime.py`, and `test_df_cc_molecular_source.py`.
The last suite uses `GENERATIVEQC_DF_CC_SOURCE_CUDA_TEST=1` and
`GENERATIVEQC_LIBRARY`, under finite Slurm `main --gres=gpu:5090:1`.
It includes exact-budget/one-byte-short publication tests and four native H2
energy endpoints. The final binary also runs all 16 cases under CUDA memcheck.

The 230/264-AO entries supply independent converged orbitals and amplitudes;
they test native source sensitivity and expanded residuals, not cold solves.
Their original direct per-factor 3e-10 gates remain unqualified. Full ignored
logs/adapters and rejected precision experiments are retained in the isolated
worktree `.artifacts/conditioned/`; no raw arrays or new external archives are
published. The two library identities have byte-identical generated packing
CPU/CUDA source. Neither represents a native DF CCSD(T)/force qualification.
