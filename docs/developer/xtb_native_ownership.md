# Native xTB execution and compiler ownership

`Calculator(method="gfn2-xtb")` executes the native molecular GFN2 runtime in
`src/xtb/native/`. Its private descriptor types and CPU/CUDA implementation use
VibeQC identity. The imported xTBloom C entry-point declarations, version header
and `include/xtbloom/` interface are retired. The public VibeQC ABI is unchanged.

The private execution adapter accepts one molecule, fresh SCC, energy and
analytic forces. It rejects periodic cells, external charge/response/field
attachments and other property flags before staging any input bytes. The
unsupported ALPB model, native periodic topology/integrals/Ewald/multipole
owners, standalone CUDA request/plan API, diagnostic snapshots, DLPack/result-arena API and CPU batch
worker/checkpoint infrastructure are removed. The internal SCC graph and its
bounded fallback remain part of CUDA execution.

## Scientific ownership

The compiler emits the following production mathematics. Backend owners retain
ragged storage traversal, validation, scheduling, accumulation and publication.

| Science | Compiler owner | Production consumers |
| --- | --- | --- |
| CN and repulsion | `geometry/gfn2_pair.py` | CPU and CUDA geometry/classical terms |
| S/D/Q Cartesian primitives | `integral/gfn2_sdq.py`, `integral/gfn2_sdq_cpu.py` | CPU and CUDA integral values/coordinate response |
| Electronic pair Hamiltonian and S/D/Q adjoints | `method/gfn2_electronic_runtime.py` | CPU and CUDA electronic owners |
| ES2, ES3 and AES2 | `method/gfn2_es2_runtime.py`, `method/gfn2_es3_runtime.py`, `method/gfn2_aes2.py` | CPU and CUDA electrostatics |
| H0 shell factors, CN/radial/Cartesian adjoints and AO adjoint updates | `method/gfn2_h0_force_runtime.py` | CPU H0 values/VJP and CUDA H0 values/forces |
| Shell spin energy and potential | `method/gfn2_spin_runtime.py` | CPU and CUDA spin owners |

Paths in this table are relative to `python/vibeqc_compiler/`. H0 value and force
consumers share `generated_gfn2_h0_native.hpp`. Spin consumers share
`generated_gfn2_spin_native.hpp`; generation preserves the ordered FMA
accumulation and records the reverse-AD identity of the symmetric shell energy.
Restricted zero-output admission remains backend policy. The public method's
restricted/shared-orbital open-shell capability is unchanged; generated spin
science does not admit a new public unrestricted endpoint.

SCC iteration/mixing/convergence, occupations, generalized eigensolver provider
selection, workspace/cache lifetime, per-system errors and public method
admission remain native runtime responsibilities. Generation needs no installed
VibeQC runtime, GPU, or scientific oracle.

## Remaining native scientific work

Compiler ownership is not complete. Integral contraction/representation
transforms and multipole translation, parameter/basis binding, the D4 SCC
charge-response hot loop, and optional interaction primitives shared with the
remaining lower-level descriptor machinery still contain native scientific
arithmetic. These
need their own generated replacements and independent gates. Molecular final
D4 derivatives already reuse the shared VibeQC D4 provider.

The CUDA ownership ledger continues to count remaining handwritten science;
renaming or relocating a source is not a scientific retirement gate. Public
ragged-batch, CUDA wheel and complete endpoint performance acceptance also
remain separate from compiler replacement.

## Provenance and qualification

Upstream parameter snapshots, licenses, oracle attribution and revision IDs
retain their historical xTBloom names. In particular, the GFN1 header generator
checks its native output against the original audited digest after reversing
only the namespace substitution. Scientific table bytes and method parameter
identities are unchanged. `src/xtb/native/CUDA_SOURCE_PROVENANCE.json` retains
upstream hashes and records the current adapted source hashes.

Native CPU execution requires LP64 OpenBLAS with LAPACKE; a development build
can set `VIBEQC_XTB_CPU_LINALG_LIBRARY` to the provider's absolute path. Wheel
builds retain their pinned private OpenBLAS provider and native shim. The former
`XTBLOOM_CPU_LINALG_LIBRARY` build setting has been retired.

Relevant gates are `test_gfn2_h0_force_codegen.py`,
`test_gfn2_spin_native_codegen.py`, `test_gfn2_runtime_bridge_boundary.py`,
`test_gfn2_xtb.py`, and `test_gfn2_xtb_force_qualification.py` under
`tests/python/`. GPU endpoint tests require `VIBEQC_TEST_GFN2_CUDA=1` inside a
Slurm allocation on `main` with `--gres=gpu:5090:1` and a finite time limit.
The additional compiler graph CUDA gates use `VIBEQC_GFN2_CUDA_TEST=1` in the
same allocation; these are separate from the native endpoint opt-in.
Run `python tools/check_compiler_structure.py` and
`python tools/report_cuda_ownership.py --check` for ownership validation.

The rationale and retained boundaries are recorded in the
[retirement note](../../.agents/notes/implemented/architecture/2026-09-22-xtb-native-compiler-retirement.md).
