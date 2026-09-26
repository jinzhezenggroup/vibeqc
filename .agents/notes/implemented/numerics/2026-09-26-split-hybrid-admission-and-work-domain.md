# Decision: split-hybrid admission and work-domain validation

Status: implemented; full scientific/CUDA acceptance pending
Date: 2026-09-26

## Problems

The first public split-hybrid layer had two incompatible metadata interfaces:
legacy Libxc records were mappings, while a typing fix converted split records
to dataclasses and also rewrote legacy lookups. This broke module import and
left another dataclass subscript error in public functional-code resolution.

The native scaled-or-hybrid predicate did not cover a registered split pair
with no K term: unity X/C scales and absent exchange took the pure-semilocal
bypass. Both public single and batch preparation must reject that descriptor.

The split point emitter also called interior-only Libxc graphs directly on
physical grid inputs. It omitted work_mgga density screening, spin-density,
sigma and tau floors, the cross-sigma clamp, and the component FHC policy.

## Decisions

Preserve immutable mapping semantics with TypedDict annotations; do not rewrite
old catalog access. Add named-selector-to-C-ABI and legacy method regressions.

Keep descriptor validation in a DFT-owned, host-only header called by the common
registry before either preparation callback. Reuse source-generated component
and exact-exchange metadata. A registered pair must carry its nonzero canonical
full-range K term even when the caller would otherwise select a semilocal path.
Internal point experiments are not public SCF capability claims.

Generate each component's work adapter from its bound density threshold and
flags, with checked pinned work_mgga/functionals source snippets and hashes.
Keep raw differentiation inside the work boundary: Libxc vxc is evaluated at
work inputs; it is not automatic differentiation of the clipping operations.
Energy density uses original total density times epsilon evaluated at work
inputs. Exchange and correlation retain separate work thresholds/policies.
The point expression identity now includes both work policies and their source
identity. Laplacian independence uses exact zero, not a lossy float conversion.

No additional 0.46 scaling is applied to M06-2X semilocal exchange: its pinned
polynomial already includes this weight. Full-range exact exchange remains a
separate shared Fock contribution and must be counted exactly once.

## Evidence and limits

Added host-compiled descriptor and work-adapter regressions and an independent
Libxc 7.0.0 point oracle runner with separate host and actual CUDA modes. It
checks 50 deterministic points per method, seven feature derivatives, interior
finite differences and restricted/polarized embedding consistency. Missing
NVCC/device/oracle, nonfinite data and mismatched numerical outputs fail the
CUDA command. It records Git/source/compiler/device identity and NPZ evidence.

During the review session the isolated emitted work adapter compiled with a
host C++ compiler and passed vacuum/floor/cross-sigma/FHC checks. Input-layout,
PSD reconstruction and nonfinite-rejection checks also ran locally. The full
repository, independent Libxc oracle, NVCC and GPU endpoints were not executed
in that session. These isolated checks must not be reported as scientific or
CUDA acceptance. Source generation and compilation alone were insufficient
qualification in the original stack.

## Required completion

Run the independent host and NVIDIA CUDA point acceptance, public RKS/UKS
endpoints with matched grids, fixed-density J/K/XC decomposition, replay/batch
and rejection tests, and Compute Sanitizer before marking public production
support ready. Do not suppress failures or relax tolerances to admit a method.

Agent: ChatGPT
Model: GPT-6 Astra Pro
