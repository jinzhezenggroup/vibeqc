# Decision: share semilocal scalar lowering between DFT CPU and CUDA

Status: implemented
Date: 2026-09-23

## Problem

The native DFT CPU generator and the native CUDA r2SCAN generator already started
from the same audited FunctionalSpec, but they independently owned derivative-root
construction and ScalarCEmitter plumbing. In particular,
tools/generate_xc_r2scan_cuda.py imported tools/generate_xc_cpu.py to reuse
build_roots while still rebuilding the device ABI locally. That made a tools
entry point part of the scientific dependency chain and left room for CPU/CUDA
expression drift.

LDA/PBE are a different existing case: their ordinary-stream CPU/CUDA point
contract is already shared through src/dft/xc_point.hpp. This decision addresses
the generated semilocal path first; it does not rewrite that qualified numerical
policy in the same change.

## Decision

Move FunctionalSpec -> canonical scalar Graph -> requested derivative roots and
the common polarized GGA/MGGA scalar emitter into
python/vibeqc_compiler/xc/semilocal_codegen.py.

Host and CUDA AOT entry points consume this compiler owner. They may choose ABI
names, value-struct names and function qualifiers, but those target choices do
not participate in the mathematical expression identity.

Keep the existing r2scan_device symbol and generated_r2scan_device.cuh boundary
as a thin CUDA ABI wrapper. The native ao_cuda.py consumer therefore continues
to execute its already-qualified call path while its scientific point body now
comes from the same compiler lowering used by the CPU generator.

CMake records semilocal_codegen.py as an explicit input of both generated
artifacts so incremental builds cannot retain stale CPU or CUDA source after the
shared compiler owner changes.

## Rejected alternatives

- Copy the generic CPU MGGA emitter into a new generic CUDA script. This would
  preserve the duplicate ownership problem under a broader filename.
- Rename the native r2SCAN ABI and header immediately. Removing a method-shaped
  symbol is not itself scientific unification and would create unnecessary ABI
  and build churn.
- Fold LDA/PBE numerical-tail code into this change. Their shared point policy
  has separate numerical history and response consumers; migrating it should be
  independently qualified rather than coupled to the r2SCAN ownership cutover.
- Make CPU and CUDA emit identical scheduling/runtime code. Only the scientific
  scalar expression is shared; backend execution and resource policy remain
  target-owned.

## Invariants

- CPU and CUDA wrappers for the same FunctionalSpec/production policy have the
  same canonical expression identity.
- Target ABI spelling and __device__ qualification cannot alter scientific
  identity.
- Tool scripts do not own XC differentiation or import one another for
  scientific semantics.
- r2SCAN production-domain continuation and vtau-to-weak-form handling remain at
  their existing qualified boundaries.
- This refactor does not grant SCAN, WB97M-V, or bulk Libxc registrations new
  CUDA, SCF, force, response, or public capability.
- Independent Libxc/molecular references remain authoritative; CPU-vs-CUDA
  agreement alone is not an independent oracle.

## Evidence

The change adds structural tests that compare the actual CPU and CUDA r2SCAN
wrapper expression identities, exercise the same compiler emitter for PBE,
SCAN, and r2SCAN feature widths, and prohibit the CUDA tool from reintroducing
ScalarCEmitter/build_roots ownership.

Existing native DFT CUDA tests continue to compare the executed r2SCAN energy
and potential against the CPU native endpoint; build/CI evidence is recorded on
the implementation PR.

## Consequences

Adding another rho/sigma/tau semilocal device evaluator no longer requires a
second differentiation/code-emission implementation. Backend-specific
integration, residency, fusion and scheduling can still specialize independently.

A later DFT consolidation slice may move the retained LDA/PBE shared numerical
point policy behind compiler-owned generated artifacts, but only with its own
boundary/response qualification.

## Revisit when

Revisit the thin r2scan_device ABI wrapper after native CUDA dispatch is driven
entirely by a generic semilocal artifact registry and removing the symbol
reduces real maintenance cost without invalidating qualified AOT consumers.

## References

- #926 M1 CPU/CUDA DFT convergence
- #396 DFT MethodIR/compiler composition
- #1087 generic host semilocal/MGGA lowerer
- tools/generate_xc_cpu.py
- tools/generate_xc_r2scan_cuda.py
- python/vibeqc_compiler/dft/ao_cuda.py

Agent: ChatGPT
Model: GPT-5.6 Sol
