# Decision: Keep higher-l four-center components outside the fixed production catalog

Status: implemented
Date: 2026-09-22

## Problem

BASIS02 requires selected four-center higher-angular-momentum code generation,
but the production direct-HF catalog is a stable 55-class s–f contract with
64-bit masks and architecture manifests. The compiler already had a generic
`ShellSignature`, while `build_shell_class_component_kernel` still required a
legacy `ShellClassSpec`. Extending the fixed catalog merely to make one g-shell
qualification case compile would couple mathematical availability to production
admission and grow default artifacts without endpoint evidence.

## Decision

The compiler-owned scalar four-center component builder accepts either the
legacy `ShellClassSpec` or a generic four-slot `ShellSignature`. Component labels
are validated from each signature's Cartesian angular momentum. The explicit
bounded CPU/CUDA emitter accepts these generic signatures through g (`l=4`) for
full-range ERI values and first nuclear derivatives.

This does not change the native CUDA direct-HF contract. Generic higher-l
signatures receive no legacy class name, no fixed mask bit, no production
manifest row, and no default AOT compilation unit. Raw bounded emission requires
Cartesian primitive signatures; spherical transforms and arbitrary public-weight
pullbacks remain caller-owned. Native/public CUDA admission therefore continues
to fail closed for g.

## Rejected alternatives

- Expanding the default `FUSED_SHELL_SPECS` limit from f to g would grow the
  canonical catalog from 55 to 120 classes before any workload proved that those
  classes were useful. That would also require a mask/catalog versioning design
  rather than silently reinterpreting the existing 64-bit ABI.
- Inventing an out-of-catalog `ShellClassSpec` such as `gsss` would give a generic
  qualification object a production-looking identity and make later mask/table
  indexing easier to misuse.
- Applying real-spherical transforms inside the scalar recurrence emitter would
  duplicate the existing basis transform/normalization ownership and obscure
  arbitrary-weight pullback semantics.

## Invariants

- Input shell angular momentum, recurrence order, derivative order, operator
  family, basis role, representation, and backend capability remain independent.
- The 55-class s–f catalog, its masks, profiles, and default binaries retain their
  existing identities.
- Generic g four-center codegen is explicit and selected-component only; it does
  not imply native CUDA HF or public molecular admission.
- Spherical transforms, primitive contractions, normalization, and public-weight
  pullbacks remain outside the raw scalar primitive emitter.
- Unsupported signatures fail before fixed-table indexing or device execution.

## Evidence

Measured implementation commit: `1c551ddb46bdf377fcae573eb5e9ca746af0d081`, based on
`9d6d44236e4a09205afbdd1cbcc6f43b8014084d`.
The final synchronized PR candidate is `af5313627c178ab4a1cd791093dff8dc6e5f6c92`.
A final-head GPU rerun was requested but could not be scheduled: the dedicated
4090 restart and fresh 4090/H200 notebook requests all remained pending under
current qz node-memory/priority constraints and were stopped. The measured
implementation and final candidate are byte-identical in
`bounded_component.py`, `shell_class.py`, and `test_high_angular.py`; their
SHA-256 values are retained in the benchmark snapshot. This establishes source
equivalence, not a claimed rerun.

- Independent CPU g-s-s-s primitive value and all 12 center derivatives matched
  PySCF/libcint 2.14.0 at two changed geometries; translation and arbitrary
  signed weight contractions passed.
- The complete high-angular CPU suite with a fresh Release native library passed
  38 tests with 6 CUDA-only skips. Loaded orbital-g RHF and g-auxiliary DF-RHF
  energy/analytic-force endpoints remained inside their existing reference gates.
- On an allocated RTX 4090 (sm_89, driver 550.163.01, CUDA 12.9.86), six selected
  bounded high-l CUDA numerical tests passed. The new g-s-s-s component compiled
  in 2.074 s to a 1,210,768 B shared object; PTXAS used 128 registers, a 48 B
  stack frame, zero spills, 0 B shared memory and 0 B local memory.
- Existing g/g nuclear-attraction evidence remains spill-heavy (255 registers and
  a 2544 B stack frame), reinforcing that structural correctness must not trigger
  mechanical production promotion.

Detailed machine-readable evidence is under
`benchmarks/results/high-angular-170-four-center/`.

## Consequences

Higher-l four-center mathematics can be qualified independently on CPU and CUDA
without inflating the default binary or consuming the fixed mask namespace.
Production code must continue to use separately admitted s–f classes until a
versioned catalog/runtime path and complete endpoint evidence justify broader
admission.

## Revisit when

Revisit production g four-center admission only when a real loaded-basis CUDA
molecular endpoint demonstrates numerical parity, bounded resources, and a
measurable complete-endpoint benefit for a target workload. If more than 64
production classes are required, design an explicit catalog/mask version rather
than changing the current mask interpretation in place.

## References

- GitHub issue #170
- PR #270 and PR #879 for the earlier g-shell capability slices
- `docs/high_angular_momentum.md`
- `benchmarks/results/high-angular-170-four-center/`
