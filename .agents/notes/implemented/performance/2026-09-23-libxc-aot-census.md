# Decision: bounded offline Libxc AOT census before broad packaging

Status: implemented
Date: 2026-09-23
Agent: ChatGPT
Model: GPT-6 Astra Pro

## Problem

Issue #1123 separates compilation economics from the mathematical admission work
in #1119/#1120. A catalog import does not justify eagerly compiling every
registration, spin, derivative order and target. Alias-specific import identities
must not force duplicate builds of identical emitted programs, but collapsing
those identities would destroy scientific provenance.

## Decision

`vibeqc_compiler.xc.bulk_aot` and `tools/census_libxc_aot.py` implement the initial
measurement and planning slice, not a runtime executor or installable package.

- Reuse `BulkProgram.roots`, `Graph.analyze_ssa` and the existing C/CUDA emitters.
  No formula, scalar algebra, numerical-domain policy or optimizer is changed.
- Preserve each registration's import identity. Group build tasks only when
  emitted source bytes, feature/output ABI, domain, spin, derivative order and
  backend match exactly. The resulting **emission identity is not a binary cache
  key** and does not carry numerical qualification.
- Request E/vxc by default. Other derivative orders require explicit selection.
  Apply the existing deterministic symbolic-work budget to each requested
  registration/spin/order/backend. Record exhaustion instead of inventing output.
- Stream source generation. Retain catalog metadata, not every source string.
  Select unique build tasks deterministically within artifact-count and total
  source-byte budgets. Rebuild only selected representatives, verify both import
  and emission identities, and measure regeneration separately from compilation.
- Compilation is opt-in (`--compile`) and offline. CPU object compilation and CUDA
  object/resource probes do not load the QC runtime, execute a GPU kernel, or
  grant SCF/force/public support. A CUDA probe entry point keeps the point function
  observable; its ptxas resources are not complete KS endpoint resources.
- Missing compiler/resource observations remain unavailable/null, not zero or
  successful. Failed/timed-out requested probes make the CLI return nonzero.

No persistent binary cache is introduced. The diagnostic recipe includes emitted
translation-unit hash, compiler executable hash/version and flags/target, but it
does not fingerprint the entire header/libdevice/downstream-toolchain closure.
It must not be repurposed as a cross-run cache key without that additional work.

## Rejected alternatives

Grouping by registration/import identity would duplicate alias builds. Grouping
by family or assuming algebraic equivalence would be unsafe. Retaining all source
strings would make the census itself scale poorly. A device-function-only CUDA
translation unit could yield misleading near-empty objects due to elimination.

## Evidence and reproduction

Unit gates cover identity changes, provenance separation, deterministic budgets,
streaming lifetime, actual C object compilation, failures/timeouts, unknown CUDA
resources, one build per selected alias group and fail-closed census observations.
Small integration gates use existing LDA/GGA/meta-GGA registrations and compare
emitted sources byte-for-byte to their existing owners. Full-catalog sweeps are
opt-in rather than a new expensive default CI test.

```sh
python tools/census_libxc_aot.py --output /tmp/libxc-census.json
python tools/census_libxc_aot.py --name LDA_C_VWN_4 --backend cpu \
  --compile --max-artifacts 2 --output /tmp/libxc-cpu-probes.json
python tools/census_libxc_aot.py --name GGA_X_PBE_SOL --backend cuda \
  --compile --cuda-arch sm_89 --output /tmp/libxc-cuda-probes.json
```

The JSON contains the catalog hash, selected request, graph/SSA/source metrics,
per-registration provenance, exact-source groups, budget blockers, generation
observations, and optional object/ptxas measurements. Timings are observations,
not inputs to scientific or emission identity.

## Remaining scope / revisit when

Keep #1123 open for the retained full-catalog hardware census, production AOT
packaging integration, persistent cache closure/invalidation, and production CUDA
resource/numerical gates. Lane #1124 may consume these compilation observations
but must independently verify numerical/backend/endpoint qualification.
