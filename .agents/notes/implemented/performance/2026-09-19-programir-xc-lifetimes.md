# Decision: prove serial ProgramIR with one XC boundary-lifetime optimization

Status: implemented
Date: 2026-09-19

## Problem

Issue #460 asks whether a small composition graph can deliver useful dependencies
and lifetime information across existing compiler/runtime subsystems. Expanding it
immediately into an XLA-like optimizer would add unvalidated complexity.

At upstream `5a7fdeb2689553c0a304dad3338ba184d850ef60`, the dense CPU
`PreparedXCContractions` consumer and its suspended `_collocation` generator both
retain the previous AO tile while allocating the next. The previous returned
potential/electron contribution is also retained after accumulation. These are
actual Python references around native computation, not a presumed GPU bottleneck.

## Decision

Add a small immutable, serial SSA/provider description in `common.program` and
reuse #203 resource records to derive inclusive last-use intervals. Keep aliasing,
in-place writes, asynchronous leases, control flow and scientific equations out.

Describe the actual CPU fixed-density XC tile with two existing opaque providers.
The XC call already includes its density/features/coefficient/Vxc operations; do
not manufacture extra kernel stages just to make a graph look more general.
The complete borrowed quadrature owner is accounted once, not its slice views.

Have the existing CPU template implement the recognized `jets` release after XC,
release the consumed contribution, and remove the generator's independent
reference before the next tile allocation. Preserve full conservative admission
budgets; the graph describes only boundary ownership, not provider interiors.

## Rejected alternatives

- A new generic allocator or Python graph executor: existing runtime/template
  ownership is sufficient for this slice.
- Deleting AO jets immediately after feature formation: Vxc still needs them.
- Treating a single consumer as permission to fuse: provider legality, live
  resources and execution evidence must be established independently.
- Calling the retain-all graph comparison a native memory/speedup measurement:
  it is a diagnostic model, not the old implementation.
- Applying the serial rule to CUDA/spatial leases or advertising whole-SCF gains:
  no such consumer or qualification is provided here.

## Invariants

No quadrature, precision, spin convention, functional expression, feature
contraction, potential assembly, reduction order, capability or failure handling
changes. Native provider checks remain authoritative. Borrowed inputs and outputs
are retained through their declared boundaries. A release follows a completed
consumer, never precedes it. Cross-subsystem buffer names must have unique owners.

## Evidence

The same CPU library was used for both the exact upstream snapshot and candidate:
SHA-256 `7dff77e0eea8e1d37f6f0987a280fd114bfee32cfe5c3eda154beb31cfc4f76e`.
Python 3.13.9, NumPy 2.5.1; OpenBLAS and OMP threads were both set to one.
No native library rebuild or real-device CUDA execution is claimed.

Using `benchmarks/programir_xc_lifetimes.py`, PBE, 32 fixture points, tile capacity
7 (five tiles, including a partial final tile):

| Fixture | AO count | Previous boundary bytes before tiles 1..5, baseline | Candidate |
| --- | ---: | --- | --- |
| H2 | 2 | 0, 528, 528, 528, 528 | 0, 0, 0, 0, 0 |
| f-spherical | 16 | 0, 7696, 7696, 7696, 7696 | 0, 0, 0, 0, 0 |

These bytes are the payloads of still-live returned AO/potential/electron arrays,
not allocator residency or total endpoint peak. Instrumentation is excluded from
timings. Eleven ordinary warmed complete E/V executions gave medians of
5.519/5.513 ms (H2 baseline/candidate) and 5.699/5.690 ms (f-spherical). These small
cross-run differences establish **no significant speedup**. Both versions execute
five scalar calls, five coefficient calls and thirty logical matrix products.
Independent stored energy/potential fixtures pass for both versions.

Final combined validation passed 199 tests, including 64 new tests. The two
new planning modules reached 100% statement and branch coverage (146 statements,
70 branches). The compiler structure audit checked 189 modules with zero errors. Dedicated tests
also check lifetime boundaries, exported/borrowed values, overflow, deterministic
replay, malformed/cyclic/in-place graphs, repeated/changed density and native
non-potential isolation.

```bash
PYTHONPATH=python:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q \
  tests/python/test_program_ir.py tests/python/test_program_ir_xc.py \
  tests/python/test_xc_contractions_native.py tests/python/test_xc_contractions.py \
  tests/python/test_xc_integration.py tests/python/test_compiler_structure.py \
  --cov=vibeqc_compiler.common.program --cov=vibeqc_compiler.xc.program_ir \
  --cov-report=term-missing
PYTHONPATH=python:. python tools/check_compiler_structure.py
```

## Consequences and revisit conditions

This supplies one small, real composition/lifetime slice without changing GPU or
SCF scheduling. It does not prove a large memory or latency benefit. Keep it narrow
unless another consumer demonstrates a concrete need for explicit aliases,
additional provider boundaries or asynchronous regions. Issue #460 remains open;
this is not completion of a universal optimizer.

## References

- #460: minimal ProgramIR experiment.
- #203: existing resource ownership and interval accounting.
- #168: independently qualified DFT schedules/tuning.
- #350: native runtime ownership; #370: separate device-iteration work.
- `docs/program_ir.md` and `benchmarks/programir_xc_lifetimes.py`.

Agent: ChatGPT
Model: GPT-6 Astra Pro
