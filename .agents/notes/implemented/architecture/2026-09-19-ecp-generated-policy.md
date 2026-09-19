# Decision: complete the ECP CUDA scientific-ownership audit

Status: implemented
Date: 2026-09-19

## Problem

The ECP CUDA adapter called generated AO, projector/radial, weighted derivative,
grid/harmonic and normalization helpers, but still specified the two integration
grids and the finite absolute-error convergence predicate itself. Its entire
mixed translation unit remained conservatively counted as scientific. The HF
workspace estimate also repeated the refined-grid dimensions.

## Decision and invariants

The compiler's `integral/ecp_policy.py` owns the fixed 160/32 and 224/44 grid
orders and emits the value/derivative acceptance predicate into the existing
shared host/device ECP header. Resource planning imports the refined dimensions.
The thresholds stay 2e-9 for matrix values and 2e-8 for derivatives. Nonfinite
coarse or fine inputs reject; equality at the threshold accepts. FP64
subtraction, node ordering, quadrature, reduction order and buffer sizes stay
unchanged. These control predicates use standard C++ math/comparison lowering;
they do not require a second algebra or differentiation implementation.

Raw exports retain their caller-specified grids and no hidden convergence gate.
Complete CUDA calls consume both configured grids, reject failure before
publishing contributions, and use the refined result on success.

## Audit of the remaining adapter

| Native responsibility | Scientific source or contract |
| --- | --- |
| AO record expansion and primitive upload | Shared molecular basis representation; generated component normalization |
| Grid construction dispatch | Generated Gauss-Legendre, mapped radial, sphere and harmonic helpers |
| AO evaluation/projector/pair launches | Generated ECP AO, projection and local/nonlocal contraction helpers |
| Ordered radial loops, symmetry and atom scatter | Output layout and physical-center mapping; no residual/projector formula |
| Hcore/force dispatch | Generated operator addition and fixed-weight derivative consumer |
| Two-grid admission dispatch | Generated finite absolute-error predicate and grid constants |
| Arena, stream, copies, retries and status mapping | Native runtime; optional radial staging retains bounded OOM fallback |

Consequently `ecp_cuda.cu` is now runtime in the semantic ownership ledger.
The report must separate this reclassification from physical source retirement.
Retaining adapter lines does not count as deleting scientific implementations.

The independent `src/integrals/ecp.cpp` remains the public CPU oracle/fallback,
including its own quadrature, harmonics, normalization and convergence checks.
It deliberately does not import the new generated acceptance predicate. It is
not an accidental duplicate awaiting automatic removal: independence is required
by #171. Revisit replacement only if an equally independent oracle and explicit
CPU production replacement have their own complete acceptance evidence.

## Validation and limits

The host projector test and a native CUDA test consume independent literal
boundary cases: adjacent representable values below/at/above both tolerances,
signed residuals, value-versus-derivative classification, equal finite extremes,
subtraction overflow, signed zero, subnormal and NaN/Inf inputs. The tests do not
obtain their expected thresholds from the generated policy. Existing native
numerical-failure/partial-allocation/recovery tests exercise the production path.

Independent CPU/Libcint and CUDA matrix/derivative/HF force and resource
qualification is retained in the [matched evidence](../../../../benchmarks/results/ecp-policy-171/README.md).
The measured candidate is `f10e7346839f059521c4a4800ae1a0655e21f1fd` against
`2bf9a2a4784fc79f6048ec9c4e74701e1caa92ec`. CPU native tests pass 2/2, CUDA
native tests 4/4; Python passes 101 CPU checks (17 explicit CUDA skips), 76
CUDA ECP checks and all 33 resource checks with CUDA enabled. The CUDA-only
selection deselects 88 CPU cases; these are not reported as CUDA passes.
All three Compute Sanitizer checks pass with zero errors, including both
complete RHF/UHF replay cases. Removing only the emitted policy fragment leaves
the baseline generated header byte-identical. Matched resource peaks are
unchanged; complete-HF energy/force errors remain within the independent gates.
Downloaded evidence verifies 51 raw members and 1,460 baseline/1,463 candidate
source inputs against exact Git blobs, resolving the recorded symlink target.

The first CuMetal CI build rejected the standard-library `std::fabs` wrapper as
host-only when the generated predicate was called on device. Use unqualified
`fabs`, matching the existing emitted arithmetic, to resolve the CUDA device
overload. This preserves the absolute-error predicate and its independent
boundary cases; do not infer device-callability from the NVIDIA build alone.
The fix at `37d3c1230d8a6db08952aceac73d455e82bd4a32` passes the CuMetal CI
lane and NVIDIA sm_120 compilation. Separate NVIDIA requalification retains new
source/header/library identities: CPU native 2/2, CUDA native 4/4, 101 CPU
Python passes, 10 focused CUDA ECP passes, 33 resource passes and zero errors
in all three sanitizer runs. Resource peaks remain identical and independent
endpoint gates pass. The [follow-up evidence](../../../../benchmarks/results/ecp-policy-171/compatibility/summary.json)
preserves the distinction from the original measurement.

Physical ownership changes are 7 removed scientific-classed lines and 7 added
runtime lines, plus 283 unchanged adapter lines reclassified as runtime. The
ledger's scientific -290/runtime +290 is not a 290-line physical deletion.
This migration does not enable DFT/ECP forces, enlarge angular/element/resource
domains, remove the CPU oracle, or establish a performance improvement.
Refs #171 and #349.

The prior [AO/weight ownership decision](2026-09-16-ecp-ao-weight-consumers.md)
and [host-grid decision](2026-09-17-ecp-host-grid.md) retain their historical
scope and measurements; this audit supersedes their conservative mixed-adapter
classification after moving the remaining convergence predicate.
