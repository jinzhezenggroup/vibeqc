# Decision: Bound scalar ECP orbital support at f

Status: implemented
Date: 2026-09-17

## Problem

The molecular basis layer already represents Cartesian and real-spherical f
shells, but the ECP IR, generated AO/normalization tables and public preflight
stopped at d. Raising only the public check would admit incomplete generated
arithmetic. The base-four component dispatch also needs an explicit domain
check: the unsupported g component `(0,0,4)` otherwise aliases p `(0,1,0)`.

## Decision

Extend the existing scalar Gaussian DAG and generated Cartesian normalization
through f. `ECP_MAX_ORBITAL_ANGULAR` owns the compiler bound and emits the native
constructor limit. Preserve the generic molecular f spherical expansion, whose
largest expansion contains three Cartesian terms and fits existing ECP AO
metadata. Reject out-of-domain powers before encoding their dispatch key;
invalid generated calls produce nonfinite jets, never plausible zero values.

Python and C preflight apply the orbital limit to all atoms in a mixed ECP and
all-electron system. Nonlocal projectors remain s/p/d, local labels through f,
radial powers 0..4, and complete methods direct RHF/UHF. This does not qualify
g orbitals, f projectors, additional element families, spin-orbit, ECP density
fitting, MP2 or DFT.

## Rejected alternatives

Do not generate new spherical algebra or a second ECP recurrence family: the
established Cartesian DAG and molecular expansion already supply the required
mathematics. Do not extend projector harmonics simply because the orbital bound
increases; these are independent capabilities with different validation needs.
Do not replace the independent CPU setup or integral loops with generated code.

## Invariants and evidence

Keep FP64, normalization/order conventions, primitive/component accumulation,
one-radial-shell staging, physical A/B/C scatter, arbitrary real AO weights and
the complete-HF two-grid policy. The existing CPU ECP implementation remains an
independent oracle/fallback. Generation still works with `python -I -S`.

Native tests independently check f polynomial derivatives and Cartesian
normalization, reject g dispatch aliases, and exercise f/g constructor bounds
on both ECP and all-electron atoms. The native wrapper test also preserves
quadrature shape, area and append semantics.

`test_ecp_f.py` adds contracted f shells to asymmetric NaH fixtures, tests both
representations/backends, and compares matrices with normalized Libcint values.
All-center derivatives and arbitrary nonsymmetric weights use two independent
finite-difference steps. A separate detached ECP center exercises f orbitals
with d projectors. Complete RHF/UHF energies/forces use PySCF; CUDA complete
energies also undergo directional finite differences. Planned-budget prepared
execution checks changed-geometry replay and the device allocation ledger.
The default CPU/CUDA replay places f on the all-electron H atom (16 public AOs).
CUDA resource inventory v1 still rejects more than 16 public AOs; the test also
checks that explicit rejection for the two-f-center fixture. Larger CUDA f
endpoints qualify complete energies/forces without claiming budget support.
The much more expensive two-f-center CPU replay is retained behind
`VIBEQC_ECP_LARGE_CPU_TEST=1`; its initial interrupted run is not a qualification
claim. Raw all-center tests still exercise both f centers on both backends.

Acceptance gates are 2e-9 Eh for raw matrix comparisons/refinement, 2e-8 for
derivative refinement, 3e-7 absolute/3e-6 relative for raw finite differences,
2e-8 Eh for complete energies and 2e-6 Eh/bohr for complete forces. These checks
qualify the bounded fixtures; empirical two-grid agreement is not a universal
quadrature error bound. The [retained qualification](../../../../benchmarks/results/ecp-f-171/README.md)
contains exact source/library identities, raw logs, sanitizer results, matched
existing-domain timings and f endpoint/resource measurements.

## Consequences and revisit conditions

Generated source grows; native scientific CUDA and runtime LOC do not change.
No native path is retired and no speedup is claimed. Revisit broader angular
or element support only with independent raw/derivative, complete-method and
resource gates, including any increase to fixed AO expansion storage.

## References

- Issue #171; builds on PR #425.
- [Current ECP contract](../../../../docs/user/ecp.md).
- [Prior host-grid decision](../architecture/2026-09-17-ecp-host-grid.md), whose
  original s/p/d orbital boundary is superseded by this bounded extension.

The separate [f-projector decision](2026-09-18-ecp-f-projectors.md) supersedes
this note's f-projector exclusion while preserving its orbital boundary.
