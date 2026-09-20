# Decision: bounded s/p/d CPU stationary derivative scheduling

Status: implemented
Date: 2026-09-20

## Problem

The s/p CPU force consumer emitted one kernel for every ordered Cartesian
quartet and assumed one Cartesian term per public AO. Directly extending that
route to d gives 10,000 ERI kernels and loses real-spherical d normalization.

## Decision

Reuse the existing S/T/V and Hermite/AD ERI graphs. The compiler identifies
equivalent primitive kernels using exact ERI center symmetries and six global
axis permutations; the attraction nucleus remains the third center. The runtime
permutes primitive exponents and coordinates, then restores every derivative
slot before scattering to physical atoms. It expands the immutable native basis
record's component coefficients and streams all component/primitive products.
No consumer-weight symmetry, density symmetry, or zero-weight screening is assumed.

Retain the existing s/p executor. Select the component executor for d-containing
bases. Neither f-shell DFT nor CUDA d-shell forces are promoted. CPU ECP math
ownership is independent work in PR #648; this change does not duplicate it.

At most eight requests share a translation unit. Each unit is limited to 4 MiB,
the program to 64 MiB; pure generated sources cache at most four domains. Every
native load still goes through common source/header/toolchain/binary validation.
The public numeric staging cap excludes source strings, compiler subprocesses,
loaded code and Python objects and is not an RSS limit.

The primitive bound uses the sum of contraction length times component count,
raised to the fourth power for ERI and squared for each S/T/V traversal. Actual
records must equal admission. CPU ECP pair-sample admission increases from 100
to 200 million: NaH+d has 15 Cartesian AOs (143,400,960 two-grid pair-samples)
or 14 spherical AOs (125,475,840). The 16 AO / 8 atom / 128 primitive / 128 term
provider domain and the 2 million primitive-record cap remain unchanged.

## Rejected alternatives

One giant translation unit multiplies identical generated math and compiler
cost. Folding ERI weights at the consumer would add assumptions about arbitrary
ordered adjoints. A new hand-coded recurrence would duplicate scientific math.
Caching loaded libraries without common artifact checks would weaken provenance.

## Evidence

Full s/p/d generation produces 313 ERI representatives, 362 total requests and
46 units: 22,969,347 source bytes, largest unit 1,490,387 bytes. Local pure
generation took about 54 seconds; this is not a complete endpoint speed claim.
`test_first_derivative_schedule.py` independently checks monomial/exponent and
electron-pair mappings. `test_stationary_cpu_work.py` explicitly enumerates
component/primitive work and exact-limit rejection.

`test_ecp_public_cpu.py` qualifies all four LDA/PBE RKS/UKS combinations and
both representations against independent PySCF full-grid-response gradients
(1e-7 Eh/bohr), energies (2e-8 Eh) and two-step reconverged energy differences
(2e-7 Eh/bohr). It records full singlepoint and cold/warm/changed batch timing,
work/source counts, exact planned budgets, failure isolation and recovery.
CI results, rather than generation size alone, determine numerical qualification.

The initial serial public-force module hit the existing 20-minute CI cap after
all eight d-shell analytic/FD cases and three of four budgeted replays completed.
Retained evidence measured a 505-second first complete d endpoint (including
compilation), 12-16-second subsequent singlepoints and 10-15-second batch calls
on that hosted runner. Maximum analytic force error was 9.11e-11 Eh/bohr.
The remedy keeps all cases and all gates: Cartesian and spherical d wrappers
run as separate loadfile workers in a dedicated `ecp-forces` shard alongside
the s/p tests. No global timeout extension or numerical case removal is used.
This integration also exercises the generated CPU ECP provider merged in #648.

## Consequences and revisit conditions

Canonicalization bounds compilation, but execution still visits all ordered
weights and spherical components. No endpoint speedup is claimed. Revisit
schedule reuse or compiled contractions when complete endpoint evidence shows
Python orchestration or primitive replay dominates; preserve arbitrary weights,
physical-center maps, independent numerical gates and explicit work admission.
