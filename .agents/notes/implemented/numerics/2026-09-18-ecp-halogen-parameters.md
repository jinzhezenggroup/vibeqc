# Decision: qualify real LANL2DZ Br/I HF endpoints

Status: implemented
Date: 2026-09-18

## Problem and scope

Real ECP parameter qualification covered Na/K and Rb/Cs. Their atomic number
alone does not validate the seven-valence-electron halogen potentials. Add
the unmodified PySCF 2.14.0 LANL2DZ Br/I orbital and scalar-ECP records with
STO-3G hydrogen to the existing heavy-element suite. Keep parameter hashes,
the removed core and effective ionic charge explicit.

Br removes 28 electrons and I removes 46. Each has effective charge +7;
neutral HBr/HI have eight explicit electrons, and +1 doublet cations have
seven. Both molecular fixtures have nine s/p AOs and fit the existing CUDA
budget domain. This is a bounded physical-parameter qualification, not new
operator mathematics or a universal heavy-element capability.

## Decision and invariants

Reuse the existing fixture conversion, independent Libcint components and
HF endpoint gates. Extend the case table and expected charge bookkeeping
instead of copying an ECP implementation or adding a parallel test suite.
The report driver accepts an optional element subset while retaining all
qualified elements by default. Runtime code, compiler formulas, native
ownership and the independent CPU oracle are unchanged.

Each case must pass separate local/nonlocal matrix comparisons, two-grid
refinement, all-center derivatives at two finite-difference steps, arbitrary
nonsymmetric fixed weights, complete RHF/UHF energy/force comparisons and
exact-budget changed-geometry energy differences. Original Rb/Cs cases remain
in the same regression suite. PySCF supplies test parameters and independent
reference results only; normal execution consumes owned records.

## Evidence and limits

See [the qualification bundle](../../../../benchmarks/results/ecp-halogen-171/README.md)
for exact fixtures, library/source identities, independent numerical gates,
allocation observations and retained logs. The original qualification started from master
`97adc1a`, independently of the then-open radial batching PR #458. Timings describe
single calls for reproducibility, not a speedup claim.

Other halogen molecules, orbital/potential families, spin-orbit interactions,
DFT forces and density fitting require separate qualification. Revisit the
scope when those consumers have independent complete-method evidence. Refs #171.

## Measured result

The fresh CPU/CUDA builds each pass all 20 shared heavy-element tests, and
native ECP CTest passes 2/2 and 3/3 respectively. Eight complete CUDA HF cases
pass Compute Sanitizer with zero errors. For the new Br/I reports, maximum
energy error is 6.928e-14 Eh and maximum force error is 1.209e-9 Eh/bohr across
both backends. Budgeted endpoints reject no allocations and remain below
their planned bounds. All 613 build inputs and both changed source files
match their retained identity. No scientific source or tolerance changes
were needed, and no speedup or broader chemical accuracy claim is made.

## Requalification after integration (2026-09-19)

The branch was rebased onto `45af86a`, retaining the merged
Au fixture semantics and radial batching. The combined suite passes 26 CPU and
26 CUDA cases; native ECP CTest passes 2/2 and 3/3, and all ten complete CUDA
RHF/UHF cases pass Compute Sanitizer with zero errors. Every recorded source
input matches the measured Git export. See the
[refresh evidence](../../../../benchmarks/results/ecp-halogen-171/refresh/README.md)
for exact source identities, numerical/resource gates and reproduction.
