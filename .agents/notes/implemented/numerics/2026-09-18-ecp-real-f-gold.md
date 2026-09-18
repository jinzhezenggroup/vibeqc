# Decision: qualify a real LANL2DZ f projector with AuH

Status: implemented
Date: 2026-09-18

## Problem

Synthetic signed f projectors qualify the operator but do not establish that
a physical parameter set with occupied d orbitals is handled correctly. The
unmodified PySCF 2.14.0 LANL2DZ Au records provide a local g label, s/p/d/f
nonlocal differences and an s/p/d orbital basis. Au removes 60 core electrons
and has effective ionic charge +19. With STO-3G H the fixture has 23 AOs.

## Decision and invariants

Extend the existing heavy-element fixture/converter, raw component gates and
complete HF tests. Keep Rb/Cs regression coverage. Pin the Au orbital/ECP
checksum, core count and valence charge. Qualify neutral singlet AuH (20
electrons, RHF) and the -1 doublet (21 electrons, UHF) at the declared off-axis
geometry. Use tighter SCF stopping criteria for Au without changing integral
grids, production code or numerical acceptance tolerances.

An independent f-only Libcint block and its two-step all-center differences
must match full-minus-zero-f native results. The f contribution must be
nonzero and the local block must remain unchanged. Use physical atomic number
to find the Au record: BasisSet canonicalizes element order.

The 23-AO calculation exceeds the CUDA budget inventory. Explicit resource
requests must still reject the unsupported domain. Numerical CUDA replay runs
without a budget; the report records an unsupported plan and no claimed peak
bound. CPU replay and Rb/Cs CUDA retain their exact-budget gates. No resource
provider or native scientific arithmetic is added or retired.

## State selection and rejected scope

The first AuH+ doublet attempt failed complete-method agreement despite raw
ECP agreement. At the declared geometry PySCF minao, one-electron and atomic
initial guesses converge near -134.749057091896 Eh. Default native execution
gave a higher solution near -134.679379 Eh; tightening the native stopping
criteria failed to converge within 200 iterations. The original failure logs
and diagnostic probe are retained. Neither loosening the error gates nor
choosing a reference merely because it matches that result is acceptable.

AuH+ is therefore explicitly outside this qualification. AuH- provides a
separate open-shell fixture for which all three reference guesses and native
execution agree. The accepted tests require agreement across those reference
guesses for both Au states. This is bounded endpoint validation, not proof of
the global HF minimum, arbitrary gold chemistry or a fix to general SCF state
selection. Revisit the cation after the native state/convergence path can
independently reproduce its reference branch and complete force/replay gates.

## Evidence and limits

See [the retained bundle](../../../../benchmarks/results/ecp-real-f-171/README.md)
for source/library identities, endpoint errors, resource status, initial-guess
energies and reproduction. Production/native/compiler sources are unchanged by this PR relative to the
qualification base `0ed06a3`. Fresh Release CPU/CUDA binaries were built and
all 680 native/runtime/build inputs plus both binary hashes were retained.
Master later advanced to `e215b30` via #448; its changed-file set has zero
overlap with those 680 qualified inputs.

This does not qualify g projectors, g orbitals, other ECP families, spin-orbit
physics, DFT forces, density fitting or larger CUDA budget inventories. Timings
are single-call context, not performance evidence. Refs #171.

## Measured outcome

All 16 shared heavy-element cases pass on each backend; native ECP CTest
passes 2/2 CPU and 3/3 CUDA. Both complete Au CUDA endpoints pass memcheck
with zero errors. For the two declared Au states, maximum energy error across
backends is 7.390e-13 Eh and maximum force error is 1.971e-10 Eh/bohr. CUDA
resource requests remain unsupported as required. All 680 native/runtime/build
inputs plus both final test/report files match retained source identities.
The cation failure remains excluded and documented rather than accepted.
