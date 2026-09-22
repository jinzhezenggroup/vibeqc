# Decision: Qualify scalar ECP nonlocal f projectors separately from f orbitals

Status: implemented
Date: 2026-09-18

## Problem

Orbital f was qualified by #432, but the semilocal operator still rejected f
projectors. Relaxing only input checks would overrun nine-slot harmonic and
projection buffers, omit the f reduction, and leave resource estimates stale.

## Decision

Extend compiler-owned real spherical harmonics and projector reductions through
f, using the existing scalar DAG and radial term contract. Seven orthonormal f
harmonics occupy slots 9..15. The compiler emits the projector limit/count;
native validation consumes the emitted limit and CUDA checks the host record
size against it. Python permits BSE/NWChem local labels through g, which is only
the label for the local residual, not support for g orbitals or g projectors.

The independent CPU integral implementation retains its own polynomial harmonics
and reductions. It remains an oracle/fallback; it does not call generated
projector algebra. Native allocation, scheduling, physical-center scatter and
complete-method convergence policy remain explicit. The mixed CUDA adapter is
still conservatively classified as scientific; no native formula is retired.

## Invariants

Preserve FP64, normalized orbital ordering, signed radial coefficients and powers
0..4, the full nonsymmetric AO dot product, A/B/C translation recovery, and the
complete-HF coarse/fine grid gates. One radial shell remains the lifetime bound.
Projection storage grows from 9 to 16 jets per AO and host sphere records from
104 to 160 bytes; the resource inventory includes the projection increase.
This capability does not expand the existing 16-public-AO CUDA budget inventory.

## Rejected alternatives

Do not infer projector support from an orbital limit, reuse the generated path
as its own sole oracle, or advertise a heavy-element parameter family from a
synthetic channel test. Do not introduce another symbolic spherical algebra:
the existing DAG already represents the bounded polynomial harmonics.
Channel-pruned or specialized schedules require their own complete endpoint and
resource evidence; this bounded extension does not make a speedup claim.

## Evidence

The native independent Legendre addition-theorem double-node contraction checks
every channel/power, A/B/C jets, signed multi-exponent terms, center filtering and
poisoned derivative slots in value-only mode. Grid tests check the full 16x16
harmonic Gram matrix and per-channel addition theorem.

The Python suite compares f-projector matrices with Libcint for all five radial
powers and Cartesian/spherical f orbitals. All-center derivatives, including an
ECP atom without a basis, use two independent finite-difference steps and
nonsymmetric fixed weights. Complete direct RHF/UHF energies and forces use
PySCF; budgeted geometry replay also checks complete-energy differences.

Acceptance gates remain 2e-9 Eh for raw matrices/refinement, 2e-8 for derivative
refinement and complete energies, 3e-7 absolute/3e-6 relative for raw finite
differences, and 2e-6 Eh/bohr for complete forces. These are bounded-fixture
qualification gates, not universal quadrature error bounds.

See the [retained measurements](../../../../benchmarks/results/ecp-f-projector-171/README.md)
for exact source/binary identities, all samples, work counts, kernel resources,
validation results and reproduction scripts.

## Revisit when

Additional projector angular momentum, parameter families or methods have their
own independent derivative, complete-method and budget qualification. Revisit
channel pruning only when endpoint measurements justify the extra schedule.

## References

- Issue #171 and PR #432.
- [Current ECP contract](../../../../docs/user/ecp.md).
- [Orbital-f decision](2026-09-17-ecp-orbital-f.md): its orbital boundary remains;
  this decision separately supersedes its f-projector exclusion.
