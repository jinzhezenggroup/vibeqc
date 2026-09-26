# Methods and long-term scope

VibeQC's long-term mission is to cover **all quantum-chemistry methods** in one
accelerator-native system. Current executable method identity is registry-driven:
the canonical native names, aliases, declared properties, and batch capability
are generated from `manifests/public_methods.json` into the
[public method table](../public_methods.md). That generated table is authoritative
for public native discovery; backend-, basis-, grid-, and model-specific
execution constraints remain method-specific and fail closed.

## Current method status

Use the generated [public method table](../public_methods.md) for registered
names, aliases, declared properties, and batch support. This guide deliberately
does not maintain a second handwritten status table. A registered method is
not a blanket claim that every backend, basis, grid, spin state, or requested
property is qualified; the corresponding method contract and execution-time
admission checks remain authoritative for those combinations.

Internal validation helpers and compiler plans are not substitutes for native
public method registration. Planned families and development directions are
listed in the [implementation roadmap](../maintainer/roadmap.md), which does not
promise release dates or a fixed implementation order.

## CUDA global-hybrid forces

The Python `Calculator` exposes analytic forces for admitted all-electron
global-hybrid RKS/UKS compositions with direct J/K, FP64, an explicit `GridSpec`,
and device-fused XC. Force eligibility follows the actual MethodIR primitives;
the SCF owner still validates the semilocal composition. Request forces through
the ordinary single-point or prepared-batch interface:

```python
from vibeqc import Calculator, GridSpec, KsOptions

calc = Calculator(
    method="b3lyp-rks", device="cuda", precision="fp64", basis="sto-3g",
    ks_options=KsOptions(
        grid=GridSpec(radial_points=32, angular_polar=10, angular_azimuth=20),
        xc_schedule="device_fused",
    ),
)
result = calc.singlepoint(
    [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))], properties=("energy", "forces")
)
```

The first call compiles a method-specific CUDA wrapper and requires a
discoverable NVCC toolkit (`CUDACXX` or `CUDA_PATH` can select it). Later calls
reuse the compiler cache. There is no CPU scientific fallback. Density-fitted,
range-separated, nonlocal, ECP and mixed-precision hybrid forces remain outside
this contract. See the [execution and resource limits](../developer/stationary_cuda_diagnostic.md)
and [independent force acceptance gate](../maintainer/hybrid_cuda_acceptance.md#public-global-hybrid-force-gate).

## Direct CUDA HF force state

Direct RHF/UHF force solves require the maximum physical AO commutator
`|F P S - S P F|` to meet `min(1e-8, density_tolerance)` in addition to the
energy and density-update criteria. The maximum includes both UHF spins;
nonfinite residuals cannot pass. Additional SCF updates remain within the
requested iteration limit and appear in the public iteration count.
An item's mixed-precision coarse stage keeps its previous stop; the exact FP64
refinement and exact-precision neighbors apply the physical force criterion.

Finalization projects the final physical-Fock orbitals to a determinant,
rebuilds its physical Fock, and rechecks that determinant's commutator before
publishing forces. Energy, forces and the returned warm state share this P/F(P).
The Pulay weight is `P F(P) P / 2` for RHF and `P_sigma F_sigma P_sigma` for
UHF. A failed final residual returns nonconvergence. Energy-only and detached
physical-reference execution retain their existing finalization contracts.

This force finalization adds one density projection, one physical Fock build,
four matrix products for residual validation and two for the Pulay weight.
Existing device scratch holds the products; a four-byte work counter is copied
at the existing completion fence. `VIBEQC_DF_PROGRESS_TRACE` records final
updates, physical Fock builds, residual checks and rejections separately from
iterative SCF updates. Complete endpoint costs include all this work. The
[decision record](../../.agents/notes/proposed/2026-09-17-consistent-direct-pulay-weight.md)
retains the independent diagnosis and qualification boundaries.

## Acceptance standard

A method becomes supported only when all of the following are true:

1. Its public behavior and mathematical conventions are documented.
2. Energies and relevant derivatives agree with an independent implementation
   over representative systems and basis sets.
3. CPU/GPU execution boundaries and unsupported cases fail explicitly; there
   is no silent fallback to an unvalidated path.
4. Reproducible benchmark artifacts support any performance claim.
5. Batched execution preserves per-system ordering, diagnostics, and failure
   isolation where the method permits batching.

## Expansion strategy

VibeQC expands method coverage behind the same registry-driven prepared
calculation interface. New public capabilities should reuse the existing basis,
integral, SCF, batching, resource-planning, and diagnostic contracts rather than
introducing method-specific API branches.

Near-term development focuses on completing scientific and backend coverage of
the method families already present in the registry: broader DFT execution and
analytic derivatives, density-fitted derivatives, correlated-method
qualification and derivatives, and wider basis/ECP/backend coverage. Compiler
and runtime work should remove duplicated handwritten execution paths while
preserving explicit numerical ownership and fail-closed unsupported cases.

A capability is promoted only after the [acceptance standard](#acceptance-standard)
is satisfied. Implementation details belong in the
[Developer Guide](../developer/index.md), performance qualification in the
[Maintainer Guide](../maintainer/index.md), and durable historical rationale in
`.agents/notes/`.
