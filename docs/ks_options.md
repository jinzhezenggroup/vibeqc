# Prepared semilocal KS model options

`Calculator(method="lda-rks" | "pbe-rks" | "lda-uks" | "pbe-uks",
ks_options=...)` snapshots a `KsOptions` object for CPU and CUDA. Its default
grid and tile reproduce the original method behavior. The existing compiler
`FunctionalSpec` and `GridSpec` are reused, with an explicit native SCF
tail/spin domain; a second functional catalog is not introduced.

```python
from vibeqc import Calculator, GridSpec, KsOptions, ResourceBudget

options = KsOptions(
    grid=GridSpec(
        radial_points=64,
        angular_polar=20,
        angular_azimuth=40,
        element_radii=((1, 1.3), (8, 0.9)),  # Bohr, indexed by element
    ),
    tile_points=128,
)
calculator = Calculator(
    method="pbe-rks",
    device="cuda",
    ks_options=options,
    resource_budget=ResourceBudget(host_bytes=1 << 30, device_bytes=1 << 30),
)
print(calculator.ks_options.to_payload())
```

An omitted functional resolves to `LDA_X+LDA_C_PW` or
`GGA_X_PBE+GGA_C_PBE`, with unit coefficients. RKS resolves an unpolarized
composition; UKS resolves a polarized one. An explicit `FunctionalSpec` must
match those components, coefficients and spin convention. Different names
for an identical composition are descriptive; changed coefficients, exact
exchange, range separation or unsupported domain policies are rejected before
native loading. These energy methods do not advertise arbitrary functional
mixtures or hybrid support.

The effective native domain is
`semilocal-scaled-v1/pbe-spin-c2-1e-18`, documented in
[the SCF domain contract](xc_scf_domain.md). `FunctionalSpec` retains its
independent interior-reference provenance; that provenance does not silently
replace the native SCF boundary policy. The resolved options record both.
The functional's required ingredients determine AO order: LDA needs values,
GGA needs values and first spatial derivatives. SCF requires scalar energy
and its first derivative for the potential, even for an energy-only output.
Neither route computes tau or higher AO jets.

Completed results expose the resolved grid and physical iteration records in
[`ks_diagnostic`](ks_diagnostics.md).

`GridSpec` selects the radial/polar/azimuth counts, partition iterations,
coincident-center tolerance and per-element radial scales. The native grid
uses the same rational-Legendre radial rule, Legendre-trapezoid angular rule
and equal-radius Becke partition as the reference. Element radii scale radial
points and radial Jacobians; they do not add a heteronuclear partition
correction. Pruning, screening, rules and units retain the version-1 contract.
Changing the grid changes the discrete energy. Changing `tile_points` changes
the schedule and capacity, with only FP64 reduction-order differences expected.

The complete options are included in resource identity and Python prepared
model identity. Resource estimates use the actual grid dimensions and tile,
including native radius-table storage. A replaced functional/grid/tile model
requires a new prepared object before a warm density, DIIS history, or cached
physical result can run. Compatible coordinate changes keep the existing
native rebuild and spin-normalization path.

The C ABI appends a nullable `ks_options` pointer to
`vibeqc_method_descriptor`. A missing tail field or NULL preserves defaults.
`vibeqc_ks_options_version()` reports support without creating a context.
Version 1 is the original grid/tile prefix, version 2 adds explicit semilocal
scales/full-range exchange, and version 3 adds the XC execution schedule. Version
4 appends the compiler-resolved self-consistent execution selector: spin-channel
count and semilocal primitive family. The complete v3 layout, including its
trailing padding, is preserved before the v4 fields. Modern Python callers derive
those v4 fields from `MethodIR -> KsExecutionPlan`; native CPU/CUDA execution no
longer chooses RKS/UKS or the LDA/PBE/r2SCAN family from the public method ID. Old
v1/v2/v3 callers retain a narrow legacy selector fallback. The PBE-D4 correction
is retained in the snapshotted plan alongside the semilocal execution selector.

Python negotiates the highest supported descriptor version. Composition and
host-unfused scheduling still fail closed when an older library cannot represent
them. For the append-only ABI rationale, see the
[compatibility decision](../.agents/notes/implemented/compatibility/2026-09-21-ks-v4-append-only.md).

Explicit options require complete nonzero grid counts and a positive tile size.
Optional element radii are a 119-entry positive finite array indexed by atomic
number (entry zero is unused); NULL/zero selects unit radii. Preparation copies
the descriptor and all pointees. Caller storage can be released or modified as
soon as prepare returns. Other method families reject an attached KS option.
Older libraries remain usable for default KS models; Python rejects custom
options if the version query is unavailable.
