# Physical KS diagnostics

Completed LDA/PBE RKS/UKS results expose an immutable `ks_diagnostic` on both
`Result` and `BatchItemResult`. It contains the actual grid/tile/AO order,
audited SCF domain, alpha/beta occupations, electron counts, physical energy
components, convergence measures and iteration history. The enclosing result's
`executed_backend` identifies the backend that performed the calculation.

```python
from vibeqc import Calculator

result = Calculator(method="pbe-uks").singlepoint(
    [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))],
    charge=1, multiplicity=2,
)
diagnostic = result.ks_diagnostic
print(diagnostic.occupations)           # (1, 0)
print(diagnostic.components.total)      # the returned physical energy
print(diagnostic.physical_residual_max)
print(diagnostic.to_payload())          # all iteration records included
```

Energies are in Hartree. `KsEnergyComponents` contains nuclear, one-electron,
Hartree and XC terms with XC counted once; `total` sums those four terms.
`electrons` measures `Tr(D_s S)` in the AO metric. RKS reports equal spin
counts from half the total density. These are distinct from quadrature
electron counts and are independent of the quadrature's accuracy.

`density_change_max` and `physical_residual_max` are the maximum spin RMS
values used by the physical convergence gates. RKS has one total-density
matrix. UKS gates each spin separately, so an empty beta channel cannot dilute
an unconverged alpha channel. The legacy result's `density_rms` and
`physical_residual_rms` continue to combine the alpha/beta matrix entries in
one RMS.

Each `KsIteration` describes a physical E/F/D state and its proposed density
change. `energy_change` is `None` on the first iteration, because no preceding
energy exists. Later entries retain the measured absolute energy difference.
`occupation_stabilized` records whether that proposal used the stationary
UKS virtual-projector shift; its recorded energy and residual remain physical
and unshifted. CPU RKS performs additional validation/projection after its
iteration loop. The final diagnostic retains those final physical components
and residual separately from the original historical rows.

`fock_builds` includes final validation rebuilds for this solve.
`initial_density_used` describes this solve's actual initial state. When the
prepared batch retries a rejected warm attempt from a cold seed, the snapshot
describes the final attempt; it is not a count or history of all discarded
work. Complete endpoint timing and transport measurement must include every
attempt and preparation/finalization phase separately.

Valid nonconverged batch items retain their actual history. Invalid or
numerically failed items have no diagnostic. A replay invalidates cached C
records before executing; Python snapshots from earlier results remain
unchanged. Unsupported methods and older native libraries return `None`.
The public object stores frozen dataclasses and tuples, not borrowed pointers
into a mutable native history.

The additive C queries are `vibeqc_calculation_get_ks_diagnostic` and
`vibeqc_batch_get_ks_diagnostic`. Legacy result structures and batch-array
strides do not change. Query a size/ABI-initialized `vibeqc_ks_diagnostic` with
NULL history and zero capacity to learn `history_count`. To copy the history,
allocate at least that many `vibeqc_ks_iteration` descriptors, initialize every
descriptor's size and ABI, and query again. Every output is validated before
any output is written. Query and execution calls on a prepared owner must be
serialized by its caller.

The method adapter moves one exported history through to its C handle. The
shared resource inventory covers CPU vector growth or the CUDA history plus
that exported snapshot; public queries do not allocate another native history.
Python result storage is owned by the caller.

## Internal final-state handoff

The resident CUDA KS owner also provides an internal, versioned handoff for
the later stationary-gradient implementation in issue #163. It is deliberately
absent from the public C/Python result ABI and does not make force requests
supported.

After a converged solve, `methods/dft_method.hpp` can return an eligibility
token for either a prepared single calculation or one prepared batch item. The
token binds the prepared provider, geometry/basis owner, GridSpec, functional,
spin occupations, device, solve epoch and exact orbital/Fock/density
generation. Every new `begin`, including a failed or nonconverged attempt,
revokes the preceding token before CUDA work. Rebuilt geometry receives a new
owner even when all matrix dimensions are unchanged. Prepared-call exceptions
and batch items rejected before device submission explicitly revoke their old
eligibility while preserving independent warm-start ownership.

An exact-token read canonicalizes the retained physical, non-DIIS Fock on the
owner's ordinary stream and detaches `D`, `F`, `C`, orbital energies,
occupations, energy components, grid and provider identity. The read validates
the AO-metric eigenframe, density reconstruction, commutator, electron trace,
idempotency, canonicality, component energy and physical residual before
publication. `W` is constructed on the host only when explicitly requested
after every gate passes. A stale token is rejected before matrix transfer.

`CudaKsTransfers::final_state_d2h_bytes` and `final_state_reads` separate this
explicit handoff from ordinary energy execution. The existing public
`matrix_d2h_bytes` total still includes snapshot matrices, so legacy transport
accounting remains conservative without an ABI change.
