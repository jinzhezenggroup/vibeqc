# SCF proposals and reproducible traces

`tools.vibeqc_scf` provides an opt-in development interface around the native
CPU conventional and density-fitted RHF/UHF loops. Its private bridge supports
up to 12 orbital and 24 auxiliary AOs. It executes the entire native solve,
including integral preparation and final analytic forces. CUDA callbacks and
DFT are explicitly unsupported. No optional ML package, learned model,
automatic download, or data upload is involved.

## Physical state and proposals

Every snapshot owns immutable FP64 arrays for the current AO density, physical
Fock, commutator, overlap, and traditional DIIS/Aufbau next density. The physical
quantities are evaluated before DIIS extrapolation. RHF has one spin-summed
matrix; UHF stores alpha then beta. Coordinates are bohr and energies Hartree.

The physical operator is `F = H + J[P] - K[P]/2` for RHF and
`F_sigma = H + J[P_alpha+P_beta] - K[P_sigma]` for UHF, with the declared DF
metric cutoff when fitted. The AO commutator is `R = F P S - S P F`; the
safeguard uses its RMS over all spin/AO entries. This norm is coordinate
dependent, so comparisons preserve the actual basis/overlap identity.

Snapshots reuse `ResolvedModel` from the accuracy layer and the immutable-byte
array convention from the post-HF reference interface. The content identity
covers every matrix, model/geometry, item owner, solve generation and iteration.
The native generation is process-local; the portable identity includes the
Python owner's UUID and scientific content. An iteration snapshot is never
implicitly eligible as a converged post-HF reference.

Supported proposal representations are:

- `DensityProposal`: an ensemble or determinant density, including an initial
  guess supplied at iteration one, a mixing update, or an explicit reset.
- `OccupiedProposal`: occupied AO columns with the model's fixed populations.
- `RotationProposal`: occupied-virtual rotations in a supplied full orbital
  gauge. `kind="preconditioner_action"` declares an action on that same space.
  A Cayley transform of the skew generator preserves orthonormality.

For each density block, validation requires symmetry, `Tr(P S)=N_sigma`, and
eigenvalues of `S^(1/2) P S^(1/2)` between zero and its occupation weight (two
for RHF, one for UHF). Determinants require eigenvalues at zero or the weight,
equivalently `P S P = weight P`. Occupied/full orbital columns require the
corresponding `C^T S C = I`. Wrong populations, shapes, nonfinite values and
stale parents are rejected. Density validation uses an absolute `1e-7` gate;
orbital orthonormality uses `1e-8`. Singular metrics are rejected at `1e-10`.
These are representation checks, not observable error bounds.

## Acceptance and fallback

The safeguard evaluates the real target operator at fractions `1, 1/2, 1/4,
1/8` between the current density and a valid candidate. A trial must satisfy
`E_trial <= E_current + 1e-9 Eh` and
`r_trial <= max(1e-12, r_current * (1 - 1e-4*fraction))`.
Every failed trial counts as a Fock build. Damping produces an explicitly
fractional ensemble even when the original proposal was a determinant.

Rejected proposals and callback exceptions retain the already computed
traditional next density and reset subsequent DIIS history. Accepted proposals
also reset that history. After three rejected/reset proposals in one solve,
the proposer is disabled and traditional DIIS continues. This budget is local
to the solve, and its exhaustion is recorded as `traditional_fallback`.

Convergence always compares the traditional next density with the current
density, so returning the current density cannot manufacture zero change.
With an active proposer, the true residual must additionally meet the density
tolerance. With `proposer=None`, results and convergence follow the existing
path; there is no snapshot work unless capture is requested. Strict initial
seeds are validated before use and are not silently rescaled.

A small residual does not establish the lowest or intended electronic state.
Snapshots expose metric occupations, gaps, and population outside the Aufbau
occupied subspace. Stability and intended-state status remain `not_evaluated`
and `unverified`. The reference dataset separately records independent
competing starts and their internal stability checks; those checks do not
automatically certify the native solution.

## Local capture, replay and lifetime

```python
from vibeqc import Calculator
from tools.vibeqc_posthf.sources import NativeSource
from tools.vibeqc_scf import ScfItem
from tools.vibeqc_scf.proposals import diis_density
from tools.vibeqc_scf.replay import export_trace, load_trace

atoms = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
model = Calculator().resolved_model(atoms)
with NativeSource(atoms) as source:
    item = ScfItem(source, model)
    result = item.solve(proposer=diis_density, capture=True)
    export_trace(result, source, "local-run.json")  # explicit molecular export

metadata, snapshots = load_trace("local-run.json")
```

`ScfItem` serializes execution for one item. Rebinding a source/model changes
the owner epoch and clears its last result; every execution gets new native
generation and DIIS state. Separate items can execute concurrently without
sharing proposals or DIIS. Sources remain caller-owned; detached snapshots
survive their closure. Recursive execution/rebinding during a callback fails
explicitly. Geometry, spin, charge or model changes cannot reuse old proposals.

The versioned JSON/NPZ format contains plain scientific data, checksums, exact
acceptance policy, inputs and provenance. It contains no Python pickle, native
handle or device resource. The loader checks archive sizes, names, FP64 shapes,
checksums and physical invariants before use. Capture is bounded and truncation
sets `trace_complete=False`. Nonconvergence retains the last iterate and its
failure status; native exceptions retain available traces and report unavailable
work counts rather than claiming zero work.

`counterfactual` rebuilds an independent target operator from native raw
integrals and verifies the saved Fock before evaluating a proposal. It measures
one snapshot, without the full trajectory's failure budget or DIIS evolution.
Its timings are not complete-solve speed measurements. A checkpoint consumer
must rebuild runtime resources, revalidate identity, and evaluate the target
operator again when importing reusable state.

## Baselines and measurement

The benchmark covers cold core guesses with DIIS, fixed-point iteration,
safeguarded DIIS, fixed mixing, diagonal OV preconditioning, current warm
density, and explicit same-AO orbital projection/extrapolation. The existing
UHF core guess's frontier mixing is retained. Native CPU Newton/SOSCF is not
available; the diagonal OV action is not labeled as a second-order solver.
Projection and extrapolation check identity/rank, record their transformation
time, and make no derivative claim.

All timed variants use matched final controls. Results distinguish current
geometry only, prior-geometry state, two prior geometries, and a converged
current-geometry density. The last category includes the prerequisite solve's
cost separately. Whole molecule families, including every geometry and basis
variant, stay together in the training/holdout split. Failed and difficult
solves remain in the report.

Metrics include energy/force agreement under the accuracy schema, iterations,
physical Fock builds, total latency/throughput, rejected trials, generator time,
native callback time, validation, materialization and declared repair costs.
One joint UHF J/K evaluation counts as one physical build. Total latency includes
native integral preparation and final forces; independently owned raw-system
setup, reference audits and disk export lie outside that interval. Native
callback time includes snapshot copying and proposal generation; declared
repair time may overlap generation/materialization, so components are not
blindly summed. Existing FleetPlan fallback work is marked unavailable where
its older public result ABI cannot expose it.

Reproduce the dataset and benchmark with:

```bash
PYTHONPATH=python:. VIBEQC_LIBRARY="$PWD/build/libvibeqc.so" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m tools.validate_scf_proposals --output /tmp/scf-baselines --repeats 3
```

Independent reference regeneration is a separate explicit command,
`python -m tools.generate_scf_proposal_references --output PATH`, requiring
already installed PySCF 2.14.0. The committed 15-case reference archive was
generated twice with identical bytes. This interface supplies safe proposal
infrastructure and traditional baselines, without a learned-acceleration or
novel-model claim.
