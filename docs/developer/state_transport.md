# RCCSD state transport

`tools.vibeqc_cc.state_transport` is the CPU/reference compatibility layer for
moving restricted CC amplitudes between orbital frames. It does not decide when
to run a source or target calculation and it does not make a transported state a
target result. Target orchestration, projected amplitudes, residual refinement,
DIIS/Krylov history and GPU kernels remain outside this slice.

## Complete endpoint identity

`StateTransportRequest.from_snapshots` binds the actual immutable source and
target `ReferenceSnapshot` coefficient arrays to a rectangular target/source AO
overlap. Each `StateIdentity` separately records:

- reference, geometry, basis and AO-representation identities;
- reference kind, precision, screening and overlap policies, functional/grid
  identities, reference/orbital energies, spin, electron count, full
  occupations and explicit occupied/virtual partitions;
- frozen-core mask and Hamiltonian identity; and
- equation, integral-provider and sorted approximation-setting identities.

The endpoint also hashes its exact coefficient frame. Direct requests reject a
coefficient array that does not match that hash, and diagnosed transport objects
can only be created by `StateTransport.classify`; callers cannot label arbitrary
maps exact by constructing a result record themselves.

Equal array dimensions do not establish compatibility. Unscreened FP64
closed-shell RHF 2/0 occupations with an empty frozen-core mask are the only
enabled reference contract. Geometry, spin, occupied count, screening/precision,
core, Hamiltonian, equation, provider or approximation changes reject the
request. A repeated reference ID must also map to the identical complete frame;
otherwise it is treated as stale/inconsistent metadata.

## Classification and orbital maps

For source coefficients `C_s`, target coefficients `C_t` and rectangular overlap
`S_ts`, the target-from-source orbital map is

```text
M = C_t^T S_ts C_s.
```

The explicit occupied/virtual partitions produce `M_oo`, `M_vv`, `M_ov` and
`M_vo`. Singular values diagnose lost rank. The largest cross-block magnitude
diagnoses occupied-virtual mixing; exceeding the policy tolerance rejects the
reference change rather than treating it as a four-index rotation.

| Classification | Meaning | Exact amplitude output |
| --- | --- | --- |
| `identity` | Same complete endpoint and identity frame | Yes |
| `exact_orbital_rotation` | Full-rank square occupied and virtual maps are independently orthogonal and cross-block mixing is below the gate | Yes |
| `projected_warm_start` | A basis/frame change preserves occupied rank but is not a complete isometry | No; classification only in slice A |
| `incompatible` | Identity mismatch, occupied-rank loss, occupied-virtual mixing, stale same-ID frame, or same-basis nonunitary map | No |

An otherwise exact block map must also transform the occupied and virtual
diagonal orbital-energy operators into the target values and preserve the
reference energy within the exact tolerance. This prevents a changed reference
operator from being admitted merely because it reused the same coefficient
frame.

The default absolute gates are `1e-10` for rank and exact isometry and `1e-8`
for occupied-virtual mixing. Diagnostics retain occupied/virtual ranks, minimum
singular values, square-map unitarity errors, mixing and source/target virtual
nullities, orbital-energy frame errors and reference-energy difference. A
rectangular map has no unitarity/energy-frame error value rather than a
fabricated finite score.

## Exact restricted-amplitude transformation

For compatible exact maps `O = M_oo` and `V = M_vv`, the restricted spatial
amplitudes use the #148 axis and simultaneous-pair conventions:

```text
t1_target[k,c]     = sum_ia       O[k,i] V[c,a] t1_source[i,a]
t2_target[k,l,c,d] = sum_ijab O[k,i] O[l,j] V[c,a] V[d,b]
                                      t2_source[i,j,a,b].
```

`StateTransport.rotate_amplitudes` first validates the source through the
existing `AmplitudeSnapshot` contract, performs four explicit staged T2
contractions, revalidates restricted pair symmetry, and returns an immutable
snapshot bound to the target reference ID. It raises for projected and
incompatible classifications before publishing output. It never rotates DIIS
or Krylov histories.

The independent tests compare T1/T2 results with explicit index loops, exercise
phase/permutation and degenerate occupied/virtual rotations, transform the
corresponding one- and two-electron tensors, and check all three RCCSD energy
contributions. Negative tests cover rank loss, occupied-virtual mixing, changed
occupation/core/equation/provider/approximation identities and stale same-ID
frames. They also reject changed screening/reference operators, mismatched
coefficient identities, forged classification records and equal-index frozen
core masks whose physical core subspace was rotated.

Run from the repository root:

```bash
PYTHONPATH=.:python python -m pytest tests/python/test_state_transport.py -q
```

The compatibility rationale and future promotion conditions are retained in
the [StateTransport decision note](../../.agents/notes/implemented/compatibility/2026-09-22-state-transport-exact-frame-boundary.md).
