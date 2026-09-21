# DF-CCSD(T) same-Hamiltonian dense oracle

Status: issue #157 slice A. This is a qualification oracle, not the production
factorized DF-CCSD(T) solver.

## First supported method definition

The first validated DF-CCSD(T) variant deliberately separates the reference
approximation from the correlation-integral approximation:

- reference: conventional, unscreened, all-electron closed-shell RHF;
- orbitals, orbital energies, Fock matrix, and RHF reference energy: retained
  exactly from that conventional RHF calculation;
- correlation Hamiltonian: density-fitted two-electron integrals only;
- DF metric: the existing square-symmetric thresholded inverse square root;
- precision: FP64;
- frozen orbitals: unsupported in this slice;
- triples: standard canonical noniterative (T), with no local/truncated-triples
  approximation.

The method contract records the conventional reference identity, orbital and
auxiliary basis identities, geometry and AO representation, DF Hamiltonian
identity, metric convention/cutoff/effective rank/conditioning, Fock policy,
precision, frozen-orbital policy, and triples variant. These fields participate
in the contract identity instead of allowing two differently defined DF
Hamiltonians to share a cache/result identity.

Changing the correlation Hamiltonian does **not** recompute or replace the
conventional RHF Fock matrix. This is intentional: the first #157 variant is
“conventional RHF reference + correlation-only DF”.

## Dense same-Hamiltonian oracle

For small qualification systems, the oracle uses the existing DF source and
metric factor to build the whitened MO three-index tensor

```text
B[Q,p,q] = sum(mu,nu,P)
           C[mu,p] C[nu,q] A[mu,nu,P] M^(-1/2)[P,Q]
```

in the conventional RHF MO basis. It then reconstructs exactly one dense
same-DF-Hamiltonian tensor

```text
g_DF[p,q,r,s] = sum_Q B[Q,p,q] B[Q,r,s].
```

That dense tensor is exposed through a validation-only provider and passed to
the already-audited RCCSD and standard (T) equation stack. Consequently,
comparison with a later factorized implementation is a comparison of two
implementations of the **same Hamiltonian**, rather than a comparison of DF
against conventional four-center integrals.

An orthogonal rotation of the retained auxiliary B axis leaves `g_DF` and the
physical RCCSD(T) energy invariant. The slice-A tests exercise that gauge
invariance explicitly.

## Memory scope

The dense oracle is intentionally small-system-only. Before reconstruction it
preflights a numeric budget that includes the caller-owned B tensor, the
immutable owned B copy, the dense four-index result, and a conservative
full-sized contraction temporary. The production slices must not rely on this
`NMO^4` allocation.

The existing `DFProvider` remains responsible for bounded construction of B.
Slice B of #157 replaces memory-heavy CCSD four-index blocks with factorized or
bounded reconstruction. Slice C integrates standard (T), user-facing
energy-only semantics, and performance/memory evidence.

## Validation rules

Implementation tolerances and DF fitting error are separate quantities.

1. Raw/whitened B and reconstructed `g_DF` are checked for finite FP64 data
   and identity compatibility.
2. The dense oracle and any factorized path must agree tightly for fixed
   amplitudes, converged RCCSD energy, and (T) energy on the same DF
   Hamiltonian.
3. Auxiliary-gauge-equivalent B tensors must give the same `g_DF` and energy.
4. DF-versus-conventional-four-center differences are reported separately by
   `df_fitting_error`; they are not used to relax same-Hamiltonian numerical
   gates.
5. The conventional RHF Fock/orbital state is preserved and is independently
   identifiable from the correlation DF Hamiltonian.

## Python validation API

The internal validation helpers live in
`tools.vibeqc_cc.df_ccsdt_oracle`:

- `correlation_df_reference` fixes the method contract while preserving the
  conventional RHF state;
- `prepare_same_hamiltonian_dense_oracle` obtains B from the existing
  `DFProvider` and reconstructs the small dense `g_DF`;
- `dense_df_oracle_from_three_index` allows independent/synthetic B fixtures;
- `run_dense_df_ccsdt_oracle` evaluates the trusted RCCSD/(T) equations;
- `df_fitting_error` reports DF-versus-exact integral error separately.

These helpers do not register a Calculator method and do not claim native,
factorized, CUDA, or force support.

## Non-goals of slice A

- no production full-`NMO^4` DF integral storage;
- no factorized CCSD residual implementation yet;
- no native/public DF-CCSD(T) method registration;
- no DF-CCSD(T) force or gradient claim (tracked separately by #158);
- no frozen-core, open-shell, ECP, local, or DLPNO variant.
