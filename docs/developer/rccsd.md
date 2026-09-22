# RCCSD energy and physical singles equations (CG11 A)

This document preserves the A-slice contract. Complete residuals and the
subsequently implemented CPU solver are documented in [RCCSD B/C](rccsd_bc.md).

`tools.vibeqc_cc` provides the internal small-system FP64 CPU RCCSD facade:
`build_ccsd_program`, `PreparedCCSD`, `SolverOptions`, `CCSDResult`, and `solve`.
It implements complete physical T1/T2 residuals and CPU iterations in addition
to the fixed-amplitude energy/T1 interface documented below. The full A/B/C
scope and molecular acceptance are described in [RCCSD B/C](rccsd_bc.md).
It does not register a public `Calculator` method; GPU RCCSD remains #149.

## Mathematical contract

All electrons participate in a real, conventional Coulomb Hamiltonian with a
closed-shell RHF reference. The provider facade accepts #147's validated
canonical `ReferenceSnapshot` and CPU `ConventionalProvider`. It rejects a
different reference identity, DF and GPU providers. Frozen core, open shell,
complex orbitals and missing virtuals remain rejected by #147. Equations keep
every Fock block, including off-diagonal oo, vv and ov elements: arbitrary-F
equation tests intentionally do not pretend to be converged RHF snapshots.

Indices i,j,k,l are occupied spatial orbitals; a,b,c,d are virtual spatial
orbitals. MO columns are occupied then virtual, as in `docs/posthf.md`.
`g[p,q,r,s]=(pq|rs)` is a chemists' ERI. Real ERIs have both within-pair
exchanges and pair-interchange symmetry. `F[p,q]` is the full symmetric RHF
Fock, `h[p,q]+sum_i(2(pq|ii)-(pi|iq))`, not orbital energy denominators.
Normal ordering is with respect to the doubly occupied determinant Phi:
`H = E_ref + H_N`. E_ref includes nuclear repulsion; H_N contains normal-ordered
Fock and two-electron operators. Energies are Hartree.

Let `E_ai = sum_sigma a^dagger_(a sigma) a_(i sigma)`. Our exponential is

```text
|Psi> = exp(T1 + T2)|Phi>
T1 = sum_ia t1[i,a] E_ai
T2 = (1/2) sum_ijab t2[i,j,a,b] E_ai E_bj
t2[i,j,a,b] = t2[j,i,b,a]
```

There is no separate i/j or a/b antisymmetry. The equivalent spin-orbital
definition orders all alpha spatial orbitals before all beta spatial orbitals:

```text
t1[i sigma,a upsilon] = delta(sigma,upsilon) t1[i,a]
t2[i sigma,j tau,a upsilon,b omega]
  = delta(sigma,upsilon) delta(tau,omega) t2[i,j,a,b]
  - delta(sigma,omega) delta(tau,upsilon) t2[i,j,b,a]
T2 = (1/4) sum_IJAB t2[IJAB] a^dagger_A a^dagger_B a_J a_I
```

The spatial residual returned is one alpha single projection (equal to beta),
`R1[i,a]=<Phi_i_alpha^a_alpha|exp(-T) H_N exp(T)|Phi>`.
It is not twice this projection or a derivative of the correlation energy.
The reference bra has unit determinant norm and the excitation sign follows
the displayed fermionic operators.

## Energy, inventory and physical residual

```text
E_corr = 2 sum_ia F[i,a] t1[i,a]
       + sum_ijab [2(ia|jb)-(ib|ja)] t2[i,j,a,b]
       + sum_ijab [2(ia|jb)-(ib|ja)] t1[i,a] t1[j,b]
E_total = E_ref + E_corr
```

`inventory.py` is the auditable equation source: 5 named energy contractions
E01–E05 and 30 named singles contractions S01–S30, each with exact integer
coefficients, explicit Einstein indices and operand names. `build_program`
lowers that unshared inventory to #145 TensorIR, retaining all individual terms
and the following diagnostic sums:

| Output group | Contributions |
| --- | --- |
| energy_t1 / energy_t2 / energy_t1t1 | The three energy lines above |
| singles_fock | Full F including linear, quadratic T1 and F*T2 terms |
| singles_g_t1 / singles_g_t2 | ERI times one amplitude |
| singles_g_t1t1 | ERI times two singles |
| singles_g_t1t2 | ERI times a single and a double |
| singles_g_t1t1t1 | ERI times three singles |

No denominator, level shift, DIIS or iteration state occurs in this DAG. In
particular at T=0, E_corr=0, E_total=E_ref and R1=F_ov. A canonical RHF
reference makes F_ov approximately zero without deleting that dependence.

The inventory expands PySCF 2.14.0 `rccsd.energy`, the T1 part of
`rccsd.update_amps`, and `rintermediates.cc_Foo/cc_Fvv/cc_Fov`. Source URLs,
SHA-256 identities, version and Apache-2.0 license are recorded in
`tools/vibeqc_cc/source_manifest.json`, `NOTICE`, and `LICENSE.pyscf`.
The generator verifies installed upstream bytes before producing fixtures.
This is an audited inventory route, not a pdaggerq generation claim.
Program logical hashes encode factors, order, symmetry, dimensions and input
roles; separate source/inventory hashes retain provenance. JSON replay uses
the existing TensorIR schema. `validate_cc.py` exports a replayable program,
inputs, reference outputs and pack maps without adding a second IR format.

For the pinned PySCF update, let `D_ia = eps_i - eps_a - level_shift`, using
the actual `eris.mo_energy`, which need not equal diag(F). PySCF subtracts
these diagonals from its Foo/Fvv intermediates. Its numerator is therefore
`R1 + D*t1`, and its returned update is `t1_new = t1 + R1/D`.
The physical residual is **`R1 = D*(t1_new-t1)`**. Reference generation tests
two shifts (0 and 0.4) and deliberately different eps and diag(F). Reading
`t1_new` directly as R1 is explicitly rejected by the tests. The CC facade
never imports PySCF or calls an external amplitude update/solver.

## Independent coordinates and future adjoints

`amplitude_layouts(o,v)` reuses #145 `PackedLayout`. Singles use C-order
`(i,a)`. Doubles use C-order `(i,j,a,b)`; each orbit of `(i,j,a,b)->(j,i,b,a)`
stores its first flat representative, with weight 1 at fixed points and 2
otherwise. Concatenated independent vectors are `[pack(t1), pack(t2)]`.
Packing validates symmetry; it never projects a malformed array silently.

The chosen real coordinate inner product is

```text
<x,y> = sum_ia x1[i,a] y1[i,a] + sum_ijab x2[i,j,a,b] y2[i,j,a,b]
      = dot(pack(x1),pack(y1)) + sum_u weight[u] pack(x2)[u] pack(y2)[u]
```

For a packed coordinate differential, an ordinary Euclidean covector has
components `weight[u]*lambda[u]` when lambda denotes this metric's vector.
These are coordinate conventions, not an implicit spin-orbital Lambda
normalization. Future Lambda implementations must explicitly map their dual
projectors to this residual and metric. A plain unweighted packed dot product
cannot substitute for the dense contraction. Slice A defines no T2 projector
or Lambda solver and makes no derivative-validation claim.

In particular the physical excitation-state overlap is a different metric:
`<Phi|T(x)^dagger T(y)|Phi> = 2 sum x1*y1 + sum x2*(2*y2-y2_ijba)`.
Its exchange coupling and singles factor two are checked against explicit
determinant excitation norms, separately from the coordinate metric above.

## Interfaces, resource scope and verification

`evaluate(snapshot, provider, t1, t2)` requests only `ovov`, `ovvo`, `oovv`,
`ovvv` and `ovoo` with explicit `MOBlock` slots. The provider pins these blocks
for repeated evaluations. It returns E_ref, E_corr, E_total, physical R1,
per-term/group diagnostics, reference/Hamiltonian IDs and input/equation hashes.
The provider's transformation/cache budget and the interpreter's `max_bytes`
are separate bounds. The latter bounds logical retained arrays and outputs,
not Python/BLAS scratch or total process RSS. This unoptimized CPU reference
can retain many small diagnostics and is not a scalable production CC engine.

The independent determinant oracle is limited to 128 determinants before
allocation. It builds spin-summed one-body operators using explicit bit-string
fermionic signs, constructs the Hamiltonian and T, then evaluates terminating
exponential series. It neither reads the inventory nor implements the same
einsum equations. Polynomial evaluations with scaled/sign-flipped amplitudes
isolate the homogeneous diagnostic groups, preventing total cancellation from
hiding a missing term. Mutation tests remove disconnected energy and singles
contributions and exchange factors from the actual DAG.

Random cases include unequal occupied/virtual dimensions, nonzero singles,
pair-symmetric doubles and eightfold symmetric signed ERIs. Fixed PySCF
references additionally reuse H2, water and LiH from #147 with identical C,
integrals and Hamiltonian. Native provider tests check those values and run
fresh VibeQC RHF into the fixed-amplitude facade. The latter is a bridge smoke
check, not the converged HF→CCSD endpoint required by slice C.

Run from the repository root with `PYTHONPATH=.:python`:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m tools.generate_cc_references --output /tmp/cc-first.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m tools.generate_cc_references --output /tmp/cc-second.json \
  --compare /tmp/cc-first.json
python -m pytest tests/python/test_cc_equations.py tests/python/test_cc_references.py -q
python -m tools.validate_cc --output /tmp/cc-evidence --references tests/reference_data/cc/rccsd-a.json
```

Generation needs pinned PySCF; ordinary tests consume committed data. Native
bridge tests require the built CPU library via `VIBEQC_LIBRARY`. Controlled
values are checked per element with `atol=1e-11, rtol=1e-10`, alongside absolute
energy `<=1e-8` and physical residual `<=1e-9` gates; no older tolerance changes.
The evidence runner records numerical/representation success only. T2, solver,
GPU, (T), Lambda and gradients remain unexecuted/outside A.
