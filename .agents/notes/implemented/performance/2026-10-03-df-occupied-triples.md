# Decision: compiler-owned DF triples on occupied tiles

Status: implemented
Date: 2026-10-03

## Problem

Hundreds-AO native DF CCSD had bounded virtual-integral storage, but the native
conventional triples scalar traversal repeatedly recomputed W contractions
inside projection/permutation loops. Substituting a Q contraction at each
integral access would add another nested reduction. This would conserve memory
while making the actual work worse. Issues #1763–#1765 call for compiler-owned
fusion, independently gated precision and explicit bounded-buffer ownership.

## Decision

Add an internal FP64 CUDA triples energy phase using the existing audited
`W_TERMS`, `V_TERMS`, `R3`, `SLOW_TABLE`, `VP`, `OP` and inventory identity.
The compiler derives a DF panel, two W GEMMs and a fused energy scalar program.
The shared TensorIR GEMM contract owns operand ordering, transposition and
dimension expressions; the emitter supplies canonical strided-view addresses.
The shared scalar C++ emitter owns V and the energy expression. Native code
owns allocation, BLAS, stream ordering, admission and deterministic reductions.

For one occupied triple `i>=j>=k`, a panel for each distinct first occupied
index is `panel_i[a,b,f] = sum_Q Bov[Q,i,a] Bvv[Q,b,f]`. Physical Bvv symmetry
identifies this with `(ia|fb)`. Each of six occupied orders `(I,J,K)` builds
`W[a,b,c] = sum_f panel_I[a,b,f] T2[K,J,c,f]
- sum_m Ovoo[I,a,J,m] T2[m,K,b,c]`. Six W cubes are retained; V is evaluated
elementwise in the fused epilogue.

The original full ordered-occupied expression uses six virtual Z rows. After
rewriting the triangular virtual traversal as a full cube, each virtual row has
the same complete sum by dummy-index relabeling. Their sum can therefore be
replaced by six times row `abc`, with the original full-domain normalization.
Grouping occupied triples still requires **all six outer occupied orders**.
For each order the epilogue forms R3 from W+V/2 and contracts it with the six
left-W entries in row `abc`. The six resulting products carry factor two and
are divided by the physical denominator times the occupied 6/2/1 multiplicity.
Repeated occupied indices still traverse six orders before this division.

## Rejected alternatives

- Keeping just one occupied order and adjusting a degeneracy factor is invalid.
  Even physically symmetric Gram integrals and pair-symmetric T2 do not make
  individual projected contributions exchangeable. The retained prototype for
  `(o,v)=(3,4)` gave an error about 4221 in its deliberately unscaled random
  example; summing all six orders restored agreement around 1e-11. The prototype
  logs' `occupied` column is the rejected shortcut; `fullerror` is the corrected
  complete sum. Neither is a molecular energy benchmark.
- Folding both occupied and virtual triangles under this new epilogue double
  counts or drops terms. Only the occupied domain is triangular here.
- Adding a Q loop to each old scalar W access amplifies repeated contraction
  work. Building bounded panels and W moments exposes ordinary GEMMs instead.
- Lower precision and asynchronous overlap are not promoted by this change.
  FP64 is the qualification baseline for #1764; one stream gives explicit
  producer/consumer lifetimes for #1765 before any overlap is introduced.

## Invariants

- With `T=o(o+1)(o+2)/6`, W work is exactly `6T(v^4+ov^3)` summands.
  Panel work is `P Q v^3`, where the actual cache-miss count `P` is reported.
  The fused epilogue visits `T v^3` points. Formal leading scaling is unchanged.
- The one-to-three-entry LRU cache stores only occupied integral panels. For
  one-buffer fallback, consume every W GEMM using the current panel before
  reusing it. All work uses the same Context-owned stream and BLAS handle.
- Six W cubes plus at most three panels are retained; no full ovvv, vvvv, T3 or
  denominator tensor is created. Staged inputs, partial sums, tile energies,
  4-MiB BLAS workspace and 96-MiB provider allowance are included in admission.
  Composing callers must separately charge retained host/CC state.
- Check dimension/leading-dimension and complete semantic-work overflow before
  reading inputs or allocating. Infeasible requested panel counts fall back to
  one; infeasible one-panel budgets fail before device execution.
- Validate finite inputs, Bvv pair symmetry and the canonical gap. Device
  arithmetic errors are sticky. Drain/check before publishing; borrowed result
  destinations outlive exception-path cleanup. Endpoint timing includes provider
  and arena destruction.

## Evidence

`tests/python/test_df_occupied_triples.py` checks generated CPU energy against
the audited original triangular sum, including all equal/two equal/distinct
occupied cases. Individual panel and moment elements are checked against direct
index loops. Native tests cover the same energies, exact work counts, one-panel
fallback, bitwise repeated results, exact-budget/one-byte-short behavior,
nonfinite inputs, symmetry/gap rejection, arithmetic overflow and preflight
before null-input access. Exact Gram factorizations of committed molecular ERIs
also compare against pinned independent PySCF 2.14.0 H2O/NH3/CH4 triples energies.

Large diagnostics use the native symmetric DF source from #1782 at independent
converged orbital/amplitude frames. The energy acceptance gate is 3e-10 Eh.
Ethane 230 AO and benzene 264 AO qualify the complete native triples phase;
they do not qualify cold-start molecular RHF/CCSD solves or forces. Original
large per-factor gates remain unresolved as documented in the
[source-symmetry note](../numerics/2026-10-03-df-cc-source-pair-symmetry.md).
The compact [evidence record](../../../../benchmarks/results/df-occupied-triples-20261003/qualification.json)
binds final timings, work/storage counts, independent errors and binary identity.
The first-build diagnostics excluded provider destruction from internal timing;
only the final complete-phase timings are retained in that evidence record.

Reproduce CPU tests normally, or set `GENERATIVEQC_DF_TRIPLES_CUDA_TEST=1` and
`GENERATIVEQC_LIBRARY=$PWD/build-cuda/libgenerativeqc.so` inside finite
`srun --partition=main --gres=gpu:5090:1 --time=00:10:00` allocations. The native
pytest fixture builds `tests/native/df_triples_probe.cpp` with ccache; the same
adapter is accepted by `benchmarks/df_triples_native_probe.py --probe ...
--library ... --input ... --state ... --reference ... --output ...`.
The input is the native-source symmetric DF solver replay file; state/reference
come from `benchmarks/df_ccsdt_large_oracle.py`. Full ignored artifacts and failed
domain-rewrite prototypes are preserved under the isolated worktree's
`.artifacts/occupied-triples/`; no new external archive is published.

## Consequences and revisit conditions

This is an internal supplied-state energy phase. Public molecular DF CCSD(T),
Lambda/response and nuclear forces still need complete composition and their
own independent acceptance gates. The phase itself performs no CPU oracle
numerics. Host preflight and scheduling do not replace native contractions.

Revisit panel traversal/cache order when complete endpoint profiles show panel
misses dominate; storage remains explicit and a bounded fallback remains
required. Revisit mixed precision only with component-specific error budgets
and independent FP64 verification. Revisit additional streams only after
events, reuse fences and failure-path drain behavior have equivalent coverage.
