# Composed RCCSD(T) energy + analytic-force API (#150 C / #155 C binding slice)

`tools.vibeqc_cc.ccsd_t_api` composes the independently validated RCCSD and
standard perturbative-triples implementations for energy and now exposes a thin
analytic-force binding over the complete #746 RCCSD(T) gradient owner. It does
not introduce another CC, Lambda, Z-vector, or nuclear-derivative equation stack.

The supported scientific model is the same one fixed by #150: real canonical
closed-shell RHF, all electrons active, conventional four-center integrals and
standard noniterative `(T)`. Frozen-core, open-shell, ECP and alternative
triples approximations are not silently substituted.

## Execution contract

`rccsd_t_method_capabilities("rccsd(t)")` reports the internal Python
composition's `energy` and `forces`; `"ccsd(t)"` is an alias.
`native_public=False` still describes that internal force facade. Separately,
the public native registry exposes CPU and CUDA `RCCSD(T)` **energy only** through
`VIBEQC_METHOD_RCCSD_T` / `Calculator("ccsd(t)")`, including homogeneous
prepared batches. The native route and its host handoff are documented in
[`rccsd_t.md`](rccsd_t.md#native-cuda-public-interface); it does not require this
internal Python composition API.

`rccsd_t_energy(...)` remains energy-only and rejects `compute_forces=True`.
`rccsd_t_force(source, ...)` delegates directly to the qualified #746 endpoint,
which constructs one coherent fresh RHF -> RCCSD -> corrected-Lambda -> total-Z
-> derivative-consumer chain. It never reuses an unrelated energy state merely
because its dimensions match.

`rccsd_t_energy(snapshot, provider, backend=...)` runs

```text
validated RHF snapshot
  -> physical RCCSD solve
  -> independent expanded R1/R2 acceptance
  -> exact accepted CC-state identity
  -> bounded standard (T) tiles
  -> E_HF, E_CCSD, E_(T), E_corr and E_total
```

The CPU route uses the audited tiled reference. The production GPU composition
uses `backend="cuda-resident"`: the RCCSD stage is the #555 resident solver and
the triples stage is the #448 generated bounded-tile executor. The older
host-staged `backend="cuda"` RCCSD helper is deliberately not promoted by this
facade. CUDA requires an explicit `CudaCompilerAdapter` and a `pathlib.Path`
compilation cache. Invalid requests fail before the RCCSD solve; CUDA errors
never silently select CPU execution. `options=SolverOptions(...)`, raw `t1/t2`
or an identity-bearing `warm_start` are forwarded to the RCCSD facade.

The borrowed conventional CPU integral provider and validated RHF snapshot
are the existing RCCSD preparation contract, including for resident execution.
There is no new AO transformation or provider read after CCSD acceptance.
`ovvv`, `ovoo`, `ovov`, `fov` and orbital energies come from the exact retained
`CCSDResult.replay_inputs`; amplitudes come from its final `t1/t2`, never the
initial guess. The audited equations and fixed triples denominator guard are
reused unchanged on both backends. CPU oracle comparison on CUDA is opt-in via
`triples_oracle=True`, and is disabled by default.

## State and lifetime contract

A successful `(T)` calculation is impossible unless RCCSD first converges and
passes its independent physical residual replay. Resident CUDA results must
also carry `resident_solved_state_identity`; that identity is bound into the
RCCSD(T) result provenance. If RCCSD exhausts its iterations or becomes
nonfinite, no triples value, RCCSD(T) correlation energy or RCCSD(T) total
energy is published.

The result retains `ccsd` and its replayable `state`, plus tile diagnostics in
`triples` on success. Failed CCSD states may retain a last finite diagnostic
CCSD energy, but this never counts as an RCCSD(T) energy. Provider, compilation,
allocation and triples failures raise for a single point; batch execution
captures each exception independently.

The RCCSD facade's resident convenience owner closes before the triples phase.
The final amplitudes and replay inputs are host-owned. `CudaTriplesTiles`
extracts and uploads exact tile feeds into one resident owner per tile, closes
that owner, and proceeds to the next tile. This is an explicit host handoff;
there is no persistent CC-to-triples device-pointer ownership contract.

## Memory and work contract

Triples never allocate a complete `nocc^3 * nvir^3` T3 or denominator tensor.
`triples_max_bytes` is passed to every CUDA tile plan and infeasible plans fail
explicitly. `vir_chunk_size` changes execution partitioning, not method
semantics.

Occupied space is full within each tile. The virtual `a` ranges partition the
triangular `a >= b >= c` domain, so `virtual_triple_count` is
`nvir * (nvir + 1) * (nvir + 2) / 6` independently of the tile partition.
Prefix-bounded virtual labels retain full W1 summation axes. This bounds tile
storage; it does not promise constant work or constant host storage with system
size. Dense CC replay inputs and final amplitudes remain retained on the host.

`SolverOptions.max_bytes` bounds the existing CCSD logical storage contract;
`triples_max_bytes` bounds each CUDA tile plan independently. The endpoint
device peak is the maximum of the CCSD combined plan reservation and the triples
tile peak plus the caller's `provider_peak_bytes` reservation, since the solver
and tiles execute sequentially. Per-tile peaks, host triples input bytes and
CCSD logical bytes are reported separately. These are numeric-buffer accounting,
not measured process RSS or a total host/device memory cap. CPU execution reports
zero device bytes and an unknown (`null`) triples workspace peak; the CUDA tile
budget does not enforce a CPU workspace limit.

## Results and endpoint artifacts

`RCCSDTResult` reports the following quantities separately:

- RHF reference energy;
- RCCSD correlation energy;
- perturbative `(T)` correction;
- total correlation energy `E_CCSD + E_(T)`;
- total electronic+nuclear energy `E_HF + E_CCSD + E_(T)`.

The deterministic `ccsd_state_identity` binds exact replay inputs, final
amplitudes, CCSD energy, reference/integral/equation identities and, for CUDA,
`resident_solved_state_identity`. `result_identity` additionally binds the
triples inventory, backend, partition and all energy components. Elapsed times
are excluded from these scientific identities. Failure artifacts carry a
diagnostic state identity without certifying a converged root.

Provenance also records tile count, semantic work count, device-plan peaks,
generated triples artifact keys, runtime device and upstream CCSD provenance.
`timing.ccsd_s` covers the complete facade call (preparation through independent
replay and owner cleanup); `timing.triples_s` covers input extraction and tile
execution. `timing.endpoint_s` covers both phases and identity bookkeeping.
`timing.triples_detail` preserves CUDA extraction/compilation/upload/run/download
timings; CPU tiles supply no internal timing breakdown. HF generation precedes
this endpoint and is outside its timing scope.

`result.write(path)` returns and writes a compact JSON record with
`schema="vibeqc.rccsd-t.endpoint/1"`. `record_hash` is the SHA-256 of canonical
JSON of every other field, including timings. Consequently repeated identical
scientific results can have different artifact hashes. Serialization uses
`allow_nan=False`; nonfinite values fail before opening the destination.
The artifact contains energies, provenance and tile scalars, not amplitude,
integral or iteration-history arrays. Use `result.state.write(path)` separately
when a full CCSD replay artifact is needed.

## Homogeneous prepared batches

`PreparedRCCSDTBatch` and `rccsd_t_batch_energy` provide the #150 C batch-state
contract at the internal post-HF facade. All items must have the same
`(nocc, nvir)` shape, checked before any provider read or solver execution.
Empty input is valid and returns `shape=None, items=()`. Invalid global execution
settings and force requests are rejected even for empty batches.
Every item owns independent amplitudes, DIIS history,
convergence and failure status; failure of one item does not corrupt its
neighbors. The control loop is sequential today, while generated compiler
artifacts are reused through the shared cache. No ragged or padded batch claim
is made.

Preparation materializes the input pairs and borrows their snapshots/providers;
it creates no CCSD or device owner. `prepared.execute()` creates fresh per-item
state on every call and returns input-ordered indexed outcomes. Nonconvergence
retains the item's CCSD state with no triples/total energy; exceptions produce
`status="error"`, a reason, and `result=None`. Subsequent items continue. Batch
settings do not accept shared raw amplitudes or warm-start state. Providers stay
caller-owned and must remain usable for their item's CCSD solve.

The internal homogeneous session remains useful for validation. The public
native C `PreparedBatch` boundary now also executes CPU RCCSD(T) energy for
homogeneous `(nocc,nvir)` groups; it owns each item independently and does not
reuse amplitudes across changed geometries.

`PreparedRCCSDTForceBatch` and `rccsd_t_batch_forces` provide the matching
#155 binding for analytic forces. Force batches are homogeneous in `(nocc, nvir)`,
execute each source with a fresh complete-gradient owner, preserve input order,
detach successful force arrays, and isolate item failures. No amplitude, Lambda,
Krylov, or device state is shared between items.

## Public native boundary

`VIBEQC_METHOD_RCCSD_T` is now active for the qualified native **CPU energy**
path. `Calculator("rccsd(t)")` and its `"ccsd(t)"` alias execute a native owner
that reuses the existing native RCCSD solve, retains the exact canonical orbital
energies from that reference, and evaluates standard `(T)` with a C++ header
generated from the audited `tools/vibeqc_cc/triples.py` inventory. The generator
AST-reads only the literal scientific inventory and recomputes the same canonical
inventory hash, so build-time code generation has no NumPy/PySCF runtime
dependency and does not introduce a second handwritten triples table.

The native energy owner never materializes full T3. It retains the accepted
RCCSD problem/final amplitudes and allocates only six W blocks, six Z blocks and
one occupied-cube scratch block for one triangular virtual triple at a time.
The combined retained-state plus triples workspace is admitted against the
existing correlation memory budget. Diagnostics publish `E_(T)`, virtual-triple
count, workspace bytes and the audited triples inventory hash separately from
the RCCSD correlation diagnostics.

The current public boundary is deliberately narrower than the internal gradient
facade:

```text
CPU energy:                 yes
CPU homogeneous batch:      yes
native CUDA RCCSD(T):        no
native/public forces:        no
DF/frozen-core/open-shell:   no
```

CUDA requests fail rather than running the CPU evaluator under a CUDA label.
Force requests also fail rather than returning RCCSD/HF derivatives.

The internal #746 CPU force chain now executes its generated Lambda, parameter-
response, `(T)` VJP, raw-Hamiltonian, canonicalization and AO back-transform
TensorIR through the common #772 `NativeTensorProgram` backend. One bound CC
lifecycle owns a `NativeCCTensorExecutor` and caches compiled artifacts by exact
program identity; no response equation is copied into handwritten C++. The
generic backend keeps its 4096-node default. The qualified CC response owner
explicitly requests an 8192-node ceiling so the NH3 final triples-response tile
(5258 nodes) is admitted without widening unrelated TensorIR consumers.

This does not yet advertise public `forces`: the remaining #155 C work is to
bind the public C++ method-owner lifecycle/result publication to this already-
native response/gradient chain. It must not add a handwritten CCSD(T)-specific
Lambda/Z implementation.

## Validation

CPU endpoint tests cover H2, He, H2O, NH3 and CH4 using committed reference
inputs. H2/He exercise the approximately-zero triples limit; H2O/NH3/CH4 carry
nonzero `(T)` corrections and are checked independently from the CCSD energy.
The tests also cover nonconvergence, force/backend rejection, endpoint artifact
serialization, homogeneous batch admission, ragged-batch rejection and
per-item failure isolation.

```bash
PYTHONPATH=python:. python -m pytest tests/python/test_ccsd_t_api.py \
    tests/python/test_cc_triples.py tests/python/test_cc_triples_tiles.py -q
```

Real-device numerical and memory evidence for the two GPU stages remains in the
retained #555 RCCSD resident evidence and #448 triples-tile evidence. A composed
CUDA run uses those exact owners and records both identities in the endpoint
result; it does not reinterpret either benchmark as a new performance claim.
The endpoint tests cover CUDA admission/dispatch with a test double and do not
claim new composed GPU numerical qualification. Every real-GPU test or benchmark
must run through Slurm on `main`, with `--gres=gpu:5090:1` and a finite `--time`,
preserving scheduler-assigned device visibility.

---

Implementation attribution:

Agent: ChatGPT
Model: GPT-5.6 Sol
