# Complete RCCSD residuals and CPU solver

The A baseline and its reviewed commit remain documented in `rccsd.md`.
Slice B adds `tools.vibeqc_cc.doubles.build_ccsd_program(o,v,form=...)`.
All inputs/energy/singles conventions from A continue unchanged. This remains
an internal conventional all-electron real restricted CPU implementation.

## B: physical doubles and auditable algebra

The doubles projection is the normalized opposite-spin determinant

```text
R2[i,j,a,b] = <Phi_i(alpha),j(beta)^a(alpha),b(beta)| exp(-T) H_N exp(T) |Phi>
|Phi_i(alpha),j(beta)^a(alpha),b(beta)>
    = a†_(a alpha) a_(i alpha) a†_(b beta) a_(j beta) |Phi>
```

Thus R2 has the same simultaneous `ijab↔jiba` symmetry as spatial T2;
it is not separately antisymmetric. At T=0, `R2[i,j,a,b]=(ia|jb)`.
The A coordinate metric applies unchanged to the full R1/R2 vector. Lambda
implementations must map their physical dual projectors to this convention.

`doubles.DEFINITIONS` is an ordered exact-rational inventory traced to pinned
PySCF 2.14.0 `rccsd.update_amps` (CCSD branch, not CC2) and `rintermediates`
`cc_Foo`, `cc_Fvv`, `Loo`, `Lvv`, `cc_Woooo`, `cc_Wvvvv`, `cc_Wvoov`, `cc_Wvovo`.
Source bytes, Apache-2.0 license and attribution remain in A's manifest/NOTICE.
VibeQC expands those definitions with full Fock diagonals restored. No orbital
energy, denominator, level shift or update formula occurs in the inventory.

| Diagnostic | Role |
| --- | --- |
| D01_driving | Bare ovov integral |
| D02_singles_v / D03_singles_o | Singles dressing and simultaneous exchange |
| D04_oo_ladder / D05_vv_ladder | Occupied/virtual ladders contracted with tau |
| D06_virtual / D07_occupied | Full Lvv/Loo contributions and pair permutations |
| D08_ring / D09_exchange / D10_cross | Mixed ring/exchange contractions |

All intermediate definitions and group outputs are replayable TensorIR nodes.
`form='expanded'` distributes every product of sums and explicitly alpha-renames
dummy indices to produce input-only contractions. This is the higher-memory
reference DAG, with no reused intermediate inside residual contractions.
`form='shared'` retains reusable L/W/tau nodes; `form='optimized'` applies #145's
conservative optimizer. Tests compare every diagnostic, not just the final
residual, across forms. `diagnostics=False` retains only energy and full R1/R2
outputs without changing their equations.

With PySCF's actual `D1_ia=eps_i-eps_a-level_shift`, its doubles denominator is
`D2_ijab=D1_ia+D1_jb`. Restoring removed diagonal Fock terms gives
`R2=D2*(updated_t2-t2)`, including **two** virtual level shifts in D2. Reference
generation uses both zero and nonzero shift, with supplied eps different from
diag(F). Production evaluation never calls this PySCF update.

Independent validation includes explicit determinant projections over five
occupied/virtual shapes, per-intermediate pinned PySCF references, and missing
ladder/ring/exchange mutations. All forms preserve the #138 per-element gate
and absolute energy/residual limits of 1e-8/1e-9. Two reference generations
must have identical case hashes. Reproduce with:

```bash
PYTHONPATH=.:python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m tools.generate_cc_references --full --output /tmp/cc-b-1.json
PYTHONPATH=.:python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m tools.generate_cc_references --full --output /tmp/cc-b-2.json \
  --compare /tmp/cc-b-1.json
PYTHONPATH=.:python python -m pytest tests/python/test_cc_doubles.py \
  tests/python/test_cc_doubles_references.py -q
PYTHONPATH=.:python python -m tools.validate_ccsd --output /tmp/cc-b-evidence
```

## C: CPU iteration and final acceptance

`tools.vibeqc_cc.solve(snapshot, provider, options=SolverOptions(),
t1=None, t2=None)` is the internal CPU facade. It consumes the existing #147
reference and conventional CPU provider; it does not register a public method
or call an external CC solver. `PreparedCCSD` first validates the reference,
provider identity, amplitude shapes/FP64/finiteness/pair symmetry and logical
budget, then checks physical canonical denominators. Only afterward does it
dry-run the complete seven-block provider cache transition and request the
MO blocks. Pin/LRU policies and existing cache hits are included in the
collective preflight, under the provider lock; insufficient collective capacity
fails before the first AO read. This CC adapter reads #147's internal cache
accounting without extending its shared API. The A energy/T1 facade also rejects
illegal amplitudes and impossible interpreter budgets before conversion.

Default initial amplitudes are T1=0 and MP2-like T2=(ia|jb)/D2 with the
**unshifted physical** denominator. Users may supply both T1 and T2 together.
Each Jacobi trial uses `(1-damping)*R/D_shifted`; damping is the fraction of
the previous amplitude retained. D1 includes one level shift and D2 includes
two. These shifts are iteration controls only: every physical residual and
the final equation remain unshifted.

CC DIIS owns a separate bounded history. Each stored amplitude has its own
freshly evaluated physical residual; packed residuals are multiplied by the
square root of the A orbit weights to reproduce the dense coordinate metric.
Singular extrapolation systems drop the oldest entry; nonfinite or extreme
coefficients cannot be silently accepted. Setting `diis_size=0` disables DIIS.

A candidate converges only when both energy change and the maximum absolute
elements of R1/R2 meet their tolerances. Before acceptance, a fresh execution
of the fully expanded DAG must independently meet the physical-residual
threshold and reproduce the energy. This is an independent algebraic replay,
not an independent external equation source; the latter is checked separately
by the endpoint validation runner. No derivative through iteration traces is
performed. Tolerances cannot be weakened beyond 1e-8 Eh energy and 1e-9 residual.

| Condition | Behavior |
| --- | --- |
| Normal root | `status='converged'`; complete energy/R1/R2 history and final expanded checks |
| Iteration limit | `status='not_converged'`; last finite state and history retained |
| Numerical overflow/nonfinite equation | `status='nonfinite'`; last finite state retained; energy is `None` if never evaluated |
| Illegal amplitudes/options, near-zero or nonnegative physical denominator | Explicit `ValueError` before MO conversion |
| Unsupported reference, missing virtuals, frozen core/open shell | Rejected by #147 or the CC facade; never silently changed |
| Wrong reference/provider or closed provider | Explicit failure; no stale-state reuse |

`max_bytes` bounds logical interpreter storage plus a conservative allowance
for solver/DIIS arrays. The provider has a separate transformation/cache
budget. Python containers, allocator overhead and BLAS workspaces are not RSS
bounds. Explicit JSON replay exports and their Python-object storage are
outside the numeric-buffer budget. This is a small CPU correctness baseline.

`CCSDResult.write(path)` saves initial and final amplitudes, owned reference
inputs, all required MO blocks, options, source identities, energy decomposition,
per-iteration physical residuals, timings and failure reasons. A SHA-256 input
identity protects replay. `python -m tools.replay_ccsd input.json --output
replayed.json` rebuilds the same validated reference and repeats iterations
using the saved MO blocks without new AO work. Timings need not reproduce;
mathematical state and histories do. A failure replay is not convergence.

The package-level facade exposes the solver without requiring callers to know
its module layout. For a small validated RHF reference and open conventional
CPU provider:

```python
from tools.vibeqc_cc import SolverOptions, solve

result = solve(snapshot, provider, options=SolverOptions(
    energy_tolerance=1e-12, residual_tolerance=1e-10,
))
if not result.converged:
    raise RuntimeError(result.reason)
print(result.correlation_energy, result.total_energy)
result.write("ccsd-state.json")
```

This remains the internal correctness interface, with the reference/provider
domain and failure rules above. GPU, open-shell, frozen-core, triples and
derivative requests cannot be inferred from this CPU facade.

## Molecular evidence and reproduction

`tools.generate_cc_endpoints` reuses exact #138 H2, He, H2O, NH3 and CH4 inputs.
For He only, an explicit s primitive of exponent 0.25 is added to STO-3G: the
one-orbital basis otherwise has no virtual and cannot test a nonzero CC
correlation correction. This basis modification is saved in the fixture.
Two-electron H2/He are independently checked against FCI. PySCF 2.14.0 supplies
the multielectron references; full source versions/hashes, actual coefficients,
AO/MO integrals and two-generation stability evidence are retained.

Each system has both a same-C/integral-provider CCSD run and a genuinely new
VibeQC-native RHF→CCSD run. Occupied and virtual overlap rotations align
amplitudes across phase changes and degenerate subspaces; occupied/virtual
mixing is checked separately. The endpoint runner transforms independent saved
AO integrals with the **actual new C**, compares every native MO block, and
re-evaluates PySCF's physical residual at the VibeQC final amplitudes. It does
not interpret `update_amps` as a residual or use bridge smoke as an endpoint.

All five systems have at most 9 AOs, so #147's existing <=12-AO exporter is
sufficient; no shared native HF/MP2 interface changes are needed. Endpoint
tests use 1e-12 energy-change and 1e-10 residual convergence, with independent
total-energy/physical-residual acceptance at the original 1e-8/1e-9 gates.
Cross-library converged amplitude comparisons use 1e-8 absolute/relative;
raw MO tensors keep 1e-11 absolute, 1e-10 relative gates.

```bash
export PYTHONPATH=.:python OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VIBEQC_LIBRARY="$PWD/build/cpu/libvibeqc.so"
python -m tools.generate_cc_endpoints --output /tmp/cc-endpoints-first
python -m tools.generate_cc_endpoints --output /tmp/cc-endpoints-second \
  --compare /tmp/cc-endpoints-first
python -m pytest tests/python/test_cc_solver.py -q
python -m tools.validate_cc_solver --output /tmp/cc-solver-evidence
```

GPU/#149, (T), Lambda and gradients remain outside #148. No performance or
public capability promotion follows from these CPU correctness results.

## CPU milestone acceptance map

PR #215 delivered the scientific A/B/C implementation. The package facade now
exposes that complete implementation and the documentation distinguishes its
current capabilities from the historical A-only description.

| #148 requirement | Implementation and verification |
| --- | --- |
| Restricted normalization, pair metric and full Fock dependence | `equations.py`, `inventory.py`; random-amplitude, polynomial-group and pack/unpack tests in `test_cc_equations.py` |
| Complete physical R1/R2, permutation factors and shared/expanded equivalence | `doubles.py`; determinant projections, mutation tests and pinned PySCF intermediate comparisons in `test_cc_doubles*.py` |
| Independent CC iteration controls and strict final root | `SolverOptions`, `PreparedCCSD`, `solve`; physical denominator, separate DIIS and expanded final-residual tests in `test_cc_solver.py` |
| Native HF/provider integration, H2/He FCI and multielectron molecules | Five same-C and five fresh-native-HF endpoints in `tools.validate_cc_solver`; preserved independent inputs in `tests/reference_data/cc/endpoints/` |
| Nonconvergence, invalid references/amplitudes, budgets and nonfinite states | Explicit solver/provider failures, preflight-before-AO-read regressions and replay failure tests |
| Audited equations and replayable results | `source_manifest.json`, upstream license/NOTICE, `CCSDResult.write`, `tools.replay_ccsd`, `test_cc_provenance.py` |

The completion audit reran all ten endpoints against pinned PySCF 2.14.0 on
the existing scientific implementation (`ce3b6c89a418985d5b8a979e5adba5812b5a6145`).
Every independent integral, aligned-amplitude, energy, FCI and physical-residual
gate passed; the maximum independent residual was `1.1111840207404525e-11`.
Two fresh generations had identical arrays for all five molecules. The
package-facade change then passed all 24 solver/provenance tests, including
the native-HF endpoints; it changes no equation or solver arithmetic. Earlier
raw states and equation/reference provenance remain in
`benchmarks/results/rccsd-148-{a,b,c}/`; the commands above reproduce the audit
without committing another copy of temporary run logs or states.
