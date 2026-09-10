# Scientific compiler ownership

`python/vibeqc_compiler` is the importable scientific compilation subsystem.
It is installed beside `vibeqc`, whose public molecular API and native-library
selection remain separate. `tools/generate_*.py` and benchmark/reproduction
commands are clients of these packages. Source generation does not import the
public runtime, load `libvibeqc`, probe a GPU, or import PySCF, Torch or CuPy.
NumPy remains the existing dependency for recurrence/reference arithmetic.

| Owner | Responsibility | Allowed compiler dependencies |
| --- | --- | --- |
| `integral` | IntegralIR, scalar algebra, recurrence lowering, integral schedules and promotion | `common` |
| `tensor` | TensorIR, AD, optimization, planning, tensor CUDA emission/execution | `common` |
| `dft` | Discrete grids, AO jets, density ingredients, prepared tile execution | `common`; `ao_cuda` alone also uses the existing scalar `integral.expr` and `integral.cuda` |
| `xc` | Audited functional expressions, derivatives, point coefficients and XC execution | `common`, `integral`, `dft` |
| `common` | Backend/target contracts, finite compiler processes, artifacts, hashes, resources and evidence | none of the scientific or user-runtime packages |

The compiler owns mathematical IR and lowering. `src/integrals`, `src/tensor`
and `src/dft` own the corresponding native interfaces, runtime allocation and
execution templates; method and SCF code consume these interfaces. A compiler
package move does not promote a new scientific capability or retire a native
fallback. Native scientific ownership and retirement are tracked by #231.

`dft.ao.NativeAO` is an explicit adapter to the existing normalized native
basis ABI. Its runtime imports occur only during preparation. Likewise,
`dft.grid.owned_atoms` uses the public Atom conversion only when accepting
molecular input, and the fixture adapter constructs public shell records only
when requested. These three narrow exceptions are enumerated by the dependency
check. No SCF policy belongs in generic compiler code.

AO CUDA lowering borrows the same scalar graph and emitter as XC. Those two
neutral modules retain their historical IntegralIR paths; the dependency check
allows only these exact imports from `dft.ao_cuda`, without permitting DFT to
depend on integral recurrence or method scheduling. This avoids a second
scientific algebra implementation.

## Lowering and tuning modules

`integral.cuda_lowering` and `integral.cuda_emitter` retain callable facades.
The implementation resides in `integral.lowering`:

- `selection`, `algebra` and `common` handle validation, structured expression
  emission and shared CUDA support.
- `fock`, `fock_component` and `fock_tiled` own Fock consumer families.
- `force_packed`, `force_rys_thread`, `force_rys_component`,
  `force_rys_uniform`, `force_resident` and `force_subgroup` own force schedules.
- `dispatch` assembles the common kernel envelope and selects consumers;
  `legacy` retains explicit compatibility entry points.

Consumer modules depend on shared emission/algebra helpers. Shared helpers do
not call back into dispatch. Recurrence mathematics is still represented by
the existing IR and scalar graph; splitting files introduces no new equations.

`integral.autotune` delegates to `integral.tuning`. `analysis` estimates
candidate structure, `policy` enumerates/promotes schedules, `emission` writes
candidate source, `manifest` serializes promotion records, `resources` applies
resource gates, `process` handles external processes, `inputs` normalizes input,
`driver` coordinates a run, and `cli` parses arguments. The type-only reference
from analysis to policy is guarded by `TYPE_CHECKING`.

Generic CUDA targets, compilation, parsed resource records, artifact handles,
metrics and preparation synchronization are owned by `common`. Integral and
TensorIR execution no longer depend on one another's runtime classes. DFT and
XC use that same artifact cache and allocation lock. The public global resource
planner and local-profile hashing/atomic-JSON helpers are re-exported from their
original `vibeqc` APIs; their implementations are not duplicated. Shared evidence
and timing helpers do not import benchmark command modules.

## Checkout and installed usage

With the existing NumPy dependency available, CMake can run the generator
scripts directly from an uninstalled checkout. Each script bootstraps the
explicit `python/` package root; compiler libraries never manipulate `sys.path`.
CMake recursively tracks compiler leaves as generation dependencies and uses
the same source inventory as `vibeqc.autotune.source_identity`.

The wheel includes the integral manifests and copies the required native
templates from `src/tensor` and `src/dft`, plus the audited Libxc source and
license provenance from `external/libxc-7.0.0`. These inputs are included in the
sdist too. `setup.py` copies canonical inputs at build time without importing
either Python package; there is no second editable native source tree.
`common.paths` resolves each input by its stable repository-relative name in a
checkout or wheel. Independent numerical fixtures remain checkout inputs.

Ordinary user calculations do not import tuning/reference dependencies. User
autotuning still requires a matching source checkout, CUDA toolkit and the
existing optional PySCF validation dependency. A wheel may drive a byte-identical
checkout: the complete loaded compiler inventory must match before tuning.
An already imported, different compiler is rejected instead of loading a second
set of IR classes via path manipulation.

## Identities and compatibility

IntegralIR/TensorIR mathematical serialization and equation hashes, schedules,
and generated integral source are unchanged by the move. Source compatibility inventories necessarily
change because they now hash the first-class source paths and all nested leaves.
Tensor artifact and XC emission contracts explicitly use layout version **2**.
XC's expression hash already includes the exact expression-module bytes, so
its import changes intentionally alter provenance. The emitted source includes
the versioned contract identity; changing it does not change the scalar graph
or arithmetic. Across all seven functionals, both spin modes, orders 0–2 and
three schedules, 126 emitted XC variants differ only on the identity line.
Twelve TensorIR example/schedule sources remain exactly identical. Both installed and checkout compiler
inventories use the same logical paths, without absolute installation paths,
timestamps, bytecode or compatibility shim bytes. Old artifacts must be rebuilt;
binary content verification and numerical promotion gates are unchanged.

Compatibility modules under `tools/vibeqc_codegen`, `tools/vibeqc_tensor`,
`tools/vibeqc_xc` and `tools/vibeqc_dft` forward to their canonical owners. The
same applies to moved generic helpers under former integral/tensor paths.
Leaf modules alias the canonical module object, preserving enum/dataclass
identity, `isinstance` checks, monkeypatches and imports through both the former
`tools.` namespace and bare packages. Package facades delegate exports while
keeping the legacy search path confined to forwarding modules. They do not
reuse the canonical package `__path__`, which would load submodules twice.

Remove these shims after downstream callers have migrated for one release and
compatibility tests are the only remaining repository callers. Hash-pinned
reference exporters intentionally retain their exact source bytes and legacy
imports until a deliberate, independently verified reference regeneration;
they must migrate before removal too. Legacy manifest paths are symlinks to the
single canonical manifest, with the same removal condition.

## Structural verification and measured impact

Run `python tools/check_compiler_structure.py` to check import directions, or
add `--json` for a module-size/dependency inventory. The same check is a
pre-commit hook. Tests also import every compiler module in a fresh process
that rejects runtime/reference imports, check legacy module identity, and run
an uninstalled generator from an unrelated working directory.

At the #239 decomposition baseline, `cuda_lowering.py` had 6,813 lines and
approximately 282 KiB; `autotune.py` had 2,173 lines and approximately 86 KiB.
The largest lowering leaf after decomposition is `lowering/dispatch.py`
(1,274 lines, approximately 53 KiB), and the largest tuning leaf is
`tuning/driver.py` (704 lines, approximately 31 KiB). The compatibility facades
contain no second lowering or tuner implementation.

The decomposition and package migration each preserved **16 generated files /
26,629,262 bytes** exactly against revision `75472d0`: sm_120 production shell
bundles, DF, weighted ERI, and one-electron values/derivatives and inventories.
Single-run source generation measurements on the same workstation were:

| Generator | Before (s) | After package move (s) |
| --- | ---: | ---: |
| Production shell bundle | 2.727 | 2.827 |
| DF values | 0.180 | 0.187 |
| Weighted ERI | 0.154 | 0.158 |
| One-electron values | 2.141 | 2.149 |
| One-electron derivatives | 9.193 | 9.210 |

After integration with master `9a3aae0`, all 16 outputs also matched a
pristine checkout of that revision byte-for-byte (**26,652,374 bytes**). The
size difference from the earlier baseline is upstream precision-counter code.
All 13 native CPU suites and 38 selected scheduled RTX 5090 regressions passed;
the latter cover TensorIR intermediates/layouts, grid/AO execution and XC replay.

These are build-impact observations, not a runtime performance claim. Raw run
logs and generated build products belong in ignored `.artifacts/`, according to
the [evidence retention policy](evidence_retention.md).
