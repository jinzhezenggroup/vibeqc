# Experimental PBE0 Schwarz-indexed force pages

This directory now contains **synchronization-corrected** acceptance evidence.
The earlier pre-fix campaign is historical, not pooled with these observations.
Its original README, verifier and complete member-hash inventory are embedded
under `historical/` in the new bundle; every original compressed member remains
in published Git commit `0b99c6ce298f2726373f1909ad10a37b5acffc43` at this path.
Use `git show <commit>:<path>` to recover those bytes. No Release or external
archive is needed. Current-tree replacement avoids retaining superseded raw
copies under the unchanged aggregate evidence budget.

## Corrected complete endpoints

Finite n1 RTX5090 Slurm jobs 5552 (3–48 atoms) and 5551 (96). Both schedule-OFF
and schedule-ON processes use the **same corrected source/binary and allocation**:

- Source: `ffc146230c1b44103fba01a8b14a061deadbb29d17e5160c79d198fca6e7b10b`
- Library: `c43436f7e39901e01434e39d46e5a81e4d71ca3af3489c1d92a0791fe681d280`
- Measured production source corresponds to repair commit `0b99c6ce`.
  Evidence-only changes preserve that identity; the later merge of master
  `86c422bdc` does not relabel these observations as timings of the merged tree.

Each variant retains cold, five warm, moved and five moved-warm calls. All144
native calls pass **every same-geometry reference-repeat pair** under unchanged
`1e-8 Eh` / `1e-7 Eh/Bohr` gates. Maximum errors are `1.060e-10 Eh` and
`3.623e-11 Eh/Bohr`. Reference arrays are reused numerical oracles, **not fresh
reference timings**. All warm native calls take one SCF iteration.

| Atoms | OFF warm (s) | ON warm (s) | Reduction |
|---:|---:|---:|---:|
| 3 | 0.344474 | 0.266467 | 22.65% |
| 6 | 0.738196 | 0.613583 | 16.88% |
| 12 | 1.652050 | 1.377643 | 16.61% |
| 24 | 4.971842 | 4.454866 | 10.40% |
| 48 | 17.710285 | 14.971006 | 15.47% |
| 96 | 76.949750 | 63.354224 | 17.67% |

At96, moved-warm is76.872031→63.543085s. Cold is712.808365→519.115689s
(27→26iterations); moved is287.763397→243.399753s (14→12iterations).
**Not uniformly faster:** at48 moved is70.550929→76.673851s, with12→14iterations.
All stopping tolerances and every negative/variable-iteration observation are
retained; no iteration normalization or sample exclusion is applied.

Ordered processes share compiler/artifact caches, so cold measurements cannot
establish isolated cold-compilation or schedule speedups. The default remains
OFF. Neither the main README nor the large-system reference-gap conclusion is
promoted by this experiment.

## Work and correctness

The unchanged [PBE0 protocol](../pbe0-def2-svp-20261003/README.md) uses spherical
def2-SVP, unpruned moving48×16×32 grids, exact exchange and the original SCF,
screening and numerical gates. No DF, mixed precision or CPU production oracle
is introduced. `GENERATIVEQC_BOUNDED_SCHWARZ_SCHEDULE=1` is the only selector.

The schedule sorts geometry-live shell pairs, retains an exclusive row prefix,
and claims16 independent64-candidate pages per surviving product. Physical
orientation and the exact density/shell/AO predicates remain unchanged. This is
output-sensitive traversal, not universal linear-scaling ERIs. Tight prefix
budgets retain sorted triangular traversal; the explicit bounded fallback stays.
Product capacity is not actual page claims or executed primitive/root count.
Grid work remains unchanged (96atoms:21,516,784,080 partition-pair visits).

The unconditional CTA barrier **before each new shared claim write** protects
all prior consumers, including empty pages and screened/inactive skips. Job5550
checks the extracted production protocol in plain, synccheck and racecheck modes
plus its host invariant (4passes), exact source/library identity, through-f
independent CPU-ERI oracles for baseline/OFF/ON, batch/exact-prefix budgets, and
memcheck/initcheck (both0errors). Endpoints5552/5551 qualify the corrected binary
separately; old whitespace-equivalence evidence is not used for this repair.

## Recheck and reproduce

Run `python benchmarks/results/pbe0-screened-pages-20261003/verify.py`.
It verifies every stored byte identity, complete sample set, independent error
pair, same-binary/allocation identity, grid work count, median and iteration list.

`corrected-campaign.json.xz` is a JSON mapping paths to exact original UTF-8
records and receipts, losslessly compressed with XZ. `storage.json` binds every
member. Decode with Python `lzma.decompress` and `json.loads` without executing
stored content. Reproduction/qualification scripts and native receipts are
included; build the pinned source with verified ccache and explicit CXX/CUDA
launchers, then run the retained scripts through finite main/gpu:5090:1 Slurm.
All old source-only counters, intermediate experiments and pre-fix qualification
are recoverable from the historical commit, not current corrected acceptance.
