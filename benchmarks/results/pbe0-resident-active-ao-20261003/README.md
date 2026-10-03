# PBE0 resident active-AO force qualification

This is a **force-only**, explicit `resident_ao_cutoff=1e-16` experiment. SCF
remains dense; the default cutoff stays `None`. Same-binary dense/local complete
energy-plus-analytic-force endpoints retain the original 256-point tiles and
512 MiB device / 256 MiB host caps, including at 96 atoms. Local AO maps use an
optional 16 MiB reserve **inside**, not on top of, the total host allowance.

## Integrated-source rerun

The new campaign is measured from clean published `a718695de66d04af272addb61b2b28ebd6700eb4`:
source `46852852006dc81b164796e994bc335c4f336da3f03e38ba76bec3c3e0269111`,
library `c2f7c6e192a83ff09816af2e0834c3a22fb0190009e7592a9f2e5c3025b04254`.
Finite n1 jobs 5573 (3–48 atoms), 5574 (96 atoms), and 5578 (fresh zero-budget
control) retain the same protocol, optional cutoff, 256-point tiles and default
memory caps. Each pair uses the same binary and GPU allocation. The later merge
of actual master `d442177a6` leaves this production source identity unchanged;
the running measured checkout was not mutated.

All **156 new complete native calls** pass comparison against every
same-geometry reference repeat: maximum errors `1.042e-10 Eh` and
`3.336e-11 Eh/Bohr`. Reference arrays remain reused numerical oracles, not fresh
reference timings. These results do not replace or relabel the frozen campaign
below.

| Atoms | Dense warm (s) | Local warm (s) | Reduction |
|---:|---:|---:|---:|
| 3 | 0.344318 | 0.335861 | 2.46% |
| 6 | 0.741613 | 0.715882 | 3.47% |
| 12 | 1.659641 | 1.606517 | 3.20% |
| 24 | 4.996515 | 4.765913 | 4.62% |
| 48 | 17.857474 | 15.478295 | 13.32% |
| 96 | 77.158586 | 62.791596 | 18.62% |

All warm and moved-warm calls still take one SCF iteration. At 96 atoms,
moved-warm is 77.476661→63.082426 s; moved is 258.088302→248.678081 s at
12 iterations in both modes. Cold is 706.651282→512.251532 s with 26→25
iterations and shared compiler caches, so it does not isolate the force-path
gain. **Negative observations remain:** 24-atom cold is
106.795475→107.085434 s (21 iterations), moved 27.083108→27.358444 s
(12 iterations); moved also increases at 3/6/12 atoms. At 48 atoms cold
increases 185.115208→187.124945 s with 23→24 iterations. No sample is dropped.

The structural counts and cache behavior match the original campaign: at
96 atoms force contraction work is 5.8249% of dense GM², active AO mean/max is
164.516/553 of 768, and every warm map lookup hits. Initial discovery takes
21.609049 s for the same 18,119,393,280 order-two AO-jet values. All 9,216
tiles, 768 empty maps, and 21,516,784,080 partition-pair visits remain included;
the total numeric map bound stays 12,144,768 bytes inside the admitted reserve.

Run `python benchmarks/results/pbe0-resident-active-ao-20261003/verify.py --campaign current`.
The separately pinned `current-campaign.json.xz` / `current-storage.json` retain
56 exact UTF-8 members, including every force array, all force-call work, the
fresh zero-budget control and reproduction receipts. The source patch is empty
because this measured checkout is clean at its published commit. The original
bundle's bytes and hashes are unchanged. Both campaigns use the same verifier
with separate hardcoded source/library/job identities, and both receive the
semantic-corruption regression tests. This is not default promotion or evidence
that the overall GPU4PySCF gap is closed.

## Frozen measured identity

- Base: `b2ee9dd7fc6f82427ab56218d7b9d803687259be`, plus retained source patch.
- Source: `edb9bf98d454f356ef3dea4488a6a763343510bad25efb4d67f96dc877c27421`.
- Library: `473f05abacad690caa7a3e149aa309d723d021e1ca4fd4ec91b78512e16268f2`.
- n1 RTX 5090, finite `main` / `gpu:5090:1` Slurm jobs 5565 (3–48 atoms),
  5566 (96 atoms), and 5567 (3-atom zero-budget control).
- Native build on n5 with verified ccache and explicit CXX/CUDA launchers;
  library transfer directly from n5 to n1. Device visibility stays Slurm-owned.

Both modes include the triangular shared-claim synchronization repair, not the
optional indexed-page optimization. Later integration of master `86c422bdc`,
the shared API/cache parents and standalone repair #1776 is a distinct checkout
identity. These measurements are **not relabeled** as timings of that newer tree.

## Complete observations

Unchanged README PBE0/RKS, spherical def2-SVP, full unpruned moving 48×16×32
grid per atom, no density fitting or mixed precision. Each variant includes
cold, five warm, moved, and five moved-warm calls. The verifier checks every
native row against **all** same-geometry independent-reference rows, not only
the oracle paired by the original runner. All 156 native calls, including the
zero-budget control, pass `1e-8 Eh` / `1e-7 Eh/Bohr`. Maximum errors are
`1.069e-10 Eh` / `3.322e-11 Eh/Bohr`.

Reference arrays come from the preceding matched-grid planner campaign. They
are reused numerical oracles, **not fresh reference timings** for this campaign.

| Atoms | Dense warm (s) | Local warm (s) | Reduction | Force contraction GM² / dense |
|---:|---:|---:|---:|---:|
| 3 | 0.349358 | 0.338739 | 3.04% | 77.76% |
| 6 | 0.739908 | 0.723894 | 2.16% | 44.91% |
| 12 | 1.680070 | 1.629296 | 3.02% | 53.37% |
| 24 | 5.047796 | 4.805904 | 4.79% | 48.25% |
| 48 | 17.959758 | 15.602874 | 13.12% | 15.50% |
| 96 | 77.143137 | 62.791405 | 18.60% | 5.82% |

All warm and moved-warm calls take one SCF iteration. At 96 atoms, moved-warm
is 77.275831→62.915670 s. Cold is 779.258286→572.212382 s with 31→29 SCF
iterations; moved is 273.514653→248.561727 s with 13→12 iterations. Neither
variable-iteration pair isolates a force-path speedup. Ordered processes also
share disk compiler/artifact caches, so cold timing does not establish an
isolated compilation advantage.

**Negative outcomes remain:** 24-atom moved is 27.224856→27.670800 s at the
same 12 iterations; cold is 107.222327→108.039425 s at the same 21 iterations.
Moved time also increases at 3, 6 and 12 atoms. No sample is filtered or
normalized by iteration count. This does not solve the overall reference gap
and does not justify default promotion or updating the main README comparison.

## Structural work and bounded fallback

For 96 atoms / 768 AOs / 2,359,296 points, the force tiles retain mean 164.516,
maximum 553 AOs. Actual contraction work falls from 1,391,569,403,904 to
81,057,099,776 point-AO² units. This is an executed-dimension proxy, not an
allocation ratio, universal scaling proof, or complete-endpoint speedup factor.
Full-capacity device arenas remain charged for discovery and dense fallback.

Every cold/moved call discovers all 9,216 maps; all warm/moved-warm calls hit
the geometry-matched cache with **zero** discoveries. Initial 96-atom discovery
visits 18,119,393,280 sampled order-two AO-jet values and costs 21.638 s; it
performs no density contractions. Retained map bytes are 12,129,408, with
15,360 transient bytes, inside the admitted 16 MiB reserve. Geometry changes
rediscover maps; the matching density is rebound for every force call.

All 9,216 tiles, including 768 empty maps, still consume complete point and
geometry work. The 21,516,784,080 partition-pair visits are unchanged. Local AO
selection does **not** reduce the `G*A²` Becke work or the SCF/full-range
derivative work. The separate zero-budget run makes no discoveries, retains no
maps, and executes dense dimensions before any unaffordable discovery.

The cutoff is a sampled-jet heuristic, not a certified error bound. Independent
numerical gates and explicit default-off status remain necessary.

## Verify and reproduce

Run `python benchmarks/results/pbe0-resident-active-ao-20261003/verify.py`.
It authenticates every stored byte, recomputes the full summary and error gates,
and checks source/library/job identity, exact work, geometry reuse and budgets.
Host tests also corrupt forces, repeats, source identity, work and budgets to
verify that semantic failures cannot hide behind retained pass flags.

`campaign.json.xz` losslessly stores exact UTF-8 records, all force arrays,
reference arrays, all 12 force-work dictionaries per variant, source patch,
harness, native qualification and ccache/build receipts. `storage.json` binds
each member. Decompress with `lzma`/`json`; the verifier neither extracts nor
executes stored scripts. No failed outcome or cold sample is discarded.

Native qualification job 5564 separately passes 77 host/GPU/caller checks and
22 real-GPU memcheck checks with zero errors on the measured source/binary.
Reproduction requires the pinned base plus patch and retained basis input,
verified ccache, explicit launchers, and finite Slurm allocations. Do not swap
in a newer library while retaining these measured source identities.

The integrated checkout is separately rebuilt on n5 as source
`46852852006dc81b164796e994bc335c4f336da3f03e38ba76bec3c3e0269111`, library
`c2f7c6e192a83ff09816af2e0834c3a22fb0190009e7592a9f2e5c3025b04254`.
Finite n1 job 5572 verifies matching compiled/deployed identity, passes 89
host/GPU/caller tests, 34 memcheck tests with zero errors, and four triangular
claim host/plain/synccheck/racecheck tests. The exact log and script are retained
separately in `current-qualification.json` / `current-qualification.log.gz`.
That qualification receipt is device correctness evidence. The separate new
complete-endpoint campaign is described in **Integrated-source rerun** above.
