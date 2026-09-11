# Restricted local-space evidence (#182)

The [report](report.json) and [array archive](states.npz) retain native RHF
reference exports, localized occupied rotations/Fock couplings, projected
virtual spaces, pair spectra/ranks, amplitudes, integrals, actual MP2 differences,
timings and numeric-storage reservations. This is a CPU prototype of restricted
local spaces and projected canonical-MP2 diagnostics; it does not claim local
CC energies, gradients, linear scaling, or production acceleration.

## Independent and full-space gates

H2, water, LiH and an f-containing HeH+ fixture each use both conventional and
DF Hamiltonians. Native HF and canonical MP2 energies agree with #147's pinned
independent PySCF 2.14.0 references within `1e-9 Eh`. Each DF comparison uses its
own auxiliary metric identity; fitted and conventional references are never
interchanged. The largest observed reference differences are below `4.7e-13 Eh`
for HF and `1.2e-13 Eh` for MP2.

Mulliken Pipek–Mezey localization matches independently generated PySCF
objectives within `1e-9`; the native implementation's maximum Jacobi gradient
must be below `1e-10`. The independent coefficients/objectives reproduced
identically across two generation runs. Reference provenance and the repetition
record are in `tests/reference_data/local-spaces/`.

Retaining all projected virtual functions and PNO directions recovers canonical
MP2 energies to about `2.1e-17 Eh` and amplitudes to about `1.3e-16` in these runs.
The occupied Fock remains noncanonical after localization. Recovery uses exact
occupied tensor rotations, including both orientations of off-diagonal pairs.

## Truncation and memory

Every Hamiltonian is evaluated at PNO occupation thresholds `1e-6`, `1e-4` and
`1e-2`, plus explicit full retention. All occupied pairs participate. Examples
at the 9 MiB numeric budget are:

| System/operator | Full sum of pair ranks | Ranks at `1e-6` | Actual MP2 difference at `1e-6` (Eh) |
| --- | ---: | ---: | ---: |
| H2 / conventional | 1 | 1 | <`1e-16` |
| Water / conventional | 30 | 16 | `7.14e-5` |
| Water / DF | 30 | 17 | `1.53e-5` |
| LiH / conventional | 12 | 11 | `1.91e-6` |
| LiH / DF | 12 | 8 | `2.05e-6` |
| f-containing HeH+ / conventional | 8 | 6 | `9.86e-6` |
| f-containing HeH+ / DF | 8 | 3 | `1.66e-5` |

The largest threshold removes all pair spaces for these fixtures. Discarded
pair-density occupations are reported separately from total energy differences;
they are not independent error bounds. Near-threshold eigenvalue clusters are
marked as rank crossings, with no differentiability claim.

The 9 MiB and 128 MiB budgets reserve local canonical-MP2 temporaries, retained
pair arrays and publication copies, then pass the remainder to a fresh bounded
conventional/DF provider. Both budgets produce equivalent results and retain
the same ranks. Tests additionally run an exact minimum conventional-provider
budget and reject one byte less before reading an integral tile. Numeric plans
exclude Python object and library allocator overhead and do not claim exact
allocator/RSS peaks.

The prototype still constructs canonical MP2 amplitudes; it is intended to
validate local representations and truncation semantics. It does not establish
faster end-to-end MP2. The report exposes canonical provider cost, localization
and domain setup, pair transformation time, complete construction time, and
construction plus native HF export/space setup. No speedup is inferred from
smaller retained arrays alone.

Tests cover arbitrary pair-subspace gauges, occupied initial rotations, actual
atom permutations, duplicate/extremely diffuse AO failures, projected-domain
rank loss, degenerate spectra, wrong parent states, missing occupied Fock
coupling, small denominators, failed localization and explicit gradient rejection.

## Reproduction

Build the CPU library and run from the repository root:

```bash
PYTHONPATH=python:. VIBEQC_LIBRARY="$PWD/build/libvibeqc.so" \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m tools.validate_local_spaces --output /tmp/local-spaces

PYTHONPATH=python:. VIBEQC_LIBRARY="$PWD/build/libvibeqc.so" \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m pytest tests/python/test_local_spaces.py tests/python/test_local_mp2.py -q
```

Pinned reference consumption does not import PySCF. Regeneration requires an
explicit PySCF 2.14.0/NumPy/threadpoolctl environment:

```bash
PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m tools.generate_local_space_references --output /tmp/local-pm.json
```
