# BASIS02 g-shell validation, RTX 4090 / CPU

Source implementation: `e8d6a006de601370feeab4702fd5b3c0f250d206`, rebased on
`a50f93665081a7e2804f84aa617e6b3db7228aa4` (including PR #264).
See [execution contracts and reproduction](../../../docs/high_angular_momentum.md)
and [machine-readable results](results.json).

## Gates

- CPU native suite: 16 passed.
- Affected Python regression suite: 326 passed, 24 optional/allocated-device skips.
- Higher-l suite with both opt-in tiers enabled: 40 passed (120.68 seconds).
- Public imported-basis CUDA gates: 9 passed, including changed-geometry replay,
  failure isolation and unsupported high-l/ECP rejection.
- Native CUDA: 18 of 19 passed. The isolated bounded-Fock diagnostic in
  `vibeqc_mixed_precision_tests` fails identically on the unmodified base above.
  This sm_89 generic build has no promoted AOT shell coverage, which that
  diagnostic requires. This is retained as a baseline failure, not counted as
  a passing or skipped test. No AOT manifest, mask, or CUDA Fock kernel changed.

## Independent molecular endpoints

| Calculation | Representation | Energy error (Hartree) | Maximum force error (Hartree/Bohr) |
| --- | --- | ---: | ---: |
| Loaded orbital g HeH+ RHF | Cartesian, 17 AOs | 2.93e-14 | 6.14e-11 |
| Loaded orbital g HeH+ RHF | Real spherical, 11 AOs | 1.78e-14 | 7.30e-12 |
| g auxiliary HeH+ DF-RHF | Cartesian | 7.37e-14 | 1.99e-13 |
| g auxiliary HeH+ DF-RHF | Real spherical | 1.95e-14 | 1.38e-13 |

PySCF/libcint 2.14.0 provides the independent references with identical geometry,
charge, basis coefficients, representation and approximation. These are small
synthetic basis diagnostics, not chemical accuracy or large-basis benchmarks.
The native gggg value probe additionally checks coincident, near-coincident,
intermediate and large Boys arguments against libcint.

## Opt-in scalar compilation resources

CUDA 12.9, sm_89, `-O1`, one explicit component and all first derivatives.
The complete main CUDA library uses `VIBEQC_CUDA_FAST_COMPILE=ON` for numerical
regression only; its size or speed is not a release performance claim.

| Family | Compile seconds | Registers | Maximum stack bytes | Spill store bytes (summed functions) |
| --- | ---: | ---: | ---: | ---: |
| Overlap g/g | 1.05 | 50 | 0 | 0 |
| Kinetic g/g | 1.11 | 56 | 0 | 0 |
| Nuclear attraction g/g | 3.04 | 255 | 2544 | 9960 |
| Coulomb metric g/g | 1.20 | 108 | 80 | 0 |
| Three-center s/s/g | 1.25 | 60 | 48 | 0 |

Nuclear attraction remains spill-heavy. The bounded scalar route is numerical
baseline evidence and is **not promoted** into HF or an AOT profile. Compiler
resources depend on the selected Cartesian component; they are not maxima for
all g components. Source/binary bytes and spill-load counts are recorded in JSON.
The test process and compiler children peak at 553940 KiB RSS; this includes
PySCF and Python and must not be presented as native molecular memory usage.
