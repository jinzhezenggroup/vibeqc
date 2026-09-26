# CPU UKS acceptance record for PR #305

The complete compact [numerical record](endpoints.json) is retained in Git,
including all six native/reference endpoint values, full physical residuals,
separate native density-update RMS, gate decisions, independent SCF iteration
measurements, grid hashes, source
identity and software provenance. Reproduction does not depend on a local
workstation file or an expiring artifact. This preserves the validator's
existing `vibeqc.uks-endpoint-validation` schema without reinterpreting gates.

- Measured clean source: `9d43a84a2780c4df9fc63050ea513812b95e00b3`.
- Numerical identity: `semilocal-scaled-v1/pbe-spin-c2-1e-18`; CPU FP64,
  conventional total-density J, energy-only.
- Validator: `tools/validate_uks_endpoints.py`, SHA-256
  `ab6ef01e05cf4d3a728c3558cb68c7742b7a7b4ef892dff1810652e07dcbeafd`.
- Record SHA-256:
  `48184b07922233b841ebf0ea93267d17590aa26ad26fb4343af1df0c5d82ce4f`.
- GCC 11.4 Release build; Python 3.13.9, NumPy 2.5.3, PySCF 2.14.0,
  Libxc 7.0.0; one OMP/OpenBLAS thread. Full platform details are in the JSON.
- The identical GridSpec-v1 grids each have 49,152 points. Atomic inputs and
  STO-3G basis resolution are fixed by the source-identified validator.

| Case | Method | Absolute energy error (Eh) | Native residual RMS | Independent residual RMS | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| H2- doublet | LDA UKS | 1.034e-13 | 1.334e-11 | 8.330e-17 | pass |
| H2- doublet | PBE UKS | 1.344e-13 | 8.775e-12 | 6.495e-17 | pass |
| H2+ fully polarized | LDA UKS | 2.409e-14 | 2.776e-16 | 6.799e-17 | pass |
| H2+ fully polarized | PBE UKS | 1.058e-9 | 9.615e-17 | 8.777e-17 | pass |
| OH doublet | LDA UKS | 9.948e-14 | 3.356e-11 | 1.114e-12 | pass |
| OH doublet | PBE UKS | 3.268e-13 | 3.134e-11 | 4.524e-12 | pass |

All native and independent solves converged. Gates are `1e-8 Eh` absolute
energy error and `1e-9` for both physical residuals. OH's independent PySCF
calculation uses the exact C2v subgroup and its own minao guess to resolve pi
orientation. The reference gate still measures the full unrestricted AO
commutator, including rotations excluded from that iteration. Native OH has
no imposed point-group constraint. Unconstrained reference PBE DIIS probes
that failed the residual gate are not accepted measurements.

Reproduce from the measured source with the versions above:

```bash
cmake -S . -B build/cpu -G Ninja -DVIBEQC_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpu -j 8
env PYTHONPATH=python VIBEQC_LIBRARY="$PWD/build/cpu/libvibeqc.so" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python tools/validate_uks_endpoints.py --output /tmp/uks-endpoints.json
```

This record supports these small STO-3G CPU numerical endpoints. It does not
establish grid convergence, CUDA, prepared batches, gradients, density fitting,
performance or completion of #162. The previous remote and local records
remain historical; this versioned record is the current acceptance evidence.
