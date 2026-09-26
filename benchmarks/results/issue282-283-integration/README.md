# Integrated DF reuse and response evidence (#282, #283)

The final generated-resident, streamed-panel, response-cache and batch-property
integration passes every energy and force comparison in the four-case #206
matrix. Five interleaved warm repeats per engine/case on one RTX 5090, with a
fixed post-cold engine-local density and no concurrent compilation, give:

| AO / batch | VibeQC energy + forces (s) | GPU4PySCF energy + forces (s) |
|---|---:|---:|
| 96 / 1 | 0.794956 | 0.245908 |
| 96 / 4 | 2.528601 | 0.981686 |
| 192 / 1 | 6.356073 | 0.330881 |
| 192 / 4 | 24.966968 | 1.323790 |

All 20 raw pairs pass 1e-9 Eh energy and 1e-8 Eh/bohr force gates. Iteration
branches differ across engines; these endpoint measurements do not establish
an iteration-matched speed comparison. `summary.json` retains every branch,
maximum error and value-plan memory diagnostic. The declared 1-GiB DF
sub-budget is not a measurement of global peak or opaque library retention.

The frozen library was built from fba86bb4eb9dafcd3cf03f82c37191fd64609e48;
SHA-256 is `08404f960886da3842d24feff3a417ddb5065846d0ff72fda44ee4c922426605`.
Slurm job 9359 ran the final matrix after four CUDA native suites, 66 GPU
Python cases, 76 host cases (six optional GPU skips), and repository hooks
passed. The archive preserves exact samples, commands, source reconstruction
patch, CMake cache, identity and validation logs. Every member was restored and
byte-compared.

Earlier evidence supplies the independent acceptance checks:

- [Generated residency and streamed J/K](../issue282-streamed-panels/README.md)
  records capacity selection, tile reuse and a 32-MiB 192-AO case whose full
  transformed tensor requires 56,623,104 bytes. This constrained probe measures
  fixed-density J/K and the complete E2 gradient, not full-SCF force timing.
- [True energy-only matrix](../issue282-energy-only/README.md) retains the
  four energy endpoints and verifies absent force work. Response caching does
  not execute in those measurements.
- [Response panel comparison](../issue283-response-panels/README.md) records
  identical-solver-branch before/after endpoints, unchanged scratch, full-panel
  raw passes reduced from three to one, independent derivative gates and
  memcheck. Warmed component captures attribute 98.84%/98.89% of the 96-AO
  force increment and 99.27%/97.37% at 192 AO.

Occupied-factor CUDA exchange is evaluated separately in #284. This archive's
force matrix uses dense-density RI-K.

```bash
python -m tools.unpack_evidence benchmarks/results/issue282-283-integration \
  --output /tmp/issue282-283-integration
python benchmarks/results/issue282-283-integration/audit.py
```
