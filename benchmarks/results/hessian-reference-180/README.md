# Independent semi-numerical RHF Hessian reference

This record qualifies the optional bounded PySCF reference, not #178/#179
analytic integration. Source hashes, complete bases/coordinates, all three
gradient-difference steps, actual Hessians and per-case gates are retained in
`qualification.json`. All gates pass for H2/STO-3G and custom 12-AO water.

Reproduce with the optional reference dependencies installed:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=python:. \
  python tools/hessian_examples.py --case h2,water --output /tmp/hessian-reference.json
```

Maximum difference against independent analytic PySCF: 1.31e-7 (H2),
1.16e-6 (custom water), in Eh/Bohr². The tests additionally cover genuine
7-AO STO-3G water. This CPU-only reference has no performance claim.
