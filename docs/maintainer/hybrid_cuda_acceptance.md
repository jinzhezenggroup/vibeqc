# Split-global-hybrid CUDA acceptance

M06-2X and MN15 expose generated RKS/UKS selectors qualified for CUDA,
strict-FP64, direct J/K, device-fused XC, and energy-only execution. Their
component-level `cuda-point-validated` label does not admit arbitrary bulk XC
compositions or CPU execution. Source generation, ABI tests, CUDA compilation,
and interior-point agreement alone are not acceptance: retain every gate below
on the same tested head. Keep raw JSON and NPZ evidence outside tracked source.

## Independent point gate

Use a clean checkout with a host C++ compiler, GCC/libquadmath, NumPy, PySCF
2.14.0 linked to Libxc 7.0.0, and the verified Libxc 7.0.0 source archive.
The validator fails rather than skipping if the reference, compiler, archive,
or actual GPU is unavailable. It checks the reference's global Fermi-hole
curvature policy, 50 points per method (including exact-empty spin and the
density tail), seven derivatives, interior finite differences, and restricted
spin embedding. An independent 113-bit Libxc Maple E/vxc probe also checks
both exact-empty M06-2X spin points and adjacent majority-density floats.
At those two original points, the 113-bit source is authoritative: compiled
binary64 Libxc is recorded as a diagnostic because its cancellation error
exceeds the sum of both acceptance tolerances. All other original points
must pass the unchanged compiled Libxc gate. No point or field is omitted,
and the same absolute and relative tolerances apply to the wide reference.
Each report includes compiler/source/device IDs.

```bash
PYTHONPATH=python:. python tools/validate_split_hybrid_cuda.py \
  --backend host --require-libxc 7.0.0 \
  --libxc-archive /path/to/libxc-7.0.0.tar.gz \
  --output .artifacts/split-hybrid/host.json

srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:10:00 bash -lc '
    PYTHONPATH=python:. python tools/validate_split_hybrid_cuda.py \
      --backend cuda --cuda-arch sm_120 --require-libxc 7.0.0 \
      --libxc-archive /path/to/libxc-7.0.0.tar.gz \
      --output .artifacts/split-hybrid/cuda.json'
```

Set `--cuda-arch` to the allocated device's actual capability. Preserve the
visibility assigned by Slurm; do not override `CUDA_VISIBLE_DEVICES`.

## Complete-endpoint gate

- Prepare the public RKS and UKS selectors with explicit matched `GridSpec`,
  CUDA, strict FP64, direct J/K, device-fused XC, and energy-only execution.
  Reject unsupported CPU/DF, derivatives, range/nonlocal compositions, missing
  or zero K terms, and changed component weights. Exercise both single and
  batch admission, replay, and a later invalid request after successful work.
- Compare fixed-density semilocal XC and exact K separately to an independent
  Libxc/PySCF reference at *identical* grids, AO values, densities, and spin
  conventions. Then compare converged total energies and density diagnostics
  for matched-grid RKS/UKS SCF, with explicitly recorded numerical gates.
  The current `tools/validate_dft_endpoints.py` only covers LDA/PBE and is not
  an independent reference for these two methods.
- Run the corresponding NVIDIA device tests under Slurm with Compute Sanitizer;
  retain the actual device, compiler, runtime, source identities, and raw
  sanitizer output. Check complete-endpoint timings and semantic J/K/XC work
  counts before making performance claims.

The opt-in matched-grid test uses H2/He RKS and two nonempty-spin H3 UKS
geometries, with PySCF 2.14.0/Libxc 7.0.0 and explicit identical native
quadrature. It validates native Fock against independent J, K and semilocal
XC matrices, separately compares Hartree and combined XC/exact-K energy
terms, reconverges SCF, and exercises batch, single, replay and invalid-input
recovery. Exact-empty-spin derivatives belong to the independent 113-bit point
gate above, not binary64 Libxc in the complete-endpoint test. The emitted
`fock_builds × grid_points` XC work is a semantic count, not a kernel timing.

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:30:00 bash -lc '
    export VIBEQC_LIBRARY=/absolute/path/to/libvibeqc.so
    export VIBEQC_SPLIT_HYBRID_CUDA_TEST=1
    export VIBEQC_SPLIT_HYBRID_EVIDENCE=/absolute/path/to/local-evidence
    export PYTHONPATH=python:.
    python -m pytest tests/python/test_split_hybrid_endpoints_cuda.py -q'

srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:30:00 bash -lc '
    export VIBEQC_LIBRARY=/absolute/path/to/libvibeqc.so
    export VIBEQC_SPLIT_HYBRID_CUDA_TEST=1 PYTHONPATH=python:.
    compute-sanitizer --tool memcheck --error-exitcode 86 \
      python -m pytest tests/python/test_split_hybrid_endpoints_cuda.py -q' \
  > /absolute/path/to/local-sanitizer.log 2>&1
```

Failure or a skip in any stage blocks scientific qualification; CUDA compile
success and interior-only parity do not substitute for a passed complete-endpoint
oracle on the same tested head.
