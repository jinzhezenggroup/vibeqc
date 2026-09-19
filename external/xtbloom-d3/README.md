# Pinned xTBloom D3 reference-data import

This repository-only qualification input comes from xTBloom commit
`2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3`. It is not a bundled xTBloom runtime
and is not added to VibeQC wheels.

- `gfn1_d3.json` is byte-identical to xTBloom's approximately 0.6 MB canonical
  packed D3 reference data (H through Rn). No second C++ data-table copy is kept.
- `covalent_radii.json` extracts only the 86 `covalent_radius_bohr` values from
  `gfn1.json`, rather than copying the whole GFN1 parameter model.
- `manifest.json` pins original paths, Git blobs and SHA-256 digests, plus the
  local data digests. `upstream_d3_manifest.json` and
  `upstream_gfn1_manifest.json` preserve the upstream source/legal chains.

The D3 tables derive from simple-dftd3 v1.4.0 under LGPL-3.0-or-later; the
covalent-radii extraction retains the source's LGPL-3.0-or-later AND Apache-2.0
provenance. See `COPYING.LESSER`, `Apache-2.0.txt` and the repository GPLv3
`LICENSE`. The retained `CUDA_MKL_LINKING_EXCEPTION` is xTBloom's limited
additional permission, not a relicensing of upstream tables or other code.

The shared diagnostic arithmetic in `tools/vibeqc_d3/native/` is adapted from
xTBloom's real CUDA `prepare_d3_weights`, `pair_coefficient` and `add_d3`
functions, cross-checked with the CPU implementation. Changes generalize damping
and cutoff parameters, remove halogen/SCC/runtime dependencies, share the
mathematics between CPU and CUDA, and add an explicit small-fixture harness.
The source is GPL-3.0-or-later with the upstream notice/provenance retained.

Reproduce the data import using a local checkout of the pinned xTBloom source:

```sh
python tools/vibeqc_d3/import_data.py /path/to/xtbloom --check
```

Omit `--check` to restore the two canonical data files. No network request,
upstream binary, large benchmark archive or generated executable is retained.
