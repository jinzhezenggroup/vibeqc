# Third-party notices

## Implib.so

VibeQC vendors the Implib.so generator/templates under `cmake/3rdparty/implib/`
for provider-free CUDA wheel linking. Upstream: https://github.com/yugr/Implib.so.
The vendored files are MIT-licensed, Copyright (c) 2017-2023 Yury Gribov; see
`LICENSES/implib-MIT.txt`. The source snapshot and digests are recorded in
`cmake/3rdparty/implib_manifest.json`.

The generated trampoline objects are compiled into `libvibeqc.so` and lazily
resolve CUDA runtime, cuBLAS, and cuSOLVER provider SONAMEs at execution time.
The NVIDIA provider shared libraries themselves are not redistributed in the
VibeQC wheel.

## xTBloom D3 qualification baseline

The repository-only D3 tools under `tools/vibeqc_d3/native/` adapt GPL-3.0-or-later
code from xTBloom commit `2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3`.
`external/xtbloom-d3/` retains its source hashes, original additional CUDA/MKL
permission, upstream parameter manifests and LGPL/Apache license texts.
The D3 tables derive from simple-dftd3 v1.4.0 (LGPL-3.0-or-later); the extracted
covalent radii retain the source's LGPL-3.0-or-later AND Apache-2.0 provenance.
See that directory's README and manifest for the exact material and changes.
No xTBloom/simple-dftd3 binary or runtime dependency is added, and these
qualification sources/data are not included as VibeQC wheel assets.
