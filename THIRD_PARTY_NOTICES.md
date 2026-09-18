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

## xTBloom / DFT-D4 molecular qualification baseline

`src/dft/dispersion/d4_reference.hpp` and `tools/parameters/generate_d4.py`
are adapted from xTBloom commit `2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3`
(`src/model/gfn2/d4.cpp`, `src/backends/cuda/gfn2_d4.cu`, and the parameter
exporter), under GPL-3.0-or-later. The original narrowly scoped CUDA/MKL
additional permission is retained verbatim in
`LICENSES/xtbloom-CUDA_MKL_LINKING_EXCEPTION.txt`; it is not a blanket
relicensing of VibeQC or third-party data. Source hashes and scope are recorded
in `src/dft/dispersion/xtbloom_manifest.json`.

The generated GFN2-D4 reference data in `src/dft/dispersion/d4_data.hpp`
is derived from DFT-D4 commit `6e1f59c3f39d919a2dbef0601d2576727c8b30e8`,
under LGPL-3.0-or-later. See `d4_manifest.json` and `d4.NOTICE` in the same
directory, plus `LICENSES/dftd4-COPYING.txt` and
`LICENSES/dftd4-COPYING.LESSER.txt`. The retained mctc-lib constant/provenance
record has its Apache-2.0 notice in `LICENSES/mctc-lib-LICENSE.txt`.

This migration imports no xTB Hamiltonian, SCC runtime, Fortran runtime, or
external dispersion library into VibeQC. Independent upstream tools are only
used to generate test fixtures.
