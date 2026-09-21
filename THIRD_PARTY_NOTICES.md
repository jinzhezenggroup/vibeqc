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

This DFT-D4 qualification migration itself imports no xTB Hamiltonian or SCC
runtime; independent upstream tools are used to generate its test fixtures.

## Pinned dispersion parameter catalogs

The repository-only snapshots under `tools/parameters/upstream/` retain the
upstream damping-parameter tables used to generate VibeQC's static method
catalog. The simple-dftd3 snapshot is pinned to commit
`41d5a07b98ce15e97bec7a1815869725f6c7b0c2`; the DFT-D4 snapshot is pinned
to commit `82fbaf41724ab9a3c0a38ddc978ad0c38c4659b4`. Both are
LGPL-3.0-or-later data/code distributions; the corresponding license texts
already retained under `LICENSES/` apply. Exact source paths, revisions and
SHA-256 digests are recorded in
`tools/parameters/dispersion_parameter_sources.json`.

These snapshots are development-time generation inputs only. Production
VibeQC does not import, execute, or dynamically read simple-dftd3 or DFT-D4;
the generated Python/C++ parameter constants remain runtime-self-contained.
For D3, the generated catalog intentionally projects the upstream BJ pair
parameters to VibeQC's currently validated two-body model with `s9=0`;
upstream ATM availability is not claimed as a production D3 capability.

## r2SCAN-3c basis and gCP qualification data

The canonical H-Ar def2-mTZVPP snapshot shipped under
`python/vibeqc/data/r2scan3c/` is generated offline from MolSSI Basis Set
Exchange commit `4adaf1372c7101620ca1a9f3130be9ae97fb8f30`. The exact source
export, content identity, and supported-element domain are recorded in
`external/r2scan3c/manifest.json`; the BSE BSD-3-Clause text is retained as
`LICENSES/bse-data-BSD-3-Clause.txt`.

The repository-only gCP qualification data and CPU/native reference providers
are derived from the documented equations and parameter tables in simple-dftd3
commit `41d5a07b98ce15e97bec7a1815869725f6c7b0c2`, licensed
LGPL-3.0-or-later. Exact source hashes are recorded in the same manifest and
the upstream GPL/LGPL license texts are retained in
`LICENSES/simple-dftd3-COPYING.txt` and
`LICENSES/simple-dftd3-COPYING.LESSER.txt`. No simple-dftd3 binary or runtime
dependency is added. The embedded MB16-43/06 qualification geometry is from
mstore commit `a9070de01ad67e0539edc87c29ab048a60381a74` under Apache-2.0;
see `LICENSES/mstore-Apache-2.0.txt`.

## xTBloom GFN2-xTB runtime bootstrap

VibeQC issue #560 embeds a reviewed source snapshot of xTBloom commit
`5a67cc59ace94c8296e873503b2ae1298e7c2861` under
`src/xtb/gfn2_runtime/` to provide the first production GFN2-xTB runtime:
intrinsic-basis integrals, H0, ES2/ES3/AES2, generalized eigensolution,
occupations, Mulliken/multipole state, SCC mixing, repulsion, self-consistent
D4, spin terms, total energy, and analytic nuclear forces. VibeQC compiles the required common/GFN2/CPU-runtime sources from that
snapshot into `libvibeqc`. Native non-wheel CUDA builds additionally use the
GFN2-only CUDA cohort from xTBloom commit
`3c21f50195389b093941eb5ed6f1143b8802f96e`, selected from the repository
state immediately before production GFN1 CUDA execution was integrated. The
cohort reuses VibeQC's canonical packed D4 tables and carries the narrow
correctness backport from xTBloom #487 that passes CUDA kernel descriptors by
value rather than launcher-stack reference.
`src/xtb/gfn2_runtime/CUDA_SOURCE_PROVENANCE.json` records the exact source
set and per-file upstream/vendored hashes. No GFN1 runtime is compiled or
admitted, and CUDA wheel admission remains a separate gate. The former broad
xTBloom subproject is not restored, and an installed xTBloom library or
executable is not a runtime dependency. Its GPL-3.0-or-later terms, additional
CUDA/MKL permission, full third-party notices, and retained license texts ship
with the source/wheel legal material.

Linux wheels use xTBloom's reviewed private `scipy-openblas32==0.3.34.0.0`
LP64 LAPACKE/CBLAS provider boundary. A tiny sibling shim gives auditwheel one
content-qualified dependency edge; repair vendors and collision-renames the
OpenBLAS cohort into the VibeQC wheel. `scipy-openblas32` is a build input only,
not an end-user Python runtime dependency.

This is a transitional scientific owner: compiler-generated GFN2 pieces from
#504/#505 are intended to replace duplicated handwritten equations after
independent qualification, while the VibeQC-owned SCC/runtime boundary remains.
