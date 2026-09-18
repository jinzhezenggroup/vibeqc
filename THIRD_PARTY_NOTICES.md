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
