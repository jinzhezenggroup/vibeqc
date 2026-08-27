<!--
IMPORTANT: Keep this README concise and user-facing. It should contain only
the project identity, current supported methods, essential capabilities,
installation, and minimal examples. Put implementation history, kernel
details, benchmark analysis, and extended roadmaps in docs/ or
benchmarks/results/ instead of expanding this file.
-->

<p align="center">
  <img src="assets/vibeqc-logo.svg" width="180" alt="VibeQC logo">
</p>

<h1 align="center">VibeQC</h1>

<p align="center">
  GPU-native, batched quantum chemistry with analytic forces.
</p>

VibeQC combines **vibe coding** and **quantum chemistry**. It is a fully
vibe-coded quantum-chemistry program: humans set the scientific goals,
constraints, and review standards; coding agents produce and revise the code,
tests, benchmarks, and documentation. Numerical results are checked against
independent references, and performance claims require reproducible gates.

Today VibeQC provides **RHF and UHF energies and analytic nuclear forces**.
Other electronic-structure methods remain on the
[roadmap](docs/methods.md), not in the current release.

## Features

- CPU reference and CUDA backends.
- Ragged batches, per-system failure isolation, and density warm starts.
- Contracted Cartesian and real-spherical `s` through `f` Gaussian bases.
- Bundled STO-3G, def2-SVP, and def2-TZVP basis data for H-Ar.
- Python, C, and C++ interfaces; optional PyTorch analytic backward.

## Build and install

Requirements: CMake 3.24+, a C++20 compiler, Python 3.10+, and optionally CUDA
12.9 for the GPU backend.

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_CUDA_COMPILER=/path/to/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120
```

CUDA 12.9 can also build portable generic binaries for `80`, `86`, `89`, and
`90`. Only `sm_120` currently has a measured generated-shell profile; other
targets automatically keep the validated generic CUDA kernels. A distributable
fat binary can be configured with:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_CUDA_COMPILER=/path/to/cuda/bin/nvcc \
  -DVIBEQC_CUDA_ARCHITECTURES="80;90;120" \
  -DVIBEQC_AOT_PROFILES="sm_120"
```

Use `-DVIBEQC_ENABLE_AOT_SHELLS=OFF` to omit generated shell bundles entirely,
or `-DVIBEQC_AOT_PROFILE=portable` to retain an explicit empty portable profile.
Builds automatically use `sccache` or `ccache` when either is on `PATH`;
override this with `-DVIBEQC_COMPILER_CACHE=off` or an explicit executable.
Generated CUDA is split into eight stable shards by default.  Tune this with
`-DVIBEQC_AOT_SHARDS=N` when local compile parallelism or memory is limited.

For compile-only CUDA experiments,
`-DVIBEQC_CUDA_SPLIT_COMPILE_THREADS=N` enables NVCC split compilation of the
large generic translation unit.  It defaults to `1` because split compilation
can change optimizer resource choices; use the normal setting for performance
and release binaries.

For CPU only, configure with:

```bash
cmake -S . -B build -G Ninja -DVIBEQC_ENABLE_CUDA=OFF
```

Then build and install:

```bash
cmake --build build -j10
python -m pip install -e .
```

The Python package finds `build/libvibeqc.so` automatically when built in the
repository. For another build location, set `VIBEQC_LIBRARY` to the shared
library path.

## Python API

Coordinates are in Bohr, energies in Hartree, and forces in Hartree/Bohr.

```python
from vibeqc import Calculator

calc = Calculator(method="rhf", basis="sto-3g", device="cuda")
result = calc.singlepoint([
    ("H", (0.0, 0.0, -0.7)),
    ("H", (0.0, 0.0, 0.7)),
])

print(result.energy)
print(result.forces)
```

Prepared batches retain reusable topology and density state:

```python
from vibeqc import Calculator

systems = [
    [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))],
    [("He", (0.0, 0.0, 0.0))],
]

calc = Calculator(method="rhf", basis="sto-3g", device="cuda")
with calc.prepare_batch(systems, warm_start=True) as batch:
    first = batch.execute(strict=True)
    second = batch.execute(strict=True)  # reuses compatible densities

print(first.energies)
```

## Documentation

- [Documentation index](docs/index.md) — methods, batching, architecture, and
  implementation roadmap.
- [Benchmark results](benchmarks/results/README.md) — protocol, gates, and
  reproducible artifacts.

## License

VibeQC is licensed under [GPL-3.0-or-later](LICENSE).
