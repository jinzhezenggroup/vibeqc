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
vibe-coded quantum-chemistry program: its API, CPU reference implementation,
CUDA kernels, tests, benchmark tools, and documentation have all been created
and iterated through a human-directed AI coding workflow. The human supplies
the scientific goals, constraints, and review; coding agents produce and
revise the implementation.

“Vibe-coded” describes how VibeQC was built, not a relaxation of scientific
standards. Numerical results are checked against independent references, and
performance claims require reproducible benchmark gates.

The project's long-term mission is to cover **all quantum-chemistry methods**
behind one coherent, accelerator-native interface. That is a direction, not a
claim about today's release.

## Supported methods

The currently supported electronic-structure method family is
**Hartree-Fock (HF)**:

| Method family | Implemented variants | Available results |
| --- | --- | --- |
| Hartree-Fock (HF) | RHF, UHF | Energy and analytic nuclear forces |

No other electronic-structure method family is currently implemented. The
method interface is intended to expand beyond HF to DFT, post-HF,
multireference, excited-state, periodic, relativistic, and embedding methods.
See the [method roadmap](docs/methods.md) for the intended scope.

## Current functionality

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
cmake --build build -j10
python -m pip install -e .
```

For a CPU-only build:

```bash
cmake -S . -B build -G Ninja -DVIBEQC_ENABLE_CUDA=OFF
cmake --build build -j10
python -m pip install -e .
```

The Python package finds `build/libvibeqc.so` automatically when built in the
repository. For another build location, set `VIBEQC_LIBRARY` to the shared
library path.

## Single-point calculation

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

## Batched calculation

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

- [Documentation index](docs/index.md)
- [Methods and long-term scope](docs/methods.md)
- [Batched HF behavior](docs/batched_hf.md)
- [Architecture](docs/architecture.md)
- [Implementation roadmap](docs/roadmap.md)
- [Benchmark protocol and results](benchmarks/results/README.md)

## License

VibeQC is licensed under [GPL-3.0-or-later](LICENSE).
