"""Bundle the compiler's native templates without importing either package."""

from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).parent
ASSETS = (
    "src/tensor/cuda_runtime.cuh",
    "src/dft/cuda_grid.cu",
    "src/dft/xc_runtime.cuh",
    "src/integrals/range_moments.hpp",
    "src/integrals/eri_geometry.hpp",
    "src/scf/cuda_weighted_eri.hpp",
    "src/scf/weighted_eri_runtime.hpp",
    "include/vibeqc/vibeqc.h",
    *(
        str(path.relative_to(ROOT))
        for path in sorted((ROOT / "external/libxc-7.0.0").rglob("*"))
        if path.is_file()
    ),
)


class BuildCompilerAssets(build_py):
    """Copy canonical, license-audited build inputs into the installed package."""

    def run(self):
        super().run()
        for name in ASSETS:
            destination = Path(self.build_lib) / "vibeqc_compiler/assets" / name
            self.mkpath(str(destination.parent))
            self.copy_file(str(ROOT / name), str(destination))

    def get_outputs(self, include_bytecode=1):
        return super().get_outputs(include_bytecode) + [
            str(Path(self.build_lib) / "vibeqc_compiler/assets" / name)
            for name in ASSETS
        ]


setup(cmdclass={"build_py": BuildCompilerAssets})
