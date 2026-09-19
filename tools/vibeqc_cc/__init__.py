"""Internal conventional CPU RCCSD equations, prepared state and solver.

The facade accepts validated closed-shell RHF snapshots and conventional CPU
integral providers. It does not register a Calculator method or imply GPU,
native Lambda or nuclear-gradient support. An internal CPU solved-state Lambda
consumer is available separately. Generated fixed-amplitude Lambda
equation actions are available separately through ``build_lambda_programs``.
The original fixed-amplitude energy/T1 entry points remain available alongside the complete R1/R2 solver and the
audited CPU (T) triples reference (issue #150 slice A).
"""

from .api import (
    BatchItemResult,
    BatchRCCSDResult,
    Capabilities,
    RCCSDResult,
    batch_energy,
    energy,
    method_capabilities,
)
from .doubles import build_ccsd_program
from .equations import amplitude_layouts, build_program
from .evaluate import evaluate
from .lambda_equations import CCSDLambdaPrograms, build_lambda_programs
from .lambda_solver import BoundCCSDLambda, CCSDLambdaResult, LambdaOptions
from .solver import CCSDResult, PreparedCCSD, SolverOptions, solve
from .triples import (
    build_triples_program,
    triples_energy,
    triples_energy_tensorir,
    triples_fullsum,
)
from .triples_cuda import (
    CudaTriplesResult,
    CudaTriplesTiles,
    TriplesTileConfig,
    cpu_triples_tiles,
)
from .triples_tiles import (
    TileSpec,
    TriplesTileEnumerator,
    build_tile_triples_program,
    tile_triples_energy,
    tile_triples_energy_masked,
    tile_triples_energy_tensorir,
)

__all__ = [
    "BatchItemResult",
    "BatchRCCSDResult",
    "BoundCCSDLambda",
    "CCSDLambdaPrograms",
    "CCSDLambdaResult",
    "CCSDResult",
    "Capabilities",
    "CudaTriplesResult",
    "CudaTriplesTiles",
    "LambdaOptions",
    "PreparedCCSD",
    "RCCSDResult",
    "SolverOptions",
    "TileSpec",
    "TriplesTileConfig",
    "TriplesTileEnumerator",
    "amplitude_layouts",
    "batch_energy",
    "build_ccsd_program",
    "build_lambda_programs",
    "build_program",
    "build_tile_triples_program",
    "build_triples_program",
    "cpu_triples_tiles",
    "energy",
    "evaluate",
    "method_capabilities",
    "solve",
    "tile_triples_energy",
    "tile_triples_energy_masked",
    "tile_triples_energy_tensorir",
    "triples_energy",
    "triples_energy_tensorir",
    "triples_fullsum",
]
