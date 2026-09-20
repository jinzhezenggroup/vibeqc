"""Internal conventional RCCSD/RCCSD(T) equations, state and validation APIs.

The facade accepts validated closed-shell RHF snapshots and conventional CPU
integral providers. RCCSD supports CPU, ordinary-stream CUDA and resident CUDA;
RCCSD(T) composes CPU or resident CUDA RCCSD with audited bounded triples tiles
and offers an isolated homogeneous Python batch helper. These internal APIs do
not register Calculator methods, native CC prepared owners or force support.
Complete-gradient validation, Lambda/response consumers and fixed-amplitude
generated actions remain separate explicit boundaries.
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
from .ccsd_t_api import (
    BatchRCCSDTResult,
    PreparedRCCSDTBatch,
    RCCSDTBatchItemResult,
    RCCSDTCapabilities,
    RCCSDTResult,
    rccsd_t_batch_energy,
    rccsd_t_energy,
    rccsd_t_method_capabilities,
)
from .complete_gradient import (
    BoundCCSDGradient,
    CCSDGradientCapabilities,
    CCSDGradientOptions,
    CCSDGradientResult,
    complete_gradient_validation,
    gradient_capabilities,
)
from .doubles import build_ccsd_program
from .equations import amplitude_layouts, build_program
from .evaluate import evaluate
from .lambda_cuda import PreparedCUDALambda
from .lambda_equations import (
    CCSDLambdaPrograms,
    build_lambda_programs,
    build_parameter_vjp,
)
from .lambda_response import BoundCCSDResponse, CCSDParameterWeight
from .lambda_solver import BoundCCSDLambda, CCSDLambdaResult, LambdaOptions
from .resident_solver import PreparedResidentCCSD, solve_gpu_resident
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
from .triples_lambda_response import (
    BoundCCSDTResponse,
    CCSDTParameterWeight,
    CorrectedLambdaResult,
    solve_corrected_lambda,
)
from .triples_response import (
    TRIPLES_RESPONSE_INPUTS,
    accumulate_tile_triples_vjp,
    build_full_triples_vjp,
    build_tile_triples_vjp,
    full_triples_vjp,
    tile_triples_vjp,
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
    "TRIPLES_RESPONSE_INPUTS",
    "BatchItemResult",
    "BatchRCCSDResult",
    "BatchRCCSDTResult",
    "BoundCCSDGradient",
    "BoundCCSDLambda",
    "BoundCCSDResponse",
    "BoundCCSDTResponse",
    "CCSDGradientCapabilities",
    "CCSDGradientOptions",
    "CCSDGradientResult",
    "CCSDLambdaPrograms",
    "CCSDLambdaResult",
    "CCSDParameterWeight",
    "CCSDResult",
    "CCSDTParameterWeight",
    "Capabilities",
    "CorrectedLambdaResult",
    "CudaTriplesResult",
    "CudaTriplesTiles",
    "LambdaOptions",
    "PreparedCCSD",
    "PreparedCUDALambda",
    "PreparedRCCSDTBatch",
    "PreparedResidentCCSD",
    "RCCSDResult",
    "RCCSDTBatchItemResult",
    "RCCSDTCapabilities",
    "RCCSDTResult",
    "SolverOptions",
    "TileSpec",
    "TriplesTileConfig",
    "TriplesTileEnumerator",
    "accumulate_tile_triples_vjp",
    "amplitude_layouts",
    "batch_energy",
    "build_ccsd_program",
    "build_full_triples_vjp",
    "build_lambda_programs",
    "build_parameter_vjp",
    "build_program",
    "build_tile_triples_program",
    "build_tile_triples_vjp",
    "build_triples_program",
    "complete_gradient_validation",
    "cpu_triples_tiles",
    "energy",
    "evaluate",
    "full_triples_vjp",
    "gradient_capabilities",
    "method_capabilities",
    "rccsd_t_batch_energy",
    "rccsd_t_energy",
    "rccsd_t_method_capabilities",
    "solve",
    "solve_corrected_lambda",
    "solve_gpu_resident",
    "tile_triples_energy",
    "tile_triples_energy_masked",
    "tile_triples_energy_tensorir",
    "tile_triples_vjp",
    "triples_energy",
    "triples_energy_tensorir",
    "triples_fullsum",
]
