"""Internal conventional CPU RCCSD equations, prepared state and solver.

The facade accepts validated closed-shell RHF snapshots and conventional CPU
integral providers. It does not register a Calculator method or imply GPU,
Lambda or nuclear-gradient support. The original fixed-amplitude energy/T1
entry points remain available alongside the complete R1/R2 solver and the
audited CPU (T) triples reference (issue #150 slice A).
"""

from .doubles import build_ccsd_program
from .equations import amplitude_layouts, build_program
from .evaluate import evaluate
from .solver import CCSDResult, PreparedCCSD, SolverOptions, solve
from .triples import (
    build_triples_program,
    triples_energy,
    triples_energy_tensorir,
    triples_fullsum,
)

__all__ = [
    "CCSDResult",
    "PreparedCCSD",
    "SolverOptions",
    "amplitude_layouts",
    "build_ccsd_program",
    "build_program",
    "build_triples_program",
    "evaluate",
    "solve",
    "triples_energy",
    "triples_energy_tensorir",
    "triples_fullsum",
]
