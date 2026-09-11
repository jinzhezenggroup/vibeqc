"""Fixed-amplitude RCCSD on the existing #146 FP64 CUDA executor.

This boundary intentionally makes no solver or public method claim. Its input
and output transfers are suitable for equation validation, not a resident CC
iteration. Equations are constructed only by the audited #148 frontend.
"""

from vibeqc_compiler.tensor import Program
from vibeqc_compiler.tensor.cuda_execute import PreparedCuda, compile_cuda
from vibeqc_compiler.tensor.cuda_plan import plan_cuda

from .doubles import build_ccsd_program


def rccsd_program(nocc, nvir, *, form="shared", trace=False):
    """Expose every live node when auditing intermediate parity.

    Trace outputs keep the original nodes, including exact coefficients and
    tensor semantics. Retaining them changes lifetimes and is explicitly
    charged by #146; trace timings are not solver timings.
    """
    program = build_ccsd_program(nocc, nvir, form=form)
    if not trace:
        return program
    names = program.debug_names
    return Program(
        {
            **program.outputs,
            **{f"trace_{names[n]}": n for n in program.live_nodes},
        },
        provenance={
            "parent_equation": program.logical_hash,
            "scope": "fixed-amplitude RCCSD every-node audit",
        },
    )


class PreparedRCCSDResidual:
    """Own a fixed-shape ordinary-stream energy/R1/R2 evaluator.

    Feeds follow #148 (spatial restricted, FP64, t2[ijab], chemists' MO
    blocks). #146 validates layouts, symmetry, finiteness, device and budget.
    No reference, result or amplitude is cached as a physical warm start.
    """

    def __init__(
        self,
        nocc,
        nvir,
        compiler,
        cache,
        *,
        max_bytes=256 << 20,
        form="shared",
        trace=False,
        device=0,
        **plan_options,
    ):
        self.program = rccsd_program(nocc, nvir, form=form, trace=trace)
        self.plan = plan_cuda(
            self.program, compiler.target, max_bytes=max_bytes, **plan_options
        )
        self.artifact = compile_cuda(self.plan, compiler, cache)
        self.executor = PreparedCuda(self.plan, self.artifact, device=device)

    def execute(self, feeds, *, profile=False):
        return self.executor.execute(feeds, profile=profile)

    def close(self):
        self.executor.close()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()
