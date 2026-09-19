"""Deterministic per-class compilation of the existing DF shell mathematics.

Policy stays in the light host dispatcher. Each CUDA unit includes only its own
class specializations, so adding an auxiliary-f implementation cannot invalidate
all numerical objects. Execution templates remain native runtime-owned.
"""

from .df_rys_shell import emit_df_rys_policy_cpp, emit_df_rys_shell_cuda
from .df_shell_derivatives import SHELL_CLASSES, emit_df_shell_derivatives_cuda


def emit_df_shell_units():
    """Yield stable relative paths and contents, without runtime or GPU imports."""
    registry = [
        "// Generated class registry; independent of mathematical and tuning policy bytes.",
        "#pragma once",
        '#include "scf/cuda/df_shell_dispatch.cuh"',
        "namespace vibeqc::scf {",
    ]
    for angular in SHELL_CLASSES:
        name = "".join(map(str, angular))
        registry.append(f"extern const DfShellDispatch df_shell_{name};")
    registry += [
        "namespace generated_df_dispatch {",
        "template<class Function> void for_each_class(Function function) {",
    ]
    for angular in SHELL_CLASSES:
        name = "".join(map(str, angular))
        parameters = ",".join(map(str, angular))
        registry.append(
            f"  function.template operator()<{parameters}>(df_shell_{name});"
        )
    registry += ["}", "}", "} // namespace vibeqc::scf", ""]
    yield "generated_df_shell_dispatch.hpp", "\n".join(registry)
    for angular in SHELL_CLASSES:
        name = "".join(map(str, angular))
        parameters = ",".join(map(str, angular))
        polynomial = f"df_shell_polynomial_{name}.cuh"
        policy = f"df_shell_policy_{name}.hpp"
        math = f"df_shell_math_{name}.cuh"
        yield polynomial, emit_df_shell_derivatives_cuda(classes=(angular,))
        yield policy, emit_df_rys_policy_cpp(classes=(angular,))
        yield (
            math,
            emit_df_rys_shell_cuda(
                classes=(angular,), shell_header=polynomial, policy_header=policy
            ),
        )
        yield (
            f"df_shell_{name}.cu",
            f'''// Generated instantiation; equations and native launch bodies are shared.
#define VIBEQC_DF_SHELL_MATH_HEADER "{math}"
#include "scf/cuda/df_shell_launch.cuh"
namespace vibeqc::scf {{
extern const DfShellDispatch df_shell_{name}{{
  launch_panel_class<{parameters}>,
  launch_group_class<{parameters}>,
  launch_packets_class<{parameters}>
}};
}} // namespace vibeqc::scf
''',
        )
