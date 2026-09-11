"""Small host-driver generation around separately compiled production kernels."""

import re
from pathlib import Path

from .f_shell import f_shell_plan, source_audit


def emit_numerical_driver(
    name: str,
    architecture: str = "sm_120",
    *,
    plan=None,
    source: str | None = None,
    consumer: str | None = None,
) -> str:
    """Extract the emitted task ABI and declare the eight generated launch entries.

    Keeping the host driver separate lets numerical-fixture changes reuse the
    expensive class object and its exact resource report. No device recurrence
    or contraction implementation is copied into this host-only translation unit.
    """
    if source is None:
        _, source = source_audit(name, architecture)
    plan = plan or f_shell_plan(name, architecture)
    consumers = (consumer,) if consumer is not None else ("fock", "force")
    class_name = name[0].upper() + name[1:]
    start = source.index(f"struct Generated{class_name}Vec3")
    stop = source.index(f"struct Generated{class_name}PrimitiveGeometry")
    types = source[start:stop]
    declarations, table = [], []
    for selected_consumer in consumers:
        constant = f"kGenerated{class_name}{'Fock' if selected_consumer == 'fock' else ''}BlockThreads"
        matches = re.findall(r"constexpr unsigned " + constant + r" = (\d+)U;", source)
        if len(matches) != 1:
            raise ValueError("generated block-thread declaration changed")
        threads = int(matches[0])
        for spin in ("rhf", "uhf"):
            for persistent in (False, True):
                suffix = "_persistent" if persistent else ""
                symbol = f"generated_{name}_shell_class_{selected_consumer}_{spin}{suffix}_kernel"
                tail = (
                    "const std::uint32_t*, const std::uint32_t*, std::uint32_t*"
                    if persistent
                    else "std::size_t"
                )
                declarations.append(f"""extern "C" __global__ void {symbol}(
                    const Generated{class_name}ShellTask*, const Generated{class_name}PrimitivePairData*,
                    const std::int64_t*, const double*, const Generated{class_name}Vec3*, double,
                    const double*, const double*, double*, {tail});""")
                table.append(
                    f'{{"{spin}_{selected_consumer}{suffix}", reinterpret_cast<const void*>({symbol}), '
                    f"{threads}, {str(spin == 'uhf').lower()}, {str(selected_consumer == 'force').lower()}, {str(persistent).lower()}"
                    + "}"
                )
    if 4 * len(consumers) != len(table):
        raise ValueError("numerical driver does not cover the complete wrapper set")
    template = Path(__file__).with_name("f_shell_driver.cu.in").read_text()
    substitutions = {
        "@TASK_TYPES@": types,
        "@KERNEL_DECLARATIONS@": "\n".join(declarations),
        "@KERNEL_TABLE@": ",\n".join(table),
        "@CLASS@": class_name,
        "@ARCH_NUMBER@": architecture.removeprefix("sm_"),
        "@AO_COUNT@": str(sum(len(c) for c in plan.spec.center_components)),
        "@COMPONENT_COUNTS@": ", ".join(
            str(len(c)) for c in plan.spec.center_components
        ),
    }
    for key, value in substitutions.items():
        template = template.replace(key, value)
    if re.search(r"@[A-Z_]+@", template):
        raise ValueError("unexpanded numerical driver marker")
    return template
