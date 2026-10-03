"""Bind shared emitted arithmetic to host or CUDA device math declarations."""


def emit_host_device_math_cpp() -> str:
    """Keep one expression body while selecting target-compatible overloads."""
    functions = ("cbrt", "exp", "fabs", "fmax", "isfinite", "log", "pow", "sqrt")
    return "\n".join(
        [
            "#ifndef GENERATIVEQC_GENERATED_HOST_DEVICE_MATH",
            "#define GENERATIVEQC_GENERATED_HOST_DEVICE_MATH",
            "#include <cmath>",
            "namespace generativeqc::generated_math {",
            "#if defined(__CUDA_ARCH__)",
            *(f"using ::{name};" for name in functions),
            "#else",
            *(f"using std::{name};" for name in functions),
            "#endif",
            "} // namespace generativeqc::generated_math",
            "#endif",
            "",
        ]
    )
