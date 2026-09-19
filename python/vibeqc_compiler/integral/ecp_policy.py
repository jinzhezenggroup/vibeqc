"""Fixed two-grid acceptance policy for the validated scalar ECP consumer.

This is an empirical discretization gate, not an error estimator or an adaptive
grid prescription. Keep the independent CPU oracle's policy/tests separate.
"""

COARSE_RADIAL_POINTS = 160
COARSE_POLAR_POINTS = 32
REFINED_RADIAL_POINTS = 224
REFINED_POLAR_POINTS = 44
MATRIX_ABS_TOLERANCE = 2e-9
DERIVATIVE_ABS_TOLERANCE = 2e-8


def emit_ecp_policy_cpp():
    """Emit host/device admission and the grids used by native orchestration.

    Nonfinite inputs must reject even when equal or when subtraction yields
    NaN. Equality at the absolute tolerance remains accepted. These control
    predicates lower directly to C++ math/comparison operations, not an AD DAG.
    As in the emitted arithmetic, unqualified fabs resolves the CUDA device
    overload; CuMetal's standard-library fabs wrapper is host-only.
    """
    constants = (
        ("coarse_radial_points", COARSE_RADIAL_POINTS),
        ("coarse_polar_points", COARSE_POLAR_POINTS),
        ("refined_radial_points", REFINED_RADIAL_POINTS),
        ("refined_polar_points", REFINED_POLAR_POINTS),
    )
    return [
        *(
            f"inline constexpr unsigned ecp_{name} = {value};"
            for name, value in constants
        ),
        "VIBEQC_ECP_INLINE bool ecp_grid_pair_accepted(double coarse, double fine,",
        "    bool derivative) {",
        f"  const double tolerance = derivative ? {DERIVATIVE_ABS_TOLERANCE!r} : {MATRIX_ABS_TOLERANCE!r};",
        "  return std::isfinite(coarse) && std::isfinite(fine) &&",
        "      fabs(coarse - fine) <= tolerance;",
        "}",
    ]
