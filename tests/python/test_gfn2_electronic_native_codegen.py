"""Production cutover tests for compiler-owned GFN2 CPU electronic arithmetic."""

from pathlib import Path

import numpy as np
from vibeqc_compiler.method.gfn2_electronic_runtime import (
    build_gfn2_core_energy_update_program,
    build_gfn2_multipole_hamiltonian_update_program,
    build_gfn2_multipole_integral_vjp_program,
    build_gfn2_population_update_program,
    build_gfn2_scalar_hamiltonian_update_program,
    build_gfn2_scalar_integral_vjp_program,
)
from vibeqc_compiler.tensor import Program, execute

ROOT = Path(__file__).resolve().parents[2]


def _scalar(program: Program, feeds: dict[str, float], output: str) -> float:
    result = execute(
        program,
        {name: np.asarray(value, dtype=np.float64) for name, value in feeds.items()},
    ).outputs[output]
    return float(np.asarray(result))


def test_runtime_bound_population_and_hamiltonian_math() -> None:
    population = build_gfn2_population_update_program()
    assert _scalar(
        population,
        {"density": 0.37, "integral": -0.21, "accumulator": 0.13},
        "updated",
    ) == np.float64(0.13 - 0.37 * -0.21)

    core_energy = build_gfn2_core_energy_update_program()
    np.testing.assert_allclose(
        _scalar(
            core_energy,
            {"density": 0.37, "h0": -0.28, "accumulator": 0.13},
            "updated",
        ),
        0.13 + 0.37 * -0.28,
        rtol=0,
        atol=2e-17,
    )

    scalar = build_gfn2_scalar_hamiltonian_update_program()
    scalar_inputs = {
        "overlap": 0.17,
        "row_vat": 0.03,
        "row_vsh": -0.07,
        "column_vat": 0.11,
        "column_vsh": 0.05,
        "accumulator": -0.02,
    }
    expected_scalar = -0.02 - 0.5 * 0.17 * (0.03 - 0.07 + 0.11 + 0.05)
    np.testing.assert_allclose(
        _scalar(scalar, scalar_inputs, "updated"),
        expected_scalar,
        rtol=0,
        atol=2e-17,
    )

    multipole = build_gfn2_multipole_hamiltonian_update_program()
    multipole_inputs = {
        "forward_integral": 0.09,
        "reverse_integral": -0.04,
        "row_potential": 0.23,
        "column_potential": -0.19,
        "accumulator": 0.08,
    }
    expected_multipole = 0.08 - 0.5 * (0.09 * -0.19 + -0.04 * 0.23)
    np.testing.assert_allclose(
        _scalar(multipole, multipole_inputs, "updated"),
        expected_multipole,
        rtol=0,
        atol=2e-17,
    )


def test_runtime_bound_sdq_vjp_is_generated_from_same_program() -> None:
    pair_density = -0.61
    row_scalar = 0.14
    column_scalar = -0.22
    scalar_vjp = build_gfn2_scalar_integral_vjp_program().program
    actual_overlap = _scalar(
        scalar_vjp,
        {
            "row_vat": 0.0,
            "row_vsh": row_scalar,
            "column_vat": 0.0,
            "column_vsh": column_scalar,
            "bar_updated": pair_density,
        },
        "bar_overlap",
    )
    np.testing.assert_allclose(
        actual_overlap,
        -0.5 * pair_density * (row_scalar + column_scalar),
        rtol=0,
        atol=2e-17,
    )

    row_potential = 0.31
    column_potential = -0.27
    multipole_vjp = build_gfn2_multipole_integral_vjp_program().program
    feeds = {
        "row_potential": np.asarray(row_potential),
        "column_potential": np.asarray(column_potential),
        "bar_updated": np.asarray(pair_density),
    }
    outputs = execute(multipole_vjp, feeds).outputs
    np.testing.assert_allclose(
        outputs["bar_forward_integral"],
        -0.5 * pair_density * column_potential,
        rtol=0,
        atol=2e-17,
    )
    np.testing.assert_allclose(
        outputs["bar_reverse_integral"],
        -0.5 * pair_density * row_potential,
        rtol=0,
        atol=2e-17,
    )


def test_native_runtime_consumers_do_not_restore_handwritten_505_formulas() -> None:
    targets = (
        ROOT / "src/xtb/gfn2_runtime/src/model/gfn2/mulliken.cpp",
        ROOT / "src/xtb/gfn2_runtime/src/model/gfn2/mulliken_kernels_impl.hpp",
        ROOT / "src/xtb/gfn2_runtime/src/model/gfn2/force.cpp",
        ROOT / "src/xtb/gfn2_runtime/src/model/gfn2/scc_driver.cpp",
    )
    sources = [path.read_text() for path in targets]
    assert all("generated_gfn2_electronic_native.hpp" in source for source in sources)
    joined = "\n".join(sources)
    assert "add_product(-density_value" not in joined
    assert "-0.5 * pair_density" not in joined
    assert "-0.5 * task.dipole_integrals" not in joined
    assert "-0.5 * integrals.dipole" not in joined
    assert "std::fma(geometry.h0" not in joined
