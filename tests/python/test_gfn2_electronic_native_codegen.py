"""Production cutover tests for compiler-owned GFN2 CPU and CUDA electronic arithmetic."""

import subprocess
import sys
from pathlib import Path

import numpy as np
from test_gfn2_electronic_ir import _restricted_feeds, _topology
from vibeqc_compiler.method import build_gfn2_electronic_program
from vibeqc_compiler.method.gfn2_electronic import (
    GFN2_DIPOLE_COMPONENTS,
    GFN2_QUADRUPOLE_COMPONENTS,
)
from vibeqc_compiler.method.gfn2_electronic_runtime import (
    build_gfn2_core_energy_update_program,
    build_gfn2_multipole_hamiltonian_update_program,
    build_gfn2_multipole_integral_vjp_program,
    build_gfn2_population_update_program,
    build_gfn2_runtime_electronic_pair_primal,
    build_gfn2_runtime_electronic_pair_vjp,
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
        ROOT / "src/xtb/native/src/model/gfn2/mulliken.cpp",
        ROOT / "src/xtb/native/src/model/gfn2/mulliken_kernels_impl.hpp",
        ROOT / "src/xtb/native/src/model/gfn2/force.cpp",
        ROOT / "src/xtb/native/src/model/gfn2/scc_driver.cpp",
    )
    sources = [path.read_text() for path in targets]
    assert all("generated_gfn2_electronic_native.hpp" in source for source in sources)
    joined = "\n".join(sources)
    assert "add_product(-density_value" not in joined
    assert "-0.5 * pair_density" not in joined
    assert "-0.5 * task.dipole_integrals" not in joined
    assert "-0.5 * integrals.dipole" not in joined
    assert "std::fma(geometry.h0" not in joined


def _pair_case() -> tuple[object, dict[str, np.ndarray], dict[str, object], int, int]:
    topology = _topology()
    feeds = _restricted_feeds()
    system = 1
    orbitals = (
        topology.system_orbital_offsets[system + 1]
        - topology.system_orbital_offsets[system]
    )
    row = topology.system_orbital_offsets[system]
    column = row + 5
    matrix_begin = topology.matrix_offsets[system]
    forward = matrix_begin + 5
    reverse = matrix_begin + 5 * orbitals
    pair = {
        "overlap": feeds["overlap"][forward],
        "row_scalar_potential": feeds["shell_scalar_potential"][
            topology.orbital_to_shell[row]
        ],
        "column_scalar_potential": feeds["shell_scalar_potential"][
            topology.orbital_to_shell[column]
        ],
    }
    for index, component in enumerate(GFN2_DIPOLE_COMPONENTS):
        pair[f"dipole_forward_{component}"] = feeds["dipole_integrals"][index, forward]
        pair[f"dipole_reverse_{component}"] = feeds["dipole_integrals"][index, reverse]
        pair[f"dipole_row_potential_{component}"] = feeds["atomic_dipole_potential"][
            topology.orbital_to_atom[row], index
        ]
        pair[f"dipole_column_potential_{component}"] = feeds["atomic_dipole_potential"][
            topology.orbital_to_atom[column], index
        ]
    for index, component in enumerate(GFN2_QUADRUPOLE_COMPONENTS):
        pair[f"quadrupole_forward_{component}"] = feeds["quadrupole_integrals"][
            index, forward
        ]
        pair[f"quadrupole_reverse_{component}"] = feeds["quadrupole_integrals"][
            index, reverse
        ]
        pair[f"quadrupole_row_potential_{component}"] = feeds[
            "atomic_quadrupole_potential"
        ][topology.orbital_to_atom[row], index]
        pair[f"quadrupole_column_potential_{component}"] = feeds[
            "atomic_quadrupole_potential"
        ][topology.orbital_to_atom[column], index]
    return topology, feeds, pair, forward, reverse


def test_runtime_pair_primal_matches_full_505_graph() -> None:
    topology, feeds, pair, forward, _ = _pair_case()
    full = build_gfn2_electronic_program("GFN2-xTB", topology)
    expected = (
        execute(full.program, feeds).outputs["hamiltonian"][forward]
        - feeds["h0"][forward]
    )
    actual = execute(build_gfn2_runtime_electronic_pair_primal(), pair).outputs["shift"]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=8e-16)


def test_runtime_pair_vjp_matches_full_505_graph() -> None:
    topology, feeds, pair, forward, reverse = _pair_case()
    full = build_gfn2_electronic_program("GFN2-xTB", topology)
    bar_shift = 0.37
    bar_hamiltonian = np.zeros(topology.matrix_count)
    bar_hamiltonian[forward] = bar_shift
    expected = execute(
        full.integral_vjp().program,
        {**feeds, "bar_hamiltonian": bar_hamiltonian},
    ).outputs
    program = build_gfn2_runtime_electronic_pair_vjp()
    live_inputs = {
        node.attrs["name"] for node in program.live_nodes if node.op == "input"
    }
    actual = execute(
        program,
        {
            **{name: value for name, value in pair.items() if name in live_inputs},
            "bar_shift": bar_shift,
        },
    ).outputs
    np.testing.assert_allclose(
        actual["bar_overlap"], expected["bar_overlap"][forward], rtol=0, atol=8e-16
    )
    for index, component in enumerate(GFN2_DIPOLE_COMPONENTS):
        np.testing.assert_allclose(
            actual[f"bar_dipole_forward_{component}"],
            expected["bar_dipole_integrals"][index, forward],
            rtol=0,
            atol=8e-16,
        )
        np.testing.assert_allclose(
            actual[f"bar_dipole_reverse_{component}"],
            expected["bar_dipole_integrals"][index, reverse],
            rtol=0,
            atol=8e-16,
        )
    for index, component in enumerate(GFN2_QUADRUPOLE_COMPONENTS):
        np.testing.assert_allclose(
            actual[f"bar_quadrupole_forward_{component}"],
            expected["bar_quadrupole_integrals"][index, forward],
            rtol=0,
            atol=8e-16,
        )
        np.testing.assert_allclose(
            actual[f"bar_quadrupole_reverse_{component}"],
            expected["bar_quadrupole_integrals"][index, reverse],
            rtol=0,
            atol=8e-16,
        )


def test_generated_cuda_header_and_consumers_own_the_pair_science(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "generated_gfn2_electronic_native.cuh"
    subprocess.run(
        [
            sys.executable,
            str(root / "tools/generate_gfn2_electronic_cuda.py"),
            "--output",
            str(output),
        ],
        check=True,
    )
    source = output.read_text()
    assert "__device__ inline bool gfn2_electronic_pair_tensor" in source
    assert "__device__ inline bool gfn2_electronic_pair_vjp_tensor" in source
    assert "evaluate_gfn2_electronic_overlap_vjp" in source

    hamiltonian = (
        root / "src/xtb/native/src/backends/cuda/gfn2_hamiltonian.cu"
    ).read_text()
    force = (
        root / "src/xtb/native/src/backends/cuda/gfn2_hamiltonian_force.cu"
    ).read_text()
    assert "evaluate_gfn2_electronic_pair(" in hamiltonian
    assert "-0.5 * input.dipole_integrals" not in hamiltonian
    assert "evaluate_gfn2_electronic_pair_vjp(" in force
    assert "-0.5 * pair_density *" not in force
