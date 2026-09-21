"""Production cutover checks for compiler-owned GFN2 CUDA electronic science."""

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
    build_gfn2_runtime_electronic_pair_primal,
    build_gfn2_runtime_electronic_pair_vjp,
)
from vibeqc_compiler.tensor import execute


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
        root / "src/xtb/gfn2_runtime/src/backends/cuda/gfn2_hamiltonian.cu"
    ).read_text()
    force = (
        root / "src/xtb/gfn2_runtime/src/backends/cuda/gfn2_hamiltonian_force.cu"
    ).read_text()
    assert "evaluate_gfn2_electronic_pair(" in hamiltonian
    assert "-0.5 * input.dipole_integrals" not in hamiltonian
    assert "evaluate_gfn2_electronic_pair_vjp(" in force
    assert "-0.5 * pair_density *" not in force
