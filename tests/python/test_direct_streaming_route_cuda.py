"""Primary streaming must preserve complete bounded HF endpoints and replay."""

import os
from dataclasses import asdict

import numpy as np
import pytest
from vibeqc import Calculator

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)


@pytest.mark.parametrize("method", ["rhf", "uhf"])
@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
def test_primary_streaming_partition_replay_matches_libcint(
    method: str, representation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing the mask must recapture both routes, including active batch segments.

    Bent H2O/H2O+ supplies s/p/d classes without an open-shell degeneracy.
    Independent libcint SCF energies/forces catch missing or duplicated work;
    repeated and changed-geometry calls exercise the cached graph's lifetime.
    """
    from pyscf import gto, scf

    assert os.environ.get("SLURM_JOB_ID")
    monkeypatch.setenv("VIBEQC_BOUNDED_DIRECT_STREAMING", "force")
    monkeypatch.delenv("VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD", raising=False)
    separate = method == "uhf"
    atoms = [("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.8)), ("H", (1.7, 0.0, -0.6))]
    coordinates = np.array([position for _, position in atoms])
    displaced = coordinates.copy()
    displaced[1, 0] += 0.015
    other = [
        (symbol, tuple(position))
        for (symbol, _), position in zip(atoms, displaced, strict=True)
    ]
    references = []
    for geometry in (atoms, other):
        molecule = gto.M(
            atom=geometry,
            basis="def2-svp",
            unit="Bohr",
            charge=int(separate),
            spin=int(separate),
            cart=representation == "cartesian",
            verbose=0,
        )
        reference = (scf.UHF if separate else scf.RHF)(molecule)
        reference.conv_tol = 1e-13
        reference.conv_tol_grad = 1e-10
        # The cation needs about 60 cycles at this gradient tolerance; PySCF's
        # default limit of 50 would stop before the independent oracle converges.
        reference.max_cycle = 100
        reference.kernel()
        assert reference.converged
        references.append((reference.e_tot, -reference.nuc_grad_method().kernel()))

    calculator = Calculator(
        method=method,
        basis="def2-svp",
        basis_representation=representation,
        device="cuda",
        density_fitting="none",
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-14,
    )
    with calculator.prepare_batch(
        [atoms, other],
        charges=[int(separate)] * 2,
        multiplicities=[2 if separate else 1] * 2,
        shell_class_profiling=True,
    ) as prepared:
        work = None
        for mask in ("all", "all", "0x15", "none", "all"):
            monkeypatch.setenv("VIBEQC_BOUNDED_DIRECT_PRIMARY_STREAMING_MASK", mask)
            result = prepared.execute(strict=True)
            for item, (energy, forces) in zip(result.items, references, strict=True):
                assert item.executed_backend == "cuda"
                assert item.energy == pytest.approx(energy, abs=2e-9)
                np.testing.assert_allclose(item.forces, forces, atol=2e-8, rtol=0)
            current = [asdict(row) for row in prepared.last_shell_class_profile()]
            if work is None:
                work = current
            else:
                assert current == work

        changed = prepared.execute([displaced, coordinates], strict=True)
        for item, (energy, forces) in zip(
            changed.items, reversed(references), strict=True
        ):
            assert item.energy == pytest.approx(energy, abs=2e-9)
            np.testing.assert_allclose(item.forces, forces, atol=2e-8, rtol=0)
