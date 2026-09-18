"""The real #162 producer, source binding and lifetime gates (Slurm on CUDA)."""

import os
from dataclasses import replace

import numpy as np
import pytest
from vibeqc import Calculator, GridSpec, KsOptions
from vibeqc._dft_gradient import (
    StableGridMotion,
    StationaryDerivativeContract,
    StationaryKsState,
    bind_generated_xc_geometry,
    scf_regularization_identity,
    xc_regularization_identity,
)
from vibeqc_compiler.dft import ExplicitGrid, NativeAO
from vibeqc_compiler.xc import functional

from tools.vibeqc_validation.dft_gradient import h2_overlap

ATOMS = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
GRID = GridSpec(radial_points=24, angular_polar=8, angular_azimuth=16)


def test_cpu_plan_cannot_publish_a_native_stationary_proof():
    calculator = Calculator(
        method="lda-rks", device="cpu", ks_options=KsOptions(grid=GRID)
    )
    with calculator.prepare_batch([ATOMS]) as batch, NativeAO(ATOMS) as basis:
        batch.execute(strict=True)
        with pytest.raises(NotImplementedError):
            StationaryKsState.from_native(batch, basis)


@pytest.mark.skipif(
    os.environ.get("VIBEQC_DFT_CUDA_TEST") != "1", reason="Slurm CUDA gate"
)
@pytest.mark.parametrize("method", ["lda-rks", "pbe-rks", "lda-uks", "pbe-uks"])
def test_native_snapshot_rejects_relabeling_and_replay(method):
    unrestricted = method.endswith("uks")
    charge, multiplicity = (-1, 2) if unrestricted else (0, 1)
    calculator = Calculator(
        method=method,
        device="cuda",
        ks_options=KsOptions(grid=GRID),
        max_iterations=200,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    with calculator.prepare_batch(
        [ATOMS], charges=[charge], multiplicities=[multiplicity]
    ) as batch:
        with NativeAO(ATOMS, charge=charge, multiplicity=multiplicity) as basis:
            with pytest.raises(RuntimeError, match="invalid argument"):
                StationaryKsState.from_native(batch, basis)
            batch.execute(strict=True)
            state = StationaryKsState.from_native(batch, basis)
            contract = StationaryDerivativeContract(state.identity)
            assert contract.validate(state) is state
            with pytest.raises(AttributeError, match="provenance is immutable"):
                state._source.metadata = tuple([1] * 16)
            np.testing.assert_allclose(state.overlap, h2_overlap(basis), atol=2e-14)
            np.testing.assert_allclose(
                np.trace(state.density @ state.overlap, axis1=1, axis2=2),
                [2, 1] if unrestricted else [2],
                atol=1e-9,
            )

            # A self-consistent change of AO metric must still fail the native
            # content proof. Algebra alone accepts this congruence transform.
            transformed = replace(
                state,
                overlap=2 * state.overlap,
                density=state.density / 2,
                weighted_density=state.weighted_density / 2,
                coefficients=state.coefficients / np.sqrt(2),
                fock=2 * state.fock,
            )
            assert contract._validate_arrays(transformed) is transformed
            with pytest.raises(ValueError, match="snapshot content"):
                contract.validate(transformed)

            spec = functional(
                "PBE" if method.startswith("pbe") else "LDA_XC_PW",
                spin="polarized" if unrestricted else "unpolarized",
            )
            generated = bind_generated_xc_geometry(
                contract, state, spec, basis, state.grid
            )
            assert generated.regularization_identity == scf_regularization_identity()
            shift = np.array([0.13, -0.07, 0.05])
            translation = StableGridMotion(
                topology_identity=state.identity.topology_identity,
                centers=np.broadcast_to(shift, generated.partials.centers.shape),
                points=np.broadcast_to(shift, generated.partials.points.shape),
                weights=np.zeros_like(generated.partials.weights),
            )
            assert generated.directional(translation).total == pytest.approx(
                0.0, abs=2e-10
            )

            # Relabeling the exact native state as the interior diagnostic
            # domain is still forbidden even though #163-A now has a real bridge.
            relabeled = replace(
                state,
                identity=replace(
                    state.identity,
                    regularization_identity=xc_regularization_identity(spec),
                ),
            )
            with pytest.raises(ValueError, match="native stationary state identity"):
                StationaryDerivativeContract(relabeled.identity).validate(relabeled)

            # Same AO count and geometry, different exponents: labels and sizes
            # cannot substitute for the exact normalized source basis.
            shells = list(basis.shells)
            shell = shells[0]
            primitive = replace(
                shell.primitives[0], exponent=1.1 * shell.primitives[0].exponent
            )
            shells[0] = replace(shell, primitives=(primitive, *shell.primitives[1:]))
            with NativeAO(
                ATOMS, basis=shells, charge=charge, multiplicity=multiplicity
            ) as wrong:
                with pytest.raises(ValueError, match="basis/overlap source"):
                    StationaryKsState.from_native(batch, wrong)
                forged = replace(
                    state,
                    identity=replace(state.identity, basis_identity=wrong.identity),
                )
                with pytest.raises(
                    ValueError, match="native stationary state identity"
                ):
                    StationaryDerivativeContract(forged.identity).validate(forged)

            wrong_grid = ExplicitGrid(
                state.grid.points,
                state.grid.weights * 1.01,
                state.grid.owners,
                {"test": "wrong weights"},
            )
            with pytest.raises(ValueError, match="grid source"):
                StationaryKsState.from_native(batch, basis, wrong_grid)

            with calculator.prepare_batch(
                [ATOMS], charges=[charge], multiplicities=[multiplicity]
            ) as other_batch:
                other_batch.execute(strict=True)
                other = StationaryKsState.from_native(other_batch, basis)
                with pytest.raises(
                    ValueError, match="native stationary state identity"
                ):
                    contract.validate(replace(state, _source=other._source))

            batch.execute(strict=True)
            with pytest.raises(ValueError, match="stale"):
                contract.validate(state)
            current = StationaryKsState.from_native(batch, basis)
            assert current.identity.solve_epoch > state.identity.solve_epoch
            stale_handle = state._source._handle
            assert type(stale_handle) is int  # No mutable ctypes .value alias.
            with pytest.raises(AttributeError, match="provenance is immutable"):
                state._source._handle = current._source._handle
            with pytest.raises(AttributeError, match="provenance is immutable"):
                del state._source._handle
            assert state._source._handle == stale_handle
            with pytest.raises(ValueError, match="stale"):
                contract.validate(state)
            assert (
                StationaryDerivativeContract(current.identity).validate(current)
                is current
            )

            # A rejected geometry update revokes the next proof too, before a
            # new successful solve. This calls the actual native invalidation.
            result = batch.execute(coordinates=[[0.0]], strict=False)
            assert not result.items[0].succeeded
            with pytest.raises(ValueError, match="stale"):
                StationaryDerivativeContract(current.identity).validate(current)

            batch.execute(strict=True)
            final = StationaryKsState.from_native(batch, basis)
        # Source AO lifetime does not own the native snapshot; batch lifetime does.
        assert StationaryDerivativeContract(final.identity).validate(final) is final
        final._source.close()
        assert final._source._handle == 0
        final._source.close()  # Explicit lease closure is idempotent.
        with pytest.raises(ValueError, match="stale"):
            StationaryDerivativeContract(final.identity).validate(final)
    with pytest.raises(RuntimeError, match="closed"):
        StationaryDerivativeContract(final.identity).validate(final)
