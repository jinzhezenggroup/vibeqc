"""Native bounded providers must recover canonical MP2 before truncation."""

from dataclasses import replace

import numpy as np
import pytest
from vibeqc import Primitive, Shell

from tools.vibeqc_local_cc.localization import localize_occupied
from tools.vibeqc_local_cc.mp2 import (
    build_local_mp2,
    plan_local_spaces,
    recover_canonical_amplitudes,
)
from tools.vibeqc_local_cc.spaces import projected_virtual_space
from tools.vibeqc_posthf.conventions import MOBlock
from tools.vibeqc_posthf.df import MetricFactor
from tools.vibeqc_posthf.export import export_rhf
from tools.vibeqc_posthf.fixtures import (
    fixture_snapshot,
    load_fixture,
    source_arguments,
)
from tools.vibeqc_posthf.providers import ConventionalProvider
from tools.vibeqc_posthf.sources import NativeSource


def ao_atoms(source):
    return tuple(
        sh.atom_index
        for sh in source.shells
        for _ in range(
            (sh.angular_momentum + 1) * (sh.angular_momentum + 2) // 2
            if source.representation == "cartesian"
            else 2 * sh.angular_momentum + 1
        )
    )


@pytest.mark.parametrize("name", ["h2", "water", "lih", "f_heh"])
@pytest.mark.parametrize("label", ["conventional", "df"])
def test_full_space_recovery_with_independent_same_hamiltonian_mp2(name, label):
    meta, arrays = load_fixture(name)
    with NativeSource(**source_arguments(meta)) as source:
        metric = MetricFactor.from_source(source) if label == "df" else None
        snapshot = fixture_snapshot(meta, arrays, label=label, metric=metric)
        local = localize_occupied(snapshot, ao_atoms(source))
        domain = projected_virtual_space(snapshot)
        result = build_local_mp2(
            snapshot, source, local, domain, metric=metric, keep_full_space=True
        )
        assert result.full_space_recovery
        assert (
            abs(
                result.correlation_energy - meta["records"][label]["correlation_energy"]
            )
            < 1e-9
        )
        recovered = recover_canonical_amplitudes(snapshot, local, result)
        np.testing.assert_allclose(
            recovered, arrays[label + "_t2"], atol=1e-11, rtol=1e-10
        )
        np.testing.assert_allclose(
            recovered, recovered.transpose(1, 0, 3, 2), atol=1e-12
        )
        assert (
            result.provider_peak_bytes + result.plan.local_numeric_bytes
            <= result.plan.budget_bytes
        )
        assert result.retained_numeric_bytes < result.plan.local_numeric_bytes
        assert result.derivative_status == "unsupported"
        with pytest.raises(ValueError, match="response"):
            replace(result, derivative_status="analytic")
        if name == "lih" and label == "conventional":
            # Re-express every retained pair in a different orthogonal gauge.
            # Both amplitude legs and the integral weights must transform.
            rng = np.random.default_rng(182)
            changed = []
            for pair in result.pairs:
                rotation, _ = np.linalg.qr(
                    rng.normal(size=(pair.space.rank, pair.space.rank))
                )
                changed.append(
                    replace(
                        pair,
                        space=replace(
                            pair.space, columns=pair.space.columns @ rotation
                        ),
                        amplitudes=rotation.T @ pair.amplitudes @ rotation,
                        integrals=rotation.T @ pair.integrals @ rotation,
                    )
                )
            transported = replace(result, pairs=tuple(changed))
            np.testing.assert_allclose(
                recover_canonical_amplitudes(snapshot, local, transported),
                recovered,
                atol=1e-11,
                rtol=0,
            )


def test_constrained_budget_and_truncation_do_not_change_parent_hamiltonian(
    monkeypatch,
):
    meta, arrays = load_fixture("water")
    with NativeSource(**source_arguments(meta)) as source:
        s = fixture_snapshot(meta, arrays)
        local = localize_occupied(s, ao_atoms(source))
        domain = projected_virtual_space(s)
        reserve = plan_local_spaces(s, local, domain).local_numeric_bytes
        with ConventionalProvider(s, source) as provider:
            minimum = provider.plan(MOBlock.from_spaces(s, "ovov")).peak_bytes + reserve
        with monkeypatch.context() as patch:
            patch.setattr(
                source,
                "tile",
                lambda *_: pytest.fail("integral read before budget rejection"),
            )
            with pytest.raises(MemoryError):
                build_local_mp2(s, source, local, domain, budget_bytes=minimum - 1)
        exact = build_local_mp2(
            s, source, local, domain, keep_full_space=True, budget_bytes=minimum
        )
        loose = build_local_mp2(
            s, source, local, domain, occupation_threshold=1e8, budget_bytes=minimum
        )
        assert (
            exact.full_space_recovery and abs(exact.observed_energy_difference) < 1e-10
        )
        assert loose.correlation_energy == 0 and not loose.full_space_recovery
        assert all(p.space.rank == 0 for p in loose.pairs)
        assert exact.reference_id == loose.reference_id == s.identity
        assert exact.hamiltonian_id == loose.hamiltonian_id == s.hamiltonian_id
        assert (
            abs(loose.observed_energy_difference - abs(exact.correlation_energy))
            < 1e-10
        )
        with pytest.raises(MemoryError):
            recover_canonical_amplitudes(s, local, exact, budget_bytes=1)


def test_stale_localization_and_malformed_occupied_coupling_are_rejected():
    meta, arrays = load_fixture("lih")
    with NativeSource(**source_arguments(meta)) as source:
        s = fixture_snapshot(meta, arrays)
        local = localize_occupied(s, ao_atoms(source))
        domain = projected_virtual_space(s)
        with pytest.raises(ValueError, match="stale"):
            build_local_mp2(replace(s, generation_id="changed"), source, local, domain)
        with pytest.raises(ValueError, match="parent reference"):
            build_local_mp2(
                s,
                source,
                replace(local, occupied_fock=np.diag(np.diag(local.occupied_fock))),
                domain,
            )
        with pytest.raises(ValueError, match="atomic partition"):
            build_local_mp2(
                s,
                source,
                replace(local, ao_atoms=tuple(1 - a for a in local.ao_atoms)),
                domain,
            )


def test_near_zero_denominators_fail_before_any_truncated_local_result():
    meta, arrays = load_fixture("h2")
    with NativeSource(**source_arguments(meta)) as source:
        s = fixture_snapshot(meta, arrays)
        eps = np.array([-1.0, -1.0 + 1e-12])
        fock = (s.overlap @ s.coefficients * eps) @ s.coefficients.T @ s.overlap
        s = replace(s, orbital_energies=eps, fock=fock)
        local = localize_occupied(s, ao_atoms(source))
        domain = projected_virtual_space(s)
        with pytest.raises(ValueError, match="no regularization"):
            build_local_mp2(s, source, local, domain)


@pytest.mark.parametrize("exponent", [1.0, 1e-12])
def test_duplicate_and_extremely_diffuse_native_bases_cannot_supply_a_local_reference(
    exponent,
):
    atoms = [("H", (0, 0, 0)), ("H", (0, 0, 1.4))]
    basis = [
        Shell(0, 0, (Primitive(exponent, 1.0),)),
        Shell(1, 0, (Primitive(exponent, 1.0),)),
    ]
    if exponent == 1.0:
        basis.insert(1, basis[0])
    with NativeSource(atoms, basis) as source:
        assert np.linalg.eigvalsh(source.one_electron()[0])[0] < 1e-10
        with pytest.raises(RuntimeError, match="linearly dependent"):
            export_rhf(source)


def test_native_atom_permutation_preserves_local_truncation_with_new_identity():
    metadata, _ = load_fixture("water")
    args = source_arguments(metadata)
    reversed_args = dict(args)
    reversed_args["atoms"] = tuple(reversed(args["atoms"]))
    for name in ("basis", "auxiliary_basis"):
        reversed_args[name] = tuple(
            replace(shell, atom_index=len(args["atoms"]) - 1 - shell.atom_index)
            for shell in args[name]
        )
    results = []
    for inputs in (args, reversed_args):
        with NativeSource(**inputs) as source:
            snapshot, _ = export_rhf(source, generation_id="atom-permutation")
            local = localize_occupied(snapshot, ao_atoms(source))
            result = build_local_mp2(
                snapshot,
                source,
                local,
                projected_virtual_space(snapshot),
                occupation_threshold=1e-6,
            )
            results.append(result)
    assert results[0].reference_id != results[1].reference_id
    assert abs(results[0].correlation_energy - results[1].correlation_energy) < 1e-10
