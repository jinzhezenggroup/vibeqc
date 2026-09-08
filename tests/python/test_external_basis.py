"""Offline basis identity, independent integral/HF and unsupported-domain gates."""

import json
from dataclasses import replace
from decimal import localcontext
from pathlib import Path

import numpy as np
import pytest
from vibeqc import (
    Atom,
    BasisProvenance,
    BasisShell,
    Calculator,
    Primitive,
    Shell,
    basis_capability,
    electron_state,
    import_bse,
    load_basis,
)
from vibeqc.basis import NORMALIZATION, decimal_text, read_local_json
from vibeqc.elements import SYMBOLS
from vibeqc.profiles import canonical_hash, file_hash

from tools.vibeqc_dft import GridSpec, MolecularGrid, NativeAO
from tools.vibeqc_posthf.sources import NativeSource
from tools.vibeqc_validation.schema import block_error

ROOT = Path(__file__).resolve().parents[1] / "data/external_basis"


def imported(name="sto-3g-ho", representation="cartesian"):
    return import_bse(
        ROOT / (name + ".json"),
        source="pinned test data",
        source_version="4adaf1372c7101620ca1a9f3130be9ae97fb8f30",
        license="BSD-3-Clause AND GPL-3.0-or-later"
        if name == "synthetic-fe-h"
        else "BSD-3-Clause",
        representation=representation,
    )


def check(actual, expected, *, atol=1e-11, rtol=1e-10):
    result = block_error(actual, expected, atol=atol, rtol=rtol)
    assert result["passed"], result


def test_source_hashes_and_complete_canonical_roundtrip(tmp_path):
    metadata = json.loads((ROOT / "manifest.json").read_text())
    for name, checksum in metadata["files"].items():
        assert file_hash(ROOT / name) == checksum
    assert (
        file_hash(
            Path(__file__).resolve().parents[2]
            / "tools/generate_external_basis_references.py"
        )
        == metadata["generator_sha256"]
    )
    for name in ("sto-3g-ho", "cc-pvtz-fe", "def2-tzvp-au", "synthetic-fe-h"):
        basis = imported(name)
        path = tmp_path / (name + ".json")
        basis.write(path)
        other = load_basis(path)
        assert other == basis and other.identity == basis.identity
        assert other.to_payload() == basis.to_payload()
        assert other.provenance.checksum == file_hash(ROOT / (name + ".json"))


def test_combined_shell_groups_general_columns_and_zero_coefficients_survive():
    oxygen = imported().by_element[8]
    assert [s.angular_momentum for s in oxygen.shells] == [0, 0, 1]
    assert oxygen.shells[1].source_group == oxygen.shells[2].source_group
    fe = imported("synthetic-fe-h")
    data = fe.by_element[26].shells[0]
    assert len(data.coefficients) == 2 and data.coefficients[0][-1] == "0"
    shells = fe.shells_for([Atom.from_value(("Fe", (0, 0, 0)))])
    assert [s.angular_momentum for s in shells] == [0, 0, 1, 2]
    assert len(shells[0].primitives) == len(shells[1].primitives) == 3
    assert shells[0].primitives[-1].coefficient == 0


def test_decimal_precision_does_not_depend_on_decimal_context():
    value = "0.123456789012345678901234567890123456789"
    with localcontext() as context:
        context.prec = 5
        a = decimal_text(value, "coefficient")
    with localcontext() as context:
        context.prec = 50
        b = decimal_text(value, "coefficient")
    assert a == b and float(a) == float(value)
    for value in ("1e-999", "NaN", "Infinity", "1e999", True):
        with pytest.raises((ValueError, TypeError)):
            decimal_text(value, "coefficient")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"exponents": ("0",)},
        {"exponents": ("-1",)},
        {"exponents": ("nan",)},
        {"exponents": ("inf",)},
        {"exponents": ("1e-400",)},
        {"coefficients": (("1", "2"),)},
        {"coefficients": (("0",),)},
        {"coefficients": (("inf",),)},
        {"angular_momentum": 2**32},
    ],
)
def test_malformed_primitive_and_contraction_records(kwargs):
    arguments = {"angular_momentum": 0, "exponents": ("1",), "coefficients": (("1",),)}
    with pytest.raises(ValueError):
        BasisShell(**(arguments | kwargs))


def test_checksums_duplicate_keys_records_and_normalization_are_not_ignored(tmp_path):
    basis = imported()
    for kwargs in (
        {"normalization": "already-radially-normalized"},
        {"exponent_units": "angstrom^-2"},
        {"elements": (basis.elements[0], basis.elements[0])},
    ):
        with pytest.raises(ValueError):
            replace(basis, **kwargs)
    assert basis.normalization == NORMALIZATION
    path = tmp_path / "basis.json"
    path.write_text('{"elements": {}, "elements": {}}')
    with pytest.raises(ValueError, match="duplicate"):
        read_local_json(path)
    basis.write(path)
    payload = json.loads(path.read_text())
    payload["name"] = "changed"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="checksum"):
        load_basis(path)
    with pytest.raises(ValueError):
        BasisProvenance("source", "", "BSD-3-Clause", "0" * 64)
    with pytest.raises(ValueError):
        replace(basis.by_element[1], ecp_core_electrons=1)


def test_ambiguous_combined_and_malformed_ecp_imports(tmp_path):
    data = json.loads((ROOT / "sto-3g-ho.json").read_text())
    shell = data["elements"]["8"]["electron_shells"][1]
    shell["coefficients"].append(shell["coefficients"][0])
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="ambiguous"):
        import_bse(path, source="test", source_version="1", license="BSD-3-Clause")

    data = json.loads((ROOT / "def2-tzvp-au.json").read_text())
    data["elements"]["79"]["ecp_potentials"][0]["r_exponents"] = []
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="radial"):
        import_bse(path, source="test", source_version="1", license="BSD-3-Clause")


@pytest.mark.parametrize("version", [True, 1.0, "1", None])
def test_schema_version_requires_an_exact_integer(tmp_path, version):
    payload = imported().to_payload()
    payload.pop("checksum")
    payload["schema_version"] = version
    payload["checksum"] = canonical_hash(payload)
    path = tmp_path / "bad-version.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="schema"):
        load_basis(path)


@pytest.mark.parametrize("arrays", [("12", [[1, 2]]), ([1, 2], ["12"]), ([1], "1")])
def test_strings_are_not_primitive_or_contraction_arrays(arrays):
    with pytest.raises(TypeError, match="arrays"):
        BasisShell(0, *arrays)


def test_auxiliary_representation_cannot_be_lost_by_posthf_source():
    atoms = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    with pytest.raises(ValueError, match="representation conflicts"):
        NativeSource(atoms, auxiliary_basis=imported(representation="spherical"))
    with NativeSource(atoms, basis=imported(), auxiliary_basis=imported()) as source:
        assert source.nbf == source.naux == 2


def test_auxiliary_high_l_and_representation_capability_checks():
    basis = imported("synthetic-fe-h")
    atoms = [("Fe", (0, 0, 0))]
    for auxiliary in (
        imported("cc-pvtz-fe"),
        replace(
            basis,
            elements=(
                replace(basis.by_element[26], shells=(BasisShell(5, [1], [[1]]),)),
            ),
        ),
    ):
        with pytest.raises(NotImplementedError, match="auxiliary.*l=[45]"):
            Calculator(
                basis=basis, auxiliary_basis=auxiliary, density_fitting="cpu"
            ).singlepoint(atoms, charge=24)
    assert not basis_capability(basis, atoms, representation="spherical")["eligible"]


@pytest.mark.parametrize(
    "kwargs", [{"charge": 0.5}, {"multiplicity": True}, {"charge": 2**32}]
)
def test_public_metadata_rejects_noninteger_native_fields(kwargs):
    with pytest.raises(ValueError, match="integer"):
        Calculator(basis=imported()).basis_metadata([("H", (0, 0, 0))], **kwargs)


def test_elements_nuclear_charge_ecp_cores_and_requested_electrons_are_separate():
    assert len(SYMBOLS) == 118
    assert Atom.from_value(("Fe", (0, 0, 0))).atomic_number == 26
    assert Atom.from_value(("Og", (0, 0, 0))).atomic_number == 118
    state = electron_state([("Fe", (0, 0, 0))], charge=24)
    assert (
        state.atomic_numbers,
        state.nuclear_charges,
        state.ecp_core_electrons,
        state.electron_count,
        state.nalpha,
        state.nbeta,
    ) == ((26,), (26,), (0,), 2, 1, 1)
    au = imported("def2-tzvp-au")
    state = electron_state(
        [("Au", (0, 0, 0))],
        multiplicity=2,
        element_metadata=au.by_element,
        electron_count=19,
    )
    assert state.nuclear_charges == (79,) and state.ecp_core_electrons == (60,)
    assert (state.nalpha, state.nbeta) == (10, 9)
    with pytest.raises(ValueError, match="conflicts"):
        electron_state(
            [("Au", (0, 0, 0))],
            multiplicity=2,
            element_metadata=au.by_element,
            electron_count=79,
        )
    for z in (0, 119, -1, 1.5, True, 2**40):
        with pytest.raises(ValueError):
            Atom.from_value((z, (0, 0, 0)))
    with pytest.raises(ValueError):
        Atom.from_value(("Fe", (float("nan"), 0, 0)))
    with pytest.raises(ValueError):
        electron_state([("H", (0, 0, 0))], charge=-(2**31))
    with pytest.raises(ValueError, match="spin occupations"):
        electron_state([("H", (0, 0, 0))], multiplicity=1)


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
@pytest.mark.parametrize("role", ["orbital", "auxiliary"])
def test_realistic_high_l_and_ecp_are_loadable_but_have_precise_missing_routes(
    backend, role
):
    fe = imported("cc-pvtz-fe")
    for operator in (
        "overlap",
        "kinetic",
        "nuclear_attraction",
        "eri",
        "df_metric",
        "df_three_center",
    ):
        report = basis_capability(
            fe,
            [("Fe", (0, 0, 0))],
            backend=backend,
            operator=operator,
            derivative_order=1,
            role=role,
        )
        assert report["data_loadable"] and not report["eligible"]
        assert any(
            "Z=26 shell 4" in reason and "l=4" in reason and operator in reason
            for reason in report["reasons"]
        )
        assert report["shells_checked"] == 20
    au = imported("def2-tzvp-au")
    report = basis_capability(au, [("Au", (0, 0, 0))], backend=backend, role=role)
    assert any("ECP with 60 core electrons" in r for r in report["reasons"])
    assert not report["eligible"]
    assert not basis_capability(
        imported(), [("H", (0, 0, 0))], backend=backend, derivative_order=2
    )["eligible"]
    assert basis_capability(
        imported(),
        [("H", (0, 0, 0))],
        backend=backend,
        operator="ao",
        derivative_order=3,
    )["eligible"]


def test_public_rejections_happen_without_truncating_or_changing_charge():
    for name, atoms, text in (
        ("cc-pvtz-fe", [("Fe", (0, 0, 0))], "l=4"),
        ("def2-tzvp-au", [("Au", (0, 0, 0))], "ECP"),
    ):
        with pytest.raises(NotImplementedError, match=text):
            Calculator(basis=imported(name)).singlepoint(atoms)
    with pytest.raises(NotImplementedError, match="ECP"):
        imported("def2-tzvp-au").shells_for([Atom.from_value(("Au", (0, 0, 0)))])
    with pytest.raises(NotImplementedError, match="Z=26"):
        Calculator().singlepoint([("Fe", (0, 0, 0))])
    with pytest.raises(ValueError, match="conflicts"):
        Calculator(
            basis=imported(representation="spherical"), basis_representation="cartesian"
        )


@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
def test_imported_equivalent_preserves_bundled_energy_forces_and_mathematical_hash(
    tmp_path, representation
):
    basis = imported(representation=representation)
    path = tmp_path / "water.json"
    basis.write(path)
    atoms = [("O", (0, 0, 0)), ("H", (0, -1.43, 1.1)), ("H", (0, 1.43, 1.1))]
    a = Calculator(basis="sto-3g", basis_representation=representation).singlepoint(
        atoms
    )
    b = Calculator(basis=path).singlepoint(atoms)
    assert a.energy == b.energy
    np.testing.assert_array_equal(a.forces, b.forces)
    assert (
        a.basis_metadata["orbital"]["mathematical_identity"]
        == b.basis_metadata["orbital"]["mathematical_identity"]
    )
    assert (
        a.basis_metadata["orbital"]["basis_identity"]
        != b.basis_metadata["orbital"]["basis_identity"]
    )
    assert (
        b.basis_metadata["orbital"]["provenance"]["checksum"]
        == basis.provenance.checksum
    )


def test_prepared_snapshot_file_changes_and_auxiliary_model_invalidation(tmp_path):
    basis = imported()
    path = tmp_path / "basis.json"
    basis.write(path)
    atoms = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    calculator = Calculator(basis=path)
    with calculator.prepare_batch([atoms, atoms]) as prepared:
        first = prepared.execute(strict=True)
        # Overwriting the source file cannot mutate an already owned snapshot.
        element = basis.elements[0]
        shell = element.shells[0]
        changed = replace(
            basis,
            elements=(
                replace(
                    element,
                    shells=(replace(shell, exponents=("4", *shell.exponents[1:])),),
                ),
                *basis.elements[1:],
            ),
        )
        changed.write(path)
        again = prepared.execute(strict=True)
        check(again.energies, first.energies, atol=1e-9)
        assert again.items[0].basis_metadata == first.items[0].basis_metadata
        metadata = prepared.basis_metadata
        metadata[0]["model_identity"] = "tampered detached result"
        assert (
            prepared.basis_metadata[0]["model_identity"]
            != metadata[0]["model_identity"]
        )
        calculator._basis = load_basis(path)
        with pytest.raises(RuntimeError, match="identity changed"):
            prepared.execute()
    with Calculator(
        basis=basis, density_fitting="cpu", auxiliary_basis=basis
    ).prepare_batch([atoms]) as prepared:
        assert "auxiliary" in prepared.basis_metadata[0]
        prepared._calculator._auxiliary_basis = changed
        with pytest.raises(RuntimeError, match="identity changed"):
            prepared.execute()


def test_explicit_caller_shell_storage_is_owned():
    primitives = [Primitive(1, 1)]
    shells = [Shell(0, 0, primitives)]
    calculator = Calculator(basis=shells)
    primitives[0] = Primitive(2, 1)
    shells.clear()
    result = calculator.singlepoint([("He", (0, 0, 0))])
    expected = Calculator(basis=(Shell(0, 0, (Primitive(1, 1),)),)).singlepoint(
        [("He", (0, 0, 0))]
    )
    assert result.energy == expected.energy


@pytest.mark.parametrize(
    "case",
    [
        "fe_ion_cartesian",
        "fe_ion_spherical",
        "fe_h_ion_cartesian",
        "fe_h_ion_spherical",
    ],
)
def test_supported_synthetic_transition_metal_ion_matches_independent_scf(case):
    metadata = json.loads((ROOT / "manifest.json").read_text())
    spec = next(c for c in metadata["cases"] if c["name"] == case)
    basis = imported("synthetic-fe-h", spec["representation"])
    result = Calculator(
        basis=basis,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-14,
        max_iterations=150,
    ).singlepoint(spec["atoms"], charge=spec["charge"])
    with np.load(ROOT / "oracles.npz") as oracle:
        check(np.array(result.energy), oracle[case + "_energy"], atol=1e-8, rtol=0)
        check(result.forces, oracle[case + "_forces"], atol=1e-7, rtol=0)
    assert result.basis_metadata["orbital"]["electrons"]["electron_count"] == 2


@pytest.mark.parametrize(
    "case", ["water_cartesian", "fe_ion_cartesian", "fe_ion_spherical"]
)
def test_independent_overlap_and_kinetic_from_native_ao_quadrature(case):
    metadata = json.loads((ROOT / "manifest.json").read_text())
    spec = next(c for c in metadata["cases"] if c["name"] == case)
    basis = imported(spec["source"].removesuffix(".json"), spec["representation"])
    # One-center Gaussian angular products are low-degree polynomials; use
    # radial refinement there. The molecular Becke partition needs a finer
    # angular mesh independently of the imported shell normalization.
    grid_spec = (
        GridSpec(160, 64, 128)
        if case.startswith("water")
        else GridSpec(160, 8, 16, element_radii=((26, 0.08),))
    )
    grid = MolecularGrid(spec["atoms"], grid_spec, charge=spec["charge"])
    with NativeAO(
        spec["atoms"],
        basis,
        representation=spec["representation"],
        charge=spec["charge"],
    ) as ao:
        overlap = np.zeros((ao.nao, ao.nao))
        kinetic = np.zeros_like(overlap)
        for tile in grid.tiles(256):
            jets = ao.evaluate(tile.points, order=1)
            overlap += np.einsum("p,pm,pn->mn", tile.weights, jets[0], jets[0])
            kinetic += 0.5 * np.einsum(
                "p,kpm,kpn->mn", tile.weights, jets[1:4], jets[1:4]
            )
    with np.load(ROOT / "oracles.npz") as oracle:
        check(overlap, oracle[case + "_overlap"], atol=2e-10, rtol=2e-10)
        check(kinetic, oracle[case + "_kinetic"], atol=2e-9, rtol=2e-10)


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_numpy_integer_metadata_preserves_native_occupation_diagnostics(dtype):
    atoms = [[("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))], [("H", (0, 0, 0))]]
    calculator = Calculator(basis=imported())
    metadata = calculator.basis_metadata(
        atoms[1], charge=dtype(0), multiplicity=dtype(1)
    )
    assert type(metadata["orbital"]["electrons"]["ionic_charge"]) is int
    assert "occupation_error" in metadata["orbital"]["electrons"]
    json.dumps(metadata)
    for charges in ([0, 0], np.array([0, 0], dtype=dtype)):
        with pytest.raises(RuntimeError, match="invalid argument"):
            calculator.batch_singlepoint(atoms, charges=charges)
    with calculator.prepare_batch(
        [atoms[0], atoms[0]],
        charges=np.array([0, 0], dtype=dtype),
        multiplicities=np.array([1, 1], dtype=dtype),
    ) as prepared:
        result = prepared.execute([None, np.zeros((1, 3))])
        assert result.failure_indices == (1,)
