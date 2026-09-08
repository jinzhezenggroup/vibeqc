"""Per-shell/operator execution boundaries, separate from basis-data availability."""

import math
from dataclasses import asdict

from .basis import BasisSet
from .elements import checked_integer, electron_state
from .profiles import canonical_hash

# These are the native generic Cartesian/real-spherical public boundaries,
# not the broader symbolic shell catalog or target-specific tuned profiles.
# Values and first nuclear derivatives have independent registered entries.
LIMITS = {
    backend: {
        operator: {0: 3, 1: 3}
        for operator in (
            "overlap",
            "kinetic",
            "nuclear_attraction",
            "eri",
            "df_metric",
            "df_three_center",
        )
    }
    for backend in ("cpu", "cuda")
}
for _operators in LIMITS.values():
    _operators["ao"] = {0: 3, 1: 3, 2: 3, 3: 3}


def basis_capability(
    basis,
    atoms,
    *,
    backend="cpu",
    operator="eri",
    derivative_order=0,
    role="orbital",
    representation=None,
):
    """Inspect every supplied orbital/auxiliary shell without device execution.

    ``eligible`` means an implemented generic mathematical route exists. It
    does not imply a GPU is present, an AOT schedule is promoted, a particular
    SCF state converges, or the chemical model has been validated for an element.
    """
    from .calculator import Atom

    atoms = tuple(Atom.from_value(a) for a in atoms)
    if role not in ("orbital", "auxiliary"):
        raise ValueError("basis role must be orbital or auxiliary")
    if type(derivative_order) is not int or derivative_order < 0:
        raise ValueError("derivative order must be a nonnegative integer")
    backend = "cpu" if backend == "cpu_reference" else backend
    reasons, rows = [], []
    maximum = LIMITS.get(backend, {}).get(operator, {}).get(derivative_order)
    if maximum is None:
        reasons.append(
            f"{backend}/{operator}: derivative order {derivative_order} has no registered execution route"
        )
    if isinstance(basis, BasisSet):
        mode = basis.representation
        if representation is not None and representation != mode:
            reasons.append("requested representation conflicts with the loaded record")
        for atom_index, atom in enumerate(atoms):
            element = basis.by_element.get(atom.atomic_number)
            if element is None:
                reasons.append(
                    f"{role} basis {basis.name!r}: atom {atom_index} Z={atom.atomic_number} has no data"
                )
                continue
            if element.ecp_core_electrons or element.ecp_data:
                reasons.append(
                    f"{role} atom {atom_index} Z={atom.atomic_number}: ECP with {element.ecp_core_electrons} core electrons has no implemented {backend} Hamiltonian"
                )
            if element.nuclear_charge != atom.atomic_number:
                reasons.append(
                    f"{role} atom {atom_index} Z={atom.atomic_number}: nuclear charge {element.nuclear_charge} requires an unsupported alchemical Hamiltonian"
                )
            for shell_index, shell in enumerate(element.shells):
                for column in range(len(shell.coefficients)):
                    rows.append(
                        (atom_index, shell_index, column, shell.angular_momentum)
                    )
    else:
        mode = representation or "cartesian"
        for index, shell in enumerate(basis):
            checked_integer(shell.atom_index, "shell atom index", high=len(atoms) - 1)
            checked_integer(
                shell.angular_momentum, "shell angular momentum", high=2**32 - 1
            )
            if not shell.primitives:
                raise ValueError("shell requires primitives")
            for primitive in shell.primitives:
                if (
                    not math.isfinite(primitive.exponent)
                    or primitive.exponent <= 0
                    or not math.isfinite(primitive.coefficient)
                ):
                    raise ValueError(
                        "primitive exponents must be positive finite and coefficients finite"
                    )
            rows.append((shell.atom_index, index, 0, shell.angular_momentum))
    if mode not in ("cartesian", "spherical"):
        reasons.append(f"unsupported AO representation {mode!r}")
    for atom_index, shell_index, column, l in rows:
        if maximum is not None and l > maximum:
            reasons.append(
                f"{role} atom {atom_index} Z={atoms[atom_index].atomic_number} shell {shell_index} contraction {column} l={l}: {backend}/{operator} derivative {derivative_order} supports only l<={maximum}"
            )
    if not rows:
        reasons.append(f"{role} basis has no shells")
    return {
        "eligible": not reasons,
        "data_loadable": bool(rows),
        "canonical_record": isinstance(basis, BasisSet),
        "backend": backend,
        "operator": operator,
        "derivative_order": derivative_order,
        "representation": mode,
        "role": role,
        "shells_checked": len(rows),
        "reasons": reasons,
        "validated_element_model": False,
        "aot_promoted": False,
    }


def require_basis(basis, atoms, **request):
    """Fail before native expansion/execution with the exact missing shell route."""
    report = basis_capability(basis, atoms, **request)
    if not report["eligible"]:
        raise NotImplementedError("; ".join(report["reasons"]))
    return report


def resolved_basis_metadata(
    basis, shells, atoms, *, representation, charge, multiplicity, role="orbital"
):
    """Hash complete native mathematical inputs separately from source provenance.

    Coordinates enter the prepared model identity separately. Decimal spelling
    and source names must not distinguish identical expanded FP64 input shells.
    Every coefficient, including zero entries, enters this mathematical hash.
    """
    # Accepted NumPy integral scalars must become JSON-native integers even
    # when invalid occupations take the diagnostic path below.
    charge = checked_integer(charge, "ionic charge", low=-(2**31), high=2**31 - 1)
    multiplicity = checked_integer(multiplicity, "multiplicity", low=1, high=2**31 - 1)
    metadata = basis.by_element if isinstance(basis, BasisSet) else {}
    numbers = tuple(a.atomic_number for a in atoms)
    nuclei = tuple(metadata[z].nuclear_charge if z in metadata else z for z in numbers)
    cores = tuple(
        metadata[z].ecp_core_electrons if z in metadata else 0 for z in numbers
    )
    try:
        state = asdict(
            electron_state(
                atoms,
                charge=charge,
                multiplicity=multiplicity,
                element_metadata=metadata,
            )
        )
    except ValueError as error:
        # Occupation inspection is separate from native method preparation.
        # Preserve its existing rejection behavior instead of replacing it
        # with an unrelated metadata/JSON exception.
        state = {
            "atomic_numbers": numbers,
            "nuclear_charges": nuclei,
            "ecp_core_electrons": cores,
            "ionic_charge": charge,
            "electron_count": sum(nuclei) - sum(cores) - charge,
            "multiplicity": multiplicity,
            "occupation_error": str(error),
        }
    payload = {
        "representation": representation,
        "elements": {
            "atomic_numbers": numbers,
            "nuclear_charges": nuclei,
            "ecp_core_electrons": cores,
        },
        "shells": [
            (
                s.atom_index,
                s.angular_momentum,
                [
                    (float(p.exponent).hex(), float(p.coefficient).hex())
                    for p in s.primitives
                ],
            )
            for s in shells
        ],
        "normalization": "normalized-primitives; native-normalized-contractions",
    }
    return {
        "name": basis.name if isinstance(basis, BasisSet) else "explicit-shells",
        "role": role,
        "mathematical_identity": canonical_hash(payload),
        "basis_identity": basis.identity
        if isinstance(basis, BasisSet)
        else canonical_hash(payload),
        "representation": representation,
        "provenance": asdict(basis.provenance)
        if isinstance(basis, BasisSet)
        else {
            "source": "explicit native Shell input",
            "version": "1",
            "license": "caller-supplied",
            "checksum": canonical_hash(payload),
        },
        "electrons": state,
        "shell_count": len(shells),
        "primitive_count": sum(len(s.primitives) for s in shells),
    }
