"""Independently audit every stored warm pair; this performs no GPU work."""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np


def main():
    """Numerical thresholds apply to all repeats, regardless of SCF branch."""
    directory = Path(__file__).resolve().parent
    with ZipFile(directory / "raw-evidence.zip") as archive:
        names = sorted(
            name
            for name in archive.namelist()
            if name.startswith("df-final-integrated-force-matrix/")
            and "ao-b" in name
            and name.endswith(".json")
        )
        if len(names) != 4:
            raise ValueError("expected the complete four-case force matrix")
        for name in names:
            data = json.loads(archive.read(name))
            pairs = list(
                zip(
                    data["vibeqc"]["warm_samples"],
                    data["gpu4pyscf"]["warm_samples"],
                    strict=True,
                )
            )
            if len(pairs) != 5:
                raise ValueError(f"{name}: expected five repeats per engine")
            errors = {"energies_hartree": 0.0, "forces_hartree_per_bohr": 0.0}
            for first, second in pairs:
                if not all(
                    item["converged"]
                    for sample in (first, second)
                    for item in sample["convergence"]
                ):
                    raise ValueError(f"{name}: unconverged SCF")
                for key, previous in errors.items():
                    a, b = np.asarray(first[key]), np.asarray(second[key])
                    if (
                        a.shape != b.shape
                        or not np.isfinite(a).all()
                        or not np.isfinite(b).all()
                    ):
                        raise ValueError(f"{name}: invalid {key} arrays")
                    errors[key] = max(previous, float(np.max(np.abs(a - b))))
            if (
                errors["energies_hartree"] > 1e-9
                or errors["forces_hartree_per_bohr"] > 1e-8
            ):
                raise ValueError(f"{name}: numerical parity failed: {errors}")
            print(name, errors)


if __name__ == "__main__":
    main()
