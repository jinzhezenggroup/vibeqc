"""Audit every archived energy repeat and confirm that no force was measured."""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np


def main():
    """Evaluate numerical gates independently of the selected SCF branch."""
    with ZipFile(Path(__file__).with_name("raw-evidence.zip")) as archive:
        names = sorted(
            n
            for n in archive.namelist()
            if n.startswith("df-energy-only-matrix/")
            and "ao-b" in n
            and n.endswith(".json")
        )
        if len(names) != 4:
            raise ValueError("expected four matrix cases")
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
                raise ValueError("expected five paired repeats")
            maximum = 0.0
            for first, second in pairs:
                for sample in (first, second):
                    if sample["forces_hartree_per_bohr"] is not None:
                        raise ValueError("energy-only sample contains forces")
                    if not all(c["converged"] for c in sample["convergence"]):
                        raise ValueError("unconverged SCF")
                a = np.asarray(first["energies_hartree"])
                b = np.asarray(second["energies_hartree"])
                if (
                    a.shape != b.shape
                    or not a.size
                    or not np.isfinite(a).all()
                    or not np.isfinite(b).all()
                ):
                    raise ValueError("invalid energy arrays")
                maximum = max(maximum, float(np.max(np.abs(a - b))))
            if maximum > 1e-9:
                raise ValueError(f"energy gate failed: {maximum}")
            print(name, maximum)


if __name__ == "__main__":
    main()
