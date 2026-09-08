"""Pin identical-grid PySCF/Libxc XC integrals; never call our XC consumer.

PySCF (Apache-2.0), libcint (BSD-2-Clause), Libxc (MPL-2.0) are reference-only
dependencies. This exporter reuses DFT01's explicit geometry/basis/grid inputs,
not its AO, feature or potential evaluation. No SCF solution is needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

import numpy as np
from vibeqc.profiles import canonical_hash, file_hash

from tools.generate_validation_references import pyscf_molecule
from tools.vibeqc_dft.fixtures import load_fixture

CASES = ("h2", "water", "f_cartesian", "f_spherical")
FUNCTIONALS = {"LDA_XC_PW": "LDA_X,LDA_C_PW", "PBE": "GGA_X_PBE,GGA_C_PBE"}


def generate(directory):
    import pyscf
    from pyscf.dft import gen_grid, libxc, numint

    if pyscf.__version__ != "2.14.0" or libxc.__version__ != "7.0.0":
        raise RuntimeError("reference export requires PySCF 2.14.0 / Libxc 7.0.0")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in CASES:
        source, old = load_fixture(name)
        mol, scale, actual = pyscf_molecule(source["inputs"])
        grids = gen_grid.Grids(mol)
        # Exact inputs from DFT01, with no value-based point selection,
        # generated-grid substitution, pruning or weight renormalization.
        grids.coords = old["partitioned_grid_points"].copy()
        grids.weights = old["partitioned_grid_weights"].copy()
        grids.non0tab = None
        grids.cutoff = 1e-30
        spin_density = old["density"] + 0.15 * np.eye(mol.nao_nr())[None]
        total_density = spin_density.sum(axis=0)
        arrays = {
            "points": grids.coords,
            "weights": grids.weights,
            "density_total": total_density,
            "density_spin": spin_density,
            "ao_scale": scale,
        }
        ni = numint.NumInt()
        ni.cutoff = 1e-30
        for label, code in FUNCTIONALS.items():
            for layout, density in (("total", total_density), ("spin", spin_density)):
                dm = density * scale[:, None] * scale[None, :]
                run = ni.nr_rks if layout == "total" else ni.nr_uks
                electrons, energy, matrix = run(mol, grids, code, dm, hermi=1)
                prefix = f"{label}_{layout}"
                arrays[f"{prefix}_energy"] = np.asarray([energy])
                arrays[f"{prefix}_potential"] = matrix * scale[:, None] * scale[None, :]
                arrays[f"{prefix}_electrons"] = np.atleast_1d(electrons)
        arrays = {
            k: np.asarray(v, dtype=np.float64, order="C") for k, v in arrays.items()
        }
        metadata = {
            "schema": "vibeqc.xc-integration-reference",
            "version": 1,
            "inputs": source["inputs"],
            "inputs_hash": source["inputs_hash"],
            "source_grid_identity": source["grid_hash"],
            "reference": {
                "PySCF": pyscf.__version__,
                "Libxc": libxc.__version__,
                "NumPy": np.__version__,
                "python": sys.version,
                "actual_angular_momenta": actual,
                "pyscf_numint_sha256": file_hash(Path(numint.__file__)),
                "exporter_sha256": file_hash(Path(__file__)),
                "basis_adapter_sha256": file_hash(
                    ROOT / "tools/generate_validation_references.py"
                ),
                "functionals": FUNCTIONALS,
                "ao_cutoff": ni.cutoff,
                "threads": pyscf.lib.num_threads(),
                "oracle": "PySCF NumInt.nr_rks/nr_uks (independent AO, density, XC and matrix assembly)",
            },
            "conventions": {
                "coordinates": "Bohr",
                "energy": "Hartree",
                "weights": "Bohr^3, already partitioned",
                "density": "arbitrary PSD with 0.15 identity per spin; no SCF or renormalization",
                "AO": "unit-normalized contracted Cartesian or real spherical; exact DFT01 shells",
                "reference_density": "D_pyscf = scale[:,None] * D * scale[None,:]",
                "reference_potential": "V = scale[:,None] * V_pyscf * scale[None,:]",
                "energy_density": "Libxc epsilon times total rho, integrated by PySCF",
            },
            "arrays": {
                k: {"shape": list(v.shape), "sha256": sha256(v.tobytes()).hexdigest()}
                for k, v in arrays.items()
            },
        }
        metadata["identity"] = canonical_hash(metadata)
        np.savez_compressed(directory / f"{name}.npz", **arrays)
        (directory / f"{name}.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(
            f"{name}: {mol.nao_nr()} AO, {len(grids.weights)} explicit points",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    generate(parser.parse_args().output)
