"""Generate test-only libcint/PySCF references; no dependency is auto-installed."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import import_module, metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from tools.vibeqc_validation.fixtures import (
    REFERENCE_VERSION,
    load_fixtures,
    mathematical_hash,
    molecular_inputs,
    small_inputs,
)
from tools.vibeqc_validation.schema import block_error, canonical_hash, file_hash


def pyscf_molecule(inputs):
    """Preserve per-atom shells and convert libcint Cartesian normalization."""
    from pyscf import gto

    labels = [
        f"{gto.mole._symbol(z)}{i}" for i, z in enumerate(inputs["atomic_numbers"])
    ]
    basis = {label: [] for label in labels}
    for shell in inputs["shells"]:
        basis[labels[shell["atom_index"]]].append(
            [shell["angular_momentum"], *shell["primitives"]]
        )
    mol = gto.M(
        atom=list(zip(labels, inputs["coordinates"], strict=True)),
        # Atoms owning only the other DF basis have no shells in this molecule.
        # PySCF accepts missing basis keys, but rejects an explicit empty list.
        basis={label: shells for label, shells in basis.items() if shells},
        unit="Bohr",
        cart=inputs["basis_representation"] == "cartesian",
        charge=inputs["charge"],
        spin=inputs["multiplicity"] - 1,
        verbose=0,
    )
    actual = [int(mol.bas_angular(i)) for i in range(mol.nbas)]
    if actual != [s["angular_momentum"] for s in inputs["shells"]]:
        raise ValueError("PySCF reordered the requested shells")
    scale = 1 / np.sqrt(np.diag(mol.intor("int1e_ovlp")))
    return mol, scale, actual


def quartet_data(mol, scale):
    """Store (ij|kl) and four independent shell-center nuclear derivatives.

    libcint ip1 differentiates the electronic coordinate on the first AO:
    negate it for a moving Gaussian center. Permutations put each center in
    that slot; transposes restore the original i,j,k,l tensor order.
    """
    offsets = mol.ao_loc_nr()
    norms = [scale[offsets[i] : offsets[i + 1]] for i in range(4)]
    normalization = np.einsum("i,j,k,l->ijkl", *norms)
    value = mol.intor_by_shell("int2e", (0, 1, 2, 3)) * normalization
    derivatives = []
    for permutation in ((0, 1, 2, 3), (1, 0, 2, 3), (2, 3, 0, 1), (3, 2, 0, 1)):
        raw = -mol.intor_by_shell("int2e_ip1", permutation)
        axes = (0, *(1 + permutation.index(i) for i in range(4)))
        derivatives.append(raw.transpose(axes) * normalization)
    return {
        "eri": value.tolist(),
        "gradient": np.asarray(derivatives).tolist(),
        "ao_labels": mol.ao_labels(),
        "libcint_to_unit_cartesian": scale.tolist(),
    }


def molecular_data(inputs, mol, scale):
    """Save canonical occupied/virtual orbitals and converged independent HF/CC."""
    from pyscf import cc, scf

    mf = (scf.RHF if inputs["method"] == "rhf" else scf.UHF)(mol)
    policy = inputs["scf_settings"]
    mf.conv_tol = policy["energy_tolerance"]
    mf.conv_tol_grad = policy["gradient_tolerance"]
    mf.max_cycle = policy["max_iterations"]
    mf.direct_scf_tol = policy["direct_scf_tol"]
    history = []
    mf.callback = lambda env: history.append(
        {
            "iteration": int(env["cycle"] + 1),
            "energy": float(env["e_tot"]),
            "orbital_gradient_norm": float(env["norm_gorb"]),
            "density_change_norm": float(env["norm_ddm"]),
        }
    )
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"reference SCF failed: {inputs['name']}")
    coefficients = np.asarray(mf.mo_coeff).copy()
    blocks = coefficients[None] if coefficients.ndim == 2 else coefficients
    for block in blocks:
        for column in range(block.shape[1]):
            pivot = np.argmax(abs(block[:, column] / scale))
            if block[pivot, column] < 0:
                block[:, column] *= -1
    mf.mo_coeff = coefficients
    gradient = mf.nuc_grad_method().kernel()
    dm = np.asarray(mf.make_rdm1())
    # The reference uses normalized Cartesian AOs, including d/f components.
    unit_mo = coefficients / scale[:, None]
    unit_dm = dm / (scale[:, None] * scale[None, :])
    overlap = mf.get_ovlp()
    fock = np.asarray(mf.get_fock(dm=dm))
    residual = fock @ dm @ overlap - overlap @ dm @ fock
    data = {
        "energy": float(mf.e_tot),
        "gradient": gradient.tolist(),
        "forces": (-gradient).tolist(),
        "mo_coeff": unit_mo.tolist(),
        "mo_energy": np.asarray(mf.mo_energy).tolist(),
        "mo_occ": np.asarray(mf.mo_occ).tolist(),
        "density": unit_dm.tolist(),
        "overlap": (overlap * scale[:, None] * scale[None, :]).tolist(),
        "hcore": (mf.get_hcore() * scale[:, None] * scale[None, :]).tolist(),
        "nuclear_repulsion": float(mol.energy_nuc()),
        "ao_labels": mol.ao_labels(),
        "scf_iterations": history,
        "scf_residual_max": float(abs(residual).max()),
    }
    if inputs["cc_settings"]:
        policy = inputs["cc_settings"]
        coupled = cc.CCSD(mf, frozen=policy["frozen_core"])
        coupled.conv_tol, coupled.conv_tol_normt = (
            policy["conv_tol"],
            policy["conv_tol_normt"],
        )
        coupled.max_cycle = policy["max_cycle"]
        coupled.kernel()
        if not coupled.converged:
            raise RuntimeError("reference CCSD failed")
        triples = float(coupled.ccsd_t())
        if abs(triples) < 1e-7:
            raise RuntimeError(
                "triples fixture must have a nontrivial (T) contribution"
            )
        updated = coupled.update_amps(coupled.t1, coupled.t2, coupled.ao2mo())
        data["ccsd_t"] = {
            "correlation_energy": float(coupled.e_corr),
            "triples_energy": triples,
            "t1": coupled.t1.tolist(),
            "t2": coupled.t2.tolist(),
            "amplitude_update_max": float(
                max(
                    abs(updated[0] - coupled.t1).max(),
                    abs(updated[1] - coupled.t2).max(),
                )
            ),
            "residual_note": "PySCF amplitude-update diagnostic; not an independently evaluated CC equation residual",
        }
    return data


def _generate(destination: Path, compare: Path | None = None) -> dict:
    """Generate a traceable run; comparison reports all numerical data changes."""
    import pyscf
    from pyscf import lib
    from threadpoolctl import threadpool_info

    lib.num_threads(1)
    config = io.StringIO()
    with contextlib.redirect_stdout(config):
        np.show_config()
    libcint = Path(pyscf.__file__).parent / "lib/deps/lib/libcint.so"
    if not libcint.exists():
        raise RuntimeError(
            "cannot identify loaded libcint; record its library path before generating"
        )
    provenance = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "pyscf": pyscf.__version__,
        "numpy": np.__version__,
        "scipy": metadata.version("scipy"),
        "h5py": metadata.version("h5py"),
        "threadpoolctl": metadata.version("threadpoolctl"),
        "loaded_threadpools": threadpool_info(),
        "blas": config.getvalue(),
        "threads": 1,
        "generator_sha256": file_hash(__file__),
        "fixture_definition_sha256": file_hash(
            ROOT / "tools/vibeqc_validation/fixtures.py"
        ),
        "libcint_sha256": file_hash(libcint),
        "basis_pack_sha256": file_hash(ROOT / "python/vibeqc/data/basis_pack.json"),
        "purpose": "independent test/reference generation only",
    }
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for inputs in small_inputs() + molecular_inputs():
        mol, scale, angular = pyscf_molecule(inputs)
        data = (
            quartet_data(mol, scale)
            if inputs["kind"] == "quartet"
            else molecular_data(inputs, mol, scale)
        )
        row = {
            "schema": "vibeqc.reference",
            "schema_version": REFERENCE_VERSION,
            "inputs": inputs,
            "inputs_hash": mathematical_hash(inputs),
            "data": data,
            "data_hash": canonical_hash(data),
            "angular_momenta_loaded": angular,
            "provenance": provenance,
            "provenance_hash": canonical_hash(provenance),
        }
        path = destination / (inputs["name"] + ".json")
        path.write_text(
            json.dumps(row, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        rows.append(
            {
                "file": path.name,
                "inputs_hash": row["inputs_hash"],
                "data_hash": row["data_hash"],
                "record_hash": canonical_hash(row),
            }
        )
    manifest = {
        "schema_version": REFERENCE_VERSION,
        "provenance": provenance,
        "fixtures": rows,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    if compare:
        old = {r["inputs"]["name"]: r for r in load_fixtures(compare)}
        stability = []
        for new in load_fixtures(destination):
            prior = old[new["inputs"]["name"]]
            if prior["inputs_hash"] != new["inputs_hash"]:
                raise ValueError(
                    "reference mathematical inputs changed between generations"
                )
            errors = {}

            def compare_data(a, b, key, errors=errors):
                if isinstance(a, dict):
                    if a.keys() != b.keys():
                        raise ValueError("reference data structure changed")
                    for k in a:
                        compare_data(a[k], b[k], key + "/" + k)
                elif isinstance(a, (float, int)) or (
                    isinstance(a, list) and a and not isinstance(a[0], (str, dict))
                ):
                    errors[key] = block_error(a, b, atol=1e-12, rtol=1e-12)
                elif a != b:
                    raise ValueError(f"reference metadata/history changed: {key}")

            compare_data(new["data"], prior["data"], "data")
            stability.append(
                {
                    "name": new["inputs"]["name"],
                    "inputs_hash": new["inputs_hash"],
                    "first_record_hash": canonical_hash(prior),
                    "second_record_hash": canonical_hash(new),
                    "first_provenance": prior["provenance"],
                    "second_provenance": new["provenance"],
                    "first_data_hash": prior["data_hash"],
                    "second_data_hash": new["data_hash"],
                    "errors": errors,
                }
            )
        (destination / "stability.json").write_text(
            json.dumps(
                {"schema_version": 1, "generations": stability},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        if not all(e["passed"] for r in stability for e in r["errors"].values()):
            raise ValueError("reference stability gate failed")
    return manifest


def generate(destination: Path, compare: Path | None = None) -> dict:
    """Limit every loaded BLAS/OpenMP runtime during reference generation."""
    from threadpoolctl import threadpool_limits

    # NumPy and PySCF can load distinct BLAS implementations. Load CC libraries
    # before inspecting/limiting pools, then restore the caller's thread policy.
    import_module("pyscf.cc")
    with threadpool_limits(limits=1):
        return _generate(destination, compare)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="directory for pinned fixtures and manifest",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        help="previous generation; require identical inputs and write stability.json",
    )
    args = parser.parse_args()
    generate(args.output, args.compare)


if __name__ == "__main__":
    main()
