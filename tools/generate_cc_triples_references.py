"""Generate the committed (T) reference JSON from the pinned PySCF 2.14.0.

This generator runs ONLY in an environment with PySCF 2.14.0 (the qz Python
3.11 setup described in ``docs/qz-development.md``).  It reads committed
``tests/reference_data/cc/endpoints/*.npz`` (all-MO ``S,h,F,C,eps,occ,g,ao,t1,
t2``), reconstructs the ``(T)`` inputs, computes the energy two independent
ways, and writes ``tests/reference_data/cc/rccsd-t.json``:

* ``via_triples_energy`` -- this repository's pure-NumPy audited reference
  ``tools.vibeqc_cc.triples_energy`` (no PySCF).
* ``via_pyscf_ccsd_t`` -- pinned PySCF ``cc.CCSD(...).ccsd_t()``, which
  internally calls the production ``ccsd_t.kernel`` (source/hash checked).

The two must agree to |dE| <= 1e-9 and produce the issue's ground-truth table.
Re-run with ``--compare`` to assert two-generation byte-for-byte stability.

Correct invocation (from the repository root, on qz):

    cd /inspire/.../vibeqc
    source .venv/bin/activate
    export PYTHONPATH=.:python
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python -m tools.generate_cc_triples_references \
        --output tests/reference_data/cc/rccsd-t.json
    python -m tools.generate_cc_triples_references \
        --output /tmp/rccsd-t-second.json \
        --compare tests/reference_data/cc/rccsd-t.json
"""

import argparse
import importlib
import json
import platform
from pathlib import Path

import numpy as np

from tools.cc_endpoint_fixtures import array_hash
from tools.vibeqc_cc.triples import triples_energy
from tools.vibeqc_validation.schema import canonical_hash, file_hash

ROOT = Path(__file__).resolve().parents[1]
ENDPOINTS = ROOT / "tests/reference_data/cc/endpoints"

# Authoritative E_T from pinned PySCF 2.14.0 ccsd_t_slow.kernel (issue #150).
GROUND_TRUTH = {
    "h2": (1, 1, 8.392021714075268e-49),
    "he": (1, 1, 0.0),
    "h2o": (5, 2, -6.731393342463869e-05),
    "nh3": (5, 3, -1.122922812723691e-04),
    "ch4": (5, 4, -1.555665872715297e-04),
}


def _triples_feeds(name):
    with np.load(ENDPOINTS / f"{name}.npz", allow_pickle=False) as data:
        eps = data["eps"]
        occ = data["occ"]
        C = data["C"]
        F = data["F"]
        g = data["g"]
        t1 = data["t1"]
        t2 = data["t2"]
    nocc = int(np.sum(occ > 0))
    nvir = len(eps) - nocc
    fov = (C.T @ F @ C)[:nocc, nocc:]
    return (
        nocc,
        nvir,
        g[:nocc, nocc:, nocc:, nocc:],
        g[:nocc, nocc:, :nocc, :nocc],
        g[:nocc, nocc:, :nocc, nocc:],
        fov,
        t1,
        t2,
        eps[:nocc],
        eps[nocc:],
    )


def _mo_fock(name):
    with np.load(ENDPOINTS / f"{name}.npz", allow_pickle=False) as data:
        return data["C"].T @ data["F"] @ data["C"]


def _endpoint_inputs(name):
    """Geometry/basis inputs recorded by generate_cc_endpoints.py."""
    return json.loads((ENDPOINTS / f"{name}.json").read_text())["inputs"]


def _solve_rhf(mol):
    """Converge RHF exactly as generate_cc_endpoints.py does (the endpoint MO
    phase normalization is irrelevant to the (T) energy)."""
    from pyscf import scf

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-11
    mf.max_cycle = 150
    mf.kernel()
    if not mf.converged:
        raise ValueError("(T) reference RHF did not converge")
    return mf


def _inputs_hash(name):
    _nocc, _nvir, ovvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v = _triples_feeds(name)
    return array_hash(
        {
            "ovvv": ovvv,
            "ovoo": ovoo,
            "ovov": ovov,
            "fov": fov,
            "t1": t1,
            "t2": t2,
            "eps_o": eps_o,
            "eps_v": eps_v,
        }
    )


def _check_energies(name, et_numpy, et_pyscf, truth):
    """Enforce both the independent agreement gate and the stored target.

    Two results on opposite sides of the target can each satisfy its
    tolerance while differing from each other by almost twice that amount.
    """
    if (
        not np.all(np.isfinite([et_numpy, et_pyscf, truth]))
        or abs(et_numpy - et_pyscf) > 1e-9
        or abs(et_numpy - truth) > 1e-9
        or abs(et_pyscf - truth) > 1e-9
    ):
        raise ValueError(
            f"{name} (T) diverged: numpy={et_numpy:.15e} "
            f"pyscf={et_pyscf:.15e} truth={truth:.15e}"
        )


def generate(output, compare=None):
    import pyscf
    from threadpoolctl import threadpool_limits

    if pyscf.__version__ != "2.14.0":
        raise ValueError("(T) references require pinned PySCF 2.14.0")

    manifest = json.loads((ROOT / "tools/vibeqc_cc/source_manifest.json").read_text())
    upstream = {record["path"]: record for record in manifest["files"]}
    for path in ("pyscf/cc/ccsd_t_slow.py", "pyscf/cc/ccsd_t.py"):
        # import_module returns the leaf module, whose __file__ is the pinned
        # source; __import__("a.b.c") would return the top-level package "a".
        module = importlib.import_module(
            "pyscf.cc.ccsd_t_slow"
            if path.endswith("ccsd_t_slow.py")
            else "pyscf.cc.ccsd_t"
        )
        if file_hash(module.__file__) != upstream[path]["sha256"]:
            raise ValueError(f"upstream source hash mismatch: {path}")

    molecules = []
    with threadpool_limits(limits=1):
        for name, (nocc_expected, nvir_expected, truth) in GROUND_TRUTH.items():
            feeds = _triples_feeds(name)
            nocc, nvir = feeds[0], feeds[1]
            if (nocc, nvir) != (nocc_expected, nvir_expected):
                raise ValueError(f"{name} (o,v) changed: {nocc},{nvir}")

            # Independent derivation #1: this repo's audited NumPy reference
            # over the committed endpoint amplitudes/integrals (no PySCF).
            et_numpy = triples_energy(*feeds)

            # Independent derivation #2: pinned PySCF production (T).  This
            # re-converges RHF/CCSD from the endpoint's recorded atoms/shells
            # then calls production ``ccsd_t.kernel`` (``_sort_eri``,
            # ``_sort_t2_vooo_``, C ``CCsd_t_contract``): it shares no code
            # with our NumPy transcription.
            from pyscf import cc
            from pyscf.cc import ccsd_t

            from tools.generate_validation_references import pyscf_molecule

            mol, _, _ = pyscf_molecule(_endpoint_inputs(name))
            mf = _solve_rhf(mol)
            coupled = cc.CCSD(mf)
            coupled.conv_tol = 1e-13
            coupled.conv_tol_normt = 1e-11
            coupled.max_cycle = 150
            coupled.kernel()
            if not coupled.converged:
                raise ValueError(f"{name} reference CCSD did not converge")
            eris = coupled.ao2mo()
            et_pyscf = ccsd_t.kernel(
                coupled, eris, coupled.t1, np.ascontiguousarray(coupled.t2)
            )

            _check_energies(name, et_numpy, et_pyscf, truth)

            molecules.append(
                {
                    "name": name,
                    "nocc": nocc,
                    "nvir": nvir,
                    "et_numpy": float(et_numpy),
                    "et_pyscf_ccsd_t": float(et_pyscf),
                    "et_ground_truth": truth,
                    "et_agreement": abs(et_numpy - et_pyscf),
                    "inputs_hash": _inputs_hash(name),
                }
            )

    result = {
        "schema": "vibeqc.rccsd-t.reference",
        "version": 1,
        "pyscf": pyscf.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "upstream": manifest,
        "generator_sha256": file_hash(__file__),
        "molecules_hash": canonical_hash(molecules),
        "molecules": molecules,
    }
    if compare:
        previous = json.loads(compare.read_text())
        if result["molecules_hash"] != previous["molecules_hash"]:
            raise ValueError("two-generation (T) reference hash differs")
        result["stability"] = {
            "identical_molecules": True,
            "previous_file_sha256": file_hash(compare),
        }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(f"wrote {len(molecules)} (T) references: {result['molecules_hash']}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    generate(args.output, args.compare)


if __name__ == "__main__":
    main()
