"""Generate fixed-amplitude PySCF 2.14.0 references, never production CC solves.

Run with PYTHONPATH=.:python and one BLAS/OpenMP thread. Re-run with --compare
to check exact input/output stability on the recorded dependency stack.
"""

import argparse
import json
import platform
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tools.vibeqc_cc.oracle import random_case
from tools.vibeqc_posthf.fixtures import fixture_snapshot, load_fixture
from tools.vibeqc_validation.schema import canonical_hash, file_hash

ROOT = Path(__file__).resolve().parents[1]


def generate(*, full=False):
    import pyscf
    import scipy
    from pyscf.cc import rccsd, rintermediates
    from threadpoolctl import threadpool_info, threadpool_limits

    manifest = json.loads((ROOT / "tools/vibeqc_cc/source_manifest.json").read_text())
    if pyscf.__version__ != manifest["version"]:
        raise ValueError("reference generation requires pinned PySCF 2.14.0")
    for module, record in zip((rccsd, rintermediates), manifest["files"]):
        if file_hash(module.__file__) != record["sha256"]:
            raise ValueError(f"upstream source hash mismatch: {record['path']}")
    cases = []
    with threadpool_limits(limits=1):
        for label in ("random", "h2", "water", "lih"):
            if label == "random":
                f, g, t1, t2 = random_case()
                source = {"kind": "random", "seed": 148}
            else:
                meta, arrays = load_fixture(label)
                snapshot = fixture_snapshot(meta, arrays)
                C = snapshot.coefficients
                f = C.T @ snapshot.fock @ C
                g = arrays["conventional_mo"]
                _, _, t1, t2 = random_case(snapshot.nocc, snapshot.nmo - snapshot.nocc)
                source = {
                    "kind": "posthf-147",
                    "name": label,
                    "array_hash": meta["array_hash"],
                }
            o, v = t1.shape
            eris = rccsd._ChemistsERIs()
            eris.fock = f
            # Deliberately different from diag(F): denominators are solver
            # choices and must cancel when reconstructing the physical R1.
            eris.mo_energy = (
                np.linspace(-2, -0.7, o).tolist() + np.linspace(0.4, 1.5, v).tolist()
            )
            eris.mo_energy = np.array(eris.mo_energy)
            for name in ("oooo", "ovoo", "ovov", "oovv", "ovvo", "ovvv", "vvvv"):
                eris.__dict__[name] = g[
                    tuple(slice(0, o) if x == "o" else slice(o, None) for x in name)
                ]
            shifts = []
            for shift in (0.0, 0.4):
                cc = SimpleNamespace(level_shift=shift, cc2=False)
                update, update2 = rccsd.update_amps(cc, t1, t2, eris)
                denominator = (
                    eris.mo_energy[:o, None] - eris.mo_energy[None, o:] - shift
                )
                shifts.append(
                    {
                        "level_shift": shift,
                        "denominator": denominator.tolist(),
                        "updated_t1": update.tolist(),
                        "residual": (denominator * (update - t1)).tolist(),
                    }
                )
                if full:
                    d2 = denominator[:, None, :, None] + denominator[None, :, None, :]
                    shifts[-1].update(
                        denominator2=d2.tolist(),
                        updated_t2=update2.tolist(),
                        residual2=(d2 * (update2 - t2)).tolist(),
                    )
            inputs = {
                "fock": f.tolist(),
                "eri": g.tolist(),
                "t1": t1.tolist(),
                "t2": t2.tolist(),
            }
            cases.append(
                {
                    "name": label,
                    "source": source,
                    "inputs": inputs,
                    "inputs_hash": canonical_hash(inputs),
                    "correlation_energy": float(rccsd.energy(None, t1, t2, eris)),
                    "updates": shifts,
                }
            )
            if full:
                functions = {
                    "Foo": "cc_Foo",
                    "Fvv": "cc_Fvv",
                    "Loo": "Loo",
                    "Lvv": "Lvv",
                    "Woooo": "cc_Woooo",
                    "Wvvvv": "cc_Wvvvv",
                    "Wvoov": "cc_Wvoov",
                    "Wvovo": "cc_Wvovo",
                }
                cases[-1]["intermediates"] = {
                    name: getattr(rintermediates, fn)(t1, t2, eris).tolist()
                    for name, fn in functions.items()
                }
        threads = threadpool_info()
    return {
        "schema": "vibeqc.rccsd.fixed-amplitude-reference",
        "version": 1,
        "pyscf": pyscf.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "python": platform.python_version(),
        "threads": threads,
        "upstream": manifest,
        "generator_sha256": file_hash(__file__),
        "cases_hash": canonical_hash(cases),
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument(
        "--full",
        action="store_true",
        help="also record physical doubles and shared intermediates",
    )
    args = parser.parse_args()
    result = generate(full=args.full)
    if args.compare:
        previous = json.loads(args.compare.read_text())
        if result["cases_hash"] != previous["cases_hash"]:
            raise ValueError("two-generation input/output hashes differ")
        result["stability"] = {
            "identical_cases": True,
            "previous_file_sha256": sha256(args.compare.read_bytes()).hexdigest(),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(f"4 fixed-amplitude references: {result['cases_hash']}")


if __name__ == "__main__":
    main()
