"""Regenerate tiny independent D3 fixtures; simple-dftd3 is oracle-only."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from dftd3.interface import DispersionModel, RationalDampingParam
from dftd3.parameters import get_damping_param


def main():
    if importlib.metadata.version("dftd3") != "1.4.0":
        raise RuntimeError("fixture generation requires dftd3==1.4.0")
    geometries = {
        "water-asymmetric": (
            [8, 1, 1],
            [[0.1, -0.1, 0.2], [1.5, 0.3, 0.4], [-0.5, 1.4, 0.1]],
        ),
        "heteroatomic": (
            [6, 8, 1, 17],
            [[0.1, 0.3, -0.2], [2.4, -0.6, 0.8], [-1.7, 0.8, 0], [3.7, 2.1, 1.3]],
        ),
        "copper-nitrogen": (
            [29, 7, 1],
            [[0.2, 0.1, 0], [3.5, -0.4, 0.8], [4.2, 1.1, 1.2]],
        ),
    }
    profiles = {"gfn1-two-body": {"s6": 1.0, "s8": 2.4, "a1": 0.63, "a2": 5.0}}
    for name in ("pbe", "pbe0"):
        upstream = get_damping_param(name, defaults=["bj"])
        profiles[name + "-bj-two-body"] = {
            k: upstream[k] for k in ("s6", "s8", "a1", "a2")
        }
    fixtures = []
    for profile, parameters in profiles.items():
        for name, (z, x) in geometries.items():
            model = DispersionModel(np.asarray(z), np.asarray(x))
            # Explicit two-body model: s-dftd3's table default s9=1 is not inherited.
            result = model.get_dispersion(
                RationalDampingParam(**parameters, s9=0.0), grad=True
            )
            fixtures.append(
                {
                    "name": profile + "/" + name,
                    "parameters": {**parameters, "s9": 0.0},
                    "numbers": z,
                    "positions": x,
                    "energy": float(result["energy"]),
                    "gradient": result["gradient"].tolist(),
                }
            )
    from dftd3 import _libdftd3

    package = Path(__file__).resolve().parents[2]
    payload = {
        "oracle": "simple-dftd3",
        "version": "1.4.0",
        "atm": False,
        "oracle_library_sha256": hashlib.sha256(
            Path(_libdftd3.__file__).read_bytes()
        ).hexdigest(),
        "units": "bohr; Hartree; gradient=dE/dR in Hartree/bohr",
        "fixtures": fixtures,
    }
    (package / "tests/data/d3_bj_reference.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
