"""Generate independent PM localization references with pinned PySCF 2.14.0."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from vibeqc.profiles import canonical_hash

from tools.generate_validation_references import pyscf_molecule
from tools.vibeqc_posthf.fixtures import load_fixture


def generate(output):
    """Use public PySCF PM APIs on the exact canonical #147 occupied subspaces."""
    import pyscf
    from pyscf import lo
    from threadpoolctl import threadpool_limits

    if pyscf.__version__ != "2.14.0":
        raise RuntimeError("local-space reference generation requires PySCF 2.14.0")
    rows = []
    with threadpool_limits(limits=1):
        for name in ("h2", "water", "lih", "f_heh"):
            metadata, arrays = load_fixture(name)
            mol, scale, _ = pyscf_molecule(metadata["inputs"])
            no = metadata["records"]["conventional"]["electron_count"] // 2
            canonical = arrays["conventional_C"][:, :no] * scale[:, None]
            localizer = lo.PM(mol, canonical, pop_method="mulliken")
            localizer.conv_tol, localizer.conv_tol_grad = 1e-13, 1e-10
            localizer.max_cycle, localizer.verbose = 200, 0
            localized = localizer.kernel()
            # kernel updates mo_coeff, so identity evaluates the final gauge.
            objective = float(localizer.cost_function(np.eye(no)))
            gradient = (
                float(np.max(np.abs(localizer.get_grad(np.eye(no))))) if no > 1 else 0.0
            )
            if gradient > 1e-7:
                raise RuntimeError(
                    f"independent PM reference failed: {name}/{gradient}"
                )
            coefficients = localized / scale[:, None]
            gram = coefficients.T @ arrays["conventional_S"] @ coefficients
            if np.max(np.abs(gram - np.eye(no))) > 1e-10:
                raise RuntimeError(
                    "independent localized orbitals lost metric normalization"
                )
            rows.append(
                {
                    "name": name,
                    "parent_array_hash": metadata["array_hash"],
                    "objective": objective,
                    "gradient_max": gradient,
                    "coefficients": coefficients.tolist(),
                }
            )
    record = {
        "schema": "vibeqc.local_space_reference",
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pyscf": pyscf.__version__,
        "numpy": np.__version__,
        "method": "Pipek-Mezey with Mulliken populations",
        "rows": rows,
    }
    record["record_hash"] = canonical_hash(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    generate(parser.parse_args().output)
