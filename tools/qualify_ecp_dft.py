"""Record independently checked LDA/PBE ECP energy endpoints, without force claims."""

import argparse
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pyscf
from vibeqc.profiles import file_hash


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fixture = root / "tests/python/test_ecp_dft.py"
    sys.path.insert(0, str(fixture.parent))
    spec = importlib.util.spec_from_file_location("ecp_dft_oracle", fixture)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    record = {
        "device": args.device,
        "python": platform.python_version(),
        "pyscf": pyscf.__version__,
        "numpy": np.__version__,
        "library_sha256": file_hash(Path(os.environ["VIBEQC_LIBRARY"])),
        "source_sha256": {
            name: file_hash(root / name)
            for name in (
                "tests/python/test_ecp.py",
                "tests/python/test_ecp_dft.py",
                "python/vibeqc/basis_capabilities.py",
                "tools/qualify_ecp_dft.py",
            )
        },
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "cases": [],
    }
    for representation in ("cartesian", "spherical"):
        for method in oracle.METHODS:
            record["cases"].append(oracle.endpoint(method, args.device, representation))
            args.output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
