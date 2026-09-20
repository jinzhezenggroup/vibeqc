"""Complete CPU/CUDA RI-MP2 endpoint and device-work ledger for issue #367."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
import time
from pathlib import Path

from vibeqc import Calculator

ATOMS = [
    ("O", (0.0, 0.0, 0.0)),
    ("H", (0.0, -1.432, 1.107)),
    ("H", (0.0, 1.432, 1.107)),
]
DEFAULT_BASES = ("sto-3g", "def2-svp", "def2-tzvp")


def sample(calc: Calculator) -> tuple[float, object]:
    started = time.perf_counter()
    result = calc.singlepoint(ATOMS, properties=("energy",))
    return time.perf_counter() - started, result


def endpoint(basis: str, device: str, repeats: int) -> dict:
    mode = "cuda" if device == "cuda" else "cpu"
    calc = Calculator(method="mp2", basis=basis, device=device, density_fitting=mode)
    cold, result = sample(calc)
    warm = []
    for _ in range(repeats):
        elapsed, result = sample(calc)
        warm.append(elapsed)
    diag = result.correlation
    return {
        "cold_s": cold,
        "warm_s": warm,
        "warm_median_s": statistics.median(warm),
        "energy": result.energy,
        "opposite_spin": diag.opposite_spin_energy,
        "same_spin": diag.same_spin_energy,
        "numeric_capacity_bytes": diag.numeric_capacity_bytes,
        "owned_device_bytes": diag.correlation_owned_device_bytes,
        "resident_b_bytes": diag.correlation_provider_retained_bytes,
        "transfer_bytes": diag.mo_transfer_bytes,
        "mo_host_staging": diag.mo_host_staging,
    }


def traced_cuda(basis: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="vibeqc-ri-mp2-") as directory:
        path = Path(directory) / "trace.jsonl"
        previous = os.environ.get("VIBEQC_DF_TRACE")
        os.environ["VIBEQC_DF_TRACE"] = str(path)
        try:
            calc = Calculator(
                method="mp2", basis=basis, device="cuda", density_fitting="cuda"
            )
            sample(calc)
        finally:
            if previous is None:
                os.environ.pop("VIBEQC_DF_TRACE", None)
            else:
                os.environ["VIBEQC_DF_TRACE"] = previous
        rows = [json.loads(line) for line in path.read_text().splitlines()]
    matches = [row for row in rows if row["operation"] == "ri_mp2_energy"]
    if len(matches) != 1:
        raise RuntimeError("expected one CUDA RI-MP2 component trace")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis", action="append", dest="bases")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    records = []
    for basis in args.bases or DEFAULT_BASES:
        cpu = endpoint(basis, "cpu", args.repeats)
        cuda = endpoint(basis, "cuda", args.repeats)
        if abs(cpu["energy"] - cuda["energy"]) > 1e-9:
            raise RuntimeError(f"{basis}: CPU/CUDA RI-MP2 energy mismatch")
        trace = traced_cuda(basis)
        records.append(
            {
                "basis": basis,
                "cpu": cpu,
                "cuda": cuda,
                "warm_speedup": cpu["warm_median_s"] / cuda["warm_median_s"],
                "trace": trace,
            }
        )
    payload = {"case": "H2O", "coordinates_unit": "bohr", "records": records}
    rendered = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
