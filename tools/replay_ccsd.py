"""Replay saved fixed-Hamiltonian CCSD states without AO work or external CC."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tools.vibeqc_cc.solver import SolverOptions, solve
from tools.vibeqc_posthf import MOBlock, ReferenceSnapshot
from tools.vibeqc_posthf.providers import BlockResult, ConventionalProvider
from tools.vibeqc_validation.schema import canonical_hash


class _SavedProvider(ConventionalProvider):
    """Validation-only replay of immutable explicitly indexed saved MO blocks."""

    def __init__(self, snapshot, inputs):
        self.snapshot = snapshot
        self.backend = "cpu"
        self.source = SimpleNamespace(_check_open=lambda: None)
        self.blocks = {
            MOBlock.from_spaces(snapshot, k).slots: np.asarray(inputs[k], dtype=float)
            for k in ("ovov", "ovvo", "oovv", "ovvv", "ovoo", "oooo", "vvvv")
        }

    def get(self, block):
        return BlockResult(
            block,
            self.blocks[block.slots],
            self.snapshot.identity,
            self.snapshot.hamiltonian_id,
            {"source": "saved MO replay"},
        )


def replay(path):
    record = json.loads(Path(path).read_text())
    if record.get("record_hash") != canonical_hash(
        {k: v for k, v in record.items() if k != "record_hash"}
    ):
        raise ValueError("CCSD replay record identity mismatch")
    if (
        record["schema"] != "vibeqc.ccsd.result"
        or record["version"] != 1
        or canonical_hash(record["inputs"]) != record["inputs_hash"]
    ):
        raise ValueError("CCSD replay input identity mismatch")
    inputs = record["inputs"]
    snapshot = ReferenceSnapshot(**inputs["snapshot"])
    if snapshot.identity != record["provenance"]["reference_id"]:
        raise ValueError("CCSD replay reference identity mismatch")
    return solve(
        snapshot,
        _SavedProvider(snapshot, inputs),
        options=SolverOptions(**record["provenance"]["options"]),
        t1=np.asarray(inputs["initial_t1"]),
        t2=np.asarray(inputs["initial_t2"]),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.input)
    result.write(args.output)
    print(result.status, result.reason)
