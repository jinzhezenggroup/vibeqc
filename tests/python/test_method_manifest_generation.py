from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from vibeqc import _generated_methods

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_ABI_IDS = {
    "rhf": 1,
    "uhf": 2,
    "wb97m-v": 3,
    "rccsd(t)": 4,
    "mp2": 5,
    "lda-rks": 6,
    "pbe-rks": 7,
    "lda-uks": 8,
    "pbe-uks": 9,
}


def test_public_method_generated_metadata_is_fresh():
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/generate_method_manifest.py"),
            "--check",
        ],
        cwd=ROOT,
        check=True,
    )


def test_public_method_abi_ids_are_explicit_and_stable():
    payload = json.loads(
        (ROOT / "methods/public_methods.json").read_text(encoding="utf-8")
    )
    actual = {entry["name"]: entry["abi_id"] for entry in payload["methods"]}
    assert actual == EXPECTED_ABI_IDS
    assert dict(_generated_methods.METHOD_NAME_TO_ID) == EXPECTED_ABI_IDS


def test_public_method_provider_sets_are_generated():
    assert _generated_methods.HF_METHOD_IDS == frozenset({1, 2})
    assert _generated_methods.NATIVE_DFT_METHOD_IDS == frozenset({6, 7, 8, 9})
