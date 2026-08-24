#!/usr/bin/env python3
"""Generate the small runtime Gaussian basis pack from Basis Set Exchange.

The generated JSON is intentionally data-only: the native engine still owns
normalization, validation, and all scientific execution. Regenerate it only
from the pinned Basis Set Exchange revision recorded in
``references/manifest.toml`` and review any numerical diff.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import basis_set_exchange as bse


def _expanded_shells(shell: dict[str, Any]) -> list[dict[str, Any]]:
    """Split BSE combined/general contractions into native single-l shells."""

    angular = [int(value) for value in shell["angular_momentum"]]
    exponents = list(shell["exponents"])
    coefficients = list(shell["coefficients"])
    if len(angular) == 1:
        angular = angular * len(coefficients)
    elif len(angular) != len(coefficients):
        raise ValueError(
            "unsupported BSE shell: angular momenta and contractions cannot "
            "be paired unambiguously"
        )

    expanded: list[dict[str, Any]] = []
    for momentum, contraction in zip(angular, coefficients, strict=True):
        if momentum > 3:
            raise ValueError(
                f"basis contains l={momentum}, above the executable s-p-d-f scope"
            )
        if len(contraction) != len(exponents):
            raise ValueError("BSE exponent/coefficient lengths differ")
        # Keep decimal strings so pack regeneration does not lose source
        # precision before the runtime converts values to IEEE double.
        expanded.append(
            {
                "angular_momentum": momentum,
                "exponents": exponents,
                "coefficients": contraction,
            }
        )
    return expanded


def make_pack(names: list[str], maximum_element: int) -> dict[str, Any]:
    """Return a deterministic pack for all elements through ``maximum_element``."""

    elements = list(range(1, maximum_element + 1))
    packed_bases: dict[str, Any] = {}
    for requested_name in names:
        basis = bse.get_basis(requested_name, elements=elements)
        canonical_name = requested_name.lower().replace("_", "-")
        packed_elements: dict[str, Any] = {}
        for atomic_number in elements:
            source = basis["elements"].get(str(atomic_number))
            if source is None:
                continue
            if source.get("ecp_potentials") or source.get("ecp_electrons"):
                raise ValueError(
                    f"{requested_name} element {atomic_number} requires an ECP"
                )
            shells: list[dict[str, Any]] = []
            for shell in source.get("electron_shells", []):
                shells.extend(_expanded_shells(shell))
            packed_elements[str(atomic_number)] = shells
        packed_bases[canonical_name] = {"elements": packed_elements}

    return {
        "schema_version": 1,
        "source": "Basis Set Exchange 0.11",
        "maximum_element": maximum_element,
        "bases": packed_bases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--basis",
        action="append",
        dest="bases",
        default=None,
        help="basis name to include; may be repeated",
    )
    parser.add_argument("--maximum-element", type=int, default=18)
    args = parser.parse_args()
    names = args.bases or ["sto-3g", "def2-svp", "def2-tzvp"]
    if not 1 <= args.maximum_element <= 118:
        raise ValueError("--maximum-element must be in [1, 118]")

    pack = make_pack(names, args.maximum_element)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(pack, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
