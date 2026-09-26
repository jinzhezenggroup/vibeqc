"""Sharded catalog campaign planning and reporting for imported Libxc XC."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from vibeqc_compiler.xc.libxc_bulk_capabilities import functional_capability

from tools import qualify_libxc_production_catalog as catalog

if TYPE_CHECKING:
    from pathlib import Path


def test_catalog_shards_are_deterministic_complete_and_disjoint() -> None:
    all_items = catalog.select_capabilities()
    shards = [
        catalog.select_capabilities(shard_count=4, shard_index=index)
        for index in range(4)
    ]

    names = [item.name for item in all_items]
    flattened = [item.name for shard in shards for item in shard]
    assert len(flattened) == len(names)
    assert len(set(flattened)) == len(names)
    assert set(flattened) == set(names)

    for index, shard in enumerate(shards):
        assert [item.name for item in shard] == names[index::4]


def test_catalog_name_filter_is_case_insensitive_and_bounded() -> None:
    selected = catalog.select_capabilities(
        names=["gga_x_pbe_sol", "lda_c_vwn_4"],
        maximum=1,
    )

    assert len(selected) == 1
    assert selected[0].name in {"GGA_X_PBE_SOL", "LDA_C_VWN_4"}

    with pytest.raises(ValueError, match="unknown imported"):
        catalog.select_capabilities(names=["NOT_A_LIBXC_REGISTRATION"])


@pytest.mark.parametrize(
    ("count", "index", "message"),
    [
        (0, 0, "positive integer"),
        (2, -1, "0 <= index"),
        (2, 2, "0 <= index"),
    ],
)
def test_catalog_rejects_invalid_shards(count: int, index: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        catalog.select_capabilities(shard_count=count, shard_index=index)


def test_catalog_summary_counts_status_family_and_blocked_cases() -> None:
    rows = [
        {
            "name": "A",
            "family": "lda",
            "status": "pass",
            "blocked_case_ids": [],
        },
        {
            "name": "B",
            "family": "gga",
            "status": "fail",
            "blocked_case_ids": ["sigma/zero", "density/vacuum"],
        },
        {
            "name": "C",
            "family": "gga",
            "status": "fail",
            "blocked_case_ids": ["sigma/zero"],
        },
    ]

    summary = catalog.summarize_rows(
        rows,
        catalog_identity="a" * 64,
        shard_count=2,
        shard_index=1,
        rtol=2.0e-6,
        atol=1.0e-8,
    )

    assert summary["selected_functionals"] == 3
    assert summary["status_counts"]["pass"] == 1
    assert summary["status_counts"]["fail"] == 2
    assert summary["by_family"]["gga"]["fail"] == 2
    assert summary["blocked_case_counts"] == {
        "density/vacuum": 1,
        "sigma/zero": 2,
    }


def test_catalog_run_retains_campaign_and_structural_blocker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    eligible = functional_capability("GGA_X_PBE_SOL")
    blocked = functional_capability("MGGA_X_JK")

    def fake_qualify(name: str, **_: object) -> dict:
        assert name == eligible.name
        return {
            "stage_evidence": {
                "status": "fail",
                "reason": "polarized:sigma/zero: candidate mismatch",
            },
            "receipt": {
                "identity": "b" * 64,
                "cases": [
                    {
                        "case_id": "sigma/zero",
                        "status": "fail",
                    }
                ],
            },
            "execution": {"identity": "c" * 64},
        }

    monkeypatch.setattr(catalog, "qualify_functional", fake_qualify)
    summary = catalog.run_catalog(
        (eligible, blocked),
        output=tmp_path,
        evidence_prefix="artifact://test/libxc-domain",
        pyscf_version="2.14.0",
        libxc=SimpleNamespace(__version__="7.0.0"),
        shard_count=1,
        shard_index=0,
        rtol=2.0e-6,
        atol=1.0e-8,
    )

    assert summary["status_counts"]["fail"] == 1
    assert summary["status_counts"]["structural-blocked"] == 1
    assert summary["blocked_case_counts"] == {"sigma/zero": 1}
    eligible_row = summary["functionals"][eligible.name]
    blocked_row = summary["functionals"][blocked.name]
    assert eligible_row["campaign"] == f"campaigns/{eligible.name}.json"
    assert blocked_row["campaign"] is None
    assert blocked_row["blocker"] == "unsupported-ingredients:laplacian"

    campaign_path = tmp_path / eligible_row["campaign"]
    assert campaign_path.is_file()
    assert json.loads(campaign_path.read_text())["receipt"]["identity"] == "b" * 64
    assert (tmp_path / "summary.json").is_file()


@pytest.mark.parametrize("field", ("rtol", "atol"))
@pytest.mark.parametrize("value", (float("nan"), float("inf"), -1.0, True))
def test_catalog_rejects_invalid_tolerances_before_any_functional_runs(
    tmp_path: Path, field: str, value: float
) -> None:
    kwargs = {"rtol": 2.0e-6, "atol": 1.0e-8}
    kwargs[field] = value
    with pytest.raises(ValueError, match="finite and nonnegative"):
        catalog.run_catalog(
            (),
            output=tmp_path,
            evidence_prefix="artifact://test/libxc-domain",
            pyscf_version="2.14.0",
            libxc=SimpleNamespace(__version__="7.0.0"),
            shard_count=1,
            shard_index=0,
            **kwargs,
        )
