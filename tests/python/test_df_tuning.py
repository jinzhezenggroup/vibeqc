"""DF selection rejects incomplete, incompatible and unsafe evidence."""

import copy
import math
import typing

import pytest
from vibeqc_compiler.integral.df_tuning.emission import emit_candidate
from vibeqc_compiler.integral.df_tuning.policy import (
    DfDerivativeTrial,
    enumerate_trials,
    rank_profiles,
    read_profile,
)
from vibeqc_compiler.integral.tuning.process import _runtime_environment


def fixture() -> typing.Any:
    work = {
        "shell_tasks": 20,
        "active_shell_tasks": 20,
        "primitive_products": 60,
        "active_component_products": 60,
    }
    profile = {
        "classes": [
            {
                "angular": [0, 0, 0],
                "signatures": [{"primitives": [1, 1, 3], "work": work}],
            }
        ]
    }
    rows = []
    for name in ("384", "768"):
        for trial in enumerate_trials(((0, 0, 0),)):
            rows.append(
                {
                    "profile": name,
                    "candidate": trial.key,
                    "primitives": [1, 1, 3],
                    "work": dict(work),
                    "eligible": True,
                    "numerical_passed": True,
                    "milliseconds": [1.0 if trial.lowering == "rys" else 2.0] * 5,
                }
            )
    return {"384": copy.deepcopy(profile), "768": copy.deepcopy(profile)}, rows


def rank(profiles: typing.Any, rows: typing.Any) -> typing.Any:
    return rank_profiles(profiles, rows, baselines={"000": "000:polynomial:compact"})


def test_available_trials_and_identity() -> None:
    trials = enumerate_trials()
    assert len(trials) == 42
    assert len({t.key for t in trials}) == len(trials)
    assert {t.angular for t in trials if t.lowering == "rys"} == {
        t.angular for t in trials
    }
    with pytest.raises(ValueError, match="unsupported"):
        DfDerivativeTrial((3, 1, 1), "rys", 2)
    trial = trials[0]
    identity = {
        "generator_sha256": "abc",
        "architecture": "sm_120",
        "toolchain": "nvcc A",
    }
    key = trial.artifact_key(**identity)
    for field in identity:
        assert trial.artifact_key(**(identity | {field: "different"})) != key
    assert trials[1].artifact_key(**identity) != key
    source = emit_candidate(trial)
    assert '"scf/cuda/df_shell_kernel.cuh"' in source
    assert "shell_packet<0,0,0,0,false>" in source
    assert "prepare_geometry" not in source


def test_profiles_remain_separate_and_conflicts_block_combination() -> None:
    profiles, rows = fixture()
    result = rank(profiles, rows)
    assert result["proposed_mapping"] == {"000": "000:rys:compact"}
    assert result["production_promoted"] is False
    for row in rows:
        if row["profile"] == "768" and ":rys:" in row["candidate"]:
            row["milliseconds"] = [3.0] * 5
    result = rank(profiles, rows)
    assert result["proposed_mapping"] == {}
    assert result["conflicts"]["000"] == {
        "384": "000:rys:compact",
        "768": "000:polynomial:compact",
    }


def test_explicit_high_angular_campaign_preserves_both_profiles() -> None:
    """A hot-class campaign must not silently discard classes outside the old seven."""
    profiles, rows = fixture()
    for payload in profiles.values():
        payload["classes"][0]["angular"] = [1, 1, 1]
    for row in rows:
        row["candidate"] = row["candidate"].replace("000:", "111:")
    result = rank_profiles(
        profiles,
        rows,
        baselines={"111": "111:polynomial:compact"},
        angular=((1, 1, 1),),
    )
    assert set(result["profiles"]) == {"384", "768"}
    assert result["proposed_mapping"] == {"111": "111:rys:compact"}
    rows[-1]["work"]["primitive_products"] += 1
    result = rank_profiles(
        profiles,
        rows,
        baselines={"111": "111:polynomial:compact"},
        angular=((1, 1, 1),),
    )
    assert result["profiles"]["768"]["111"]["rejections"][rows[-1]["candidate"]] == [
        "work domain differs"
    ]


@pytest.mark.parametrize(
    "fault", ["missing", "work", "numeric", "compile", "nan", "repeats"]
)
def test_bad_signature_rejects_whole_candidate(fault: typing.Any) -> None:
    profiles, rows = fixture()
    for row in rows[:]:
        if ":rys:" not in row["candidate"]:
            continue
        if fault == "missing":
            rows.remove(row)
        elif fault == "work":
            row["work"]["shell_tasks"] += 1
        elif fault == "numeric":
            row["numerical_passed"] = False
        elif fault == "compile":
            row["eligible"] = False
        elif fault == "nan":
            row["milliseconds"][0] = math.nan
        else:
            row["milliseconds"] = [1.0]
    assert rank(profiles, rows)["proposed_mapping"] == {"000": "000:polynomial:compact"}


def test_missing_baseline_cannot_select_a_winner() -> None:
    profiles, rows = fixture()
    rows = [r for r in rows if r["candidate"] != "000:polynomial:compact"]
    result = rank(profiles, rows)
    assert not result["proposed_mapping"]
    assert result["conflicts"]


def test_profile_frequency_validation_and_duplicate_rows() -> None:
    profiles, rows = fixture()
    with pytest.raises(ValueError, match="duplicate"):
        rank(profiles, rows + [rows[0]])
    profiles["384"]["classes"][0]["signatures"][0]["work"]["primitive_products"] += 1
    with pytest.raises(ValueError, match="primitive work"):
        read_profile(profiles["384"])


def test_runtime_preserves_scheduler_visibility(
    monkeypatch: typing.Any, tmp_path: typing.Any
) -> None:
    for value in ("", "0", "GPU-unique-token"):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
        assert (
            _runtime_environment(tmp_path / "bin/nvcc")["CUDA_VISIBLE_DEVICES"] == value
        )


def test_manifests_require_complete_independent_qualification(
    tmp_path: typing.Any,
) -> None:
    """A partial numerical run cannot turn on either automatic production policy."""
    import copy
    import json

    from vibeqc_compiler.integral.df_tuning.manifest import MANIFEST, load_manifest
    from vibeqc_compiler.integral.df_tuning.value_manifest import (
        VALUE_MANIFEST,
        load_value_manifest,
    )

    for source, load, derivative in (
        (MANIFEST, load_manifest, True),
        (VALUE_MANIFEST, load_value_manifest, False),
    ):
        original = json.loads(source.read_text())
        payload = copy.deepcopy(original)
        profile = payload["architectures"]["sm_120"] if derivative else payload
        profile["qualified"] = True
        profile["provenance"] = {"candidate_report": "partial.json"}
        path = tmp_path / "partial.json"
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="provenance|evidence"):
            load(path)


def campaign_manifest() -> typing.Any:
    """Keep the already-promoted SSS choice in both synthetic comparison arms."""
    baseline = {
        "qualified": True,
        "kernels": [
            {"class": "000", "lowering": "rys", "schedule": "compact"},
            {"class": "001", "lowering": "polynomial", "schedule": "compact"},
        ],
        "provenance": dict.fromkeys(
            [
                "generator_sha256",
                "toolchain",
                "profile_sha256",
                "candidate_report",
                "endpoint_384",
                "endpoint_768",
                "sanitizer",
                "gradient_fixtures",
            ],
            "unit-test fixture",
        ),
    }
    candidate = copy.deepcopy(baseline)
    candidate["qualified"] = False
    candidate["kernels"][1]["lowering"] = "rys"
    candidate["baseline"] = baseline
    return {"schema_version": 1, "architectures": {"sm_120": candidate}}


@pytest.mark.parametrize("promoted", [False, True])
def test_compiled_campaign_selection_preserves_qualified_baseline(
    tmp_path: typing.Any, promoted: typing.Any
) -> None:
    """Promotion removes the comparison arm without changing candidate math."""
    import json
    import shutil
    import subprocess

    from vibeqc_compiler.integral.df_tuning.manifest import emit_policy

    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    path = tmp_path / "manifest.json"
    manifest = campaign_manifest()
    if promoted:
        profile = manifest["architectures"]["sm_120"]
        del profile["baseline"]
        profile["qualified"] = True
    path.write_text(json.dumps(manifest))
    source = tmp_path / "policy.cpp"
    source.write_text(
        emit_policy(path)
        + """
using namespace vibeqc::scf::generated_df_shell;
static_assert(DfProductionPolicy<0,0,0>::select(120).rys);
static_assert(DfProductionPolicy<0,0,0>::select(120,true).rys);
static_assert(DfProductionPolicy<0,0,1>::select(120).qualified);
static_assert(DfProductionPolicy<0,0,1>::select(120,true).rys);
static_assert(!DfProductionPolicy<0,0,1>::select(80,true).available);
static_assert(!DfProductionPolicy<1,1,1>::select(120,true).available);
"""
        + f"static_assert(DfProductionPolicy<0,0,1>::select(120).rys == {str(promoted).lower()});\n"
        + f"static_assert(DfProductionPolicy<0,0,1>::select(120,true).qualified == {str(promoted).lower()});\n"
    )
    subprocess.run([compiler, "-std=c++20", "-fsyntax-only", str(source)], check=True)


@pytest.mark.parametrize(
    "fault", ["baseline", "candidate", "nested", "evidence", "math"]
)
def test_campaign_baseline_requires_qualified_independent_evidence(
    tmp_path: typing.Any, fault: typing.Any
) -> None:
    import json

    from vibeqc_compiler.integral.df_tuning.manifest import load_manifest

    payload = campaign_manifest()
    candidate = payload["architectures"]["sm_120"]
    baseline = candidate["baseline"]
    if fault == "baseline":
        baseline["qualified"] = False
    elif fault == "candidate":
        candidate["qualified"] = True
    elif fault == "nested":
        baseline["baseline"] = copy.deepcopy(baseline)
    elif fault == "evidence":
        del baseline["provenance"]["endpoint_768"]
    else:
        baseline["kernels"][0]["class"] = "311"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_manifest(path)


def test_value_identity_and_profile_disagreements() -> None:
    """Class scoring retains both workloads and excludes incomplete/changed work."""
    import copy

    from vibeqc_compiler.integral.df_tuning.values import (
        DfValueTrial,
        enumerate_value_trials,
    )

    from tools.benchmark_df_values import rank_values

    trials = [t for t in enumerate_value_trials() if t.angular == (0, 0, 0)]
    assert len(enumerate_value_trials()) == 90
    with pytest.raises(ValueError):
        DfValueTrial((False, 0, 0), "rys", 4)
    groups = [
        {
            "profile": p,
            "angular": (0, 0, 0),
            "primitives": (1, 1, 1),
            "shell_frequency": 100,
            "sampled_shell_blocks": 8,
            "sampled_components": 8,
            "replicas": 1024,
            "measured_primitive_components": 8192,
        }
        for p in ("384", "768")
    ]
    rows = []
    for i, group in enumerate(groups):
        for t in trials:
            winner = (t.lowering, t.lanes) == (
                ("polynomial", 4) if i == 0 else ("rys", 4)
            )
            rows.append(
                {
                    **group,
                    "group": i,
                    "candidate": t.key,
                    "numerical_passed": True,
                    "milliseconds": [1 if winner else 3] * 5,
                }
            )
    compiled = [{"key": t.key, "eligible": True} for t in trials]
    ranked = rank_values(groups, rows, compiled)
    assert "000" in ranked["conflicts"]
    assert not ranked["production_promoted"]
    altered = copy.deepcopy(rows)
    for row in altered:
        if ":rys:" in row["candidate"]:
            row["measured_primitive_components"] -= 1
    ranked = rank_values(groups, altered, compiled)
    for p in ("384", "768"):
        assert not any(
            ":rys:" in k for k in ranked["profiles"][p]["000"]["estimated_class_ms"]
        )
    baseline = "raw_cartesian:000:generic:lanes1"
    missing = [r for r in rows if r["candidate"] != baseline]
    assert not rank_values(groups, missing, compiled)["proposed_mapping"]
