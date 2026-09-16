"""DF selection rejects incomplete, incompatible and unsafe evidence."""

import copy
import math

import pytest
from vibeqc_compiler.integral.df_tuning.emission import emit_candidate
from vibeqc_compiler.integral.df_tuning.policy import (
    DfDerivativeTrial,
    enumerate_trials,
    rank_profiles,
    read_profile,
)
from vibeqc_compiler.integral.tuning.process import _runtime_environment


def fixture():
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


def rank(profiles, rows):
    return rank_profiles(profiles, rows, baselines={"000": "000:polynomial:compact"})


def test_available_trials_and_identity():
    trials = enumerate_trials()
    assert len(trials) == 42
    assert len({t.key for t in trials}) == len(trials)
    assert {t.angular for t in trials if t.lowering == "rys"} == {
        t.angular for t in trials
    }
    with pytest.raises(ValueError, match="unsupported"):
        DfDerivativeTrial((1, 1, 1), "rys", 2)
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


def test_profiles_remain_separate_and_conflicts_block_combination():
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


@pytest.mark.parametrize(
    "fault", ["missing", "work", "numeric", "compile", "nan", "repeats"]
)
def test_bad_signature_rejects_whole_candidate(fault):
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


def test_missing_baseline_cannot_select_a_winner():
    profiles, rows = fixture()
    rows = [r for r in rows if r["candidate"] != "000:polynomial:compact"]
    result = rank(profiles, rows)
    assert not result["proposed_mapping"]
    assert result["conflicts"]


def test_profile_frequency_validation_and_duplicate_rows():
    profiles, rows = fixture()
    with pytest.raises(ValueError, match="duplicate"):
        rank(profiles, rows + [rows[0]])
    profiles["384"]["classes"][0]["signatures"][0]["work"]["primitive_products"] += 1
    with pytest.raises(ValueError, match="primitive work"):
        read_profile(profiles["384"])


def test_runtime_preserves_scheduler_visibility(monkeypatch, tmp_path):
    for value in ("", "0", "GPU-unique-token"):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
        assert (
            _runtime_environment(tmp_path / "bin/nvcc")["CUDA_VISIBLE_DEVICES"] == value
        )


def test_manifests_require_complete_independent_qualification(tmp_path):
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


def test_value_identity_and_profile_disagreements():
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
