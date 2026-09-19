"""Deterministic compilation boundaries for unchanged DF derivative mathematics."""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from vibeqc_compiler.integral import df_rys_shell
from vibeqc_compiler.integral.df_rys_shell import (
    RYS_SHELL_CLASSES,
    emit_df_rys_shell_cuda,
)
from vibeqc_compiler.integral.df_shell_derivatives import (
    SHELL_CLASSES,
    emit_df_shell_derivatives_cuda,
    select_shell_classes,
)
from vibeqc_compiler.integral.df_shell_units import emit_df_shell_units

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def units():
    return dict(emit_df_shell_units())


def specialization(source, declaration):
    """Extract one complete C++ struct, including nested method/loop bodies."""
    begin = source.index(declaration)
    brace = source.index("{", begin)
    depth = 0
    for index in range(brace, len(source)):
        depth += (source[index] == "{") - (source[index] == "}")
        if depth == 0:
            return source[begin : index + 2]
    raise AssertionError("unterminated emitted specialization")


def test_units_cover_all_classes_once_and_are_deterministic(units):
    assert units == dict(emit_df_shell_units())
    assert len(units) == 1 + 4 * len(SHELL_CLASSES)
    registry = units["generated_df_shell_dispatch.hpp"]
    calls = re.findall(r"operator\(\)<(\d),(\d),(\d)>\(df_shell_(\d{3})\)", registry)
    assert [tuple(map(int, row[:3])) for row in calls] == list(SHELL_CLASSES)
    assert all("".join(row[:3]) == row[3] for row in calls)
    for angular in SHELL_CLASSES:
        name = "".join(map(str, angular))
        source = units[f"df_shell_{name}.cu"]
        assert f'VIBEQC_DF_SHELL_MATH_HEADER "df_shell_math_{name}.cuh"' in source
        for kind in ("panel", "group", "packets"):
            assert f"launch_{kind}_class<{','.join(map(str, angular))}>" in source
        assert "production" not in source


def test_selected_class_math_and_schedules_are_byte_identical(units):
    polynomial = emit_df_shell_derivatives_cuda()
    rys = emit_df_rys_shell_cuda()
    for angular in SHELL_CLASSES:
        name = "".join(map(str, angular))
        parameters = ",".join(map(str, angular))
        selected = units[f"df_shell_polynomial_{name}.cuh"]
        assert (
            "for_each_class" not in selected
        )  # No differing external template definitions.
        for variant in range(3):
            declaration = f"template<> struct Schedule<{parameters},{variant}>"
            assert specialization(selected, declaration) == specialization(
                polynomial, declaration
            )
        if max(angular) <= 1:
            declaration = f"template<> struct Shell<{parameters}>"
            assert specialization(selected, declaration) == specialization(
                polynomial, declaration
            )
        if angular in RYS_SHELL_CLASSES:
            declaration = f"template<> struct RysShell<{parameters}>"
            assert specialization(
                units[f"df_shell_math_{name}.cuh"], declaration
            ) == specialization(rys, declaration)
        policy = units[f"df_shell_policy_{name}.hpp"]
        mask = (
            1 << (16 * angular[0] + 4 * angular[1] + angular[2])
            if angular in RYS_SHELL_CLASSES
            else 0
        )
        assert f"rys_available_mask={mask}ULL" in policy
        assert (
            "inline constexpr" not in policy
        )  # Per-TU capabilities have internal linkage.


def test_class_local_math_change_does_not_invalidate_neighbors(units, monkeypatch):
    original = df_rys_shell._emit_cooperative_shell

    def changed(angular):
        lines = original(angular)
        return lines + (["// class-local edit"] if angular == (0, 0, 3) else [])

    monkeypatch.setattr(df_rys_shell, "_emit_cooperative_shell", changed)
    after = dict(emit_df_shell_units())
    assert {name for name in units if units[name] != after[name]} == {
        "df_shell_math_003.cuh"
    }


@pytest.mark.parametrize("classes", [[], [(4, 0, 0)], [(0, 0)], [(0, 0, 0)] * 2])
def test_invalid_subsets_are_rejected(classes):
    with pytest.raises(ValueError, match="supported s/p/d/f"):
        select_shell_classes(classes)


def test_subset_order_is_canonical():
    assert select_shell_classes([(2, 1, 3), (0, 0, 0)]) == ((0, 0, 0), (2, 1, 3))


def test_policy_edit_preserves_numerical_source_bytes_and_mtimes(tmp_path):
    manifest = json.loads(
        (
            ROOT / "python/vibeqc_compiler/integral/production_df_derivatives.json"
        ).read_text()
    )
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps(manifest))
    command = [
        sys.executable,
        str(ROOT / "tools/generate_df_kernels.py"),
        "--derivatives",
        "--output",
        str(tmp_path / "generated_df_derivatives.cuh"),
        "--shell-output",
        str(tmp_path / "generated_df_shell_derivatives.cuh"),
        "--shell-units-directory",
        str(tmp_path / "units"),
        "--df-production-manifest",
        str(policy),
    ]
    subprocess.run(command, check=True, capture_output=True, timeout=120)
    before = {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in (tmp_path / "units").iterdir()
    }
    original_policy = (tmp_path / "generated_df_production.hpp").read_bytes()
    manifest["architectures"]["sm_120"]["qualified"] = False
    policy.write_text(json.dumps(manifest))
    subprocess.run(command, check=True, capture_output=True, timeout=120)
    after = {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in (tmp_path / "units").iterdir()
    }
    assert before == after
    assert (tmp_path / "generated_df_production.hpp").read_bytes() != original_policy


def test_ci_cache_snapshots_version_compiler_and_build_inputs():
    for workflow in ("ci.yml", "wheels.yml"):
        source = (ROOT / ".github/workflows" / workflow).read_text()
        keys = [line for line in source.splitlines() if "key: ccache-" in line]
        assert keys
        for key in keys:
            assert "'cmake/**'" in key
            assert "'python/vibeqc_compiler/**'" in key
