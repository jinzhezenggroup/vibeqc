"""Runtime-shape RCCSD Lambda AOT stays identical to generated TensorIR."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.tensor import execute

from tools.vibeqc_cc.lambda_equations import (
    PARAMETERS,
    build_lambda_programs,
    build_parameter_vjp,
)
from tools.vibeqc_cc.oracle import dense_feeds, random_case

FIELDS = (
    "foo",
    "fov",
    "fvv",
    "ovov",
    "ovvo",
    "oovv",
    "ovvv",
    "ovoo",
    "oooo",
    "vvvv",
    "t1",
    "t2",
)


def _literal(value: float) -> str:
    return float(value).hex()


def _array(name: str, value: np.ndarray) -> str:
    flat = np.asarray(value, dtype=np.float64).reshape(-1)
    body = ",".join(_literal(x) for x in flat)
    return f"static const double {name}[{flat.size}]={{{body}}};"


def _reference(o: int, v: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    f, g, t1, t2 = random_case(o, v, 1700 + 10 * o + v)
    feeds = dense_feeds(f, g, t1, t2)
    programs = build_lambda_programs(o, v)
    rhs = execute(
        programs.energy_vjp.program,
        {**feeds, "bar_correlation_energy": np.asarray(-1.0)},
    ).outputs
    jt = execute(
        programs.residual_vjp.program,
        {
            **feeds,
            "bar_singles_residual": t1,
            "bar_doubles_residual": t2,
        },
    ).outputs
    seeds = {
        "bar_correlation_energy": np.asarray(1.0),
        "bar_singles_residual": t1,
        "bar_doubles_residual": t2,
    }
    parameter = {
        name: np.asarray(
            execute(
                build_parameter_vjp(programs.primal, name).program,
                {**feeds, **seeds},
            ).outputs[f"bar_{name}"]
        )
        for name in PARAMETERS
    }
    return feeds, {
        "rhs_t1": np.asarray(rhs["bar_t1"]),
        "rhs_t2": np.asarray(rhs["bar_t2"]),
        "jt_t1": np.asarray(jt["bar_t1"]),
        "jt_t2": np.asarray(jt["bar_t2"]),
        "bar_r1": t1,
        "bar_r2": t2,
        **{f"param_{name}": value for name, value in parameter.items()},
    }


def _parameter_size(name: str) -> str:
    axes = name.removeprefix("f")
    factors = ["o" if axis == "o" else "v" for axis in axes]
    return "*".join(factors) if factors else "1"


def _case_cpp(o: int, v: int) -> str:
    feeds, reference = _reference(o, v)
    declarations = [_array(name, feeds[name]) for name in FIELDS]
    declarations += [
        _array(name, reference[name])
        for name in ("rhs_t1", "rhs_t2", "jt_t1", "jt_t2", "bar_r1", "bar_r2")
    ]
    declarations += [
        _array(f"param_{name}", reference[f"param_{name}"]) for name in PARAMETERS
    ]
    assignments = [f"inputs.{name}={name};" for name in FIELDS]
    parameter_arenas = ",".join(
        f"parameter_{name}_arena_elements(o,v)" for name in PARAMETERS
    )
    parameter_checks = [
        (
            f"auto p_{name}=run_parameter_{name}_cpu(o,v,inputs,&response_seed,"
            f"bar_r1,bar_r2,arena.data(),arena.size());"
            f"if(!close(p_{name}.values,param_{name},{_parameter_size(name)})) return {20 + index};"
        )
        for index, name in enumerate(PARAMETERS)
    ]
    return "\n".join(
        [
            f"static int case_{o}_{v}(){{",
            *declarations,
            "vibeqc::cc::generated::Inputs inputs{};",
            *assignments,
            f"constexpr std::size_t o={o},v={v};",
            "using namespace vibeqc::cc::generated;",
            "const auto arena_count=std::max({lambda_rhs_arena_elements(o,v),"
            "lambda_transpose_arena_elements(o,v)," + parameter_arenas + "});",
            "std::vector<double> arena(arena_count);",
            "const double energy_seed=-1.0;",
            "auto rhs=run_lambda_rhs_cpu(o,v,inputs,&energy_seed,arena.data(),arena.size());",
            "if(!close(rhs.t1,rhs_t1,o*v)||!close(rhs.t2,rhs_t2,o*o*v*v)) return 1;",
            (
                "auto jt=run_lambda_transpose_cpu(o,v,inputs,bar_r1,bar_r2,"
                "arena.data(),arena.size());"
            ),
            "if(!close(jt.t1,jt_t1,o*v)||!close(jt.t2,jt_t2,o*o*v*v)) return 2;",
            "const double response_seed=1.0;",
            *parameter_checks,
            "return 0;",
            "}",
        ]
    )


CPP_PREFIX = r"""
#include "generated_rccsd_cpu.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>
static bool close(const double* actual,const double* expected,std::size_t n){
  for(std::size_t i=0;i<n;++i)
    if(!std::isfinite(actual[i]) || !std::isfinite(expected[i]) || std::abs(actual[i]-expected[i])>2e-11*(1.0+std::abs(expected[i]))) return false;
  return true;
}
"""


def test_runtime_shape_lambda_codegen_matches_tensorir(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    root = Path(__file__).resolve().parents[2]
    header = tmp_path / "generated_rccsd_cpu.hpp"
    subprocess.run(
        [
            sys.executable,
            str(root / "tools/generate_rccsd_native.py"),
            "--cpu-header",
            str(header),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    cases = ((1, 2), (2, 2))
    source = tmp_path / "lambda.cpp"
    source.write_text(
        CPP_PREFIX
        + "\n".join(_case_cpp(o, v) for o, v in cases)
        + "\nint main(){"
        + "".join(
            f"if(case_{o}_{v}()) return {10 + i};" for i, (o, v) in enumerate(cases)
        )
        + 'std::cout<<"runtime-shape Lambda TensorIR parity passed\\n";return 0;}\n'
    )
    executable = tmp_path / "lambda"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O0",
            "-I" + str(tmp_path),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    result = subprocess.run(
        [str(executable)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "runtime-shape Lambda TensorIR parity passed" in result.stdout
