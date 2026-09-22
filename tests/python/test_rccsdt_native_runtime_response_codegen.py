"""Runtime-shape standard-(T) response AOT matches generated TensorIR."""

from __future__ import annotations

import shutil
import subprocess
import sys
import typing
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.tensor import execute, transpose_program

from tools.vibeqc_cc.triples_tiles import build_runtime_tile_triples_program

INPUTS = ("ovvv", "ovoo", "ovov", "fov", "t1", "t2", "eps_o", "eps_v")


def _scientific(o: int, v: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    t2 = rng.normal(size=(o, o, v, v))
    t2 = (t2 + t2.transpose(1, 0, 3, 2)) / 2
    return {
        "ovvv": rng.normal(size=(o, v, v, v)),
        "ovoo": rng.normal(size=(o, v, o, o)),
        "ovov": rng.normal(size=(o, v, o, v)),
        "fov": rng.normal(size=(o, v)),
        "t1": rng.normal(size=(o, v)),
        "t2": t2,
        "eps_o": np.linspace(-1.2, -0.5, o),
        "eps_v": np.linspace(0.4, 1.4, v),
    }


def _case(o: int, v: int, q: int, seed: int) -> typing.Any:
    feeds = _scientific(o, v, seed)
    a = np.arange(q, dtype=np.int64) % v
    b = np.arange(q, dtype=np.int64)[::-1] % v
    c = (np.arange(q, dtype=np.int64) * 2) % v
    feeds.update(
        a_map=a,
        b_map=b,
        c_map=c,
        active=np.where(np.arange(q) == q - 1, 0.0, 1.0),
        degeneracy=np.linspace(1.0, 3.0, q),
        bar_triples_energy=np.asarray(0.73),
    )
    primal = build_runtime_tile_triples_program(o, v, capacity=q)
    response = transpose_program(
        primal,
        ("triples_energy",),
        inputs=INPUTS,
        max_elements=100_000_000,
    ).program
    expected = execute(response, feeds).outputs
    return feeds, expected


def _literal(value: float) -> str:
    return float(value).hex()


def _array(name: str, value: np.ndarray) -> str:
    flat = np.asarray(value).reshape(-1)
    if flat.dtype == np.int64:
        body = ",".join(str(int(x)) for x in flat)
        return f"static const std::int64_t {name}[{flat.size}]={{{body}}};"
    body = ",".join(_literal(x) for x in flat.astype(np.float64))
    return f"static const double {name}[{flat.size}]={{{body}}};"


def _case_cpp(o: int, v: int, q: int, seed: int) -> str:
    feeds, expected = _case(o, v, q, seed)
    declarations = [_array(name, value) for name, value in feeds.items()]
    declarations += [
        _array(f"expected_{name}", expected[f"bar_{name}"]) for name in INPUTS
    ]
    assignments = [f"inputs.{name}={name};" for name in sorted(feeds)]
    checks = [
        (
            f"if(!close(out.{name},expected_{name},"
            f"{np.asarray(expected[f'bar_{name}']).size})) return {20 + i};"
        )
        for i, name in enumerate(INPUTS)
    ]
    return "\n".join(
        [
            f"static int case_{o}_{v}_{q}(){{",
            *declarations,
            "using namespace vibeqc::cc::generated;",
            "TriplesResponseInputs inputs{};",
            *assignments,
            f"constexpr std::size_t o={o},v={v},q={q};",
            "std::vector<double> arena(triples_response_arena_elements(o,v,q));",
            "auto out=run_triples_response_cpu(o,v,q,inputs,arena.data(),arena.size());",
            *checks,
            "return 0;",
            "}",
        ]
    )


CPP_PREFIX = r"""
#include "generated_rccsd_cpu.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>
static bool close(const double* actual,const double* expected,std::size_t n){
  for(std::size_t i=0;i<n;++i)
    if(!std::isfinite(actual[i]) || !std::isfinite(expected[i]) || std::abs(actual[i]-expected[i])>5e-11*(1.0+std::abs(expected[i]))) {
      std::cerr<<"mismatch "<<i<<" "<<actual[i]<<" "<<expected[i]<<"\n";
      return false;
    }
  return true;
}
"""


def test_runtime_shape_triples_response_matches_tensorir(tmp_path: Path) -> None:
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
        timeout=120,
    )
    cases = ((1, 2, 3, 801), (2, 2, 4, 802))
    source = tmp_path / "triples_response.cpp"
    source.write_text(
        CPP_PREFIX
        + "\n".join(_case_cpp(*case) for case in cases)
        + "\nint main(){"
        + "".join(
            f"if(const int rc=case_{o}_{v}_{q}()) return rc;" for o, v, q, _ in cases
        )
        + 'std::cout<<"runtime-shape triples response parity passed\\n";return 0;}\n'
    )
    executable = tmp_path / "triples_response"
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
        timeout=120,
    )
    result = subprocess.run(
        [str(executable)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "runtime-shape triples response parity passed" in result.stdout
