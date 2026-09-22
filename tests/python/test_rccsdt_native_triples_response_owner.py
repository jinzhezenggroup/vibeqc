"""Native bounded standard-(T) response owner matches the full VJP oracle."""

from __future__ import annotations

import shutil
import subprocess
import sys
import typing
from pathlib import Path

import numpy as np
import pytest

from tools.vibeqc_cc.triples_response import full_triples_vjp

NAMES = ("ovvv", "ovoo", "ovov", "fov", "t1", "t2", "eps_o", "eps_v")


def _case(o: int, v: int, seed: int) -> typing.Any:
    rng = np.random.default_rng(seed)
    t2 = rng.normal(size=(o, o, v, v))
    t2 = (t2 + t2.transpose(1, 0, 3, 2)) / 2
    arrays = {
        "ovvv": rng.normal(size=(o, v, v, v)),
        "ovoo": rng.normal(size=(o, v, o, o)),
        "ovov": rng.normal(size=(o, v, o, v)),
        "fov": rng.normal(size=(o, v)),
        "t1": rng.normal(size=(o, v)),
        "t2": t2,
        "eps_o": np.linspace(-1.2, -0.8, o),
        "eps_v": np.linspace(0.5, 1.1, v),
    }
    expected = full_triples_vjp(
        o,
        v,
        *(arrays[name] for name in NAMES),
        inputs=NAMES,
        max_elements=10_000_000,
    )
    return arrays, expected


def _literal(value: float) -> str:
    return float(value).hex()


def _array(name: str, value: np.ndarray) -> str:
    flat = np.asarray(value, dtype=np.float64).reshape(-1)
    return f"static const double {name}[{flat.size}]={{{','.join(_literal(x) for x in flat)}}};"


def _cpp_case(o: int, v: int, seed: int) -> str:
    arrays, expected = _case(o, v, seed)
    declarations = [_array(name, arrays[name]) for name in NAMES]
    declarations += [_array(f"expected_{name}", expected[name]) for name in NAMES]
    sizes = {
        "ovvv": o * v * v * v,
        "ovoo": o * v * o * o,
        "ovov": o * v * o * v,
        "fov": o * v,
        "t1": o * v,
        "t2": o * o * v * v,
        "eps_o": o,
        "eps_v": v,
    }
    checks = [
        f"if(!close(result.{name},expected_{name},{sizes[name]})) return {20 + i};"
        for i, name in enumerate(NAMES)
    ]
    return "\n".join(
        [
            f"static int case_{o}_{v}(){{",
            *declarations,
            "using namespace vibeqc::cc;",
            "Problem p;",
            f"p.nocc={o}; p.nvir={v};",
            "p.reference_energy=0.0;",
            f"p.foo.assign({o * o},0.0);",
            f"p.fov.assign(fov,fov+{o * v});",
            f"p.fvv.assign({v * v},0.0);",
            f"p.ovov.assign(ovov,ovov+{sizes['ovov']});",
            f"p.ovvo.assign({o * v * v * o},0.0);",
            f"p.oovv.assign({o * o * v * v},0.0);",
            f"p.ovvv.assign(ovvv,ovvv+{sizes['ovvv']});",
            f"p.ovoo.assign(ovoo,ovoo+{sizes['ovoo']});",
            f"p.oooo.assign({o**4},0.0);",
            f"p.vvvv.assign({v**4},0.0);",
            f"p.d1.assign({o * v},-1.0);",
            f"p.d2.assign({o * o * v * v},-2.0);",
            f"p.initial_t1.assign({o * v},0.0);",
            f"p.initial_t2.assign({o * o * v * v},0.0);",
            "SolverResult cc;",
            "cc.status=SolveStatus::Converged;",
            f"cc.t1.assign(t1,t1+{sizes['t1']});",
            f"cc.t2.assign(t2,t2+{sizes['t2']});",
            f"std::vector<double> eo(eps_o,eps_o+{o}), ev(eps_v,eps_v+{v});",
            "TriplesResponseOptions options; options.batch_capacity=2; options.max_bytes=64ULL<<20;",
            "auto result=triples_response_cpu(p,cc,eo,ev,options);",
            *checks,
            "if(result.pages<2) return 40;",
            "if(!result.program_hash) return 41;",
            "return 0;",
            "}",
        ]
    )


CPP_PREFIX = r"""
#include "cc/triples_response.hpp"
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>
static bool close(const std::vector<double>& actual,const double* expected,std::size_t n){
  if(actual.size()!=n) return false;
  for(std::size_t i=0;i<n;++i)
    if(!std::isfinite(actual[i]) || !std::isfinite(expected[i]) || std::abs(actual[i]-expected[i])>8e-11*(1.0+std::abs(expected[i]))) {
      std::cerr<<"mismatch "<<i<<" "<<actual[i]<<" "<<expected[i]<<"\n";
      return false;
    }
  return true;
}
"""


def test_native_triples_response_owner_matches_full_vjp(tmp_path: Path) -> None:
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
    source = tmp_path / "owner.cpp"
    source.write_text(
        CPP_PREFIX
        + _cpp_case(1, 2, 901)
        + "\nint main(){const int rc=case_1_2();"
        + 'if(!rc) std::cout<<"native triples response owner parity passed\\n";return rc;}\n'
    )
    executable = tmp_path / "owner"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O0",
            "-DVIBEQC_HAS_CUDA=0",
            "-I" + str(root / "src"),
            "-I" + str(tmp_path),
            str(root / "src/cc/solver.cpp"),
            str(root / "src/cc/triples_response.cpp"),
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
    assert "native triples response owner parity passed" in result.stdout
