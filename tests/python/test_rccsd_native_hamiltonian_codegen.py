"""Runtime-shape CC Hamiltonian response AOT matches TensorIR exactly."""

from __future__ import annotations

import shutil
import subprocess
import sys
import typing
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.tensor import PackedLayout, Program, execute

from tools.vibeqc_cc.gradient_equations import (
    build_fock_weight_program,
    build_hamiltonian_programs,
)


def _literal(value: float) -> str:
    return float(value).hex()


def _array(name: str, value: np.ndarray) -> str:
    flat = np.asarray(value, dtype=np.float64).reshape(-1)
    return (
        f"static const double {name}[{flat.size}]="
        + "{"
        + ",".join(_literal(x) for x in flat)
        + "};"
    )


def _feeds(
    program: Program, o: int, v: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    result = {}
    n = o + v
    for node in program.live_nodes:
        if node.op != "input":
            continue
        name = node.attrs["name"]
        if name in result:
            continue
        if name == "density":
            value = np.zeros((n, n))
            value[np.arange(o), np.arange(o)] = 2.0
        elif name == "rotation":
            value = np.eye(n)
        elif name == "bar_reference_electronic_energy":
            value = np.asarray(0.73)
        elif node.spec.symmetries:
            layout = PackedLayout.from_spec(node.spec)
            value = layout.unpack(rng.normal(size=layout.size).astype(np.float64))
        else:
            value = rng.normal(size=node.spec.shape)
        result[name] = np.asarray(value, dtype=np.float64)
    return result


def _reference(o: int, v: int) -> typing.Any:
    rng = np.random.default_rng(9100 + 10 * o + v)
    programs = build_hamiltonian_programs(o, v, explicit_density_input=True)
    weights_feeds = _feeds(programs.weights, o, v, rng)
    weights = execute(programs.weights, weights_feeds).outputs

    jvp_feeds = _feeds(programs.orbital_jvp.program, o, v, rng)
    for name in ("h", "g", "density", "rotation"):
        jvp_feeds[name] = weights_feeds[name]
    jvp = execute(programs.orbital_jvp.program, jvp_feeds).outputs

    fock_program = build_fock_weight_program(o, v, explicit_density_input=True)
    fock_feeds = _feeds(fock_program, o, v, rng)
    for name in ("h", "g", "density", "rotation"):
        fock_feeds[name] = weights_feeds[name]
    fock = execute(fock_program, fock_feeds).outputs
    return (weights_feeds, weights), (jvp_feeds, jvp), (fock_feeds, fock)


def _case_cpp(o: int, v: int) -> str:
    groups = _reference(o, v)
    declarations: list[str] = []
    sections: list[str] = []
    structs = ("HamiltonianWeightInputs", "OrbitalJvpInputs", "FockWeightInputs")
    runners = (
        "run_hamiltonian_weights_cpu",
        "run_orbital_jvp_cpu",
        "run_fock_weights_cpu",
    )
    arenas = (
        "hamiltonian_weights_arena_elements",
        "orbital_jvp_arena_elements",
        "fock_weights_arena_elements",
    )
    for group, ((feeds, expected), struct, runner, arena) in enumerate(
        zip(groups, structs, runners, arenas, strict=True)
    ):
        input_names = sorted(feeds)
        for name in input_names:
            declarations.append(_array(f"g{group}_{name}", feeds[name]))
        for name, value in expected.items():
            declarations.append(_array(f"g{group}_expected_{name}", value))
        section = [
            f"vibeqc::cc::generated::{struct} in{group}{{}};",
            *[f"in{group}.{name}=g{group}_{name};" for name in input_names],
            f"std::vector<double> arena{group}({arena}(o,v));",
            f"auto out{group}={runner}(o,v,in{group},arena{group}.data(),arena{group}.size());",
        ]
        for name, value in expected.items():
            size = np.asarray(value).size
            section.append(
                f"if(!close(out{group}.{name},g{group}_expected_{name},{size})) "
                f"return {10 * (group + 1) + len(section)};"
            )
        sections.extend(section)
    return "\n".join(
        [
            f"static int case_{o}_{v}(){{",
            *declarations,
            "using namespace vibeqc::cc::generated;",
            f"constexpr std::size_t o={o},v={v};",
            *sections,
            "return 0;",
            "}",
        ]
    )


CPP_PREFIX = r"""
#include "generated_rccsd_cpu.hpp"
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>
static bool close(const double* actual,const double* expected,std::size_t n){
  for(std::size_t i=0;i<n;++i)
    if(!std::isfinite(actual[i]) || !std::isfinite(expected[i]) || std::abs(actual[i]-expected[i])>3e-11*(1.0+std::abs(expected[i]))) {
      std::cerr<<"mismatch "<<i<<" "<<actual[i]<<" "<<expected[i]<<"\n";
      return false;
    }
  return true;
}
"""


def test_runtime_shape_hamiltonian_response_matches_tensorir(tmp_path: Path) -> None:
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
        timeout=90,
    )
    cases = ((1, 2), (2, 2))
    source = tmp_path / "hamiltonian.cpp"
    source.write_text(
        CPP_PREFIX
        + "\n".join(_case_cpp(o, v) for o, v in cases)
        + "\nint main(){"
        + "".join(f"if(const int rc=case_{o}_{v}()) return rc;" for o, v in cases)
        + 'std::cout<<"runtime-shape Hamiltonian TensorIR parity passed\\n";return 0;}\n'
    )
    executable = tmp_path / "hamiltonian"
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
    assert "runtime-shape Hamiltonian TensorIR parity passed" in result.stdout
