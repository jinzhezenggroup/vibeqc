"""The native force consumer's actual W and all forces match independent PySCF."""

import json
import os
import shutil
import subprocess
import typing
from pathlib import Path

import pytest

from benchmarks.df_scf_state_checks import validate_scf_export
from benchmarks.issue308_stage_probe import prepare

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)


@pytest.fixture(scope="module")
def scf_export_probe(tmp_path_factory: typing.Any) -> typing.Any:
    assert os.environ.get("SLURM_JOB_ID")
    pytest.importorskip("pyscf")
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("host C++ compiler unavailable")
    root = Path(__file__).resolve().parents[2]
    library = Path(os.environ["VIBEQC_LIBRARY"]).resolve()
    directory = tmp_path_factory.mktemp("force-state")
    fixture = directory / "input"
    prepare("water-def2-svp-spherical", fixture, None)
    binary = directory / "probe"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-I" + str(root / "src"),
            "-I" + str(root / "include"),
            str(root / "benchmarks/df_scf_probe.cpp"),
            str(library),
            "-Wl,-rpath," + str(library.parent),
            "-ldl",
            "-o",
            str(binary),
        ],
        check=True,
    )
    return binary, fixture


@pytest.mark.parametrize("mode", ("cold", "seeded"))
@pytest.mark.parametrize("forces", (False, True))
@pytest.mark.parametrize("budget", (0, 64 << 20))
def test_actual_force_weighted_density(
    scf_export_probe: typing.Any,
    mode: typing.Any,
    forces: typing.Any,
    budget: typing.Any,
    tmp_path: typing.Any,
) -> None:
    binary, fixture = scf_export_probe
    arrays = tmp_path / "state.bin"
    completed = subprocess.run(
        [
            str(binary),
            str(fixture / "input.txt"),
            str(arrays),
            mode,
            "forces" if forces else "energy",
            str(budget),
            "100",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    rows = [json.loads(line) for line in completed.stdout.splitlines()]
    endpoint = next(row for row in rows if row["operation"] == mode)
    checked = validate_scf_export(fixture, arrays, endpoint, forces=forces)
    assert checked["passed"], checked
    if forces:
        assert checked["errors"]["actual_weighted_density_vs_pyscf"] < 1e-8
        assert checked["errors"]["forces_vs_pyscf"] < 1e-8
        assert checked["maximum_net_force"] < 1e-9
