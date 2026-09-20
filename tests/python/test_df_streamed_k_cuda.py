"""Executed raw work and both K routes agree with independently qualified PySCF D."""

import json
import os
import shutil
import subprocess
import typing
from pathlib import Path

import numpy as np
import pytest

from benchmarks.issue308_stage_probe import prepare

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)


@pytest.fixture(scope="module")
def fixed_density_probe(tmp_path_factory: typing.Any) -> typing.Any:
    assert os.environ.get("SLURM_JOB_ID")
    pytest.importorskip("pyscf")
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("host C++ compiler unavailable")
    library = Path(os.environ["VIBEQC_LIBRARY"]).resolve()
    root = Path(__file__).resolve().parents[2]
    directory = tmp_path_factory.mktemp("streamed-k")
    fixture = directory / "input"
    prepare("water-tetramer-def2-svp-spherical", fixture, None)
    binary = directory / "probe"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-I" + str(root / "src"),
            "-I" + str(root / "include"),
            str(root / "benchmarks/df_stage_probe.cpp"),
            str(library),
            "-Wl,-rpath," + str(library.parent),
            "-ldl",
            "-o",
            str(binary),
        ],
        check=True,
    )
    return binary, fixture


@pytest.mark.parametrize(
    "pairs,auxiliary,passes,occupied_rows,retained",
    [
        (8192, 96, 2, 96, False),
        (3552, 13, 20, 24, False),
        (9216, 5, 0, 0, True),
        (9216, 1, 0, 0, True),
    ],
)
@pytest.mark.parametrize("exchange", ["auto", "full"])
def test_streamed_raw_reuse_matches_independent_jk(
    fixed_density_probe: typing.Any,
    pairs: typing.Any,
    auxiliary: typing.Any,
    passes: typing.Any,
    occupied_rows: typing.Any,
    retained: typing.Any,
    exchange: typing.Any,
    tmp_path: typing.Any,
) -> None:
    binary, fixture = fixed_density_probe
    trace = tmp_path / "trace.jsonl"
    arrays = tmp_path / "arrays.bin"
    env = dict(
        os.environ, VIBEQC_DF_TRACE=str(trace), VIBEQC_DF_RESIDENT_EXCHANGE=exchange
    )
    subprocess.run(
        [
            str(binary),
            str(fixture / "input.txt"),
            "1",
            str(auxiliary),
            str(pairs),
            str(arrays),
            "all",
            str(int(retained)),
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    actual = np.fromfile(arrays).reshape(4, 96, 96)
    with np.load(fixture / "reference.npz") as reference:
        for values, key in zip(actual, ["density", "j", "k", "k"], strict=True):
            np.testing.assert_allclose(values, reference[key], atol=1e-9, rtol=0)
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    contractions = [r for r in records if r["operation"] in ("ri_k", "ri_k_occupied")]
    assert len(contractions) == 4
    for record in contractions:
        assert record["execution"] == "stream"
        counts = record["counters"]
        assert record["streamed"] != retained
        source_elements = 96**3 * passes
        if not retained and record["operation"] == "ri_k_occupied":
            # All-Q occupied projections need one block (96 rows) or four
            # 24-row blocks. Full K rereads each row four times; triangular
            # K needs 24*(4+3+2+1) generated rows, including diagonal reuse.
            blocks = 96 // occupied_rows
            generated_rows = (
                occupied_rows * blocks * (blocks + 1) // 2
                if exchange == "auto"
                else 96 * blocks
            )
            source_elements = generated_rows * 96 * 96
            assert counts["streamed_occupied_source_first"] == 1
            assert counts["streamed_occupied_raw_generation_rows"] == generated_rows
            assert counts["streamed_occupied_row_blocks"] == blocks
            assert counts["streamed_occupied_retained_projection_capacity_bytes"] == (
                2 * occupied_rows * 20 * 96 * 8
            )
        assert (
            counts.get("raw_panel_source_auxiliary_evaluations", 0) == source_elements
        )
        # Q=5 has a final Q=1 tail with room for P=5 raw reuse. The old
        # <=4 branch silently repeated recurrences on that tail.
        assert counts.get("fused_source_auxiliary_evaluations", 0) == 0
        assert counts.get("fused_metric_panel_productions", 0) == 0
    for record in (r for r in records if r["operation"] == "ri_j"):
        assert record["counters"].get("transformed_tile_productions", 0) == 0
    if retained:
        setup = [
            r
            for r in records
            if r["operation"] == "resident_three_center_materialization"
        ]
        assert len(setup) == 1
        assert setup[0]["counters"]["raw_value_bytes"] == 8 * 96**3
