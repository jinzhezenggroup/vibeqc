"""Compare actual streamed XC boundary retention on separately selected checkouts.

Run with PYTHONPATH pointing at the checkout under test and VIBEQC_LIBRARY at an
explicit CPU library. Instrumented lifetime checks and ordinary endpoint timings
are separate. This reports no process/device peak or SCF/GPU speedup.
"""

import argparse
import json
import os
import shutil
import statistics
import tempfile
import weakref
from pathlib import Path
from time import perf_counter

import numpy as np
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.common.provenance import file_hash
from vibeqc_compiler.dft import NativeAO
from vibeqc_compiler.dft.fixtures import basis_arguments
from vibeqc_compiler.xc import functional
from vibeqc_compiler.xc.integration_fixtures import load_integration_fixture
from vibeqc_compiler.xc.native import NativeContractionProgram
from vibeqc_compiler.xc.prepared import PreparedXCContractions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("h2", "f_spherical"), default="h2")
    parser.add_argument("--functional", choices=("LDA_XC_PW", "PBE"), default="PBE")
    parser.add_argument("--tile-points", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=11)
    args = parser.parse_args()
    if args.repeats < 3 or args.tile_points < 1:
        parser.error("repeats must be at least 3 and tile-points positive")
    compiler = shutil.which("c++")
    if compiler is None or not os.environ.get("VIBEQC_LIBRARY"):
        parser.error("c++ and an explicit VIBEQC_LIBRARY are required")
    meta, data, grid = load_integration_fixture(args.case)
    refs, before = [], []
    with tempfile.TemporaryDirectory(prefix="programir-xc-") as cache:
        program = NativeContractionProgram(
            functional(args.functional),
            compiler=CppCompilerAdapter(Path(compiler)),
            cache=cache,
        )
        with (
            NativeAO(**basis_arguments(meta)) as basis,
            PreparedXCContractions(
                program, basis, grid, tile_points=args.tile_points
            ) as prepared,
        ):
            original_ao, original_xc = basis.evaluate, program.evaluate

            def track_ao(*a, **kw):
                before.append(
                    {
                        "live_boundary_arrays": sum(
                            ref() is not None for _, _, ref in refs
                        ),
                        "live_boundary_bytes": sum(
                            size for _, size, ref in refs if ref() is not None
                        ),
                    }
                )
                jets = original_ao(*a, **kw)
                refs.append(("jets", jets.nbytes, weakref.ref(jets)))
                return jets

            def track_xc(*a, **kw):
                result = original_xc(*a, **kw)
                for name in ("potential", "electrons"):
                    value = result[name]
                    refs.append((name, value.nbytes, weakref.ref(value)))
                return result

            basis.evaluate, program.evaluate = track_ao, track_xc
            try:
                actual = prepared.execute(data["density_spin"])
            finally:
                basis.evaluate, program.evaluate = original_ao, original_xc
            np.testing.assert_allclose(
                actual["energy"],
                data[f"{args.functional}_spin_energy"][0],
                rtol=1e-10,
                atol=1e-11,
            )
            np.testing.assert_allclose(
                actual["potential"],
                data[f"{args.functional}_spin_potential"],
                rtol=1e-10,
                atol=1e-11,
            )
            energy = actual["energy"]
            del actual
            # Warm once, then time complete fixed-density E/V executions with
            # no weakref hooks, assertions, compilation or retained history.
            prepared.execute(data["density_spin"])
            samples = []
            for _ in range(args.repeats):
                started = perf_counter()
                result = prepared.execute(data["density_spin"])
                samples.append(perf_counter() - started)
                del result
            report = {
                "case": args.case,
                "functional": args.functional,
                "nao": basis.nao,
                "points": len(grid.weights),
                "tile_points": args.tile_points,
                "energy": energy,
                "before_collocation": before,
                "warm_complete_endpoint_seconds": samples,
                "warm_median_seconds": statistics.median(samples),
                "semantic_work": {
                    k: v
                    for k, v in prepared.statistics.items()
                    if k.endswith(("calls", "products")) or k == "tiles"
                },
                "host_source_hashes": program.metadata["host_source_hashes"],
                "native_library_sha256": file_hash(Path(os.environ["VIBEQC_LIBRARY"])),
                "scope": "live returned AO/potential/electron ndarray payload before each new collocation; excludes provider interiors, caller history, allocator/RSS and GPU/SCF claims",
            }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
