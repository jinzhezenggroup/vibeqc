"""Measure complete CPU XC endpoints and independent derivative projections.

Requires a clean source tree, matching native library and pinned development-only
PySCF/Libxc. External oracle calls occur outside production timing. Native
point code uses the existing shared cache; fresh --cache records cold compilation.
"""

import argparse
import ctypes
import json
import os
import platform
import resource
import subprocess
import sys
import time
import tracemalloc
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

import numpy as np
from vibeqc.autotune import source_identity
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.common.evidence import block_error
from vibeqc_compiler.common.provenance import canonical_hash, file_hash
from vibeqc_compiler.common.resources import ResourceBudget
from vibeqc_compiler.dft import NativeAO
from vibeqc_compiler.dft.ao import jet_indices
from vibeqc_compiler.dft.fixtures import basis_arguments
from vibeqc_compiler.dft.spatial import SpatialPolicy
from vibeqc_compiler.dft.spatial_prepared import PreparedSpatialGrid
from vibeqc_compiler.xc import functional
from vibeqc_compiler.xc.contractions import ContractionProgram
from vibeqc_compiler.xc.integration_fixtures import load_integration_fixture
from vibeqc_compiler.xc.native import NativeContractionProgram
from vibeqc_compiler.xc.prepared import PreparedXCContractions

from tools.generate_validation_references import pyscf_molecule

CASES = ("h2", "f_cartesian", "f_spherical", "separated_f")
FUNCTIONALS = {"LDA_XC_PW": "LDA_X,LDA_C_PW", "PBE": "GGA_X_PBE,GGA_C_PBE"}
OBSERVABLES = ("energy", "potential", "response", "geometry")
STEPS = (1e-3, 3e-4, 1e-4)


def capture(argv):
    return subprocess.check_output(argv, text=True, timeout=60).strip()


def gate(actual, expected, *, atol=1e-11, rtol=1e-10):
    error = block_error(
        np.atleast_1d(actual), np.atleast_1d(expected), atol=atol, rtol=rtol
    )
    if not error["passed"]:
        raise AssertionError(error)
    return error


def compare(actual, expected):
    assert set(actual) == set(expected)
    errors = {}
    for key, value in actual.items():
        if key == "geometry":
            errors.update(
                {
                    key + "/" + field: gate(
                        getattr(value, field), getattr(expected[key], field)
                    )
                    for field in ("centers", "points", "weights")
                }
            )
        else:
            errors[key] = gate(value, expected[key])
    return errors


def oracle(inputs, points, weights, density, code):
    """Independent normalized AO, feature, Libxc and AO-potential evaluation."""
    from pyscf.dft import gen_grid, numint

    mol, scale, _ = pyscf_molecule(inputs)
    grid = gen_grid.Grids(mol)
    grid.coords, grid.weights, grid.non0tab, grid.cutoff = points, weights, None, 1e-30
    ni = numint.NumInt()
    ni.cutoff = 1e-30
    dm = density * scale[:, None] * scale[None, :]
    _, energy, matrix = ni.nr_uks(mol, grid, code, dm, hermi=1)
    return float(energy), matrix * scale[:, None] * scale[None, :]


def load_case(case):
    """Use pinned fixtures plus a reproducible case with genuinely local f masks."""
    if case != "separated_f":
        return load_integration_fixture(case)
    from vibeqc_compiler.dft import ExplicitGrid

    rng = np.random.default_rng(23640)
    centers = np.array([[8.0 * i, 0.1 * (i % 2), 0] for i in range(4)])
    inputs = {
        "name": case,
        "atomic_numbers": [1] * 4,
        "coordinates": centers.tolist(),
        "shells": [
            {
                "atom_index": i,
                "angular_momentum": angular,
                "primitives": [[0.7, 1.0], [1.5, -0.1]],
            }
            for i in range(4)
            for angular in (0, 1, 3)
        ],
        "basis_representation": "cartesian",
        "charge": 0,
        "multiplicity": 1,
    }
    points = np.concatenate([r + 0.2 * rng.normal(size=(16, 3)) for r in centers])
    weights = np.full(len(points), 0.03)
    grid = ExplicitGrid(
        points, weights, tuple(np.repeat(np.arange(4), 16).tolist()), {"case": case}
    )
    matrix = rng.normal(size=(2, 56, 7)) * 0.05
    density = matrix @ matrix.swapaxes(-1, -2) + 0.2 * np.eye(56)[None]
    data = {"density_spin": density}
    for name, code in FUNCTIONALS.items():
        energy, potential = oracle(inputs, points, weights, density, code)
        data[name + "_spin_energy"] = np.array([energy])
        data[name + "_spin_potential"] = potential
    return {"inputs": inputs, "inputs_hash": canonical_hash(inputs)}, data, grid


def ao_atoms(basis):
    return np.repeat(
        [s.atom_index for s in basis.shells],
        [
            2 * s.angular_momentum + 1
            if basis.representation == "real_spherical"
            else (s.angular_momentum + 1) * (s.angular_momentum + 2) // 2
            for s in basis.shells
        ],
    )


def independent_derivatives(meta, data, grid, basis, programs, name):
    """Retain raw external finite-difference projections at every declared step."""
    from tools.vibeqc_posthf.pair_space import PairSpace

    density = data["density_spin"]
    rng = np.random.default_rng(23610)
    directions = rng.normal(size=(2, *density.shape)) * 0.015
    directions = (directions + directions.swapaxes(-1, -2)) / 2
    values = {}
    for observable in ("potential", "response", "geometry"):
        program = programs[name + "/" + observable]
        options = (
            {"delta_density": directions[0]}
            if observable == "response"
            else {"ao_atoms": ao_atoms(basis), "natom": basis.natom}
            if observable == "geometry"
            else {}
        )
        values[observable] = program.evaluate(
            basis.evaluate(grid.points, program.contract.ao_order),
            density,
            grid.weights,
            **options,
        )
    rows = []
    potential = values["potential"]["potential"]
    action = values["response"]["response"]
    for step in STEPS:
        plus = oracle(
            meta["inputs"],
            grid.points,
            grid.weights,
            density + step * directions[0],
            FUNCTIONALS[name],
        )
        minus = oracle(
            meta["inputs"],
            grid.points,
            grid.weights,
            density - step * directions[0],
            FUNCTIONALS[name],
        )
        for axis, analytic, p, m in (
            ("density", float(np.sum(potential * directions[0])), plus[0], minus[0]),
            (
                "response",
                float(np.sum(action * directions[1])),
                float(np.sum(plus[1] * directions[1])),
                float(np.sum(minus[1] * directions[1])),
            ),
        ):
            finite = (p - m) / (2 * step)
            rows.append(
                {
                    "axis": axis,
                    "step": step,
                    "analytic": analytic,
                    "plus": p,
                    "minus": m,
                    "finite_difference": finite,
                    "error": gate(analytic, finite, atol=3e-8, rtol=0),
                }
            )
    response = programs[name + "/response"]
    jets = basis.evaluate(grid.points, response.contract.ao_order)
    second = response.evaluate(
        jets, density, grid.weights, delta_density=directions[1]
    )["response"]
    pairs = PairSpace(basis.nao)
    pack = lambda array: np.concatenate([pairs.pack(s) for s in array])
    transpose = gate(
        pack(directions[0]) @ pack(second),
        pack(action) @ pack(directions[1]),
        atol=1e-12,
        rtol=1e-10,
    )
    dc = rng.normal(size=(basis.natom, 3)) * 0.03
    dp = rng.normal(size=grid.points.shape) * 0.02
    dw = rng.normal(size=grid.weights.shape) * 0.001
    geometry = values["geometry"]["geometry"]
    for axis, motion in (
        ("centers", (dc, dp * 0, dw * 0)),
        ("points", (dc * 0, dp, dw * 0)),
        ("weights", (dc * 0, dp * 0, dw)),
        ("combined", (dc, dp, dw)),
    ):
        centers, points, weights = motion
        analytic = geometry.directional(centers=centers, points=points, weights=weights)
        for step in STEPS:
            samples = []
            for sign in (1, -1):
                inputs = {
                    **meta["inputs"],
                    "coordinates": (
                        np.asarray(meta["inputs"]["coordinates"])
                        + sign * step * centers
                    ).tolist(),
                }
                samples.append(
                    oracle(
                        inputs,
                        grid.points + sign * step * points,
                        grid.weights + sign * step * weights,
                        density,
                        FUNCTIONALS[name],
                    )[0]
                )
            finite = (samples[0] - samples[1]) / (2 * step)
            rows.append(
                {
                    "axis": axis,
                    "step": step,
                    "analytic": analytic,
                    "plus": samples[0],
                    "minus": samples[1],
                    "finite_difference": finite,
                    "error": gate(analytic, finite, atol=3e-8, rtol=0),
                }
            )
    return {"projections": rows, "svec_transpose": transpose}


def diagnostic(program, basis, grid, density, tile, spatial, direction):
    """Identical-mask arithmetic reference; zeros precede nonlinear evaluation."""
    observable = program.contract.request.observable
    nspin = 2
    result = {"energy": 0.0, "electrons": np.zeros(2)}
    if observable in ("potential", "response"):
        result[observable] = np.zeros((nspin, basis.nao, basis.nao))
    if observable == "geometry":
        from vibeqc_compiler.xc.contractions import GeometryPartials

        centers, points, weights = (
            np.zeros((basis.natom, 3)),
            np.zeros_like(grid.points),
            np.zeros_like(grid.weights),
        )
    maps = (
        [(np.arange(len(grid.points)), None)]
        if spatial is None
        else [(t.point_ids, t.ao_ids) for t in spatial.tasks.tasks]
    )
    for ids, active in maps:
        for begin in range(0, len(ids), tile):
            selected = ids[begin : begin + tile]
            if active is not None and not len(active):
                continue
            jets = basis.evaluate(grid.points[selected], program.contract.ao_order)
            if active is not None:
                jets = jets.copy()
                omitted = np.ones(basis.nao, dtype=bool)
                omitted[active] = False
                jets[:, :, omitted] = 0
            options = (
                {"delta_density": direction}
                if observable == "response"
                else {"ao_atoms": ao_atoms(basis), "natom": basis.natom}
                if observable == "geometry"
                else {}
            )
            value = program.evaluate(jets, density, grid.weights[selected], **options)
            for key in ("energy", "electrons", "potential", "response"):
                if key in result:
                    result[key] += value[key]
            if observable == "geometry":
                centers += value["geometry"].centers
                points[selected] = value["geometry"].points
                weights[selected] = value["geometry"].weights
    if observable == "geometry":
        result["geometry"] = GeometryPartials(centers, points, weights)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists() or args.cache.exists():
        raise FileExistsError("use fresh output and cache paths")
    if args.samples < 3 or capture(["git", "-C", str(ROOT), "status", "--porcelain"]):
        raise ValueError("at least three samples and a clean source checkout required")
    os.environ["VIBEQC_LIBRARY"] = str(args.library.resolve())
    library = ctypes.CDLL(str(args.library.resolve()))
    library.vibeqc_get_source_identity.restype = ctypes.c_char_p
    identity = source_identity(ROOT)
    if library.vibeqc_get_source_identity().decode() != identity:
        raise ValueError("native library/source identity mismatch")
    import pyscf
    from pyscf.dft import libxc, numint

    if pyscf.__version__ != "2.14.0" or libxc.__version__ != "7.0.0":
        raise RuntimeError("independent oracle requires PySCF 2.14.0 / Libxc 7.0.0")
    report = {
        "schema": "vibeqc.xc-contraction-benchmark.v1",
        "revision": capture(["git", "rev-parse", "HEAD"]),
        "dirty": False,
        "source_identity": identity,
        "library_sha256": file_hash(args.library),
        "hardware": platform.platform()
        + "\n"
        + next(
            line
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        "toolchain": {
            "python": sys.version,
            "numpy": np.__version__,
            "cxx": capture(["c++", "--version"]),
            "pyscf": pyscf.__version__,
            "libxc": libxc.__version__,
            "numint_sha256": file_hash(Path(numint.__file__)),
            "threads": {
                k: os.environ.get(k)
                for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            },
        },
        "programs": {},
        "cases": [],
        "independent_derivatives": {},
    }
    programs = {}
    for name in FUNCTIONALS:
        for observable in OBSERVABLES:
            started = time.perf_counter()
            program = NativeContractionProgram(
                functional(name),
                observable,
                compiler=CppCompilerAdapter(Path("c++")),
                cache=args.cache,
            )
            elapsed = time.perf_counter() - started
            key = name + "/" + observable
            programs[key] = program
            compact = ContractionProgram(functional(name), observable)
            coefficients = compact.response_coefficients or compact.coefficients
            report["programs"][key] = {
                "metadata": program.metadata,
                "artifact": program.artifact.metadata,
                "construction_seconds": elapsed,
                "source_bytes": program.metadata["source_bytes"],
                "coefficients_before": coefficients.before,
                "coefficients_after": coefficients.after,
                "ao_pullback_before": None
                if compact.jet_pullback is None
                else compact.jet_pullback.before,
                "ao_pullback_after": None
                if compact.jet_pullback is None
                else compact.jet_pullback.after,
            }
    for case in CASES:
        meta, data, grid = load_case(case)
        density, direction = data["density_spin"], 0.02 * data["density_spin"]
        with NativeAO(**basis_arguments(meta)) as basis:
            for name in FUNCTIONALS:
                report["independent_derivatives"][case + "/" + name] = (
                    independent_derivatives(meta, data, grid, basis, programs, name)
                )
        for name in FUNCTIONALS:
            for observable in OBSERVABLES:
                program = programs[name + "/" + observable]
                reference = ContractionProgram(functional(name), observable)
                for local in (False, True):
                    for tile, budget in ((7, 4 << 20), (19, 8 << 20)):
                        row = {
                            "case": f"{case}/{name}/{observable}/{'local' if local else 'dense'}/{tile}",
                            "program": name + "/" + observable,
                            "fixture": case,
                            "inputs_hash": meta["inputs_hash"],
                            "grid_identity": grid.identity,
                            "density_hash": canonical_hash(density.tolist()),
                            "tile_points": tile,
                            "budget_bytes": budget,
                            "local": local,
                            "samples": [],
                        }
                        for sample in range(args.samples):
                            with ExitStack() as stack:
                                started = time.perf_counter()
                                basis = stack.enter_context(
                                    NativeAO(**basis_arguments(meta))
                                )
                                spatial = (
                                    stack.enter_context(
                                        PreparedSpatialGrid(
                                            basis,
                                            grid,
                                            policy=SpatialPolicy(
                                                screening="absolute_ao_jet",
                                                cutoff=1e-4,
                                                region_points=5,
                                                derivatives=jet_indices(2),
                                            ),
                                            tile_points=tile,
                                            resource_budget=ResourceBudget(
                                                host_bytes=budget
                                            ),
                                        )
                                    )
                                    if local
                                    else None
                                )
                                prepared = stack.enter_context(
                                    PreparedXCContractions(
                                        program,
                                        basis,
                                        grid,
                                        tile_points=tile,
                                        spatial=spatial,
                                        resource_budget=ResourceBudget(
                                            host_bytes=budget
                                        ),
                                    )
                                )
                                construction = time.perf_counter() - started
                                options = (
                                    {"delta_density": direction}
                                    if observable == "response"
                                    else {}
                                )
                                started = time.perf_counter()
                                result = prepared.execute(density, **options)
                                elapsed = time.perf_counter() - started
                                statistics = dict(prepared.statistics)
                                expected = diagnostic(
                                    reference,
                                    basis,
                                    grid,
                                    density,
                                    tile,
                                    spatial,
                                    direction,
                                )
                                errors = compare(result, expected)
                                if not local:
                                    errors["external_energy"] = gate(
                                        result["energy"], data[name + "_spin_energy"][0]
                                    )
                                    if observable == "potential":
                                        errors["external_potential"] = gate(
                                            result["potential"],
                                            data[name + "_spin_potential"],
                                        )
                                if sample == 0:
                                    row["nao"] = basis.nao
                                    row["approximation_delta"] = {
                                        "energy": result["energy"]
                                        - float(data[name + "_spin_energy"][0]),
                                        "potential_max_abs": float(
                                            np.max(
                                                np.abs(
                                                    result["potential"]
                                                    - data[name + "_spin_potential"]
                                                )
                                            )
                                        )
                                        if observable == "potential"
                                        else None,
                                        "scope": "difference from unscreened independent collocation; diagnostic only, not an approximation error bound",
                                    }
                                    row["resource_plan"] = (
                                        prepared.resource_plan.to_dict()
                                    )
                                    row["mask_identity"] = (
                                        None
                                        if spatial is None
                                        else spatial.tasks.identity
                                    )
                                    row["active_ao_sizes"] = (
                                        [basis.nao]
                                        if spatial is None
                                        else [
                                            len(t.ao_ids) for t in spatial.tasks.tasks
                                        ]
                                    )
                                    tracemalloc.start()
                                    traced_result = prepared.execute(density, **options)
                                    row["traced_execute_peak_bytes"] = (
                                        tracemalloc.get_traced_memory()[1]
                                    )
                                    tracemalloc.stop()
                                    del traced_result
                                row["samples"].append(
                                    {
                                        "sample": sample,
                                        "construction_seconds": construction,
                                        "execute_seconds": elapsed,
                                        "statistics": statistics,
                                        "errors": errors,
                                    }
                                )
                        report["cases"].append(row)
                        print(row["case"], flush=True)
    report["memory"] = {
        "process_lifetime_rss_high_water_bytes": resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        * 1024,
        "scope": "Linux worker lifetime RSS includes Python, DAGs, loaded libraries, all previous cases and independent reference work; not per-endpoint delta. Tracemalloc peaks measure separate untimed calls and include traced Python/NumPy allocations, excluding untraced native/BLAS storage. Resource plans bound numeric capacities with explicit exclusions.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
