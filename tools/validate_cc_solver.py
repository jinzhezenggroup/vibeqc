"""Native HF->CCSD acceptance with independent same-Hamiltonian final residuals."""

# Source-tree CLI bootstrap for transitive compiler clients.
import sys as _compiler_sys
from pathlib import Path as _CompilerPath

_compiler_sys.path.insert(
    0, str(_CompilerPath(__file__).resolve().parents[1] / "python")
)

import argparse
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

from tools.cc_endpoint_fixtures import ROOT as DATA
from tools.cc_endpoint_fixtures import load, snapshot_from_fixture, source_arguments
from tools.vibeqc_cc.solver import SolverOptions, solve
from tools.vibeqc_posthf import MOBlock
from tools.vibeqc_posthf.export import export_rhf
from tools.vibeqc_posthf.providers import ConventionalProvider
from tools.vibeqc_posthf.sources import NativeSource
from tools.vibeqc_validation.schema import (
    block_error,
    canonical_hash,
    file_hash,
    new_evidence,
    outcome,
    write_evidence,
)

ROOT = Path(__file__).resolve().parents[1]


def run(output):
    from types import SimpleNamespace

    import pyscf
    from pyscf.cc import rccsd

    if pyscf.__version__ != "2.14.0":
        raise ValueError("validation requires pinned PySCF 2.14.0")
    manifest = json.loads((ROOT / "tools/vibeqc_cc/source_manifest.json").read_text())
    if file_hash(rccsd.__file__) != manifest["files"][0]["sha256"]:
        raise ValueError("independent residual source hash mismatch")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    sources = {
        str(p.relative_to(ROOT)).replace("\\", "/"): file_hash(p)
        for folder in (
            "tools/vibeqc_cc",
            "python/vibeqc_compiler/tensor",
            "tools/vibeqc_posthf",
        )
        for p in sorted((ROOT / folder).glob("*.py"))
    }
    sources["tools/validate_cc_solver.py"] = file_hash(__file__)
    for name in ("h2", "he", "h2o", "nh3", "ch4"):
        meta, a = load(name)
        with NativeSource(**source_arguments(meta["inputs"])) as source:
            for fresh in (False, True):
                s, stats = (
                    export_rhf(source, tolerance=1e-12, max_iterations=150)
                    if fresh
                    else (snapshot_from_fixture(source, meta, a), {})
                )
                label = name + ("-native-HF" if fresh else "-same-C")
                with ConventionalProvider(s, source) as p:
                    result = solve(
                        s,
                        p,
                        options=SolverOptions(
                            energy_tolerance=1e-12, residual_tolerance=1e-10
                        ),
                    )
                    result.write(output / (label + "-state.json"))
                    record = new_evidence(
                        tier="cpu",
                        subject="RCCSD-C/" + label,
                        inputs_hash=canonical_hash(result.replay_inputs),
                    )
                    record.update(
                        revision=subprocess.check_output(
                            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                        ).strip(),
                        backend_selected="numpy-cpu-CCSD/native-HF",
                        device={"kind": "cpu", "machine": platform.machine()},
                        hardware=outcome("pass"),
                        toolchain={
                            "python": platform.python_version(),
                            "numpy": np.__version__,
                            "pyscf": pyscf.__version__,
                        },
                        settings={
                            "fresh_native_HF": fresh,
                            "HF": stats,
                            "source_files": sources,
                            "upstream": manifest,
                            "fixture_sha256": file_hash(DATA / (name + ".npz")),
                            "fixture_metadata_sha256": file_hash(
                                DATA / (name + ".json")
                            ),
                            "reference_id": s.identity,
                            "integral_hash": result.provenance["integral_hash"],
                            "options": result.provenance["options"],
                            "dirty": bool(
                                subprocess.check_output(
                                    ["git", "status", "--porcelain"],
                                    cwd=ROOT,
                                    text=True,
                                ).strip()
                            ),
                        },
                    )
                    record["hashes"].update(
                        equation=result.provenance["equation_hash"],
                        ir=result.provenance["independent_equation_hash"],
                        source=canonical_hash(sources),
                    )
                    record["hash_reasons"]["schedule"] = "CPU reference interpreter"
                    record["solver_iterations"] = list(result.history)
                    record["solver_trace_reason"] = "complete history recorded"
                    record["stages"]["representation"] = outcome("pass")
                    if not result.converged:
                        record["stages"]["numerical"] = outcome("fail", result.reason)
                        write_evidence(output / (label + ".json"), record)
                        raise ValueError(f"CCSD did not converge: {label}")
                    c = s.coefficients
                    independent_g = np.einsum(
                        "uvwx,up,vq,wr,xs->pqrs", a["ao"], c, c, c, c, optimize=True
                    )
                    eris = rccsd._ChemistsERIs()
                    eris.fock = c.T @ s.fock @ c
                    eris.mo_energy = s.orbital_energies
                    for blockname in (
                        "oooo",
                        "ovov",
                        "oovv",
                        "ovoo",
                        "ovvo",
                        "ovvv",
                        "vvvv",
                    ):
                        block = MOBlock.from_spaces(s, blockname)
                        g = independent_g[np.ix_(*block.slots)]
                        setattr(eris, blockname, g)
                        record["block_errors"]["integrals/" + blockname] = block_error(
                            p.get(block).to_host(), g, atol=1e-11, rtol=1e-10
                        )
                    u1, u2 = rccsd.update_amps(
                        SimpleNamespace(level_shift=0.0, cc2=False),
                        result.t1,
                        result.t2,
                        eris,
                    )
                    o = s.nocc
                    d1 = s.orbital_energies[:o, None] - s.orbital_energies[None, o:]
                    d2 = d1[:, None, :, None] + d1[None, :, None, :]
                    r1 = d1 * (u1 - result.t1)
                    r2 = d2 * (u2 - result.t2)
                    e = rccsd.energy(None, result.t1, result.t2, eris)
                    record["residuals"] = {
                        "independent_r1_max": float(np.max(np.abs(r1))),
                        "independent_r2_max": float(np.max(np.abs(r2))),
                    }
                    record["block_errors"]["energy_same_amplitudes"] = block_error(
                        np.array(result.correlation_energy),
                        np.array(e),
                        atol=1e-8,
                        rtol=0,
                    )
                    record["block_errors"]["total_energy"] = block_error(
                        np.array(result.total_energy),
                        np.array(meta["total_energy"]),
                        atol=1e-8,
                        rtol=0,
                    )
                    if "fci_total_energy" in meta:
                        record["block_errors"]["two_electron_FCI"] = block_error(
                            np.array(result.total_energy),
                            np.array(meta["fci_total_energy"]),
                            atol=1e-8,
                            rtol=0,
                        )
                    U = a["C"].T @ a["S"] @ c
                    record["settings"]["occupied_virtual_mixing"] = float(
                        np.max(np.abs(U[:o, o:]))
                    )
                    t1 = np.einsum("ki,kc,ca->ia", U[:o, :o], a["t1"], U[o:, o:])
                    t2 = np.einsum(
                        "ki,lj,klcd,ca,db->ijab",
                        U[:o, :o],
                        U[:o, :o],
                        a["t2"],
                        U[o:, o:],
                        U[o:, o:],
                    )
                    for key, actual, expected in (
                        ("t1", result.t1, t1),
                        ("t2", result.t2, t2),
                    ):
                        record["block_errors"]["aligned_" + key] = block_error(
                            actual, expected, atol=1e-8, rtol=1e-8
                        )
                    passed = (
                        all(v["passed"] for v in record["block_errors"].values())
                        and max(record["residuals"].values()) <= 1e-9
                        and record["settings"]["occupied_virtual_mixing"] < 1e-7
                    )
                    record["stages"]["numerical"] = outcome(
                        "pass" if passed else "fail",
                        None if passed else "independent endpoint gate failed",
                    )
                    record["stages"]["endpoint"] = outcome(
                        "pass" if passed else "fail",
                        None if passed else "independent endpoint gate failed",
                    )
                    write_evidence(output / (label + ".json"), record)
                    records.append(record)
                    print(
                        label,
                        result.status,
                        result.total_energy,
                        record["residuals"],
                        passed,
                        flush=True,
                    )
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = run(args.output)
    raise SystemExit(
        0 if all(r["stages"]["endpoint"]["status"] == "pass" for r in records) else 1
    )
