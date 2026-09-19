"""Prepare independent fixed-D inputs and run bounded Slurm-only stage probes.

These are diagnostic experiments, not cold SCF or GPU4PySCF parity results.
The imported orbitals must pass independent overlap and metric-policy checks.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

try:
    from benchmarks._retention import raw_output_path
except ModuleNotFoundError:
    from _retention import raw_output_path

from benchmarks._cases import benchmark_cases
from benchmarks.df_progress_ledger import read_progress, summarize_progress
from benchmarks.issue206_df_force_probe import _source_metadata


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(case_name: str, output: Path, checkpoint: Path | None) -> None:
    """Use libcint/PySCF for independent S, D, J/K and metric policy validation."""
    import pyscf
    from pyscf import df, gto, lib, scf

    lib.num_threads(2)
    output.mkdir(parents=True, exist_ok=False)
    case = benchmark_cases()[case_name]
    mol = gto.M(
        atom=case.atoms, basis=case.pyscf_basis, unit="Bohr", cart=False, verbose=0
    )
    mf = scf.RHF(mol).density_fit(auxbasis=case.pyscf_basis)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 100
    if checkpoint:
        saved_mol = lib.chkfile.load_mol(str(checkpoint))
        assert saved_mol._basis == mol._basis
        np.testing.assert_array_equal(saved_mol.atom_charges(), mol.atom_charges())
        np.testing.assert_allclose(
            saved_mol.atom_coords(), mol.atom_coords(), atol=1e-14, rtol=0
        )
        saved = lib.chkfile.load(str(checkpoint), "scf")
        mf.mo_coeff, mf.mo_occ, mf.mo_energy = (
            saved[k] for k in ("mo_coeff", "mo_occ", "mo_energy")
        )
        energy = float(saved["e_tot"])
    else:
        energy = float(mf.kernel())
        if not mf.converged:
            raise ValueError("independent fixed-D SCF did not converge")
    overlap = mf.get_ovlp()
    density = mf.make_rdm1()
    j, k = mf.get_jk(mol, density)
    fock = mf.get_hcore() + j - 0.5 * k
    commutator = fock @ density @ overlap - overlap @ density @ fock
    residual = fock @ mf.mo_coeff - (overlap @ mf.mo_coeff) * mf.mo_energy
    metric = df.addons.make_auxmol(mol, case.pyscf_basis).intor("int2c2e")
    eigenvalues = np.linalg.eigvalsh(metric)
    # This fixture uses the same full positive metric space for PySCF Cholesky
    # and VibeQC's relative 1e-10 spectral policy. Reject a rank-policy mismatch.
    rank = int(np.count_nonzero(eigenvalues > 1e-10 * eigenvalues[-1]))
    if rank != len(eigenvalues):
        raise ValueError(
            "independent metric rank differs from the native fixture policy"
        )
    errors = {
        "commutator_max": float(np.max(np.abs(commutator))),
        "eigen_residual_max": float(np.max(np.abs(residual))),
        "orthogonality_max": float(
            np.max(np.abs(mf.mo_coeff.T @ overlap @ mf.mo_coeff - np.eye(mol.nao)))
        ),
        "electron_trace_error": float(
            abs(np.einsum("ij,ji", density, overlap) - mol.nelectron)
        ),
        "metric_idempotency_max": float(
            np.max(np.abs(density @ overlap @ density - 2 * density))
        ),
    }
    if max(errors.values()) > 1e-8:
        raise ValueError(f"independent seed is not qualified: {errors}")
    coefficients = mf.mo_coeff[:, mf.mo_occ > 0]
    shells = []
    for atom in range(mol.natm):
        for shell in mol._basis[mol.atom_symbol(atom)]:
            angular, rows = shell[0], shell[1:]
            for contraction in range(1, len(rows[0])):
                shells.append((atom, angular, [(r[0], r[contraction]) for r in rows]))
    with (output / "input.txt").open("x") as stream:
        stream.write(
            f"vibeqc-stage-v1 {mol.natm} {len(shells)} 1 {coefficients.shape[1]}\n"
        )
        for z, xyz in zip(mol.atom_charges(), mol.atom_coords(), strict=True):
            stream.write(f"{z} " + " ".join(format(v, ".17g") for v in xyz) + "\n")
        for atom, angular, primitives in shells:
            stream.write(f"{atom} {angular} {len(primitives)}\n")
            for exponent, coefficient in primitives:
                stream.write(f"{exponent:.17g} {coefficient:.17g}\n")
        np.savetxt(stream, coefficients, fmt="%.17g")
        np.savetxt(stream, overlap, fmt="%.17g")
    np.savez_compressed(
        output / "reference.npz", density=density, j=j, k=k, overlap=overlap
    )
    record = {
        "schema": "vibeqc.issue308.fixed-density-input.v1",
        "case": case_name,
        "nbf": mol.nao,
        "naux": len(eigenvalues),
        "energy": energy,
        "engine": "PySCF",
        "engine_version": pyscf.__version__,
        "metric_relative_threshold": 1e-10,
        "metric_rank": rank,
        "metric_min_max": [float(eigenvalues[0]), float(eigenvalues[-1])],
        "seed_checks": errors,
        "scope": "independently converged fixed D; not cold-start timing",
        "checkpoint_sha256": sha256(checkpoint) if checkpoint else None,
        "input_sha256": sha256(output / "input.txt"),
        "reference_sha256": sha256(output / "reference.npz"),
    }
    (output / "input.json").write_text(json.dumps(record, indent=2) + "\n")


def completed_native_rows(path: Path) -> tuple[list[dict], bool]:
    """A killed unit-buffered C++ writer may stop in the middle of a JSON row."""
    lines = path.read_bytes().splitlines(keepends=True)
    truncated = bool(lines and not lines[-1].endswith(b"\n"))
    if truncated:
        lines.pop()
    return [json.loads(line) for line in lines], truncated


def run(args: argparse.Namespace) -> None:
    """Retain launch identity, bounded disposition, memory samples and partial stages."""
    if not os.environ.get("SLURM_JOB_ID") or "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise RuntimeError("run this probe through a finite srun allocation")
    if args.timeout <= 0:
        raise ValueError("a positive probe timeout is required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    library, probe, fixture = (
        args.library.resolve(),
        args.probe.resolve(),
        args.input.resolve(),
    )
    metadata = json.loads((fixture / "input.json").read_text())
    for name, key in [
        ("input.txt", "input_sha256"),
        ("reference.npz", "reference_sha256"),
    ]:
        if sha256(fixture / name) != metadata[key]:
            raise ValueError("input identity mismatch")
    native = ctypes.CDLL(str(library))
    native.vibeqc_get_source_identity.restype = ctypes.c_char_p
    from vibeqc.autotune import source_identity

    identity = native.vibeqc_get_source_identity().decode()
    if identity != source_identity(Path(__file__).resolve().parents[1]):
        raise ValueError("loaded library does not match the current native source")
    cuda_version = ctypes.c_int()
    if native.cudaRuntimeGetVersion(ctypes.byref(cuda_version)) != 0:
        raise RuntimeError("cannot identify the loaded CUDA runtime")
    env = dict(os.environ)
    ambient = {k: v for k, v in env.items() if k.startswith("VIBEQC_DF_")}
    if any(
        ambient.get(k)
        for k in ("VIBEQC_DF_TRACE", "VIBEQC_DF_HOST_TRACE", "VIBEQC_DF_PROGRESS_TRACE")
    ):
        raise ValueError("trace paths must be owned by this fresh probe")
    if args.progress:
        env["VIBEQC_DF_PROGRESS_TRACE"] = str(output / "progress.jsonl")
        env["VIBEQC_DF_TRACE"] = str(output / "cuda.jsonl")
        if args.scf_mode:
            env["VIBEQC_DF_HOST_TRACE"] = str(output / "host.jsonl")
    env["LD_LIBRARY_PATH"] = str(library.parent) + ":" + env.get("LD_LIBRARY_PATH", "")
    command = [
        str(probe),
        str(fixture / "input.txt"),
        str(args.repeats),
        str(args.auxiliary_tile),
        str(args.ao_pairs),
        str(output / "arrays.bin"),
        args.operation,
    ]
    if args.scf_mode:
        command = [
            str(probe),
            str(fixture / "input.txt"),
            str(output / "arrays.bin"),
            args.scf_mode,
            args.property,
            str(args.scf_budget),
            str(args.max_iterations),
        ]
    elif args.value_budget:
        command.extend(["0", str(args.value_budget)])
    elif args.retain_b:
        command.append("1")
    record = {
        "schema": "vibeqc.issue308.stage-probe.v1",
        "status": "running",
        "command": command,
        "input": metadata,
        "source": _source_metadata(library),
        "native_source_identity": identity,
        "probe_sha256": sha256(probe),
        "cuda_runtime_version": cuda_version.value,
        "slurm_job": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
        "diagnostic_environment": {
            k: v for k, v in env.items() if k.startswith("VIBEQC_DF_")
        },
        "driver": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True,
        ).strip(),
        "scope": "diagnostic wall times with memory sampling; fixed qualified D, not cold SCF or GPU parity",
        "memory_samples": [],
        "independent_checks": "not_run",
    }
    if args.scf_mode:
        record["scope"] = (
            f"Instrumented native RHF {args.scf_mode} {args.property} with physical export; "
            "DF subbudget, not whole-HF global-resource acceptance or GPU4PySCF parity."
        )
    cache = library.parent / "CMakeCache.txt"
    if cache.exists():
        shutil.copyfile(cache, output / "CMakeCache.txt")
        record["cmake_cache_sha256"] = sha256(cache)
    patch = subprocess.check_output(["git", "diff", "--binary", "HEAD"])
    (output / "measured-source.patch").write_bytes(patch)

    def save():
        (output / "result.json").write_text(json.dumps(record, indent=2) + "\n")

    save()
    started = time.monotonic()
    with (
        (output / "native.jsonl").open("x") as stdout,
        (output / "stderr.txt").open("x") as stderr,
    ):
        process = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr)
        record["pid"] = process.pid
        while process.poll() is None:
            if time.monotonic() - started > args.timeout:
                process.kill()
                record["status"] = "timeout"
                break
            procdir = Path(f"/proc/{process.pid}")
            try:
                memory = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                )
                own = [
                    line
                    for line in memory.splitlines()
                    if line.split(",")[0].strip() == str(process.pid)
                ]
                record["memory_samples"].append(
                    {
                        "seconds": time.monotonic() - started,
                        "gpu_process_memory": own,
                        "host": [
                            line
                            for line in (procdir / "status").read_text().splitlines()
                            if line.startswith(("VmRSS:", "VmHWM:"))
                        ],
                    }
                )
                record["loaded_native_paths"] = sorted(
                    {
                        line.split()[-1]
                        for line in (procdir / "maps").read_text().splitlines()
                        if "libvibeqc.so" in line
                    }
                )
                save()
            except (OSError, subprocess.CalledProcessError):
                pass  # Process exit can race the final diagnostic sample.
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
        record["returncode"] = process.wait()
    native_rows, truncated_native = completed_native_rows(output / "native.jsonl")
    record["native_truncated_tail"] = truncated_native
    observed = next(
        (row for row in native_rows if row.get("operation") == "identity"), None
    )
    record["native_rows"] = native_rows
    if (
        observed is None
        or observed["source_identity"] != identity
        or Path(observed["library"]).resolve() != library
    ):
        record["status"] = "library_identity_failed"
        save()
        raise ValueError(
            "probe loaded a different library, or did not report its identity"
        )
    record["seconds"] = time.monotonic() - started
    if record["status"] != "timeout":
        record["status"] = "completed" if record["returncode"] == 0 else "failed"
    if args.progress and (output / "progress.jsonl").exists():
        record["progress"] = summarize_progress(
            read_progress(output / "progress.jsonl")
        )
    if record["returncode"] == 0 and args.scf_mode:
        from benchmarks.df_scf_state_checks import validate_scf_export

        endpoint = next(r for r in native_rows if r["operation"] == args.scf_mode)
        try:
            record["independent_checks"] = validate_scf_export(
                fixture,
                output / "arrays.bin",
                endpoint,
                forces=args.property == "forces",
                checkpoint=args.checkpoint,
                reference_forces=args.reference_forces,
            )
            if not record["independent_checks"]["passed"]:
                record["status"] = "independent_comparison_failed"
        except (ValueError, AssertionError) as error:
            record["status"] = "independent_comparison_failed"
            record["independent_checks"] = {"error": str(error)}
    elif record["returncode"] == 0:
        n = metadata["nbf"]
        fields = [("density", "density")]
        fields.extend(
            (name, key)
            for name, key, mode in [
                ("j", "j", "j"),
                ("dense_k", "k", "dense"),
                ("occupied_k", "k", "occupied"),
            ]
            if args.operation in ("all", mode)
        )
        values = np.fromfile(output / "arrays.bin").reshape(len(fields), n, n)
        with np.load(fixture / "reference.npz") as reference:
            errors = {
                name: float(np.max(np.abs(value - reference[key])))
                for (name, key), value in zip(fields, values, strict=True)
            }
        record["independent_checks"] = errors
        if not np.all(np.isfinite(values)) or max(errors.values()) > 1e-9:
            record["status"] = "independent_comparison_failed"
    save()
    print(
        json.dumps({k: record[k] for k in ("status", "seconds", "independent_checks")})
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--case", default="water-tetramer-def2-svp-spherical")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=raw_output_path, required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--scf-mode", choices=("cold", "seeded"))
    parser.add_argument("--property", choices=("energy", "forces"), default="energy")
    parser.add_argument("--scf-budget", type=int, default=0)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--reference-forces", type=Path)
    parser.add_argument("--auxiliary-tile", type=int, default=0)
    parser.add_argument("--ao-pairs", type=int, default=0)
    parser.add_argument(
        "--value-budget",
        type=int,
        default=0,
        help="plan with actual source bytes under this DF value allowance",
    )
    parser.add_argument(
        "--retain-b",
        action="store_true",
        help="retain B with the requested bounded K scratch",
    )
    parser.add_argument(
        "--operation", choices=("all", "setup", "j", "dense", "occupied"), default="all"
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    if args.scf_budget < 0 or args.max_iterations < 1:
        parser.error("SCF budget must be nonnegative and iteration limit positive")
    if args.scf_mode and (
        args.value_budget or args.retain_b or args.ao_pairs or args.auxiliary_tile
    ):
        parser.error(
            "SCF probes use --scf-budget instead of explicit J/K plan controls"
        )
    if args.value_budget < 0 or (
        args.value_budget and (args.ao_pairs or args.auxiliary_tile or args.retain_b)
    ):
        parser.error(
            "--value-budget must be positive and cannot accompany explicit tiles/storage"
        )
    if args.prepare:
        prepare(args.case, args.output, args.checkpoint)
    else:
        if not args.input or not args.library or not args.probe:
            parser.error("--input, --library and --probe are required to run")
        run(args)


if __name__ == "__main__":
    main()
