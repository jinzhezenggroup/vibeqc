"""Write the reviewed v1 contract from immutable input files and basis expansion."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from vibeqc import Atom, GridPolicy
from vibeqc.calculator import _named_basis_shells
from vibeqc_compiler.dft.grid import molecular_grid_identity

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_digest(data: bytes) -> str:
    """Bind versioned text as the LF Git blob on Windows and Unix alike."""

    return digest(data.replace(b"\r\n", b"\n"))


def canonical(value: dict) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def build() -> dict:
    grid = GridPolicy("tight").resolve("pbe", derivative_order=1)
    basis_bytes = (REPO / "python/vibeqc/data/basis_pack.json").read_bytes()
    cases = {}
    for path in sorted((ROOT / "inputs").glob("*.json")):
        if path.stem.endswith("-changed"):
            continue
        raw = path.read_bytes()
        value = json.loads(raw)
        changed = ROOT / "inputs" / f"{value['id']}-changed.json"
        atoms = [Atom.from_value((element, xyz)) for element, xyz in value["atoms"]]
        changed_atoms = [
            Atom.from_value((element, xyz))
            for element, xyz in json.loads(changed.read_text(encoding="utf-8"))["atoms"]
        ]
        shells = _named_basis_shells("def2-svp", atoms)
        aos = sum(2 * shell.angular_momentum + 1 for shell in shells)
        cases[value["id"]] = {
            "input": f"inputs/{path.name}",
            "input_sha256": source_digest(raw),
            "changed_input": f"inputs/{changed.name}",
            "changed_input_sha256": source_digest(changed.read_bytes()),
            "atom_count": len(atoms),
            "ao_count_spherical": aos,
            "charge": value["charge"],
            "multiplicity": value["multiplicity"],
            "device": "RTX 5090/sm_120",
            "memory_budget_bytes": 20 * 1024**3,
            "grid_identity": molecular_grid_identity(
                atoms, grid, charge=value["charge"], multiplicity=value["multiplicity"]
            ),
            "changed_grid_identity": molecular_grid_identity(
                changed_atoms,
                grid,
                charge=value["charge"],
                multiplicity=value["multiplicity"],
            ),
            "classification": "performance_holdout"
            if value["id"] in {"caffeine", "ace_glygly_nme"}
            else "performance_tuning"
            if value["id"] in {"water8", "water16", "water32"}
            else "correctness_sentinel",
        }
    methods = {
        "pbe": {"version": "PBE-GGA/libxc-7.0", "role": "core_mixed"},
        "r2scan": {"version": "r2SCAN-MGGA/libxc-7.0", "role": "core_mixed"},
        "pbe0": {"version": "PBE0-global-hybrid/libxc-7.0", "role": "core_mixed"},
        "b3lyp": {"version": "B3LYP-VWN-RPA-tail-v1/libxc-7.0", "role": "strict_reuse"},
        "lda": {"version": "LDA-XC-PW/libxc-7.0", "role": "regression"},
    }
    rows = []
    closed = [
        "water",
        "water_dimer",
        "benzene",
        "water8",
        "water16",
        "water32",
        "caffeine",
        "ace_glygly_nme",
    ]
    open_spin = ["oh", "o2"]
    for method, meta in methods.items():
        selected = (
            (closed + open_spin)
            if meta["role"] == "core_mixed"
            else (["water", "benzene", "oh"] if method == "b3lyp" else ["water", "oh"])
        )
        if method == "b3lyp":
            selected.append("o2")  # explicitly optional root-selection stress case
        for case in selected:
            spin = "uks" if case in open_spin else "rks"
            levels = ["fp64_energy", "fp64_energy_forces"]
            if meta["role"] == "core_mixed":
                levels.append("mixed_correct")
                if case in {
                    "water8",
                    "water16",
                    "water32",
                    "caffeine",
                    "ace_glygly_nme",
                }:
                    levels.append("promoted")
            for level in levels:
                rows.append(
                    {
                        "id": f"{method}/{spin}/{case}/{level}",
                        "method": method,
                        "spin": spin,
                        "case": case,
                        "level": level,
                        "required": not (method == "b3lyp" and case == "o2"),
                        "backend": "cuda",
                        "provider": "native-dft",
                        "j_k": "direct",
                        "basis": "def2-svp/spherical",
                        "product": "energy"
                        if level == "fp64_energy"
                        else "energy+analytic_forces",
                    }
                )
    return {
        "schema_version": 1,
        "contract_id": "DFT-MP-v1",
        "version": "1.0.0",
        "source_issue": 1185,
        "product_issue": 1191,
        "final_acceptance_issue": 1190,
        "amendments": [],
        "amendment_policy": "new version and hash; record affected gates and migration mapping in issue and PR",
        "basis": {
            "name": "def2-svp",
            "representation": "real_spherical",
            "basis_pack_sha256": source_digest(basis_bytes),
            "j_k": "direct",
            "auxiliary": None,
            "all_electron": True,
        },
        "model": {
            "grid_spec": asdict(grid),
            "grid_selection": "explicit_common_pbe_tight_derivative_v2_not_per_method_default",
            "grid_source": "GridPolicy('tight').resolve('pbe', derivative_order=1) at contract freeze",
            "grid_points_weights": "molecular_grid_identity per case",
            "grid_convergence": "independent finer grid required",
            "scf": {
                "max_iterations": 100,
                "energy_tolerance_eh": 1e-10,
                "density_tolerance": 1e-8,
                "physical_residual_max": 1e-8,
                "screening": "production default, receipt must record numeric thresholds",
                "state_selection": "same attained state required",
            },
            "methods": methods,
        },
        "scope": {
            "elements": ["H", "C", "N", "O"],
            "periodic": False,
            "hardware": {
                "device": "RTX 5090",
                "sm": 120,
                "memory_budget_bytes": 20 * 1024**3,
                "driver_toolchain_build": "pin exact identities in campaign before timing",
            },
            "batch_latency": 1,
            "batch_throughput": 4,
            "optional_domains": [
                "density_fitting",
                "def2-tzvp",
                "other_gpu",
                "ecp",
                "periodic",
                "hessians",
            ],
        },
        "gates": {
            "independent_total_energy_abs_eh": 1e-7,
            "independent_force_component_max_eh_per_bohr": 1e-6,
            "mixed_energy_abs_eh": 1e-7,
            "mixed_force_component_max_eh_per_bohr": 1e-6,
            "existing_stricter_gate_wins": True,
            "minimum_paired_samples": 5,
            "preferred_paired_samples": 10,
            "pair_order": "interleaved alternating AB/BA with recorded order and raw samples",
            "statistic": "geometric mean of per-case median strict/mixed ratios; stratified paired bootstrap 10000 resamples per case with seed 1185, percentile 2.5 lower bound",
            "minimum_geomean_speedup": 1.20,
            "minimum_ci95_lower_exclusive": 1.0,
            "maximum_unexplained_case_regression": 0.05,
            "performance_cases": [
                "water8",
                "water16",
                "water32",
                "caffeine",
                "ace_glygly_nme",
            ],
            "timing_boundaries": ["cold", "warm", "changed_geometry"],
            "complete_endpoint": "prepare+SCF+final strict verification/refinement+analytic forces+transfers+synchronization+selection; changed geometry includes rebuild",
            "no_online_scientific_cuda_compile": True,
        },
        "required_ablations": [
            "strict_matched",
            "algorithm_layout_only",
            "arithmetic_only",
            "combined",
        ],
        "required_checks": [
            "independent_energy_force_oracle",
            "independent_grid_convergence",
            "multi_step_reconverged_finite_difference",
            "changed_geometry",
            "warm_replay",
            "batch_isolation",
            "failure_recovery",
            "strict_fallback",
            "public_no_online_compile",
        ],
        "cases": cases,
        "rows": rows,
    }


def main() -> None:
    contract = build()
    contract["contract_sha256"] = digest(canonical(contract))
    (ROOT / "manifest.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
