import json
import os
import py_compile
import subprocess
import sys
from importlib.machinery import SourceFileLoader
from importlib.util import cache_from_source
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tools.dft_mp_v1 import qualify_capacity

ROOT = Path(__file__).resolve().parents[2]
SOURCE_SHA = "f" * 40
SPD_CONTRACT_FILES = (
    "src/molecule/basis.cpp",
    "src/dft/ao_grid.cpp",
    "src/dft/bridge.cpp",
    "python/generativeqc/_stationary_cuda.py",
)
PUBLIC_ROUTE_FILES = (
    "python/generativeqc/calculator.py",
    "python/generativeqc/batch.py",
)
GRID_CONTRACT_FILES = (
    "python/generativeqc/ks.py",
    "python/generativeqc_compiler/dft/grid.py",
    "python/generativeqc_compiler/xc/quadrature_cuda.py",
    "src/dft/cuda_quadrature.cu",
    "src/dft/grid.hpp",
    "src/methods/dft_method.cpp",
)


def report(*, aot_directory: Path | None = None) -> dict:
    return qualify_capacity._build_report(
        ROOT,
        source_sha=SOURCE_SHA,
        aot_directory=aot_directory,
    )


def spd_contract_tree(tmp_path: Path) -> None:
    for relative in SPD_CONTRACT_FILES:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            (ROOT / relative).read_text(encoding="utf-8"), encoding="utf-8"
        )


def copy_contract_files(tmp_path: Path, files: tuple[str, ...]) -> None:
    for relative in files:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            (ROOT / relative).read_text(encoding="utf-8"), encoding="utf-8"
        )


def stationary_contract_tree(tmp_path: Path, source: str) -> None:
    copy_contract_files(
        tmp_path,
        (
            "src/dft/stationary_gradient_cuda.cuh",
            "python/generativeqc/_ks_snapshot.py",
            "python/generativeqc/_snapshot_grid_cache.py",
            "python/generativeqc_compiler/method/stationary_resources.py",
        ),
    )
    target = tmp_path / "python/generativeqc/_stationary_cuda.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")


def test_frozen_capacity_report_uses_actual_basis_and_grid_identities() -> None:
    result = report()

    assert result["schema"] == "generativeqc.dft-mp-v1.stationary-capacity.v2"
    assert result["source"]["sha"] == SOURCE_SHA
    assert result["source"]["qualifier_sha256"] == qualify_capacity._lf_sha256(
        (ROOT / "tools/dft_mp_v1/qualify_capacity.py").read_bytes()
    )
    assert result["source"]["imported_module_sha256"][
        "generativeqc_compiler.dft.ao"
    ] == qualify_capacity._lf_sha256(
        (ROOT / "python/generativeqc_compiler/dft/ao.py").read_bytes()
    )
    assert result["contract"] == {
        "id": "DFT-MP-v1",
        "version": "1.0.0",
        "sha256": "de6c847b1ed93e537c1422679ae3df53a4cca3cd13e7268483e419afbf2faa00",
    }

    assert result["basis"]["manifest_ao_counts_match"] is True
    assert result["basis"]["basis_pack_sha256_match"] is True
    assert result["basis"]["packed_capacity_definition"] == (
        "np.empty(3 * self.natom + 2 * self.nprimitive + 16 * self.nao)"
    )
    assert result["basis"]["numeric_capacity_definition"] == (
        "2 * self.packed.nbytes + 32 * self.natom + "
        "32 * len(self.shells) + 16 * self.nprimitive"
    )
    assert result["basis"]["spd_expansion_contract_sha256"] == (
        "a482b3ace40fd01758f504c1e47bfa795dcabb5483149873fcd36e71c6b47ae0"
    )
    assert result["basis"]["sparse_spherical_component_terms"] == {
        "s": 1,
        "p": 3,
        "d": 8,
    }
    assert result["basis"]["ao_packer_contract_sha256"] == (
        "07858ba7f9a78fe6348bbcb9430eb4f8321db8774ea3ce1ecef495629abe2a1c"
    )
    assert result["basis"]["ao_pack_bridge_contract_sha256"] == (
        "aba8691ce81438c66517604c5b2e1e0d39df474888763b4d44686ca8816fabbc"
    )
    assert result["basis"]["native_ao_constructor_contract_sha256"] == (
        "9e210b215cd43f21af6899b328ee738b87f53eaa9b20654b16de760274a8c372"
    )
    assert result["basis"]["basis_snapshot_contract_sha256"] == (
        "4dea9a2041897bf843012c01f64c580b6b8696f444611a54c0d161141f1894fd"
    )
    assert result["basis"]["basis_shell_expansion_contract_sha256"] == (
        "300a64c1815273cf31ed5b463eac2e32f24bf4a1db5938975ffbad5cfb27f60c"
    )
    assert result["basis"]["calculator_shell_forwarding_contract_sha256"] == (
        "af7bd2da7d571fb6d92dee7f984bf8de95f32b263c8b75fe69555772d60a44b6"
    )
    assert result["basis"]["native_system_basis_forwarding_contract_sha256"] == (
        "120936d57b90062a8a888892ce4d514ca8a62a0b2ced671afe3fce660b0414d5"
    )
    assert result["basis"]["production_shell_expansion"] == (
        "snapshot_basis('def2-svp', 'spherical').shells_for(atoms)"
    )
    assert result["basis"]["stationary_layout_contract_sha256"] == (
        "fa8c4ff2a644fd45ab4eb828a995c4e42c49c80adbe712b32d50f90b3d98fb74"
    )
    assert result["basis"]["native_spherical_ao_count_contract_sha256"] == (
        "b6e7a3a70accf7f4abeb82f0168634ae33b7c58f282044b8a9cd0462672200f0"
    )
    assert result["grid"] == {
        "source_only_molecular_grid_sha256": (
            "03a43444cd793167823c0c30c0b66b51c2a464d8f65946118dd781813dc7f0a4"
        ),
        "public_grid_abi_sha256": (
            "b64b0ca7be75b9425c32220476d5eda9b5f013168bd68c856a448fdd320a679e"
        ),
        "native_grid_abi_sha256": (
            "d4930bf86b781cd4a77f152380439ac8a6b168d1846f42325bb2d6a3e7e638e4"
        ),
        "generated_quadrature_layout_sha256": (
            "1674152aa312f3769d8b78be60aff491cc52f577d74b8d889d133bff3064ff23"
        ),
        "native_cuda_grid_sha256": (
            "0f5c74f638f51833f3242d369df019362e0df08c3602f2c2cc645266c46d53ed"
        ),
        "native_cuda_grid_route_sha256": (
            "98b6763435fcd84c6306ea034c330a9864ecf4d33d0009031230f7b64e77b71e"
        ),
        "native_grid_point_count_sha256": (
            "92cd50078b7a96f371ed8d4fcdb77930b8c472134bd1e97bba803ac445d85867"
        ),
        "point_count_definition": (
            "atom_count * radial_points * angular_polar * angular_azimuth"
        ),
    }
    assert result["public_route"] == {
        "whole_grid_work_limits": {"grid_points": None, "grid_pair_visits": None},
        "registry_manifest_sha256": qualify_capacity._lf_sha256(
            (ROOT / "manifests/public_methods.json").read_bytes()
        ),
        "semilocal_force_predicate_sha256": (
            "fba0a84cb3d993919caf6e6d10391239598ef876cda41123d683479fccf767e0"
        ),
        "global_hybrid_force_predicate_sha256": (
            "18f4f010596672eb47b8d085e28b8a26373c41178ac1c6a5ff4fa705ef2f3944"
        ),
        "force_capability_promotion_sha256": (
            "49f903598301e16b11be96d1b24eb084aa7bee3194942702b174b41e59d4b01c"
        ),
        "cuda_force_method_sha256": (
            "dcfcbef93e798c62cc5669e93190a9b73184ffe120a8afc43730a9dbc74cb448"
        ),
        "prepared_aot_selection_sha256": (
            "a9d5f920839f2a020b17addaddb8e00c674bf78e37f112a113751d67ef336224"
        ),
    }

    cases = {item["id"]: item for item in result["cases"]}
    assert set(cases) == {
        "ace_glygly_nme",
        "benzene",
        "caffeine",
        "o2",
        "oh",
        "water",
        "water_dimer",
        "water8",
        "water16",
        "water32",
    }
    assert cases["water8"]["shape"] == {
        "atom_count": 24,
        "ao_count_spherical": 192,
        "shell_count": 96,
        "basis_primitive_count": 176,
        "ao_primitive_count_peak": 5,
        "component_primitive_sum": 328,
        "grid_points": 1_990_656,
    }
    assert cases["water16"]["shape"]["ao_count_spherical"] == 384
    assert cases["water32"]["shape"]["ao_count_spherical"] == 768
    assert cases["caffeine"]["shape"]["ao_count_spherical"] == 246
    assert cases["ace_glygly_nme"]["shape"]["ao_count_spherical"] == 247
    assert cases["water32"]["identities"]["grid"] == (
        "462361d06e44c372a4b116599ecde2e4bf6b6bd4416f26fea2f3b47fa6cb0ece"
    )
    assert cases["water8"]["identities"]["changed_input_sha256"] == (
        "5c2813a5e647040a55cc6e4fc7f4879759c896c906e53bed49472e73dd7e5d69"
    )
    assert cases["water8"]["identities"]["changed_grid"] == (
        "f0f51a6ba5dd355d91f95b0bdcd5d0354b889b3a257a7970c39aaf7c3ce36c0d"
    )
    assert all(item["identities"]["basis"] for item in cases.values())


def test_qualifier_and_dependencies_use_current_source_not_stale_bytecode(
    tmp_path: Path,
) -> None:
    source = tmp_path / "stale_fixture.py"
    bytecode = Path(cache_from_source(str(source)))
    bytecode.parent.mkdir()
    source.write_text("VALUE = 'stale'\n", encoding="utf-8")
    py_compile.compile(
        str(source),
        cfile=str(bytecode),
        doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
    )
    source.write_text("VALUE = 'current'\n", encoding="utf-8")

    stale_module = ModuleType("stale_fixture")
    SourceFileLoader("stale_fixture", str(source)).exec_module(stale_module)
    assert stale_module.VALUE == "stale"  # type: ignore[attr-defined]

    current_module = ModuleType("current_fixture")
    qualify_capacity._DftMpSourceOnlyLoader("current_fixture", str(source)).exec_module(
        current_module
    )
    assert current_module.VALUE == "current"  # type: ignore[attr-defined]


def test_report_exposes_exact_first_gate_and_all_losing_work() -> None:
    result = report()
    cases = {item["id"]: item for item in result["cases"]}

    assert result["admission_limits"]["native_owner_capacity"] == {
        "atom_count": 128,
        "ao_count": 1024,
        "basis_primitive_count": 16384,
    }
    assert result["admission_limits"]["ao_task_fallback_capacity"] == {
        "atom_count": 32,
        "ao_count": 128,
        "basis_primitive_count": 4096,
    }
    assert result["admission_limits"]["primitive_records"] == 16_000_000
    assert (
        result["admission_limits"]["primitive_records_scope"]
        == "per_native_page_on_ao_task_fallback_only"
    )
    assert result["admission_limits"]["primitive_logical_metric_limit"] == (2**64 - 1)
    assert result["admission_limits"]["primitive_page_budget_bindings"] == [
        "max_primitive_records",
        "max_primitive_records",
    ]
    assert result["admission_limits"]["fixed_task_capacity_definition"] == (
        "min(integral_terms, primitive_tile)"
    )
    assert result["admission_limits"]["primitive_page_execution_definition"] == (
        "task_executor.execute_pages(domain, submit_page)"
    )
    assert result["admission_limits"]["primitive_page_contract_sha256"] == {
        "geometry_resources_sha256": "51730953ed9a62807442f672013fcc40787ed3c0b0cca838d6c492c8b1c080b1",
        "public_wrapper_sha256": (
            "56b9ca1e1cd87f51b3cf0d57fcfbb46c8fed4b32a1808fb49e53b545019958dc"
        ),
        "ao_map_reserve_sha256": (
            "c2ba1b47e5655f75196c79384079d4ecd9c1edf9fe9aa1e97e339e96cbbaad0a"
        ),
        "resident_ao_cache_sha256": (
            "dcf02f08e8d0279dcb49f2093d0ac3bf023523a7519a0faf6537fc9140711c9f"
        ),
        "initializer_sha256": (
            "7bb03b7286a5868527104420a74f4749c8af92360f17ad12ced31051a7ae9d75"
        ),
        "flush_sha256": (
            "1c2e0bb83a12eed7113825855cbe2164f53366b6bb270dd6c1247b498737c77b"
        ),
        "bulk_sha256": (
            "b7bc1344bd86447cd6c9efcdfef944bb22c8b92b5ed5327d2028cf787d6a1729"
        ),
        "scalar_sha256": (
            "c5b8ef983462f6c56ebfe6bd6eb8d5cf98f92f36f3e5425504b596730846205f"
        ),
        "component_integral_sha256": (
            "d3f61e820c8bcf0df4bf4fce639f342936b79aceb956cb6e9f43caa3d13cdaa3"
        ),
        "nuclear_sha256": (
            "1e86737d8732ef8637378ab925f829dfe229bcf049219c2705a0a4fbf7afdb85"
        ),
        "geometry_sha256": (
            "d469560a2b776a9b86ff5082ba35d3f8ae956c0d63df76aec1ab39550ab92a30"
        ),
        "finish_span_sha256": (
            "419e21953688eb24d214e6fb43d33ca974cb94632c3797e3f4f0113704d9a1f9"
        ),
        "component_mode_sha256": (
            "8d9819961d3014d161aff8c5c798f926fe6f1d9de54b84a725fdf2f6694b76bb"
        ),
        "executor_sha256": (
            "5f4bf38658ca1da15d845e1a65ebd72195191aebf6009873f131c2cbe8a34195"
        ),
        "submit_page_sha256": (
            "2fcd280569106fe3cbcf1256a02693aa3532e7db0fa62f7c97d7319454a62a48"
        ),
        "nuclear_pair_loop_sha256": (
            "5a69bf4fd85d28b137e1ae35bce4a1d32134375bbaca9f66f60c9377a0c8f935"
        ),
        "endpoint_owner_sha256": (
            "0d9571918d55d9b9d66bd5d40fa8f38d595d1c20c6c58fb04c155467d9c97a81"
        ),
        "native_owner_sha256": (
            "47af7a99e4aadfe4386e1a798e619ff52aea3143a254c7cbffcca08ee9b1c1db"
        ),
        "native_allocation_sha256": (
            "b0e739be97cb1048b86efeaa5c9b116cce1ac76f056e1ac610971f91df08129e"
        ),
        "native_create_sha256": (
            "e3c05c687211842a710c0cada57e96b787992389c5e9a8799191655d07ae05b3"
        ),
        "native_reset_sha256": (
            "e0bdfb373199c744de1dea05e912f7bbd24dff59b7e9a27e08e899bcaec8c1fc"
        ),
        "native_tasks_sha256": (
            "5b0148f4f48019115a82e638d1d6671dd2548f3df6141da6e5254c8967bad2bc"
        ),
        "native_nuclear_sha256": (
            "be4a553ba6117c7f772882a551d50817935954c5c4d66190e86d9bf2be043902"
        ),
        "native_geometry_ao_map_sha256": (
            "d4830d6d9695219f4bf4c59611717b943c7aa1da016fdba67ceb6036241f1dc0"
        ),
        "native_geometry_external_sha256": (
            "e5a36f9b80f332b1e03a48e2b3c066e583b1ba9340b7a5feab4dab3d292be236"
        ),
        "native_geometry_enqueue_sha256": (
            "cdac623e8296338a03b3b81bd3e77fbacb36730d4fb4502a2adbb6921a5f9544"
        ),
        "native_geometry_route_sha256": (
            "3fc0a5f613dfaa01ab02104e15929680f3f61fa17c07d59d54241201f903d476"
        ),
        "native_launch_geometry_sha256": (
            "7a06757b893e51c6924675c88c66c6d0817923904283bf48378fb4e8302f47be"
        ),
        "native_configure_becke_sha256": (
            "dc844781c888d1bdd281238d4dd23c76048d17f816cb81b5a0616756a22ffe91"
        ),
        "native_metrics_sha256": (
            "21e067818117b8ebaf8eeb218aeface0680681fdd6f39285ed6f981c3cef969a"
        ),
        "native_finish_span_sha256": (
            "3f12a2c23709399c56776e34f5d7cd2394a95e153f754694bb7d523772efa431"
        ),
    }
    assert result["admission_limits"]["primitive_records_definition"] == (
        "(1 + int(has_exchange)) * primitive_sum ** 4 + "
        "(na + 2) * primitive_sum ** 2 + na * (na - 1) // 2"
    )
    assert result["admission_limits"]["grid_pair_visits_definition"] == (
        "grid_work.grid_pair_visits"
    )
    assert result["admission_limits"]["grid_derivative_order_definition"] == (
        "'sigma' in ingredients"
    )
    assert result["admission_limits"]["method_ir_definition"] == (
        "state._source.method_ir"
    )
    assert result["admission_limits"]["functional_lowering_definition"] == (
        "int(state._source.metadata[6])"
    )
    assert result["admission_limits"]["functional_ingredients_definition"] == (
        "state._source.functional.ingredients"
    )
    assert result["admission_limits"]["snapshot_functional_contract_sha256"] == {
        "init_sha256": (
            "522c7571c3d18db25685ffbffb55279deadde63df64ee4c8b330f04017f7b3ae"
        ),
        "decode_sha256": (
            "5bd874ce5ba8d4d5d78eb282244144052822bb2f0266d40d33361735302b1a97"
        ),
        "grid_cache_sha256": (
            "569705abf406d2ec00ec9526e84f23301448d5511fc2bf79ee9ef6993a794ca6"
        ),
    }
    assert result["admission_limits"]["grid_plan_definition"] == (
        "plan_tiles(basis, backend='cuda', order=2 if needs_first else 1, "
        "tile_points=tile_points, active_ao_capacity=n, "
        "budget_bytes=max_device_bytes)"
    )
    assert result["admission_limits"]["source_resources_definition"].startswith(
        "plan_stationary_cuda_resources(atoms=na"
    )
    assert result["admission_limits"]["source_bytes_definition"] == (
        "source_resources.allocation_bytes"
    )
    assert result["admission_limits"]["host_bound_definition"].startswith(
        "grid_plan.host_bytes + 8 * (34 * primitive_tile"
    )
    assert result["admission_limits"]["available_device_bytes_definition"] == (
        "max_device_bytes - grid_plan.peak_bytes - minimum_source_bytes"
    )
    assert result["admission_limits"]["additional_device_admission"] == (
        "minimum_additional_device_bytes < additional_device_budget and additional_device_peak_bound <= additional_device_budget"
    )
    assert result["admission_limits"]["gate_order"] == [
        "native_owner_capacity",
        "grid_work_capacity",
        "grid_point_work_budget",
        "grid_pair_work_budget",
        "pending_grid_pair_budget",
        "native_integral_provider_required",
        "primitive_logical_metric_range",
        "additional_device_budget",
        "additional_host_budget",
        "native_integral_result_required",
        "primitive_descriptor_page_budget",
    ]
    assert result["admission_limits"]["gate_predicates"] == {
        "primitive_metric_range": "records > np.iinfo(np.uint64).max",
        "additional_device": "available <= 0",
        "additional_host": "host_bound > max_host_bytes",
    }
    assert result["admission_limits"]["tile_points"] == 256
    assert result["admission_limits"]["primitive_tile"] == 4096
    assert result["admission_limits"]["integral_terms"] == 32
    assert result["admission_limits"]["diagnostic_work_limits"] == {
        "grid_points": 1_000_000,
        "grid_pair_visits": 100_000_000,
    }
    assert result["admission_limits"]["public_work_limits"] == {
        "grid_points": None,
        "grid_pair_visits": None,
    }
    assert result["admission_limits"]["pending_grid_tiles"] == 64
    assert result["admission_limits"]["pending_grid_pair_visits"] == 100_000_000

    assert cases["water8"]["requirements"]["primitive_records"] == 23_151_431_572
    assert cases["water16"]["requirements"]["primitive_records"] == 370_399_663_720
    assert cases["water32"]["requirements"]["primitive_records"] == 5_926_219_028_944
    assert all(
        case["requirements"]["primitive_descriptor_peak_records"] == 625
        for case in cases.values()
    )
    assert cases["caffeine"]["requirements"]["grid_pair_visits"] == 1_098_842_388
    assert cases["ace_glygly_nme"]["requirements"]["grid_pair_visits"] == 1_401_753_925

    # Public complete-grid admission is separate from the still-bounded private
    # diagnostic defaults and does not claim native provider execution succeeded.
    assert all(
        case["admission"]["outcome"] == "passes_static_stationary_caps"
        for case in cases.values()
    )
    assert all(case["admission"]["first_blocker"] is None for case in cases.values())
    for case_name in ("water8", "water16", "water32", "caffeine", "ace_glygly_nme"):
        requirement = cases[case_name]["requirements"]["native_integral_admission"]
        assert requirement["required"] is True
        assert requirement["provider_and_budget_qualification"] == "NOT_RUN"
        assert (
            "cannot use AO-task fallback"
            in requirement["enlarged_domain_failure_behavior"]
        )
    assert (
        cases["benzene"]["requirements"]["native_integral_admission"]["required"]
        is False
    )
    assert (
        cases["benzene"]["diagnostic_admission"]["first_blocker"]["gate"]
        == "grid_pair_work_budget"
    )
    for sentinel in ("water", "oh", "o2", "water_dimer"):
        assert cases[sentinel]["diagnostic_admission"]["first_blocker"] is None
    water32_gates = [
        item["gate"] for item in cases["water32"]["diagnostic_admission"]["failures"]
    ]
    assert water32_gates == ["grid_point_work_budget", "grid_pair_work_budget"]
    assert cases["water32"]["requirements"]["grid_work_plan"] == {
        "grid_points": 7_962_624,
        "tile_points": 256,
        "tile_count": 31_104,
        "chunk_points": 10_752,
        "chunk_count": 741,
        "grid_pair_visits": 72_619_135_440,
        "chunk_pair_visits": 98_058_240,
    }


def test_report_covers_every_required_fp64_force_row_and_aot_route(
    tmp_path: Path,
) -> None:
    result = report()
    rows = result["rows"]

    assert len(rows) == 35
    assert result["summary"] == {
        "required_fp64_force_rows": 35,
        "required_semilocal_fp64_force_rows": 22,
        "statically_blocked_rows": 0,
        "rows_passing_static_stationary_caps": 35,
        "scientific_qualification": "NOT_RUN",
    }
    assert {row["method"] for row in rows} == {
        "lda",
        "pbe",
        "r2scan",
        "pbe0",
        "b3lyp",
    }
    assert {row["product"] for row in rows} == {"energy+analytic_forces"}
    assert all(row["required"] is True for row in rows)
    assert all(row["public_capability"]["forces"] is True for row in rows)
    assert {
        row["public_capability"]["selector_contract"]["selector"] for row in rows
    } == {
        "lda-rks",
        "lda-uks",
        "pbe-rks",
        "pbe-uks",
        "r2scan-rks",
        "r2scan-uks",
        "pbe0-rks",
        "pbe0-uks",
        "b3lyp-rks",
        "b3lyp-uks",
    }
    assert all(
        row["public_capability"]["selector_contract"]["stationary_plan_identity"]
        == row["stationary_plan"]["identity"]
        for row in rows
    )
    for row in rows:
        hybrid = row["method"] in ("pbe0", "b3lyp")
        route = row["public_route"]
        assert route["scientific_runtime_compilation_required"] is hybrid
        assert row["packaged_aot"]["selected_by_public_route"] is not hybrid
        assert route["selection"] == (
            "runtime-compiled stationary CUDA"
            if hybrid
            else "all-electron packaged stationary CUDA"
        )
        assert route["missing_aot_behavior"] == (
            "not selected by this public route"
            if hybrid
            else "fail closed; no NVCC fallback"
        )
    assert all(row["packaged_aot"]["source_package_declared"] is True for row in rows)
    assert result["stationary_aot_source_package"] == {
        "cmake_contract_sha256": (
            "2ddd2b7a77c992bc9812914e1a31b8dbbc1a332d4fab8b3ec92c239ff6ff2228"
        ),
        "profiles": [
            "lda_rks",
            "lda_uks",
            "pbe_rks",
            "pbe_uks",
            "r2scan_rks",
            "r2scan_uks",
            "pbe0_rks",
            "pbe0_uks",
            "b3lyp_rks",
            "b3lyp_uks",
        ],
        "component_domain": "spd",
    }
    assert all(
        row["packaged_aot"]["binary_verification"]["status"] == "not_checked"
        for row in rows
    )

    pbe_uks = next(row for row in rows if row["id"] == "pbe/uks/oh/fp64_energy_forces")
    assert pbe_uks["packaged_aot"]["name"] == "pbe_uks_spd"
    assert pbe_uks["packaged_aot"]["contract_identity"]
    assert pbe_uks["stationary_plan"]["source_names"] == [
        "one_electron",
        "coulomb",
        "xc_ao",
        "xc_grid",
        "xc_weight",
        "overlap_pulay",
        "nuclear",
    ]

    b3lyp_uks = next(
        row for row in rows if row["id"] == "b3lyp/uks/oh/fp64_energy_forces"
    )
    assert b3lyp_uks["packaged_aot"]["name"] == "b3lyp_uks_spd"
    assert "exact_exchange" in b3lyp_uks["stationary_plan"]["source_names"]
    b3lyp_contract = b3lyp_uks["public_capability"]["selector_contract"]
    assert b3lyp_contract["registry_selector"] == "pbe-uks"
    assert b3lyp_contract["coefficients"] == [1.0, 1.0, -0.2]
    assert b3lyp_contract["exchange"] == [
        {
            "operator": "full-range",
            "coefficient": "1/5",
            "omega": "0",
            "fock_coefficient": "-1/5",
        }
    ]

    pbe0_rks = next(
        row for row in rows if row["id"] == "pbe0/rks/water/fp64_energy_forces"
    )
    assert pbe0_rks["packaged_aot"]["name"] == "pbe0_rks_spd"
    pbe0_contract = pbe0_rks["public_capability"]["selector_contract"]
    assert pbe0_contract["registry_selector"] == "pbe-rks"
    assert pbe0_contract["coefficients"] == [0.75, 1.0, -0.125]

    b3lyp_benzene = next(
        row for row in rows if row["id"] == "b3lyp/rks/benzene/fp64_energy_forces"
    )
    assert b3lyp_benzene["admission"]["outcome"] == "passes_static_stationary_caps"
    assert b3lyp_benzene["admission"]["first_blocker"] is None

    water32 = next(
        row for row in rows if row["id"] == "pbe/rks/water32/fp64_energy_forces"
    )
    assert water32["resource_requirements"]["additional_device_peak_bound"] == (
        356_801_792 + 48 * (96 * 95 // 2) + 4_851_008
    )
    assert water32["resource_requirements"][
        "stationary_center_geometry_bytes"
    ] == 48 * (96 * 95 // 2)
    assert water32["resource_requirements"]["additional_host_numeric_bound"] == (
        192_187_488 + 4_851_008
    )
    assert water32["resource_requirements"]["additional_device_budget"] == 512 << 20
    assert water32["resource_requirements"]["additional_host_budget"] == 256 << 20

    missing = report(aot_directory=tmp_path)
    missing_rows = missing["rows"]
    assert all(
        row["packaged_aot"]["binary_verification"]["status"] == "missing_or_invalid"
        for row in missing_rows
    )
    assert all(
        "missing packaged stationary CUDA artifact"
        in row["packaged_aot"]["binary_verification"]["detail"]
        for row in missing_rows
    )


@pytest.mark.parametrize("boundary", ["minimum", "expanded"])
def test_each_row_uses_its_own_method_memory_admission(
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    original = qualify_capacity._method_resources

    def method_resources(
        basis: object,
        *,
        atom_count: int,
        functional: int,
        spin: str,
        limits: dict,
        plan: object | None = None,
    ) -> tuple[dict, object]:
        memory, plan = original(
            basis,
            atom_count=atom_count,
            functional=functional,
            spin=spin,
            limits=limits,
            plan=plan,
        )
        memory = dict(memory)
        if functional == qualify_capacity.SEMILOCAL_FUNCTIONALS["pbe"]:
            if boundary == "minimum":
                memory["minimum_additional_device_bytes"] = limits[
                    "additional_device_bytes"
                ]
            else:
                memory["additional_device_peak_bound"] = (
                    limits["additional_device_bytes"] + 1
                )
        return memory, plan

    monkeypatch.setattr(qualify_capacity, "_method_resources", method_resources)
    result = report()
    assert (
        next(case for case in result["cases"] if case["id"] == "water")["admission"][
            "outcome"
        ]
        == "blocked"
    )
    rows = {row["id"]: row for row in result["rows"]}

    assert rows["lda/rks/water/fp64_energy_forces"]["admission"]["outcome"] == (
        "passes_static_stationary_caps"
    )
    assert rows["r2scan/rks/water/fp64_energy_forces"]["admission"]["outcome"] == (
        "passes_static_stationary_caps"
    )
    pbe = rows["pbe/rks/water/fp64_energy_forces"]["admission"]
    assert pbe["outcome"] == "blocked"
    assert pbe["first_blocker"]["gate"] == "additional_device_budget"


def test_public_selector_contract_rejects_changed_semilocal_coefficients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = qualify_capacity.resolve_ks_options

    def changed(selector: str) -> SimpleNamespace:
        options = original(selector)
        return SimpleNamespace(
            coefficients=(0.5, 1.0, 0.0),
            execution_plan=options.execution_plan,
        )

    monkeypatch.setattr(qualify_capacity, "resolve_ks_options", changed)
    plan = qualify_capacity._qualified_aot_plan(1, "unpolarized")

    with pytest.raises(RuntimeError, match="semilocal coefficients"):
        qualify_capacity._public_selector_contract(
            "pbe-rks",
            expected_functional=1,
            expected_spin="unpolarized",
            stationary_plan=plan,
        )


def test_public_selector_contract_rejects_changed_hybrid_coefficients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = qualify_capacity.resolve_ks_options

    def changed(selector: str, options: object | None = None) -> SimpleNamespace:
        resolved = original(selector, options)
        return SimpleNamespace(
            coefficients=(1.0, 1.0, -0.125),
            execution_plan=resolved.execution_plan,
        )

    monkeypatch.setattr(qualify_capacity, "resolve_ks_options", changed)
    method_ir, _ = qualify_capacity.resolve_ks_method("b3lyp-rks")
    plan = qualify_capacity.StationaryGradientPlan(
        method_ir,
        qualify_capacity.StationaryMeanField(qualify_capacity.SCF_POINT_MODEL),
    )

    with pytest.raises(RuntimeError, match="scientific coefficients"):
        qualify_capacity._public_selector_contract(
            "b3lyp-rks",
            expected_functional=3,
            expected_spin="unpolarized",
            stationary_plan=plan,
            grid_spec=qualify_capacity.GridSpec(),
        )


def test_public_selector_contract_rejects_removed_native_eligibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        qualify_capacity.generated_methods,
        "NATIVE_DFT_METHOD_IDS",
        qualify_capacity.generated_methods.NATIVE_DFT_METHOD_IDS - {7},
    )
    plan = qualify_capacity._qualified_aot_plan(1, "unpolarized")

    with pytest.raises(RuntimeError, match="native DFT eligibility"):
        qualify_capacity._public_selector_contract(
            "pbe-rks",
            expected_functional=1,
            expected_spin="unpolarized",
            stationary_plan=plan,
        )


def test_machine_readable_report_round_trips_without_nonfinite_values() -> None:
    result = report()
    encoded = json.dumps(result, allow_nan=False, sort_keys=True)
    assert json.loads(encoded) == result


def test_report_rejects_a_repository_other_than_its_import_checkout(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="checkout containing this tool"):
        qualify_capacity.build_report(tmp_path)


def test_public_report_binds_the_clean_git_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "e" * 40
    monkeypatch.setattr(qualify_capacity, "_clean_git_sha", lambda _, **__: revision)
    monkeypatch.setattr(qualify_capacity, "_PRELOADED_LOCAL_MODULES", frozenset())
    monkeypatch.setattr(qualify_capacity, "_DIRECT_SOURCE_EXECUTION", True)

    result = qualify_capacity.build_report(ROOT)

    assert result["source"]["sha"] == revision


def test_public_report_cannot_exempt_an_arbitrary_dirty_path() -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument 'output_path'"):
        qualify_capacity.build_report(  # type: ignore[call-arg]
            ROOT,
            output_path=ROOT / "python/generativeqc/calculator.py",
        )


def test_source_contract_hashes_do_not_use_version_dependent_ast_dump(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_ast_dump(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("source contracts must not depend on ast.dump formatting")

    monkeypatch.setattr(qualify_capacity.ast, "dump", reject_ast_dump)

    basis = qualify_capacity._basis_layout_contract(ROOT)
    expansion = qualify_capacity._spd_expansion_contract(ROOT)

    assert basis["native_ao_constructor_contract_sha256"] == (
        qualify_capacity.NATIVE_AO_CONSTRUCTOR_CONTRACT_SHA256
    )
    assert expansion["stationary_layout_contract_sha256"] == (
        qualify_capacity.STATIONARY_LAYOUT_CONTRACT_SHA256
    )


def test_clean_git_sha_ignores_only_the_requested_report_output(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )

    git("init")
    git("config", "user.name", "Capacity Test")
    git("config", "user.email", "capacity@example.invalid")
    (repository / "tracked.json").write_text("{}\n", encoding="utf-8")
    git("add", "tracked.json")
    git("commit", "-m", "fixture")
    revision = git("rev-parse", "HEAD").stdout.strip()

    output = repository / "capacity-report.json"
    output.write_text("first run\n", encoding="utf-8")
    assert qualify_capacity._report_output_exemption(repository, output) == (
        output.resolve()
    )
    assert qualify_capacity._clean_git_sha(repository, ignored_path=output) == revision

    with pytest.raises(ValueError, match="must not replace a tracked file"):
        qualify_capacity._report_output_exemption(
            repository, repository / "tracked.json"
        )
    with pytest.raises(ValueError, match="must be a JSON file"):
        qualify_capacity._report_output_exemption(
            repository, repository / "capacity-report.py"
        )
    aot_directory = repository / "aot"
    aot_directory.mkdir()
    with pytest.raises(ValueError, match="must not overlap AOT evidence"):
        qualify_capacity._report_output_exemption(
            repository,
            aot_directory / "pbe_rks_spd.json",
            aot_directory=aot_directory,
        )

    (repository / "unrelated.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="clean Git worktree"):
        qualify_capacity._clean_git_sha(repository, ignored_path=output)


@pytest.mark.parametrize("flag", ("--assume-unchanged", "--skip-worktree"))
def test_clean_git_sha_rejects_hidden_index_paths(
    tmp_path: Path,
    flag: str,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )

    git("init")
    git("config", "user.name", "Capacity Test")
    git("config", "user.email", "capacity@example.invalid")
    tracked = repository / "tracked.py"
    tracked.write_text("ORIGINAL = True\n", encoding="utf-8")
    git("add", "tracked.py")
    git("commit", "-m", "fixture")
    git("update-index", flag, "tracked.py")
    tracked.write_text("ORIGINAL = False\n", encoding="utf-8")

    assert git("status", "--porcelain").stdout == ""
    with pytest.raises(
        RuntimeError,
        match="rejects assume-unchanged/skip-worktree paths: tracked.py",
    ):
        qualify_capacity._clean_git_sha(repository)


def test_report_output_rejects_hard_link_to_aot_evidence(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    aot_directory = repository / "aot"
    aot_directory.mkdir()
    evidence = aot_directory / "pbe_rks_spd.json"
    evidence.write_text('{"evidence": true}\n', encoding="utf-8")
    output_directory = tmp_path / "reports"
    output_directory.mkdir()
    output = output_directory / "capacity.json"
    os.link(evidence, output)

    with pytest.raises(ValueError, match="must not be a hard link"):
        qualify_capacity._report_output_exemption(
            repository,
            output,
            aot_directory=aot_directory,
        )

    assert evidence.read_text(encoding="utf-8") == '{"evidence": true}\n'


def test_report_output_rejects_symlinked_aot_evidence(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    aot_directory = repository / "aot"
    aot_directory.mkdir()
    external = tmp_path / "external-evidence.json"
    external.write_text('{"evidence": true}\n', encoding="utf-8")
    try:
        os.symlink(external, aot_directory / "pbe_rks_spd.json")
    except OSError as error:
        pytest.skip(f"filesystem cannot create a test symlink: {error}")

    with pytest.raises(ValueError, match="AOT evidence must not contain symlinks"):
        qualify_capacity._report_output_exemption(
            repository,
            external,
            aot_directory=aot_directory,
        )
    with pytest.raises(ValueError, match="AOT evidence must not contain symlinks"):
        report(aot_directory=aot_directory)

    assert external.read_text(encoding="utf-8") == '{"evidence": true}\n'


def test_report_rejects_symlinked_aot_directory_ancestor(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    real_parent = tmp_path / "real-aot-parent"
    aot_directory = real_parent / "package"
    aot_directory.mkdir(parents=True)
    alias_parent = tmp_path / "alias-aot-parent"
    try:
        os.symlink(real_parent, alias_parent, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"filesystem cannot create a test directory symlink: {error}")
    aliased_aot = alias_parent / "package"
    assert not aliased_aot.is_symlink()

    with pytest.raises(ValueError, match="AOT evidence must not contain symlinks"):
        report(aot_directory=aliased_aot)


def test_primitive_budget_scope_fails_closed_when_whole_force_gate_returns(
    tmp_path: Path,
) -> None:
    source = (ROOT / "python/generativeqc/_stationary_cuda.py").read_text(
        encoding="utf-8"
    )
    whole_force_gate = (
        "    if records > max_primitive_records:\n"
        '        raise ValueError("primitive work budget exceeded")\n'
    )
    assert whole_force_gate not in source
    marker = "    pair_visits = grid_work.grid_pair_visits"
    assert marker in source
    stationary_contract_tree(
        tmp_path, source.replace(marker, whole_force_gate + marker, 1)
    )

    with pytest.raises(RuntimeError, match="restored a whole-force primitive cap"):
        qualify_capacity._source_limits(tmp_path)


def test_primitive_budget_scope_fails_closed_when_page_contract_moves(
    tmp_path: Path,
) -> None:
    source = (ROOT / "python/generativeqc/_stationary_cuda.py").read_text(
        encoding="utf-8"
    )
    old = "if np.any(primitive_work > self.page_work_budget):"
    assert old in source
    stationary_contract_tree(
        tmp_path,
        source.replace(old, "if np.any(primitive_work >= self.page_work_budget):", 1),
    )

    with pytest.raises(RuntimeError, match="bulk page contract changed"):
        qualify_capacity._source_limits(tmp_path)


def test_primitive_budget_scope_fails_closed_when_component_work_moves(
    tmp_path: Path,
) -> None:
    source = (ROOT / "python/generativeqc/_stationary_cuda.py").read_text(
        encoding="utf-8"
    )
    old = "primitive_work *= int(row[2])"
    assert old in source
    stationary_contract_tree(
        tmp_path, source.replace(old, "primitive_work *= int(row[2]) + 1", 1)
    )

    with pytest.raises(RuntimeError, match="component_integral page contract changed"):
        qualify_capacity._source_limits(tmp_path)


def test_primitive_page_gate_order_fails_closed_when_execution_moves(
    tmp_path: Path,
) -> None:
    source = (ROOT / "python/generativeqc/_stationary_cuda.py").read_text(
        encoding="utf-8"
    )
    original = (
        "            execution = task_executor.execute_pages(domain, submit_page)"
    )
    host_gate = "    if host_bound > max_host_bytes:"
    assert original in source and host_gate in source
    moved = source.replace(original, "            execution = None", 1)
    moved = moved.replace(
        host_gate,
        "    task_executor.execute_pages(domain, submit_page)\n" + host_gate,
        1,
    )
    stationary_contract_tree(tmp_path, moved)

    with pytest.raises(RuntimeError, match="primitive descriptor page order changed"):
        qualify_capacity._source_limits(tmp_path)


def test_primitive_page_gate_fails_closed_when_callback_bypasses_producer(
    tmp_path: Path,
) -> None:
    source = (ROOT / "python/generativeqc/_stationary_cuda.py").read_text(
        encoding="utf-8"
    )
    old = "sources.integral_page("
    assert source.count(old) == 2
    stationary_contract_tree(tmp_path, source.replace(old, "sources.integral(", 1))

    with pytest.raises(RuntimeError, match="submit-page contract changed"):
        qualify_capacity._source_limits(tmp_path)


def test_primitive_page_gate_fails_closed_when_budget_initialization_moves(
    tmp_path: Path,
) -> None:
    source = (ROOT / "python/generativeqc/_stationary_cuda.py").read_text(
        encoding="utf-8"
    )
    old = "self.page_work_budget = int(page_work_budget)"
    assert old in source
    stationary_contract_tree(
        tmp_path,
        source.replace(old, "self.page_work_budget = 2 * int(page_work_budget)", 1),
    )

    with pytest.raises(RuntimeError, match="initializer page contract changed"):
        qualify_capacity._source_limits(tmp_path)


def test_primitive_page_gate_fails_closed_when_native_consumer_moves(
    tmp_path: Path,
) -> None:
    source = (ROOT / "python/generativeqc/_stationary_cuda.py").read_text(
        encoding="utf-8"
    )
    stationary_contract_tree(tmp_path, source)
    target = tmp_path / "src/dft/stationary_gradient_cuda.cuh"
    native = target.read_text(encoding="utf-8")
    old = "size_t(work) > p->max_page_primitive_work"
    assert old in native
    target.write_text(
        native.replace(old, "size_t(work) >= p->max_page_primitive_work", 1),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="native_tasks_sha256 contract changed"):
        qualify_capacity._source_limits(tmp_path)


def test_nuclear_pair_work_fails_closed_when_python_consumer_moves(
    tmp_path: Path,
) -> None:
    source = (ROOT / "python/generativeqc/_stationary_cuda.py").read_text(
        encoding="utf-8"
    )
    old = '        self.flush()\n        kind = self.kinds["nuclear", ()]'
    assert old in source
    stationary_contract_tree(
        tmp_path,
        source.replace(old, '        kind = self.kinds["nuclear", ()]', 1),
    )

    with pytest.raises(RuntimeError, match="nuclear page contract changed"):
        qualify_capacity._source_limits(tmp_path)


def test_nuclear_pair_work_fails_closed_when_endpoint_loop_moves(
    tmp_path: Path,
) -> None:
    source = (ROOT / "python/generativeqc/_stationary_cuda.py").read_text(
        encoding="utf-8"
    )
    old = "            for other in range(atom):\n                sources.nuclear("
    assert old in source
    stationary_contract_tree(
        tmp_path,
        source.replace(
            old,
            "            for other in range(atom + 1):\n                sources.nuclear(",
            1,
        ),
    )

    with pytest.raises(RuntimeError, match="nuclear-pair loop contract changed"):
        qualify_capacity._source_limits(tmp_path)


def test_nuclear_pair_work_fails_closed_when_native_consumer_moves(
    tmp_path: Path,
) -> None:
    source = (ROOT / "python/generativeqc/_stationary_cuda.py").read_text(
        encoding="utf-8"
    )
    stationary_contract_tree(tmp_path, source)
    target = tmp_path / "src/dft/stationary_gradient_cuda.cuh"
    native = target.read_text(encoding="utf-8")
    old = "p->check_page_primitive_work(1);"
    assert native.count(old) == 1
    target.write_text(
        native.replace(old, "p->check_page_primitive_work(2);", 1),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="native_nuclear_sha256 contract changed"):
        qualify_capacity._source_limits(tmp_path)


def test_primitive_work_fails_closed_when_endpoint_enumeration_moves(
    tmp_path: Path,
) -> None:
    source = (ROOT / "python/generativeqc/_stationary_cuda.py").read_text(
        encoding="utf-8"
    )
    old = '("coulomb", 4, "four_center_eri")'
    assert source.count(old) == 1
    stationary_contract_tree(
        tmp_path,
        source.replace(old, '("coulomb", 2, "four_center_eri")', 1),
    )

    with pytest.raises(RuntimeError, match="endpoint owner contract changed"):
        qualify_capacity._source_limits(tmp_path)


def test_grid_pair_work_fails_closed_when_endpoint_tiling_moves(
    tmp_path: Path,
) -> None:
    source = (ROOT / "python/generativeqc/_stationary_cuda.py").read_text(
        encoding="utf-8"
    )
    old = "for begin in range(chunk_begin, chunk_end, tile_points):"
    assert source.count(old) == 1
    stationary_contract_tree(
        tmp_path,
        source.replace(
            old,
            "for begin in range(chunk_begin, chunk_end, 2 * tile_points):",
            1,
        ),
    )

    with pytest.raises(RuntimeError, match="endpoint owner contract changed"):
        qualify_capacity._source_limits(tmp_path)


def test_grid_pair_work_fails_closed_when_python_geometry_route_moves(
    tmp_path: Path,
) -> None:
    source = (ROOT / "python/generativeqc/_stationary_cuda.py").read_text(
        encoding="utf-8"
    )
    old = 'else "stationary_geometry_enqueue"'
    assert source.count(old) == 1
    stationary_contract_tree(
        tmp_path,
        source.replace(old, 'else "stationary_geometry"', 1),
    )

    with pytest.raises(RuntimeError, match="geometry page contract changed"):
        qualify_capacity._source_limits(tmp_path)


def test_grid_pair_work_fails_closed_when_native_geometry_consumer_moves(
    tmp_path: Path,
) -> None:
    source = (ROOT / "python/generativeqc/_stationary_cuda.py").read_text(
        encoding="utf-8"
    )
    stationary_contract_tree(tmp_path, source)
    target = tmp_path / "src/dft/stationary_gradient_cuda.cuh"
    native = target.read_text(encoding="utf-8")
    marker = "int stationary_geometry_enqueue("
    start = native.index(marker)
    old = "p->pair_visits += view->npoint * p->atoms * (p->atoms - 1);"
    position = native.index(old, start)
    target.write_text(
        native[:position]
        + "p->pair_visits += view->npoint * p->atoms;"
        + native[position + len(old) :],
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError, match="native_geometry_enqueue_sha256 contract changed"
    ):
        qualify_capacity._source_limits(tmp_path)


def test_native_finish_span_fails_closed_on_contract_drift(
    tmp_path: Path,
) -> None:
    source = (ROOT / "python/generativeqc/_stationary_cuda.py").read_text(
        encoding="utf-8"
    )
    stationary_contract_tree(tmp_path, source)
    target = tmp_path / "src/dft/stationary_gradient_cuda.cuh"
    native = target.read_text(encoding="utf-8")
    marker = "int stationary_finish_span("
    start = native.index(marker)
    old = "p->downloads += count * 8;"
    position = native.index(old, start)
    target.write_text(
        native[:position]
        + "p->downloads += count * 16;"
        + native[position + len(old) :],
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError, match="native_finish_span_sha256 contract changed"
    ):
        qualify_capacity._source_limits(tmp_path)


def test_memory_bounds_fail_closed_when_native_allocation_moves(
    tmp_path: Path,
) -> None:
    source = (ROOT / "python/generativeqc/_stationary_cuda.py").read_text(
        encoding="utf-8"
    )
    stationary_contract_tree(tmp_path, source)
    target = tmp_path / "src/dft/stationary_gradient_cuda.cuh"
    native = target.read_text(encoding="utf-8")
    old = "(3 + 18 * geometry_lanes + 3 * stationary_source_count) * na"
    assert old in native
    target.write_text(
        native.replace(
            old, "(4 + 18 * geometry_lanes + 3 * stationary_source_count) * na", 1
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="native_allocation_sha256 contract changed"):
        qualify_capacity._source_limits(tmp_path)


def test_primitive_budget_scope_fails_closed_when_work_definition_moves(
    tmp_path: Path,
) -> None:
    source = (ROOT / "python/generativeqc/_stationary_cuda.py").read_text(
        encoding="utf-8"
    )
    old = "(1 + int(has_exchange)) * primitive_sum**4"
    assert old in source
    stationary_contract_tree(tmp_path, source.replace(old, "primitive_sum**3", 1))

    with pytest.raises(RuntimeError, match="primitive-record definition"):
        qualify_capacity._source_limits(tmp_path)


def test_limit_defaults_fail_closed_when_public_forwarding_moves(
    tmp_path: Path,
) -> None:
    source = (ROOT / "python/generativeqc/_stationary_cuda.py").read_text(
        encoding="utf-8"
    )
    old = '"max_grid_points": max_grid_points,'
    assert source.count(old) == 1
    stationary_contract_tree(
        tmp_path,
        source.replace(old, '"max_grid_points": max_grid_points // 2,', 1),
    )

    with pytest.raises(RuntimeError, match="public wrapper contract changed"):
        qualify_capacity._source_limits(tmp_path)


def test_memory_bounds_fail_closed_when_production_definition_moves(
    tmp_path: Path,
) -> None:
    source = (ROOT / "python/generativeqc/_stationary_cuda.py").read_text(
        encoding="utf-8"
    )
    old = "tasks=primitive_tile,"
    assert old in source
    stationary_contract_tree(
        tmp_path, source.replace(old, "tasks=primitive_tile + 1,", 1)
    )

    with pytest.raises(RuntimeError, match="minimum-source-bytes definition"):
        qualify_capacity._source_limits(tmp_path)


@pytest.mark.parametrize(
    ("relative", "old", "new", "message"),
    [
        (
            "python/generativeqc/_model_resolution.py",
            'return _named_basis_record(basis, representation or "cartesian")',
            'return _named_basis_record(basis, representation or "spherical")',
            "snapshot_basis basis lowering contract changed",
        ),
        (
            "python/generativeqc/basis.py",
            "for row in shell.coefficients:",
            "for row in shell.coefficients[:1]:",
            "shells_for basis lowering contract changed",
        ),
        (
            "python/generativeqc/calculator.py",
            "return (\n            selected_basis.shells_for(atoms)",
            "return (\n            tuple(selected_basis.shells_for(atoms))",
            "_shells_for_atoms basis lowering contract changed",
        ),
        (
            "python/generativeqc/calculator.py",
            "shells = self._shells_for_atoms(atoms, basis)",
            "shells = tuple(self._shells_for_atoms(atoms, basis))",
            "_create_native_system basis lowering contract changed",
        ),
    ],
)
def test_basis_counts_fail_closed_when_production_lowering_moves(
    tmp_path: Path,
    relative: str,
    old: str,
    new: str,
    message: str,
) -> None:
    copy_contract_files(
        tmp_path,
        (
            "python/generativeqc_compiler/dft/ao.py",
            "python/generativeqc/_model_resolution.py",
            "python/generativeqc/basis.py",
            "python/generativeqc/calculator.py",
        ),
    )
    target = tmp_path / relative
    source = target.read_text(encoding="utf-8")
    assert old in source
    target.write_text(source.replace(old, new, 1), encoding="utf-8")

    with pytest.raises(RuntimeError, match=message):
        qualify_capacity._basis_layout_contract(tmp_path)


def test_grid_memory_fails_closed_when_production_plan_inputs_move(
    tmp_path: Path,
) -> None:
    source = (ROOT / "python/generativeqc/_stationary_cuda.py").read_text(
        encoding="utf-8"
    )
    old = "order=2 if needs_first else 1"
    first = source.index(old)
    production = source.index(old, first + len(old))
    stationary_contract_tree(
        tmp_path,
        source[:production] + "order=1" + source[production + len(old) :],
    )

    with pytest.raises(RuntimeError, match="grid-plan input definition changed"):
        qualify_capacity._source_limits(tmp_path)


def test_grid_memory_fails_closed_when_functional_lowering_moves(
    tmp_path: Path,
) -> None:
    source = (ROOT / "python/generativeqc/_stationary_cuda.py").read_text(
        encoding="utf-8"
    )
    old = "functional = int(state._source.metadata[6])"
    assert old in source
    stationary_contract_tree(tmp_path, source.replace(old, "functional = 0", 1))

    with pytest.raises(RuntimeError, match="snapshot functional-code lowering"):
        qualify_capacity._source_limits(tmp_path)


def test_public_selector_contract_rejects_changed_functional_lowering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qualify_capacity, "native_xc_functional_code", lambda _: 0)
    plan = qualify_capacity._qualified_aot_plan(1, "unpolarized")

    with pytest.raises(RuntimeError, match="native functional-family lowering"):
        qualify_capacity._public_selector_contract(
            "pbe-rks",
            expected_functional=1,
            expected_spin="unpolarized",
            stationary_plan=plan,
        )


def test_memory_bounds_fail_closed_when_host_gate_moves(tmp_path: Path) -> None:
    source = (ROOT / "python/generativeqc/_stationary_cuda.py").read_text(
        encoding="utf-8"
    )
    old = "if host_bound > max_host_bytes:"
    assert old in source
    begin = source.index("def _complete_rks_cuda_gradient_diagnostic(")
    stationary_contract_tree(
        tmp_path,
        source[:begin]
        + source[begin:].replace(old, "if host_bound >= max_host_bytes:", 1),
    )

    with pytest.raises(RuntimeError, match="additional-host predicate"):
        qualify_capacity._source_limits(tmp_path)


def test_admission_gate_order_fails_closed_when_leading_gates_move(
    tmp_path: Path,
) -> None:
    source = (ROOT / "python/generativeqc/_stationary_cuda.py").read_text(
        encoding="utf-8"
    )
    native_begin = source.index(
        "    requires_native_integrals = stationary_cuda_requires_native_integrals("
    )
    grid_begin = source.index(
        "    grid_work = plan_stationary_cuda_grid_work(", native_begin
    )
    grid_end = source.index("    if requires_native_integrals and (", grid_begin)
    native = source[native_begin:grid_begin]
    grid = source[grid_begin:grid_end]
    stationary_contract_tree(tmp_path, source.replace(native + grid, grid + native, 1))

    with pytest.raises(RuntimeError, match="admission gate order changed"):
        qualify_capacity._source_limits(tmp_path)


def test_admission_gate_order_fails_closed_when_memory_gates_move(
    tmp_path: Path,
) -> None:
    source = (ROOT / "python/generativeqc/_stationary_cuda.py").read_text(
        encoding="utf-8"
    )
    device = (
        "    if available <= 0:\n"
        '        raise ValueError("stationary additional-device budget exceeded")\n'
    )
    host = (
        "    if host_bound > max_host_bytes:\n"
        '        raise ValueError("stationary additional-host byte budget exceeded")\n'
    )
    assert device in source and host in source
    begin = source.index("def _complete_rks_cuda_gradient_diagnostic(")
    swapped = source[begin:].replace(device, "    # swapped-memory-gate\n", 1)
    swapped = swapped.replace(host, device, 1)
    swapped = swapped.replace("    # swapped-memory-gate\n", host, 1)
    stationary_contract_tree(tmp_path, source[:begin] + swapped)

    with pytest.raises(
        RuntimeError, match="native host-reserve admission order changed"
    ):
        qualify_capacity._source_limits(tmp_path)


def test_packaged_aot_claim_fails_closed_when_cmake_wiring_moves(
    tmp_path: Path,
) -> None:
    source = (ROOT / "cmake/GenerativeQCCuda.cmake").read_text(encoding="utf-8")
    old = "          --component-domain spd"
    assert old in source
    target = tmp_path / "cmake/GenerativeQCCuda.cmake"
    target.parent.mkdir(parents=True)
    target.write_text(
        source.replace(old, "          --component-domain sp", 1), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="packaged-AOT CMake contract changed"):
        qualify_capacity._source_package_inventory(tmp_path)


def test_basis_numeric_bound_fails_closed_when_production_definition_moves(
    tmp_path: Path,
) -> None:
    source = (ROOT / "python/generativeqc_compiler/dft/ao.py").read_text(
        encoding="utf-8"
    )
    old = "2 * self.packed.nbytes"
    assert old in source
    target = tmp_path / "python/generativeqc_compiler/dft/ao.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        source.replace(old, "3 * self.packed.nbytes", 1), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="numeric capacity definition"):
        qualify_capacity._basis_layout_contract(tmp_path)


def test_spherical_component_count_fails_closed_when_d_expansion_moves(
    tmp_path: Path,
) -> None:
    spd_contract_tree(tmp_path)
    target = tmp_path / "src/molecule/basis.cpp"
    source = target.read_text(encoding="utf-8")
    old = ", {{0, 2, 0}, -root_three_over_two}"
    assert old in source
    target.write_text(source.replace(old, "", 1), encoding="utf-8")

    with pytest.raises(RuntimeError, match="s/p/d expansion contract changed"):
        qualify_capacity._spd_expansion_contract(tmp_path)


def test_spherical_component_count_fails_closed_when_cartesian_generator_moves(
    tmp_path: Path,
) -> None:
    spd_contract_tree(tmp_path)
    target = tmp_path / "src/molecule/basis.cpp"
    source = target.read_text(encoding="utf-8")
    old = "components.push_back({static_cast<unsigned>(lx), ly, lz});"
    assert old in source
    target.write_text(source.replace(old, "components.push_back({0, ly, lz});", 1))

    with pytest.raises(RuntimeError, match="s/p/d expansion contract changed"):
        qualify_capacity._spd_expansion_contract(tmp_path)


def test_spherical_component_count_fails_closed_when_ao_packer_moves(
    tmp_path: Path,
) -> None:
    spd_contract_tree(tmp_path)
    target = tmp_path / "src/dft/ao_grid.cpp"
    source = target.read_text(encoding="utf-8")
    old = "record[3] = expansion.size();"
    assert old in source
    target.write_text(source.replace(old, "record[3] = 1;", 1))

    with pytest.raises(RuntimeError, match="packed-AO contract changed"):
        qualify_capacity._spd_expansion_contract(tmp_path)


def test_spherical_component_count_fails_closed_when_layout_parser_moves(
    tmp_path: Path,
) -> None:
    spd_contract_tree(tmp_path)
    target = tmp_path / "python/generativeqc/_stationary_cuda.py"
    source = target.read_text(encoding="utf-8")
    old = "for term in range(int(row[3]))"
    assert old in source
    target.write_text(source.replace(old, "for term in range(1)", 1))

    with pytest.raises(RuntimeError, match="stationary layout contract changed"):
        qualify_capacity._spd_expansion_contract(tmp_path)


def test_spherical_component_count_fails_closed_when_basis_creation_moves(
    tmp_path: Path,
) -> None:
    spd_contract_tree(tmp_path)
    target = tmp_path / "src/dft/bridge.cpp"
    source = target.read_text(encoding="utf-8")
    old = "dimensions[2] = basis->nao;"
    assert old in source
    target.write_text(source.replace(old, "dimensions[2] = 0;", 1))

    with pytest.raises(RuntimeError, match="packed-AO contract changed"):
        qualify_capacity._spd_expansion_contract(tmp_path)


def test_spherical_ao_count_fails_closed_when_native_count_moves(
    tmp_path: Path,
) -> None:
    spd_contract_tree(tmp_path)
    target = tmp_path / "src/molecule/basis.cpp"
    source = target.read_text(encoding="utf-8")
    old = "? 2 * static_cast<std::size_t>(shell.angular_momentum) + 1"
    assert old in source
    target.write_text(source.replace(old, old[:-1] + "2", 1), encoding="utf-8")

    with pytest.raises(RuntimeError, match="spherical AO count contract changed"):
        qualify_capacity._spd_expansion_contract(tmp_path)


def test_public_capability_fails_closed_when_complete_predicate_moves(
    tmp_path: Path,
) -> None:
    copy_contract_files(tmp_path, PUBLIC_ROUTE_FILES)
    target = tmp_path / "python/generativeqc/calculator.py"
    source = target.read_text(encoding="utf-8")
    old = "self._ks_options.coefficients == (1.0, 1.0, 0.0)"
    first = source.index(old)
    semilocal = source.index(old, first + len(old))
    target.write_text(
        source[:semilocal] + "False" + source[semilocal + len(old) :],
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="semilocal force predicate changed"):
        qualify_capacity._source_public_route(tmp_path)


def test_public_capability_fails_closed_when_promotion_condition_moves(
    tmp_path: Path,
) -> None:
    copy_contract_files(tmp_path, PUBLIC_ROUTE_FILES)
    target = tmp_path / "python/generativeqc/calculator.py"
    source = target.read_text(encoding="utf-8")
    old = "and self._method in _method_manifest.NATIVE_DFT_METHOD_IDS"
    assert old in source
    target.write_text(source.replace(old, "and False", 1), encoding="utf-8")

    with pytest.raises(RuntimeError, match="force capability promotion changed"):
        qualify_capacity._source_public_route(tmp_path)


def test_public_cuda_force_fails_closed_when_packaged_route_moves(
    tmp_path: Path,
) -> None:
    copy_contract_files(tmp_path, PUBLIC_ROUTE_FILES)
    target = tmp_path / "python/generativeqc/batch.py"
    source = target.read_text(encoding="utf-8")
    old = "and not state._source.method_ir.full_range_exact_exchange"
    assert old in source
    target.write_text(source.replace(old, "and False", 1), encoding="utf-8")

    with pytest.raises(RuntimeError, match="public CUDA force route changed"):
        qualify_capacity._source_public_route(tmp_path)


def test_prepared_aot_route_fails_closed_when_selection_moves(
    tmp_path: Path,
) -> None:
    relative = "python/generativeqc/_stationary_cuda.py"
    copy_contract_files(tmp_path, (relative,))
    target = tmp_path / relative
    source = target.read_text(encoding="utf-8")
    old = "if aot_directory is None or ecp"
    assert old in source
    target.write_text(source.replace(old, "if True", 1), encoding="utf-8")

    with pytest.raises(RuntimeError, match="AOT selection contract changed"):
        qualify_capacity._prepared_aot_route_contract(tmp_path)


def test_grid_count_fails_closed_when_native_cuda_shape_moves(
    tmp_path: Path,
) -> None:
    copy_contract_files(tmp_path, GRID_CONTRACT_FILES)
    target = tmp_path / "src/dft/cuda_quadrature.cu"
    source = target.read_text(encoding="utf-8")
    old = "q::product(system.atoms.size(), per_atom)"
    assert old in source
    target.write_text(
        source.replace(old, "q::product(system.atoms.size() + 1, per_atom)", 1),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="grid point-count contract changed"):
        qualify_capacity._grid_count_contract(tmp_path)


def test_grid_count_fails_closed_when_source_only_shape_moves(
    tmp_path: Path,
) -> None:
    copy_contract_files(tmp_path, GRID_CONTRACT_FILES)
    target = tmp_path / "python/generativeqc_compiler/dft/grid.py"
    source = target.read_text(encoding="utf-8")
    old = "len(atoms) * len(r) * len(angular)"
    assert old in source
    target.write_text(source.replace(old, "len(r) * len(angular)", 1), encoding="utf-8")

    with pytest.raises(RuntimeError, match="grid point-count contract changed"):
        qualify_capacity._grid_count_contract(tmp_path)


def test_grid_count_fails_closed_when_generated_layout_moves(
    tmp_path: Path,
) -> None:
    copy_contract_files(tmp_path, GRID_CONTRACT_FILES)
    target = tmp_path / "python/generativeqc_compiler/xc/quadrature_cuda.py"
    source = target.read_text(encoding="utf-8")
    old = "l.points = points;"
    assert old in source
    target.write_text(
        source.replace(old, "l.points = points + 1;", 1), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="grid point-count contract changed"):
        qualify_capacity._grid_count_contract(tmp_path)


def test_grid_count_fails_closed_when_native_backend_route_moves(
    tmp_path: Path,
) -> None:
    copy_contract_files(tmp_path, GRID_CONTRACT_FILES)
    target = tmp_path / "src/methods/dft_method.cpp"
    source = target.read_text(encoding="utf-8")
    old = "return dft::MolecularGrid::from_cuda(system, spec, device, retain_device);"
    assert old in source
    target.write_text(
        source.replace(old, "return dft::MolecularGrid(system, spec);", 1),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="grid point-count contract changed"):
        qualify_capacity._grid_count_contract(tmp_path)


def test_grid_count_fails_closed_when_public_abi_lowering_moves(
    tmp_path: Path,
) -> None:
    copy_contract_files(tmp_path, GRID_CONTRACT_FILES)
    target = tmp_path / "python/generativeqc/ks.py"
    source = target.read_text(encoding="utf-8")
    old = "        grid.radial_points,"
    assert old in source
    target.write_text(
        source.replace(old, "        grid.radial_points + 1,", 1),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="grid point-count contract changed"):
        qualify_capacity._grid_count_contract(tmp_path)


def test_grid_count_fails_closed_when_native_abi_lowering_moves(
    tmp_path: Path,
) -> None:
    copy_contract_files(tmp_path, GRID_CONTRACT_FILES)
    target = tmp_path / "src/methods/dft_method.cpp"
    source = target.read_text(encoding="utf-8")
    old = "grid.radial_points = input.radial_points;"
    assert old in source
    target.write_text(
        source.replace(old, "grid.radial_points = input.radial_points + 1;", 1),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="grid point-count contract changed"):
        qualify_capacity._grid_count_contract(tmp_path)


def test_module_import_binds_helpers_to_the_tool_checkout(tmp_path: Path) -> None:
    environment = {
        key: value for key, value in os.environ.items() if key != "PYTHONPATH"
    }
    environment["PYTHONPATH"] = str(ROOT)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import inspect; "
                "from tools.dft_mp_v1 import qualify_capacity as q; "
                "print(inspect.getfile(q.Atom))"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert Path(completed.stdout.strip()).resolve().is_relative_to(ROOT / "python")


def test_report_rejects_a_helper_imported_outside_the_tool_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qualify_capacity, "Atom", Path)
    monkeypatch.setattr(qualify_capacity, "_PRELOADED_LOCAL_MODULES", frozenset())
    monkeypatch.setattr(qualify_capacity, "_DIRECT_SOURCE_EXECUTION", True)
    with pytest.raises(RuntimeError, match="outside the tool checkout"):
        qualify_capacity.build_report(ROOT)


def test_report_rejects_same_checkout_helper_source_changed_since_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _ = qualify_capacity._IMPORTED_HELPER_SOURCES["Atom"]
    monkeypatch.setitem(
        qualify_capacity._IMPORTED_HELPER_SOURCES,
        "Atom",
        (path, "0" * 64),
    )

    with pytest.raises(RuntimeError, match="helper source changed since import: Atom"):
        qualify_capacity._assert_local_imports()


def test_report_rejects_import_time_registry_manifest_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _ = qualify_capacity._IMPORTED_DATA_DEPENDENCIES["public_methods_manifest"]
    monkeypatch.setitem(
        qualify_capacity._IMPORTED_DATA_DEPENDENCIES,
        "public_methods_manifest",
        (path, "0" * 64),
    )

    with pytest.raises(
        RuntimeError,
        match="import-time data dependency changed: public_methods_manifest",
    ):
        qualify_capacity._assert_local_imports()


def test_report_requires_a_fresh_interpreter_after_checkout_head_moves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qualify_capacity, "_IMPORTED_TOOL_HEAD", "0" * 40)

    with pytest.raises(RuntimeError, match="start a fresh interpreter"):
        qualify_capacity._assert_local_imports()


def test_report_requires_a_fresh_interpreter_after_qualifier_source_moves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qualify_capacity, "_IMPORTED_TOOL_SOURCE_SHA256", "0" * 64)

    with pytest.raises(RuntimeError, match="qualifier source changed since import"):
        qualify_capacity._assert_local_imports()


def test_report_rejects_transitive_imported_module_source_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _ = qualify_capacity._IMPORTED_LOCAL_MODULE_SOURCES[
        "generativeqc_compiler.dft.ao"
    ]
    monkeypatch.setitem(
        qualify_capacity._IMPORTED_LOCAL_MODULE_SOURCES,
        "generativeqc_compiler.dft.ao",
        (path, "0" * 64),
    )

    with pytest.raises(
        RuntimeError,
        match="imported module source changed: generativeqc_compiler.dft.ao",
    ):
        qualify_capacity._assert_local_imports()


def test_report_rejects_rebound_planner_ao_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        qualify_capacity.plan_tiles.__globals__,
        "jet_indices",
        lambda _: (),
    )

    with pytest.raises(RuntimeError, match="planner captured AO helper changed"):
        qualify_capacity._assert_local_imports()


def test_report_rejects_dependencies_preloaded_before_qualifier_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qualify_capacity, "_DIRECT_SOURCE_EXECUTION", True)
    monkeypatch.setattr(
        qualify_capacity,
        "_PRELOADED_LOCAL_MODULES",
        frozenset({"generativeqc_compiler.dft.plan"}),
    )
    with pytest.raises(RuntimeError, match="requires a fresh interpreter"):
        qualify_capacity.build_report(ROOT)


def test_imported_public_report_requires_direct_source_execution() -> None:
    with pytest.raises(RuntimeError, match="require direct source CLI execution"):
        qualify_capacity.build_report(ROOT)


def test_module_execution_cannot_emit_authoritative_report() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "python")
    completed = subprocess.run(
        [sys.executable, "-m", "tools.dft_mp_v1.qualify_capacity"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode != 0
    assert "require direct source CLI execution" in completed.stderr


def test_report_reloads_basis_data_instead_of_reusing_a_stale_cache() -> None:
    baseline = next(case for case in report()["cases"] if case["id"] == "water")
    qualify_capacity._basis_pack.cache_clear()
    qualify_capacity._named_basis_record.cache_clear()
    pack = qualify_capacity._basis_pack()
    bases = pack["bases"]
    assert isinstance(bases, dict)
    shells = bases["def2-svp"]["elements"]["1"]
    coefficients = shells[0]["coefficients"]
    original = coefficients[0]
    coefficients[0] = "999.0"
    try:
        result = report()
    finally:
        coefficients[0] = original
        qualify_capacity._basis_pack.cache_clear()
        qualify_capacity._named_basis_record.cache_clear()

    water = next(case for case in result["cases"] if case["id"] == "water")
    assert water["identities"]["basis"] == baseline["identities"]["basis"]


def test_malformed_optional_aot_manifest_is_reported_not_raised(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def malformed_loader(*_: object, **__: object) -> object:
        raise KeyError("source_identity")

    monkeypatch.setattr(
        qualify_capacity, "load_stationary_aot_artifact", malformed_loader
    )
    result = qualify_capacity._artifact_verification(
        tmp_path,
        functional=0,
        spin="unpolarized",
        plan=object(),
    )

    assert result == {
        "status": "missing_or_invalid",
        "detail": "missing AOT manifest field: source_identity",
    }


def test_non_object_optional_aot_manifest_is_reported_not_raised(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def non_object_loader(*_: object, **__: object) -> object:
        raise AttributeError("'list' object has no attribute 'get'")

    monkeypatch.setattr(
        qualify_capacity, "load_stationary_aot_artifact", non_object_loader
    )
    result = qualify_capacity._artifact_verification(
        tmp_path,
        functional=0,
        spin="unpolarized",
        plan=object(),
    )

    assert result == {
        "status": "missing_or_invalid",
        "detail": "invalid AOT manifest schema: 'list' object has no attribute 'get'",
    }


def test_unreadable_optional_aot_artifact_is_reported_not_raised(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def unreadable_loader(*_: object, **__: object) -> object:
        raise PermissionError("access denied")

    monkeypatch.setattr(
        qualify_capacity, "load_stationary_aot_artifact", unreadable_loader
    )
    result = qualify_capacity._artifact_verification(
        tmp_path,
        functional=0,
        spin="unpolarized",
        plan=object(),
    )

    assert result == {
        "status": "missing_or_invalid",
        "detail": "access denied",
    }


@pytest.mark.parametrize("minimum,blocked", [(512, True), (511, False)])
def test_device_budget_requires_a_positive_minimum_remainder(
    minimum: int, blocked: bool
) -> None:
    limits = {
        "native_owner_capacity": {
            "atom_count": 128,
            "ao_count": 1024,
            "basis_primitive_count": 16384,
        },
        "ao_task_fallback_capacity": {
            "atom_count": 32,
            "ao_count": 128,
            "basis_primitive_count": 4096,
        },
        "grid_work_capacity": {"grid_points": 1 << 40, "grid_pair_visits": 2**64 - 1},
        "public_work_limits": {"grid_points": None, "grid_pair_visits": None},
        "tile_points": 256,
        "pending_grid_pair_visits": 100_000_000,
        "primitive_records": 16_000_000,
        "primitive_logical_metric_limit": 2**64 - 1,
        "additional_device_bytes": 512,
        "additional_host_bytes": 256,
    }
    failures = qualify_capacity._case_failures(
        {
            "atom_count": 1,
            "ao_count_spherical": 1,
            "basis_primitive_count": 1,
        },
        {
            "primitive_records": 1,
            "primitive_descriptor_peak_records": 1,
            "grid_points": 1,
            "grid_pair_visits": 1,
        },
        {
            "additional_device_peak_bound": 512,
            "minimum_additional_device_bytes": minimum,
            "additional_host_numeric_bound": 1,
        },
        limits,
    )

    assert [item["gate"] for item in failures] == (
        ["additional_device_budget"] if blocked else []
    )


def test_primitive_descriptor_budget_is_page_local_and_ordered_after_host() -> None:
    limits = {
        "native_owner_capacity": {
            "atom_count": 128,
            "ao_count": 1024,
            "basis_primitive_count": 16384,
        },
        "ao_task_fallback_capacity": {
            "atom_count": 32,
            "ao_count": 128,
            "basis_primitive_count": 4096,
        },
        "grid_work_capacity": {"grid_points": 1 << 40, "grid_pair_visits": 2**64 - 1},
        "public_work_limits": {"grid_points": None, "grid_pair_visits": None},
        "tile_points": 256,
        "pending_grid_pair_visits": 100_000_000,
        "primitive_records": 16_000_000,
        "primitive_logical_metric_limit": 2**64 - 1,
        "additional_device_bytes": 512,
        "additional_host_bytes": 256,
    }
    failures = qualify_capacity._case_failures(
        {
            "atom_count": 1,
            "ao_count_spherical": 1,
            "basis_primitive_count": 1,
        },
        {
            "primitive_records": 16_000_001,
            "primitive_descriptor_peak_records": 16_000_001,
            "grid_points": 1,
            "grid_pair_visits": 1,
        },
        {
            "additional_device_peak_bound": 1,
            "additional_host_numeric_bound": 1,
        },
        limits,
    )

    assert [item["gate"] for item in failures] == ["primitive_descriptor_page_budget"]


def test_logical_primitive_metric_retains_uint64_range_gate() -> None:
    limits = {
        "native_owner_capacity": {
            "atom_count": 128,
            "ao_count": 1024,
            "basis_primitive_count": 16384,
        },
        "ao_task_fallback_capacity": {
            "atom_count": 32,
            "ao_count": 128,
            "basis_primitive_count": 4096,
        },
        "grid_work_capacity": {"grid_points": 1 << 40, "grid_pair_visits": 2**64 - 1},
        "public_work_limits": {"grid_points": None, "grid_pair_visits": None},
        "tile_points": 256,
        "pending_grid_pair_visits": 100_000_000,
        "primitive_records": 16_000_000,
        "primitive_logical_metric_limit": 2**64 - 1,
        "additional_device_bytes": 512,
        "additional_host_bytes": 256,
    }
    failures = qualify_capacity._case_failures(
        {
            "atom_count": 1,
            "ao_count_spherical": 1,
            "basis_primitive_count": 1,
        },
        {
            "primitive_records": 2**64,
            "primitive_descriptor_peak_records": 1,
            "grid_points": 1,
            "grid_pair_visits": 1,
        },
        {
            "additional_device_peak_bound": 1,
            "additional_host_numeric_bound": 1,
        },
        limits,
    )

    assert [item["gate"] for item in failures] == ["primitive_logical_metric_range"]


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("and self.owner == owner", "and True"),
        ("np.array_equal", "np.allclose"),
        ("128 << 20", "256 << 20"),
    ],
)
def test_snapshot_grid_cache_identity_and_cap_changes_fail_closed(
    tmp_path: Path, before: str, after: str
) -> None:
    source = (ROOT / "python/generativeqc/_stationary_cuda.py").read_text()
    stationary_contract_tree(tmp_path, source)
    path = tmp_path / "python/generativeqc/_snapshot_grid_cache.py"
    original = path.read_text()
    assert before in original
    path.write_text(original.replace(before, after))
    with pytest.raises(RuntimeError, match="grid-cache contract changed"):
        qualify_capacity._source_limits(tmp_path)


def test_geometry_resource_budget_changes_fail_closed(tmp_path: Path) -> None:
    source = (ROOT / "python/generativeqc/_stationary_cuda.py").read_text()
    stationary_contract_tree(tmp_path, source)
    path = tmp_path / "python/generativeqc_compiler/method/stationary_resources.py"
    path.write_text(
        path.read_text().replace(
            "GEOMETRY_MAX_SCRATCH_BYTES = 8 << 20",
            "GEOMETRY_MAX_SCRATCH_BYTES = 16 << 20",
        )
    )
    with pytest.raises(RuntimeError, match="geometry-resource contract changed"):
        qualify_capacity._source_limits(tmp_path)


@pytest.mark.parametrize(
    "marker,old,new,gate",
    [
        (
            "void launch_geometry(",
            "owner.becke_threads_per_point > 1",
            "false",
            "native_launch_geometry_sha256",
        ),
        (
            "int stationary_configure_becke(",
            "p->atoms > stationary_becke_max_atoms",
            "p->atoms > 128",
            "native_configure_becke_sha256",
        ),
    ],
)
def test_cooperative_native_schedule_contract_fails_closed(
    tmp_path: Path, marker: str, old: str, new: str, gate: str
) -> None:
    source = (ROOT / "python/generativeqc/_stationary_cuda.py").read_text(
        encoding="utf-8"
    )
    stationary_contract_tree(tmp_path, source)
    target = tmp_path / "src/dft/stationary_gradient_cuda.cuh"
    native = target.read_text(encoding="utf-8")
    position = native.index(old, native.index(marker))
    target.write_text(
        native[:position] + new + native[position + len(old) :], encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match=f"{gate} contract changed"):
        qualify_capacity._source_limits(tmp_path)


def current_admission_fixture() -> tuple[dict, dict, dict, dict]:
    limits = qualify_capacity._source_limits(ROOT)
    limits["public_work_limits"] = qualify_capacity._source_public_route(ROOT)[
        "whole_grid_work_limits"
    ]
    shape = {"atom_count": 2, "ao_count_spherical": 2, "basis_primitive_count": 2}
    requirements = {
        "primitive_records": 1,
        "primitive_descriptor_peak_records": 1,
        "grid_points": 1,
        "grid_pair_visits": 3,
    }
    memory = {
        "minimum_additional_device_bytes": 1,
        "additional_device_peak_bound": 1,
        "additional_host_numeric_bound": 1,
    }
    return limits, shape, requirements, memory


@pytest.mark.parametrize(
    "key,cap",
    [
        ("atom_count", 128),
        ("ao_count_spherical", 1024),
        ("basis_primitive_count", 16384),
    ],
)
def test_current_native_owner_capacity_inclusive_boundary(key: str, cap: int) -> None:
    limits, shape, requirements, memory = current_admission_fixture()
    shape[key] = cap
    assert qualify_capacity._case_failures(shape, requirements, memory, limits) == []
    shape[key] = cap + 1
    failures = qualify_capacity._case_failures(shape, requirements, memory, limits)
    assert [failure["gate"] for failure in failures] == ["native_owner_capacity"]
    assert (
        failures[0]["owner"]["function"] == "stationary_cuda_requires_native_integrals"
    )


@pytest.mark.parametrize(
    "key,cap",
    [("grid_points", 1_000_000), ("grid_pair_visits", 100_000_000)],
)
def test_diagnostic_guards_are_optional_and_distinct_from_public_work(
    key: str, cap: int
) -> None:
    limits, shape, requirements, memory = current_admission_fixture()
    requirements[key] = cap
    assert (
        qualify_capacity._case_failures(
            shape, requirements, memory, limits, work_mode="diagnostic"
        )
        == []
    )
    requirements[key] += 1
    failures = qualify_capacity._case_failures(
        shape, requirements, memory, limits, work_mode="diagnostic"
    )
    assert len(failures) == 1
    assert failures[0]["required"] == cap + 1
    assert failures[0]["cap"] == cap
    assert failures[0]["owner"]["function"] == "plan_stationary_cuda_grid_work"
    assert qualify_capacity._case_failures(shape, requirements, memory, limits) == []


def test_public_complete_grid_still_enforces_pending_window_work() -> None:
    limits, shape, requirements, memory = current_admission_fixture()
    shape["atom_count"] = 96
    requirements.update(grid_points=2_359_296, grid_pair_visits=21_516_784_080)
    visits_per_tile = 2 * limits["tile_points"] * (96 * 95 // 2)
    limits["pending_grid_pair_visits"] = visits_per_tile
    assert qualify_capacity._case_failures(shape, requirements, memory, limits) == []
    limits["pending_grid_pair_visits"] -= 1
    failures = qualify_capacity._case_failures(shape, requirements, memory, limits)
    assert [failure["gate"] for failure in failures] == ["pending_grid_pair_budget"]
    assert failures[0]["required"] == visits_per_tile


@pytest.mark.parametrize(
    "key,cap", [("grid_points", 1 << 40), ("grid_pair_visits", 2**64 - 1)]
)
def test_public_work_retains_representable_capacity(key: str, cap: int) -> None:
    limits, shape, requirements, memory = current_admission_fixture()
    requirements[key] = cap
    assert qualify_capacity._case_failures(shape, requirements, memory, limits) == []
    requirements[key] += 1
    failures = qualify_capacity._case_failures(shape, requirements, memory, limits)
    assert [failure["gate"] for failure in failures] == ["grid_work_capacity"]


def test_native_required_domain_never_admits_ao_descriptor_fallback_work() -> None:
    limits, shape, requirements, memory = current_admission_fixture()
    requirements["primitive_descriptor_peak_records"] = limits["primitive_records"] + 1
    assert [
        failure["gate"]
        for failure in qualify_capacity._case_failures(
            shape, requirements, memory, limits
        )
    ] == ["primitive_descriptor_page_budget"]
    # The public larger-domain route must succeed through complete native sources
    # or fail at runtime; the unused AO descriptor page is not its capacity gate.
    shape["ao_count_spherical"] = 129
    assert qualify_capacity._case_failures(shape, requirements, memory, limits) == []


def test_paired_host_reserve_is_charged_before_inclusive_host_admission() -> None:
    result = report()
    row = next(
        row
        for row in result["rows"]
        if row["id"] == "pbe0/rks/water32/fp64_energy_forces"
    )
    memory = row["resource_requirements"]
    assert memory["stationary_native_integral_host_reserve_bytes"] == 4_851_008
    assert memory["additional_host_numeric_bound"] == 197_047_712
    assert memory["additional_device_peak_bound"] == (
        memory["stationary_grid_device_peak_bound"]
        + memory["stationary_native_pair_reserve_bytes"]
    )
    assert memory["native_integral_device_budget"] == (
        memory["additional_device_budget"] - memory["stationary_grid_device_peak_bound"]
    )
    limits = result["admission_limits"]
    case = next(case for case in result["cases"] if case["id"] == "water32")
    limits["additional_host_bytes"] = memory["additional_host_numeric_bound"]
    assert qualify_capacity._case_failures(case["shape"], memory, memory, limits) == []
    limits["additional_host_bytes"] -= 1
    failures = qualify_capacity._case_failures(case["shape"], memory, memory, limits)
    assert [failure["gate"] for failure in failures] == ["additional_host_budget"]


def test_logical_primitive_work_and_native_work_are_method_specific() -> None:
    result = report()
    rows = {row["id"]: row for row in result["rows"]}
    pbe = rows["pbe/rks/water32/fp64_energy_forces"]["resource_requirements"]
    pbe0 = rows["pbe0/rks/water32/fp64_energy_forces"]["resource_requirements"]
    assert pbe["primitive_records"] == 2_963_193_862_608
    assert pbe0["primitive_records"] == 5_926_219_028_944
    for requirement in (pbe, pbe0):
        native = requirement["native_integral_admission"]
        assert native["required"] is True
        assert native["provider_and_budget_qualification"] == "NOT_RUN"
        assert native["stationary_primitive_records_if_native_complete"] == 4560
        assert native["ao_task_descriptors_if_native_complete"] == 0


@pytest.mark.parametrize(
    "old,new",
    [
        ("STATIONARY_MAX_ATOMS = 128", "STATIONARY_MAX_ATOMS = 129"),
        ("STATIONARY_MAX_AOS = 1024", "STATIONARY_MAX_AOS = 1025"),
        ("STATIONARY_MAX_PRIMITIVES = 16384", "STATIONARY_MAX_PRIMITIVES = 16385"),
        (
            "return atoms > 32 or aos > 128 or primitives > 4096",
            "return atoms > 96 or aos > 768 or primitives > 4096",
        ),
        (
            "if max_grid_points is not None and grid_points > max_grid_points:",
            "if max_grid_points is not None and grid_points >= max_grid_points:",
        ),
        ("if tile_visits > max_pending_pair_visits:", "if False:"),
        (
            "yield begin, min(begin + self.chunk_points, self.grid_points)",
            "yield begin, min(begin + self.chunk_points - 1, self.grid_points)",
        ),
        ("2 * chunk_points * pairs,", "chunk_points * pairs,"),
        ("+ 48 * atoms", "+ 24 * atoms"),
    ],
)
def test_current_resource_admission_and_work_changes_fail_closed(
    tmp_path: Path, old: str, new: str
) -> None:
    stationary_contract_tree(
        tmp_path, (ROOT / "python/generativeqc/_stationary_cuda.py").read_text()
    )
    path = tmp_path / "python/generativeqc_compiler/method/stationary_resources.py"
    source = path.read_text()
    assert old in source
    path.write_text(source.replace(old, new, 1))
    with pytest.raises(RuntimeError, match="geometry-resource contract changed"):
        qualify_capacity._source_limits(tmp_path)


@pytest.mark.parametrize(
    "old,new,message",
    [
        (
            "grid_points=len(state.grid.points),",
            "grid_points=len(state.grid.points) // 2,",
            "bounded grid-work plan definition changed",
        ),
        (
            "max_pending_tiles=max_pending_grid_tiles,",
            "max_pending_tiles=4096,",
            "bounded grid-work plan definition changed",
        ),
        (
            "max_pending_pair_visits=max_pending_grid_pair_visits,",
            "max_pending_pair_visits=1 << 40,",
            "bounded grid-work plan definition changed",
        ),
        (
            "    host_bound += native_integral_host_reserve\n",
            "",
            "native host-reserve admission order changed",
        ),
        (
            "if requires_native_integrals and not native_complete_integrals:",
            "if False:",
            "endpoint owner contract changed",
        ),
        (
            "for chunk_begin, chunk_end in grid_work.chunks():",
            "for chunk_begin, chunk_end in [(0, grid_points)]:",
            "endpoint owner contract changed",
        ),
        (
            "                sources.drain_geometry()\n",
            "                pass\n",
            "endpoint owner contract changed",
        ),
        (
            '        or work["xc_points"] != grid_work.grid_points\n',
            "",
            "endpoint owner contract changed",
        ),
        (
            'work["grid_pair_visits"] != pair_visits',
            'work["grid_pair_visits"] > pair_visits',
            "endpoint owner contract changed",
        ),
        (
            "            > native_integral_host_reserve\n",
            "            > max_host_bytes\n",
            "endpoint owner contract changed",
        ),
    ],
)
def test_current_endpoint_windows_native_requirement_and_reserve_fail_closed(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    source = (ROOT / "python/generativeqc/_stationary_cuda.py").read_text()
    assert old in source
    stationary_contract_tree(tmp_path, source.replace(old, new, 1))
    with pytest.raises(RuntimeError, match=message):
        qualify_capacity._source_limits(tmp_path)


@pytest.mark.parametrize(
    "old,new,message",
    [
        (
            "min(requested_bytes, max_host_bytes - host_bound)",
            "requested_bytes",
            "_stationary_ao_map_reserve contract changed",
        ),
        (
            "    host_bound += ao_map_reserve\n",
            "",
            "AO-map reserve admission changed",
        ),
        (
            "        grid.geometry_generation,\n",
            "        0,\n",
            "_stationary_resident_ao_cache contract changed",
        ),
        (
            "    state._source.check_current()\n    domain = ResidentAoMapDomain(",
            "    domain = ResidentAoMapDomain(",
            "_stationary_resident_ao_cache contract changed",
        ),
        (
            (
                "            sum(value.host_bytes for value in tensor_plans.values())\n"
                "            if prepared is not None\n"
            ),
            "            0\n            if prepared is not None\n",
            "AO-map reserve admission changed",
        ),
    ],
)
def test_optional_resident_ao_admission_and_lifetime_fail_closed(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    """Optional maps cannot escape the original cap or stale-owner checks."""
    source = (ROOT / "python/generativeqc/_stationary_cuda.py").read_text()
    assert old in source
    stationary_contract_tree(tmp_path, source.replace(old, new, 1))
    with pytest.raises(RuntimeError, match=message):
        qualify_capacity._source_limits(tmp_path)


def test_prepared_request_cannot_drop_the_resident_ao_policy(tmp_path: Path) -> None:
    source = (ROOT / "python/generativeqc/_stationary_cuda.py").read_text()
    old = '"resident_ao_cutoff": resident_ao_cutoff,'
    assert old in source
    stationary_contract_tree(
        tmp_path, source.replace(old, '"resident_ao_cutoff": None,', 1)
    )
    with pytest.raises(RuntimeError, match="AO request contract changed"):
        qualify_capacity._prepared_aot_route_contract(tmp_path)


@pytest.mark.parametrize("key", ["max_grid_points", "max_grid_pair_visits"])
def test_public_override_cannot_silently_inherit_diagnostic_limits(
    tmp_path: Path, key: str
) -> None:
    copy_contract_files(tmp_path, PUBLIC_ROUTE_FILES)
    path = tmp_path / "python/generativeqc/batch.py"
    source = path.read_text()
    old = f'"{key}": None,'
    assert old in source
    path.write_text(source.replace(old, f'"{key}": 1_000_000,', 1))
    with pytest.raises(
        RuntimeError, match="public complete-grid work override changed"
    ):
        qualify_capacity._source_public_route(tmp_path)


@pytest.mark.parametrize("points,blocked", [(0, True), (1, False)])
def test_empty_grid_rejected_but_one_atom_zero_pair_work_is_valid(
    points: int, blocked: bool
) -> None:
    limits, shape, requirements, memory = current_admission_fixture()
    shape["atom_count"] = 1
    requirements.update(grid_points=points, grid_pair_visits=0)
    failures = qualify_capacity._case_failures(shape, requirements, memory, limits)
    assert [failure["gate"] for failure in failures] == (
        ["grid_work_capacity"] if blocked else []
    )


@pytest.mark.parametrize(
    "before,after",
    [
        ("view.nao == aos", "view.nao <= aos"),
        ("view.nactive <= aos", "view.nactive <= aos + 1"),
        ("view.ao_ids != nullptr", "true"),
    ],
)
def test_local_ao_admission_helper_remains_bound_to_full_capacity_report(
    tmp_path: Path, before: str, after: str
) -> None:
    """Out-of-line admission drift must not silently reuse the old census."""
    source = (ROOT / "python/generativeqc/_stationary_cuda.py").read_text()
    stationary_contract_tree(tmp_path, source)
    target = tmp_path / "src/dft/stationary_gradient_cuda.cuh"
    native = target.read_text()
    start = native.index("bool valid_geometry_ao_map(")
    end = native.index("\n}", start)
    original = native[start:end]
    assert original.count(before) == 1
    target.write_text(native[:start] + original.replace(before, after) + native[end:])
    with pytest.raises(
        RuntimeError, match="native_geometry_ao_map_sha256 contract changed"
    ):
        qualify_capacity._source_limits(tmp_path)


def test_local_ao_capacity_stays_global_and_pair_work_stays_complete() -> None:
    """Admitting local panels is not authority to shrink the fixed upper bound."""
    limits = qualify_capacity._source_limits(ROOT)
    assert "active_ao_capacity=n" in limits["grid_plan_definition"]
    assert "grid_plan.host_bytes" in limits["host_bound_definition"]
    assert limits["grid_pair_visits_definition"] == "grid_work.grid_pair_visits"
    # The existing exact source bindings additionally cover allocations, launch
    # scratch, the deferred route, and every counter in the geometry body.
    native = (ROOT / "src/dft/stationary_gradient_cuda.cuh").read_text()
    for marker in (
        "int stationary_geometry_external(",
        "int stationary_geometry_enqueue(",
    ):
        start = native.index(marker)
        end = native.index("\n}\n", start)
        body = native[start:end]
        assert "p->point_count += view->npoint;" in body
        assert "p->pair_visits += view->npoint * p->atoms * (p->atoms - 1);" in body
        assert "std::min(p->geometry_lanes, view->npoint)" in body
        assert "if (!view->nactive)" not in body
