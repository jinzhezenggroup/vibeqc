"""Report frozen DFT-MP-v1 stationary CUDA capacity without running science.

The report resolves the committed inputs, bundled basis expansion, frozen grid,
current stationary plan and current fail-closed limits.  It performs no native
library load, CUDA initialization, SCF calculation or scientific compilation.
Passing this static report is therefore only a preflight result, never a
scientific qualification.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.machinery
import inspect
import json
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

SOURCE_REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE_PYTHON = SOURCE_REPOSITORY / "python"
_DIRECT_SOURCE_EXECUTION = __name__ == "__main__" and __spec__ is None

if TYPE_CHECKING:
    from collections.abc import Sequence
    from importlib.machinery import ModuleSpec
    from types import CodeType, ModuleType


class _DftMpSourceOnlyLoader(importlib.machinery.SourceFileLoader):
    """Compile current source directly, bypassing every bytecode cache."""

    def get_code(self, fullname: str) -> CodeType:
        source = self.get_data(self.path)
        return self.source_to_code(source, self.path)


class _DftMpSourceOnlyFinder:
    """Use source-only loading for repository-local qualifier dependencies."""

    @staticmethod
    def find_spec(
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        if fullname not in (
            "generativeqc",
            "generativeqc_compiler",
        ) and not fullname.startswith(("generativeqc.", "generativeqc_compiler.")):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.origin is None or not spec.origin.endswith(".py"):
            return None
        origin = Path(spec.origin).resolve()
        if not origin.is_relative_to(SOURCE_PYTHON):
            return None
        spec.loader = _DftMpSourceOnlyLoader(fullname, str(origin))
        spec.cached = None
        return spec


_SOURCE_ONLY_FINDER = _DftMpSourceOnlyFinder()
sys.meta_path.insert(0, _SOURCE_ONLY_FINDER)

_PRELOADED_LOCAL_MODULES = frozenset(
    name
    for name, module in tuple(sys.modules.items())
    if module is not None
    and (source := getattr(module, "__file__", None)) is not None
    and Path(source).resolve().is_relative_to(SOURCE_PYTHON)
)


def _git_head(repository: Path) -> str:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise RuntimeError("Git did not return a full source SHA")
    return revision


_IMPORTED_TOOL_HEAD = _git_head(SOURCE_REPOSITORY)
_IMPORTED_TOOL_SOURCE = Path(__file__).resolve()
_IMPORTED_TOOL_SOURCE_SHA256 = hashlib.sha256(
    _IMPORTED_TOOL_SOURCE.read_bytes().replace(b"\r\n", b"\n")
).hexdigest()

source_python = str(SOURCE_PYTHON)
if source_python in sys.path:
    sys.path.remove(source_python)
sys.path.insert(0, source_python)

import numpy as np
from generativeqc import Atom
from generativeqc import _generated_methods as generated_methods
from generativeqc._model_resolution import snapshot_basis
from generativeqc._stationary_cuda import (
    COMPONENT_LABELS,
    complete_rks_cuda_gradient_diagnostic,
)
from generativeqc.basis import BasisSet
from generativeqc.basis_capabilities import resolved_basis_metadata
from generativeqc.calculator import _basis_pack, _named_basis_record
from generativeqc.ks import (
    KsOptions,
    ks_coefficients,
    native_dft_carrier,
    native_xc_functional_code,
    resolve_ks_method,
    resolve_ks_options,
)
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.dft.ao import jet_indices
from generativeqc_compiler.dft.grid import GridSpec, MolecularGrid
from generativeqc_compiler.dft.plan import plan_tiles
from generativeqc_compiler.method.stationary_cuda import (
    QUALIFIED_SPD_COMPONENTS,
    STATIONARY_RUNTIME_SOURCE_NAMES,
    _qualified_aot_plan,
    _stationary_aot_name,
    load_stationary_aot_artifact,
    stationary_aot_profile_contract_identity,
    stationary_runtime_sources,
)
from generativeqc_compiler.method.stationary_gradient import (
    SCF_POINT_MODEL,
    StationaryGradientPlan,
    StationaryMeanField,
)
from generativeqc_compiler.method.stationary_resources import (
    STATIONARY_MAX_AOS,
    STATIONARY_MAX_ATOMS,
    STATIONARY_MAX_PRIMITIVES,
    plan_stationary_cuda_grid_work,
    plan_stationary_cuda_resources,
    stationary_cuda_allocation_bytes,
    stationary_cuda_requires_native_integrals,
    stationary_native_pair_reserve,
)

if _SOURCE_ONLY_FINDER in sys.meta_path:
    sys.meta_path.remove(_SOURCE_ONLY_FINDER)

_LOCAL_HELPERS = {
    "Atom": Atom,
    "generated_methods": generated_methods,
    "snapshot_basis": snapshot_basis,
    "BasisSet": BasisSet,
    "complete_rks_cuda_gradient_diagnostic": complete_rks_cuda_gradient_diagnostic,
    "resolved_basis_metadata": resolved_basis_metadata,
    "KsOptions": KsOptions,
    "ks_coefficients": ks_coefficients,
    "native_dft_carrier": native_dft_carrier,
    "native_xc_functional_code": native_xc_functional_code,
    "resolve_ks_method": resolve_ks_method,
    "resolve_ks_options": resolve_ks_options,
    "_basis_pack": _basis_pack,
    "_named_basis_record": _named_basis_record,
    "GridSpec": GridSpec,
    "MolecularGrid": MolecularGrid,
    "jet_indices": jet_indices,
    "plan_tiles": plan_tiles,
    "_qualified_aot_plan": _qualified_aot_plan,
    "load_stationary_aot_artifact": load_stationary_aot_artifact,
    "stationary_aot_profile_contract_identity": stationary_aot_profile_contract_identity,
    "stationary_runtime_sources": stationary_runtime_sources,
    "StationaryGradientPlan": StationaryGradientPlan,
    "StationaryMeanField": StationaryMeanField,
    "plan_stationary_cuda_grid_work": plan_stationary_cuda_grid_work,
    "plan_stationary_cuda_resources": plan_stationary_cuda_resources,
    "stationary_cuda_allocation_bytes": stationary_cuda_allocation_bytes,
    "stationary_cuda_requires_native_integrals": stationary_cuda_requires_native_integrals,
    "stationary_native_pair_reserve": stationary_native_pair_reserve,
}


def _helper_source_path(helper: Any) -> Path | None:
    module = helper if inspect.ismodule(helper) else inspect.getmodule(helper)
    source = None if module is None else getattr(module, "__file__", None)
    return None if source is None else Path(source).resolve()


_IMPORTED_HELPER_SOURCES = {
    name: (
        path,
        hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
    )
    for name, helper in _LOCAL_HELPERS.items()
    if (path := _helper_source_path(helper)) is not None
}
_PUBLIC_METHOD_MANIFEST = SOURCE_REPOSITORY / "manifests/public_methods.json"
_IMPORTED_DATA_DEPENDENCIES = {
    "public_methods_manifest": (
        _PUBLIC_METHOD_MANIFEST,
        hashlib.sha256(
            _PUBLIC_METHOD_MANIFEST.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
    )
}
_IMPORTED_LOCAL_MODULE_SOURCES = {
    name: (
        path,
        hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
    )
    for name, module in tuple(sys.modules.items())
    if module is not None
    and (source := getattr(module, "__file__", None)) is not None
    and (path := Path(source).resolve()).is_relative_to(SOURCE_PYTHON)
}

SCHEMA = "generativeqc.dft-mp-v1.stationary-capacity.v2"
SEMILOCAL_FUNCTIONALS = {"lda": 0, "pbe": 1, "r2scan": 2}
FP64_FORCE_METHODS = frozenset((*SEMILOCAL_FUNCTIONALS, "pbe0", "b3lyp"))
SEMILOCAL_ABI_IDS = {
    "lda-rks": 6,
    "pbe-rks": 7,
    "lda-uks": 8,
    "pbe-uks": 9,
    "r2scan-rks": 10,
    "r2scan-uks": 11,
}
SPARSE_SPHERICAL_COMPONENT_TERMS = {0: 1, 1: 3, 2: 8}
SPD_EXPANSION_CONTRACT_SHA256 = (
    "a482b3ace40fd01758f504c1e47bfa795dcabb5483149873fcd36e71c6b47ae0"
)
AO_PACKER_CONTRACT_SHA256 = (
    "07858ba7f9a78fe6348bbcb9430eb4f8321db8774ea3ce1ecef495629abe2a1c"
)
AO_PACK_BRIDGE_CONTRACT_SHA256 = (
    "aba8691ce81438c66517604c5b2e1e0d39df474888763b4d44686ca8816fabbc"
)
NATIVE_AO_CONSTRUCTOR_CONTRACT_SHA256 = (
    "9e210b215cd43f21af6899b328ee738b87f53eaa9b20654b16de760274a8c372"
)
BASIS_SNAPSHOT_CONTRACT_SHA256 = (
    "4dea9a2041897bf843012c01f64c580b6b8696f444611a54c0d161141f1894fd"
)
BASIS_SHELL_EXPANSION_CONTRACT_SHA256 = (
    "300a64c1815273cf31ed5b463eac2e32f24bf4a1db5938975ffbad5cfb27f60c"
)
CALCULATOR_SHELL_FORWARDING_CONTRACT_SHA256 = (
    "af7bd2da7d571fb6d92dee7f984bf8de95f32b263c8b75fe69555772d60a44b6"
)
NATIVE_SYSTEM_BASIS_FORWARDING_CONTRACT_SHA256 = (
    "120936d57b90062a8a888892ce4d514ca8a62a0b2ced671afe3fce660b0414d5"
)
STATIONARY_LAYOUT_CONTRACT_SHA256 = (
    "fa8c4ff2a644fd45ab4eb828a995c4e42c49c80adbe712b32d50f90b3d98fb74"
)
NATIVE_SPHERICAL_AO_COUNT_CONTRACT_SHA256 = (
    "b6e7a3a70accf7f4abeb82f0168634ae33b7c58f282044b8a9cd0462672200f0"
)
PUBLIC_SEMILOCAL_FORCE_CONTRACT_SHA256 = (
    "fba0a84cb3d993919caf6e6d10391239598ef876cda41123d683479fccf767e0"
)
PUBLIC_FORCE_PROMOTION_CONTRACT_SHA256 = (
    "49f903598301e16b11be96d1b24eb084aa7bee3194942702b174b41e59d4b01c"
)
PUBLIC_CUDA_FORCE_METHOD_CONTRACT_SHA256 = (
    "dcfcbef93e798c62cc5669e93190a9b73184ffe120a8afc43730a9dbc74cb448"
)
PUBLIC_CUDA_HYBRID_FORCE_CONTRACT_SHA256 = (
    "18f4f010596672eb47b8d085e28b8a26373c41178ac1c6a5ff4fa705ef2f3944"
)
PYTHON_GRID_CONTRACT_SHA256 = (
    "03a43444cd793167823c0c30c0b66b51c2a464d8f65946118dd781813dc7f0a4"
)
QUADRATURE_LAYOUT_CONTRACT_SHA256 = (
    "1674152aa312f3769d8b78be60aff491cc52f577d74b8d889d133bff3064ff23"
)
NATIVE_CUDA_GRID_CONTRACT_SHA256 = (
    "0f5c74f638f51833f3242d369df019362e0df08c3602f2c2cc645266c46d53ed"
)
NATIVE_GRID_ROUTE_CONTRACT_SHA256 = (
    "98b6763435fcd84c6306ea034c330a9864ecf4d33d0009031230f7b64e77b71e"
)
NATIVE_GRID_POINT_COUNT_CONTRACT_SHA256 = (
    "92cd50078b7a96f371ed8d4fcdb77930b8c472134bd1e97bba803ac445d85867"
)
PUBLIC_GRID_ABI_CONTRACT_SHA256 = (
    "b64b0ca7be75b9425c32220476d5eda9b5f013168bd68c856a448fdd320a679e"
)
NATIVE_GRID_ABI_CONTRACT_SHA256 = (
    "d4930bf86b781cd4a77f152380439ac8a6b168d1846f42325bb2d6a3e7e638e4"
)
STATIONARY_AOT_CMAKE_CONTRACT_SHA256 = (
    "2ddd2b7a77c992bc9812914e1a31b8dbbc1a332d4fab8b3ec92c239ff6ff2228"
)
STATIONARY_PAGE_FLUSH_CONTRACT_SHA256 = (
    "1c2e0bb83a12eed7113825855cbe2164f53366b6bb270dd6c1247b498737c77b"
)
STATIONARY_PAGE_INITIALIZER_CONTRACT_SHA256 = (
    "7bb03b7286a5868527104420a74f4749c8af92360f17ad12ced31051a7ae9d75"
)
STATIONARY_PAGE_BULK_CONTRACT_SHA256 = (
    "b7bc1344bd86447cd6c9efcdfef944bb22c8b92b5ed5327d2028cf787d6a1729"
)
STATIONARY_PAGE_SCALAR_CONTRACT_SHA256 = (
    "c5b8ef983462f6c56ebfe6bd6eb8d5cf98f92f36f3e5425504b596730846205f"
)
STATIONARY_PAGE_COMPONENT_INTEGRAL_CONTRACT_SHA256 = (
    "d3f61e820c8bcf0df4bf4fce639f342936b79aceb956cb6e9f43caa3d13cdaa3"
)
STATIONARY_PAGE_NUCLEAR_CONTRACT_SHA256 = (
    "1e86737d8732ef8637378ab925f829dfe229bcf049219c2705a0a4fbf7afdb85"
)
STATIONARY_PAGE_GEOMETRY_CONTRACT_SHA256 = (
    "d469560a2b776a9b86ff5082ba35d3f8ae956c0d63df76aec1ab39550ab92a30"
)
STATIONARY_PAGE_FINISH_SPAN_CONTRACT_SHA256 = (
    "419e21953688eb24d214e6fb43d33ca974cb94632c3797e3f4f0113704d9a1f9"
)
STATIONARY_COMPONENT_MODE_CONTRACT_SHA256 = (
    "8d9819961d3014d161aff8c5c798f926fe6f1d9de54b84a725fdf2f6694b76bb"
)
STATIONARY_TASK_EXECUTOR_CONTRACT_SHA256 = (
    "5f4bf38658ca1da15d845e1a65ebd72195191aebf6009873f131c2cbe8a34195"
)
STATIONARY_SUBMIT_PAGE_CONTRACT_SHA256 = (
    "2fcd280569106fe3cbcf1256a02693aa3532e7db0fa62f7c97d7319454a62a48"
)
STATIONARY_NUCLEAR_PAIR_LOOP_CONTRACT_SHA256 = (
    "5a69bf4fd85d28b137e1ae35bce4a1d32134375bbaca9f66f60c9377a0c8f935"
)
STATIONARY_ENDPOINT_OWNER_CONTRACT_SHA256 = (
    "0d9571918d55d9b9d66bd5d40fa8f38d595d1c20c6c58fb04c155467d9c97a81"
)
STATIONARY_AO_MAP_RESERVE_CONTRACT_SHA256 = (
    "c2ba1b47e5655f75196c79384079d4ecd9c1edf9fe9aa1e97e339e96cbbaad0a"
)
STATIONARY_AO_MAP_CACHE_CONTRACT_SHA256 = (
    "dcf02f08e8d0279dcb49f2093d0ac3bf023523a7519a0faf6537fc9140711c9f"
)
NATIVE_KS_SNAPSHOT_INIT_CONTRACT_SHA256 = (
    "522c7571c3d18db25685ffbffb55279deadde63df64ee4c8b330f04017f7b3ae"
)
NATIVE_KS_SNAPSHOT_DECODE_CONTRACT_SHA256 = (
    "5bd874ce5ba8d4d5d78eb282244144052822bb2f0266d40d33361735302b1a97"
)
SNAPSHOT_GRID_CACHE_CONTRACT_SHA256 = (
    "569705abf406d2ec00ec9526e84f23301448d5511fc2bf79ee9ef6993a794ca6"
)
STATIONARY_PUBLIC_WRAPPER_CONTRACT_SHA256 = (
    "56b9ca1e1cd87f51b3cf0d57fcfbb46c8fed4b32a1808fb49e53b545019958dc"
)
NATIVE_STATIONARY_OWNER_CONTRACT_SHA256 = (
    "47af7a99e4aadfe4386e1a798e619ff52aea3143a254c7cbffcca08ee9b1c1db"
)
NATIVE_STATIONARY_ALLOCATION_CONTRACT_SHA256 = (
    "b0e739be97cb1048b86efeaa5c9b116cce1ac76f056e1ac610971f91df08129e"
)
NATIVE_STATIONARY_CREATE_CONTRACT_SHA256 = (
    "e3c05c687211842a710c0cada57e96b787992389c5e9a8799191655d07ae05b3"
)
NATIVE_STATIONARY_RESET_CONTRACT_SHA256 = (
    "e0bdfb373199c744de1dea05e912f7bbd24dff59b7e9a27e08e899bcaec8c1fc"
)
NATIVE_STATIONARY_TASKS_CONTRACT_SHA256 = (
    "5b0148f4f48019115a82e638d1d6671dd2548f3df6141da6e5254c8967bad2bc"
)
NATIVE_STATIONARY_NUCLEAR_CONTRACT_SHA256 = (
    "be4a553ba6117c7f772882a551d50817935954c5c4d66190e86d9bf2be043902"
)
# Local AO admission changes only collocation/contraction domain. The census
# still charges global n AO capacity and every molecular-grid/Becke point pair.
# Bind the out-of-line predicate itself so later weakening cannot hide behind
# unchanged callers and stale favorable memory/work bounds.
NATIVE_STATIONARY_GEOMETRY_AO_MAP_CONTRACT_SHA256 = (
    "d4830d6d9695219f4bf4c59611717b943c7aa1da016fdba67ceb6036241f1dc0"
)
NATIVE_STATIONARY_GEOMETRY_EXTERNAL_CONTRACT_SHA256 = (
    "e5a36f9b80f332b1e03a48e2b3c066e583b1ba9340b7a5feab4dab3d292be236"
)
NATIVE_STATIONARY_GEOMETRY_ENQUEUE_CONTRACT_SHA256 = (
    "cdac623e8296338a03b3b81bd3e77fbacb36730d4fb4502a2adbb6921a5f9544"
)
NATIVE_STATIONARY_GEOMETRY_ROUTE_CONTRACT_SHA256 = (
    "3fc0a5f613dfaa01ab02104e15929680f3f61fa17c07d59d54241201f903d476"
)
NATIVE_STATIONARY_LAUNCH_GEOMETRY_CONTRACT_SHA256 = (
    "7a06757b893e51c6924675c88c66c6d0817923904283bf48378fb4e8302f47be"
)
NATIVE_STATIONARY_CONFIGURE_BECKE_CONTRACT_SHA256 = (
    "dc844781c888d1bdd281238d4dd23c76048d17f816cb81b5a0616756a22ffe91"
)
NATIVE_STATIONARY_METRICS_CONTRACT_SHA256 = (
    "21e067818117b8ebaf8eeb218aeface0680681fdd6f39285ed6f981c3cef969a"
)
NATIVE_STATIONARY_FINISH_SPAN_CONTRACT_SHA256 = (
    "3f12a2c23709399c56776e34f5d7cd2394a95e153f754694bb7d523772efa431"
)
PREPARED_AOT_SELECTION_CONTRACT_SHA256 = (
    "a9d5f920839f2a020b17addaddb8e00c674bf78e37f112a113751d67ef336224"
)
PREPARED_AO_REQUEST_CONTRACT_SHA256 = (
    "ef5b67eebcd197e3c470f4b244252cbaae74daa2e8aeb3e58ee486f3ecaa8355"
)
PRIMITIVE_SUM_DEFINITION = (
    "sum((int(row[2]) * len(expansion) for row, expansion in "
    "zip(aos, expansions, strict=True)))"
)
PRIMITIVE_RECORDS_DEFINITION = (
    "(1 + int(has_exchange)) * primitive_sum ** 4 + "
    "(na + 2) * primitive_sum ** 2 + na * (na - 1) // 2"
)
GRID_PAIR_VISITS_DEFINITION = "grid_work.grid_pair_visits"
GRID_WORK_DEFINITION = (
    "plan_stationary_cuda_grid_work(atoms=na, grid_points=len(state.grid.points), "
    "tile_points=tile_points, max_grid_points=max_grid_points, "
    "max_grid_pair_visits=max_grid_pair_visits, "
    "max_pending_tiles=max_pending_grid_tiles, "
    "max_pending_pair_visits=max_pending_grid_pair_visits)"
)
NATIVE_REQUIREMENT_DEFINITION = "stationary_cuda_requires_native_integrals(atoms=na, aos=n, primitives=basis.nprimitive)"
NATIVE_HOST_RESERVE_DEFINITION = (
    "stationary_native_pair_reserve(atoms=na, aos=n, primitives=basis.nprimitive) "
    "if not ecp and (not bool(getattr(state._source, 'density_fitted', False))) "
    "and callable(getattr(state._source, 'cuda_integral_derivatives', None)) else 0"
)
METHOD_IR_DEFINITION = "state._source.method_ir"
FUNCTIONAL_LOWERING_DEFINITION = "int(state._source.metadata[6])"
INGREDIENTS_DEFINITION = "state._source.functional.ingredients"
NEEDS_FIRST_DEFINITION = "'sigma' in ingredients"
GRID_PLAN_DEFINITION = (
    "plan_tiles(basis, backend='cuda', order=2 if needs_first else 1, "
    "tile_points=tile_points, active_ao_capacity=n, budget_bytes=max_device_bytes)"
)
GEOMETRY_RESOURCES_CONTRACT_SHA256 = (
    "51730953ed9a62807442f672013fcc40787ed3c0b0cca838d6c492c8b1c080b1"
)
MINIMUM_SOURCE_BYTES_DEFINITION = (
    "stationary_cuda_allocation_bytes(atoms=na, aos=n, primitives=basis.nprimitive, "
    "points=tile_points, tasks=primitive_tile, spins=plan.spin_blocks, "
    "sources=len(source_names), geometry_lanes=min(32, tile_points))"
)
SOURCE_RESOURCES_DEFINITION = (
    "plan_stationary_cuda_resources(atoms=na, aos=n, primitives=basis.nprimitive, "
    "points=tile_points, tasks=primitive_tile, spins=plan.spin_blocks, "
    "sources=len(source_names), target=target, budget_bytes=max_device_bytes - "
    "grid_plan.peak_bytes - sum((value.peak_bytes for value in tensor_plans.values())) - native_geometry_reserve)"
)
SOURCE_BYTES_DEFINITION = "source_resources.allocation_bytes"
HOST_BOUND_DEFINITION = (
    "grid_plan.host_bytes + 8 * (34 * primitive_tile + "
    "4 * plan.spin_blocks * n * n + 120 * na + "
    "12 * (len(source_names) - len(_SOURCE_NAMES)) * na + "
    "26 * integral_terms + len(COMPONENT_LABELS) ** 4 + "
    "3 * len(COMPONENT_LABELS) ** 2 + 3 * tile_points + 2 * basis.nprimitive + "
    "4 * n + 80) + max((tp.host_bytes for tp in tensor_plans.values()), default=0)"
)
AVAILABLE_DEVICE_BYTES_DEFINITION = (
    "max_device_bytes - grid_plan.peak_bytes - minimum_source_bytes"
)
GATE_PREDICATES = {
    "primitive_metric_range": "records > np.iinfo(np.uint64).max",
    "additional_device": "available <= 0",
    "additional_host": "host_bound > max_host_bytes",
}
BASIS_PACKED_CAPACITY_DEFINITION = (
    "np.empty(3 * self.natom + 2 * self.nprimitive + 16 * self.nao)"
)
BASIS_NUMERIC_CAPACITY_DEFINITION = (
    "2 * self.packed.nbytes + 32 * self.natom + "
    "32 * len(self.shells) + 16 * self.nprimitive"
)
STATIONARY_OWNER = {
    "file": "python/generativeqc/_stationary_cuda.py",
    "function": "_complete_rks_cuda_gradient_diagnostic",
}


def _assert_local_imports(*, require_fresh: bool = False) -> None:
    """Reject helpers already imported from an installed or foreign checkout."""

    if _git_head(SOURCE_REPOSITORY) != _IMPORTED_TOOL_HEAD:
        raise RuntimeError(
            "capacity tool checkout changed since import; start a fresh interpreter"
        )
    if (
        hashlib.sha256(
            _IMPORTED_TOOL_SOURCE.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        != _IMPORTED_TOOL_SOURCE_SHA256
    ):
        raise RuntimeError(
            "capacity qualifier source changed since import; start a fresh interpreter"
        )
    if require_fresh and not _DIRECT_SOURCE_EXECUTION:
        raise RuntimeError(
            "authoritative capacity reports require direct source CLI execution"
        )
    if require_fresh and _PRELOADED_LOCAL_MODULES:
        raise RuntimeError(
            "capacity qualifier requires a fresh interpreter; preloaded local modules: "
            + ", ".join(sorted(_PRELOADED_LOCAL_MODULES))
        )
    foreign = []
    stale = []
    for name, original in _LOCAL_HELPERS.items():
        helper = globals()[name]
        source = _helper_source_path(helper)
        imported = _IMPORTED_HELPER_SOURCES.get(name)
        if (
            source is None
            or not source.is_relative_to(SOURCE_PYTHON)
            or imported is None
            or source != imported[0]
        ):
            foreign.append(name)
            continue
        current_digest = hashlib.sha256(
            source.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        if helper is not original or current_digest != imported[1]:
            stale.append(name)
    if foreign:
        raise RuntimeError(
            "capacity helper imported outside the tool checkout: "
            + ", ".join(sorted(foreign))
        )
    if stale:
        raise RuntimeError(
            "capacity helper source changed since import: " + ", ".join(sorted(stale))
        )
    stale_data = [
        name
        for name, (path, imported_digest) in _IMPORTED_DATA_DEPENDENCIES.items()
        if not path.is_relative_to(SOURCE_REPOSITORY)
        or not path.is_file()
        or hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        != imported_digest
    ]
    if stale_data:
        raise RuntimeError(
            "capacity import-time data dependency changed: "
            + ", ".join(sorted(stale_data))
        )
    stale_modules = [
        name
        for name, (path, imported_digest) in _IMPORTED_LOCAL_MODULE_SOURCES.items()
        if not path.is_file()
        or hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        != imported_digest
    ]
    if stale_modules:
        raise RuntimeError(
            "capacity imported module source changed: "
            + ", ".join(sorted(stale_modules))
        )
    if plan_tiles.__globals__.get("jet_indices") is not jet_indices:
        raise RuntimeError("capacity planner captured AO helper changed since import")


def _lf_sha256(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def _canonical_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_node_sha256(source: str, node: ast.AST) -> str:
    """Hash the selected source span without Python-version AST formatting."""

    segment = ast.get_source_segment(source, node)
    if segment is None:
        raise RuntimeError("source contract segment is unavailable")
    return _lf_sha256(segment.encode())


def _source_span_sha256(
    source: str,
    *,
    begin: str,
    end: str,
    label: str,
) -> str:
    try:
        start = source.index(begin)
        stop = source.index(end, start)
    except ValueError as error:
        raise RuntimeError(f"{label} source contract is missing") from error
    return _lf_sha256(source[start:stop].encode())


def _cpp_block_sha256(source: str, marker: str) -> str:
    try:
        start = source.index(marker)
        opening = source.index("{", start)
    except ValueError as error:
        raise RuntimeError(f"native source block is missing: {marker}") from error
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return _lf_sha256(source[start : index + 1].encode())
    raise RuntimeError(f"native source block is unterminated: {marker}")


def _snapshot_functional_contract(repository: Path) -> dict[str, str]:
    """Bind native snapshot selector provenance consumed by stationary CUDA."""

    source = (repository / "python/generativeqc/_ks_snapshot.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    owners = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NativeKsSnapshot"
    ]
    if len(owners) != 1:
        raise RuntimeError("native KS snapshot owner is missing or ambiguous")
    methods = {}
    expected = {
        "__init__": NATIVE_KS_SNAPSHOT_INIT_CONTRACT_SHA256,
        "decode": NATIVE_KS_SNAPSHOT_DECODE_CONTRACT_SHA256,
    }
    for name, expected_digest in expected.items():
        candidates = [
            node
            for node in owners[0].body
            if isinstance(node, ast.FunctionDef) and node.name == name
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"native KS snapshot {name} contract is missing or ambiguous"
            )
        digest = _source_node_sha256(source, candidates[0])
        if digest != expected_digest:
            raise RuntimeError("native KS snapshot functional contract changed")
        key = "init_sha256" if name == "__init__" else f"{name}_sha256"
        methods[key] = digest
    # Decode now delegates exact grid reuse. Bind the delegated content proof
    # and retention cap as well as the caller; a caller hash alone is insufficient.
    cache_source = repository / "python/generativeqc/_snapshot_grid_cache.py"
    cache_digest = _lf_sha256(cache_source.read_bytes())
    if cache_digest != SNAPSHOT_GRID_CACHE_CONTRACT_SHA256:
        raise RuntimeError("native KS snapshot grid-cache contract changed")
    methods["grid_cache_sha256"] = cache_digest
    return methods


def _source_limits(repository: Path) -> dict[str, Any]:
    """Bind current compiler/native capacity and private diagnostic defaults.

    Public whole-grid overrides are audited independently in _source_public_route.
    A static-cap pass remains conditional on the required native provider result.
    """

    snapshot_functional_contract = _snapshot_functional_contract(repository)
    source_path = repository / STATIONARY_OWNER["file"]
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == STATIONARY_OWNER["function"]
    ]
    if len(functions) != 1:
        raise RuntimeError("stationary CUDA admission owner is missing or ambiguous")
    owner = functions[0]
    wrappers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "complete_rks_cuda_gradient_diagnostic"
    ]
    if len(wrappers) != 1:
        raise RuntimeError("stationary CUDA public wrapper is missing or ambiguous")
    wrapper_digest = _source_node_sha256(source, wrappers[0])
    if wrapper_digest != STATIONARY_PUBLIC_WRAPPER_CONTRACT_SHA256:
        raise RuntimeError("stationary CUDA public wrapper contract changed")
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    page_methods = {
        "initializer": (
            "_CudaSources",
            "__init__",
            STATIONARY_PAGE_INITIALIZER_CONTRACT_SHA256,
        ),
        "flush": ("_CudaSources", "flush", STATIONARY_PAGE_FLUSH_CONTRACT_SHA256),
        "bulk": (
            "_CudaSources",
            "integral_page",
            STATIONARY_PAGE_BULK_CONTRACT_SHA256,
        ),
        "scalar": (
            "_CudaSources",
            "_append_task",
            STATIONARY_PAGE_SCALAR_CONTRACT_SHA256,
        ),
        "component_integral": (
            "_CudaSources",
            "integral",
            STATIONARY_PAGE_COMPONENT_INTEGRAL_CONTRACT_SHA256,
        ),
        "nuclear": (
            "_CudaSources",
            "nuclear",
            STATIONARY_PAGE_NUCLEAR_CONTRACT_SHA256,
        ),
        "geometry": (
            "_CudaSources",
            "geometry",
            STATIONARY_PAGE_GEOMETRY_CONTRACT_SHA256,
        ),
        "finish_span": (
            "_CudaSources",
            "finish_span",
            STATIONARY_PAGE_FINISH_SPAN_CONTRACT_SHA256,
        ),
        "executor": (
            "_BoundedStationaryTaskExecutor",
            "execute_pages",
            STATIONARY_TASK_EXECUTOR_CONTRACT_SHA256,
        ),
    }
    resource_source = (
        repository / "python/generativeqc_compiler/method/stationary_resources.py"
    )
    resource_digest = _lf_sha256(resource_source.read_bytes())
    if resource_digest != GEOMETRY_RESOURCES_CONTRACT_SHA256:
        raise RuntimeError("stationary CUDA geometry-resource contract changed")
    page_contract = {
        "public_wrapper_sha256": wrapper_digest,
        "geometry_resources_sha256": resource_digest,
    }
    for label, (class_name, method_name, expected_digest) in page_methods.items():
        class_node = classes.get(class_name)
        methods = (
            []
            if class_node is None
            else [
                node
                for node in class_node.body
                if isinstance(node, ast.FunctionDef) and node.name == method_name
            ]
        )
        if len(methods) != 1:
            raise RuntimeError(f"stationary CUDA {label} page owner is ambiguous")
        digest = _source_node_sha256(source, methods[0])
        if digest != expected_digest:
            raise RuntimeError(f"stationary CUDA {label} page contract changed")
        page_contract[f"{label}_sha256"] = digest
    component_modes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_component_mode"
    ]
    if len(component_modes) != 1:
        raise RuntimeError("stationary CUDA component-mode owner is ambiguous")
    component_mode_digest = _source_node_sha256(source, component_modes[0])
    if component_mode_digest != STATIONARY_COMPONENT_MODE_CONTRACT_SHA256:
        raise RuntimeError("stationary CUDA component-mode contract changed")
    page_contract["component_mode_sha256"] = component_mode_digest
    native_source = (repository / "src/dft/stationary_gradient_cuda.cuh").read_text(
        encoding="utf-8"
    )
    native_blocks = {
        "native_owner_sha256": (
            "struct Owner {",
            NATIVE_STATIONARY_OWNER_CONTRACT_SHA256,
        ),
        "native_allocation_sha256": (
            "size_t allocation(",
            NATIVE_STATIONARY_ALLOCATION_CONTRACT_SHA256,
        ),
        "native_create_sha256": (
            "int stationary_create(",
            NATIVE_STATIONARY_CREATE_CONTRACT_SHA256,
        ),
        "native_reset_sha256": (
            "int stationary_reset(",
            NATIVE_STATIONARY_RESET_CONTRACT_SHA256,
        ),
        "native_tasks_sha256": (
            "int stationary_tasks(",
            NATIVE_STATIONARY_TASKS_CONTRACT_SHA256,
        ),
        "native_nuclear_sha256": (
            "int stationary_nuclear(",
            NATIVE_STATIONARY_NUCLEAR_CONTRACT_SHA256,
        ),
        "native_geometry_ao_map_sha256": (
            "bool valid_geometry_ao_map(",
            NATIVE_STATIONARY_GEOMETRY_AO_MAP_CONTRACT_SHA256,
        ),
        "native_geometry_external_sha256": (
            "int stationary_geometry_external(",
            NATIVE_STATIONARY_GEOMETRY_EXTERNAL_CONTRACT_SHA256,
        ),
        "native_geometry_enqueue_sha256": (
            "int stationary_geometry_enqueue(",
            NATIVE_STATIONARY_GEOMETRY_ENQUEUE_CONTRACT_SHA256,
        ),
        "native_geometry_route_sha256": (
            "int stationary_geometry(",
            NATIVE_STATIONARY_GEOMETRY_ROUTE_CONTRACT_SHA256,
        ),
        "native_launch_geometry_sha256": (
            "void launch_geometry(",
            NATIVE_STATIONARY_LAUNCH_GEOMETRY_CONTRACT_SHA256,
        ),
        "native_configure_becke_sha256": (
            "int stationary_configure_becke(",
            NATIVE_STATIONARY_CONFIGURE_BECKE_CONTRACT_SHA256,
        ),
        "native_metrics_sha256": (
            "int stationary_metrics(",
            NATIVE_STATIONARY_METRICS_CONTRACT_SHA256,
        ),
        "native_finish_span_sha256": (
            "int stationary_finish_span(",
            NATIVE_STATIONARY_FINISH_SPAN_CONTRACT_SHA256,
        ),
    }
    for label, (marker, expected_digest) in native_blocks.items():
        digest = _cpp_block_sha256(native_source, marker)
        if digest != expected_digest:
            raise RuntimeError(f"stationary CUDA {label} contract changed")
        page_contract[label] = digest
    if tuple(COMPONENT_LABELS) != tuple(QUALIFIED_SPD_COMPONENTS):
        raise RuntimeError("stationary CUDA component-label capacity changed")
    definition_nodes = {
        name: [
            node
            for node in owner.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ]
        for name in (
            "requires_native_integrals",
            "grid_work",
            "native_integral_host_reserve",
            "primitive_sum",
            "records",
            "pair_visits",
            "method",
            "functional",
            "ingredients",
            "needs_first",
            "grid_plan",
            "minimum_source_bytes",
            "source_resources",
            "source_bytes",
            "available",
            "host_bound",
        )
    }
    if any(len(nodes) != 1 for nodes in definition_nodes.values()):
        raise RuntimeError(
            "stationary CUDA capacity definitions are missing or ambiguous"
        )
    definitions = {
        name: ast.unparse(nodes[0].value) for name, nodes in definition_nodes.items()
    }
    expected_definitions = {
        "requires_native_integrals": NATIVE_REQUIREMENT_DEFINITION,
        "grid_work": GRID_WORK_DEFINITION,
        "native_integral_host_reserve": NATIVE_HOST_RESERVE_DEFINITION,
        "primitive_sum": PRIMITIVE_SUM_DEFINITION,
        "records": PRIMITIVE_RECORDS_DEFINITION,
        "pair_visits": GRID_PAIR_VISITS_DEFINITION,
        "method": METHOD_IR_DEFINITION,
        "functional": FUNCTIONAL_LOWERING_DEFINITION,
        "ingredients": INGREDIENTS_DEFINITION,
        "needs_first": NEEDS_FIRST_DEFINITION,
        "grid_plan": GRID_PLAN_DEFINITION,
        "minimum_source_bytes": MINIMUM_SOURCE_BYTES_DEFINITION,
        "source_resources": SOURCE_RESOURCES_DEFINITION,
        "source_bytes": SOURCE_BYTES_DEFINITION,
        "available": AVAILABLE_DEVICE_BYTES_DEFINITION,
        "host_bound": HOST_BOUND_DEFINITION,
    }
    definition_labels = {
        "requires_native_integrals": "native-integral requirement",
        "grid_work": "bounded grid-work plan",
        "native_integral_host_reserve": "native-integral host reserve",
        "primitive_sum": "primitive-sum",
        "records": "primitive-record",
        "pair_visits": "grid-pair-visits",
        "method": "stationary MethodIR",
        "functional": "native snapshot functional-code lowering",
        "ingredients": "native semilocal ingredient provenance",
        "needs_first": "grid derivative-order",
        "grid_plan": "grid-plan input",
        "minimum_source_bytes": "minimum-source-bytes",
        "source_resources": "source-resources",
        "source_bytes": "source-bytes",
        "available": "available-device-bytes",
        "host_bound": "host-bound",
    }
    for name, expected in expected_definitions.items():
        if definitions[name] != expected:
            raise RuntimeError(
                f"stationary CUDA {definition_labels[name]} definition changed"
            )
    direct_if_tests = [
        ast.unparse(node.test) for node in owner.body if isinstance(node, ast.If)
    ]
    gate_labels = {
        "primitive_metric_range": "logical primitive metric range",
        "additional_device": "positive additional-device remainder",
        "additional_host": "additional-host",
    }
    for name, predicate in GATE_PREDICATES.items():
        if direct_if_tests.count(predicate) != 1:
            raise RuntimeError(f"stationary CUDA {gate_labels[name]} predicate changed")
    if direct_if_tests.count("records > max_primitive_records") != 0:
        raise RuntimeError("stationary CUDA restored a whole-force primitive cap")
    page_budget_bindings = [
        ast.unparse(keyword.value)
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "page_work_budget"
    ]
    if page_budget_bindings.count("max_primitive_records") != 2:
        raise RuntimeError("stationary CUDA page-work budget wiring changed")
    executor_definitions = {
        name: [
            ast.unparse(node.value)
            for node in ast.walk(owner)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ]
        for name in ("fixed_task_capacity", "task_executor")
    }
    expected_executor_definitions = {
        "fixed_task_capacity": "min(integral_terms, primitive_tile)",
        "task_executor": (
            "_BoundedStationaryTaskExecutor(fixed_capacity=fixed_task_capacity, "
            "resident_capacity=primitive_tile, page_capacity=primitive_tile)"
        ),
    }
    for name, expected in expected_executor_definitions.items():
        if executor_definitions[name] != [expected]:
            raise RuntimeError(f"stationary CUDA {name} definition changed")
    page_execution_calls = [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "task_executor.execute_pages"
        and [ast.unparse(argument) for argument in node.args]
        == [
            "domain",
            "submit_page",
        ]
        and not node.keywords
    ]
    host_gates = [
        node
        for node in owner.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == GATE_PREDICATES["additional_host"]
    ]
    if (
        len(page_execution_calls) != 1
        or len(host_gates) != 1
        or page_execution_calls[0].lineno <= host_gates[0].end_lineno
    ):
        raise RuntimeError("stationary CUDA primitive descriptor page order changed")
    submit_pages = [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.FunctionDef) and node.name == "submit_page"
    ]
    if len(submit_pages) != 1:
        raise RuntimeError("stationary CUDA submit-page owner is ambiguous")
    submit_page_digest = _source_node_sha256(source, submit_pages[0])
    if submit_page_digest != STATIONARY_SUBMIT_PAGE_CONTRACT_SHA256:
        raise RuntimeError("stationary CUDA submit-page contract changed")
    page_contract["submit_page_sha256"] = submit_page_digest
    nuclear_pair_loops = [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.For)
        and len(node.body) == 1
        and isinstance(node.body[0], ast.For)
        and any(
            isinstance(call, ast.Call) and ast.unparse(call.func) == "sources.nuclear"
            for call in ast.walk(node.body[0])
        )
    ]
    if len(nuclear_pair_loops) != 1:
        raise RuntimeError("stationary CUDA nuclear-pair loop is ambiguous")
    nuclear_pair_loop_digest = _source_node_sha256(source, nuclear_pair_loops[0])
    if nuclear_pair_loop_digest != STATIONARY_NUCLEAR_PAIR_LOOP_CONTRACT_SHA256:
        raise RuntimeError("stationary CUDA nuclear-pair loop contract changed")
    page_contract["nuclear_pair_loop_sha256"] = nuclear_pair_loop_digest
    # Hashes retain complete fail-closed coverage; these structural checks explain
    # the critical admission ordering and concurrent host ownership explicitly.
    native_requirement = definition_nodes["requires_native_integrals"][0]
    grid_work = definition_nodes["grid_work"][0]
    if (
        not native_requirement.lineno
        < grid_work.lineno
        < definition_nodes["records"][0].lineno
    ):
        raise RuntimeError("stationary CUDA admission gate order changed")
    reserve_additions = [
        node
        for node in owner.body
        if isinstance(node, ast.AugAssign)
        and ast.unparse(node.target) == "host_bound"
        and isinstance(node.op, ast.Add)
        and ast.unparse(node.value) == "native_integral_host_reserve"
    ]
    if (
        len(reserve_additions) != 1
        or not definition_nodes["native_integral_host_reserve"][0].lineno
        < reserve_additions[0].lineno
        < host_gates[0].lineno
    ):
        raise RuntimeError(
            "stationary CUDA native host-reserve admission order changed"
        )
    signature = inspect.signature(complete_rks_cuda_gradient_diagnostic)
    if signature.parameters["resident_ao_cutoff"].default is not None:
        raise RuntimeError("stationary CUDA default AO membership changed")
    ao_reserves = [
        node
        for node in owner.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and ast.unparse(node.targets[0]) == "ao_map_reserve"
    ]
    ao_charges = [
        node
        for node in owner.body
        if isinstance(node, ast.AugAssign)
        and ast.unparse(node.target) == "host_bound"
        and isinstance(node.op, ast.Add)
        and ast.unparse(node.value) == "ao_map_reserve"
    ]
    cache_paths = [
        node
        for node in owner.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and ast.unparse(node.targets[0]) == "cache"
    ]
    expected_ao_reserve = ast.parse(
        "_stationary_ao_map_reserve(resident_ao_cutoff, resident_ao_cache_bytes, "
        "host_bound + (sum(value.host_bytes for value in tensor_plans.values()) "
        "if prepared is not None else 0), max_host_bytes)",
        mode="eval",
    ).body
    if (
        len(ao_reserves) != 1
        or ast.dump(ao_reserves[0].value) != ast.dump(expected_ao_reserve)
        or len(ao_charges) != 1
        or len(cache_paths) != 1
        or not host_gates[0].lineno
        < ao_reserves[0].lineno
        < ao_charges[0].lineno
        < cache_paths[0].lineno
    ):
        raise RuntimeError("stationary CUDA AO-map reserve admission changed")
    for name, expected in (
        ("_stationary_ao_map_reserve", STATIONARY_AO_MAP_RESERVE_CONTRACT_SHA256),
        ("_stationary_resident_ao_cache", STATIONARY_AO_MAP_CACHE_CONTRACT_SHA256),
    ):
        helpers = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        ]
        if len(helpers) != 1 or _source_node_sha256(source, helpers[0]) != expected:
            raise RuntimeError(f"stationary CUDA {name} contract changed")
        page_contract[f"{name.removeprefix('_stationary_')}_sha256"] = expected

    def default(name: str) -> int:
        value = signature.parameters[name].default
        if type(value) is not int:
            raise RuntimeError(f"stationary CUDA {name} default is not an integer")
        return value

    messages = (
        "enlarged stationary CUDA domains require prepared native integral derivatives",
        "primitive work count exceeds uint64 metric range",
        "stationary additional-device budget exceeded",
        "stationary additional-host byte budget exceeded",
        "native stationary host staging exceeds admitted reserve",
        "prepared native integral derivatives are unavailable within the admitted ",
    )
    endpoint_source = ast.get_source_segment(source, owner)
    assert endpoint_source is not None
    positions = [endpoint_source.find(message) for message in messages]
    if any(position < 0 for position in positions):
        raise RuntimeError("stationary CUDA admission messages are incomplete")
    if positions != sorted(positions):
        raise RuntimeError("stationary CUDA admission gate order changed")
    endpoint_owner_digest = _source_node_sha256(source, owner)
    if endpoint_owner_digest != STATIONARY_ENDPOINT_OWNER_CONTRACT_SHA256:
        raise RuntimeError("stationary CUDA endpoint owner contract changed")
    page_contract["endpoint_owner_sha256"] = endpoint_owner_digest

    return {
        "owner": STATIONARY_OWNER,
        "resource_owner": "python/generativeqc_compiler/method/stationary_resources.py",
        "native_owner_capacity": {
            "atom_count": STATIONARY_MAX_ATOMS,
            "ao_count": STATIONARY_MAX_AOS,
            "basis_primitive_count": STATIONARY_MAX_PRIMITIVES,
        },
        "ao_task_fallback_capacity": {
            "atom_count": 32,
            "ao_count": 128,
            "basis_primitive_count": 4096,
        },
        "native_integral_requirement_definition": NATIVE_REQUIREMENT_DEFINITION,
        "native_integral_host_reserve_definition": NATIVE_HOST_RESERVE_DEFINITION,
        "host_bound_total_definition": "host_bound + native_integral_host_reserve",
        "grid_work_definition": GRID_WORK_DEFINITION,
        "diagnostic_work_limits": {
            "grid_points": default("max_grid_points"),
            "grid_pair_visits": default("max_grid_pair_visits"),
        },
        "grid_work_capacity": {
            "grid_points": 1 << 40,
            "grid_pair_visits": (1 << 64) - 1,
        },
        "pending_grid_tiles": default("max_pending_grid_tiles"),
        "pending_grid_pair_visits": default("max_pending_grid_pair_visits"),
        "primitive_records": default("max_primitive_records"),
        "primitive_records_scope": "per_native_page_on_ao_task_fallback_only",
        "primitive_logical_metric_limit": (1 << 64) - 1,
        "primitive_sum_definition": PRIMITIVE_SUM_DEFINITION,
        "primitive_records_definition": PRIMITIVE_RECORDS_DEFINITION,
        "snapshot_functional_contract_sha256": snapshot_functional_contract,
        "primitive_page_contract_sha256": page_contract,
        "primitive_page_budget_bindings": list(page_budget_bindings),
        "fixed_task_capacity_definition": expected_executor_definitions[
            "fixed_task_capacity"
        ],
        "task_executor_definition": expected_executor_definitions["task_executor"],
        "primitive_page_execution_definition": (
            "task_executor.execute_pages(domain, submit_page)"
        ),
        "grid_pair_visits_definition": GRID_PAIR_VISITS_DEFINITION,
        "method_ir_definition": METHOD_IR_DEFINITION,
        "functional_lowering_definition": FUNCTIONAL_LOWERING_DEFINITION,
        "functional_ingredients_definition": INGREDIENTS_DEFINITION,
        "grid_derivative_order_definition": NEEDS_FIRST_DEFINITION,
        "grid_plan_definition": GRID_PLAN_DEFINITION,
        "minimum_source_bytes_definition": MINIMUM_SOURCE_BYTES_DEFINITION,
        "source_resources_definition": SOURCE_RESOURCES_DEFINITION,
        "source_bytes_definition": SOURCE_BYTES_DEFINITION,
        "host_bound_definition": HOST_BOUND_DEFINITION,
        "available_device_bytes_definition": AVAILABLE_DEVICE_BYTES_DEFINITION,
        "additional_device_admission": (
            "minimum_additional_device_bytes < additional_device_budget and additional_device_peak_bound <= additional_device_budget"
        ),
        "gate_predicates": dict(GATE_PREDICATES),
        "tile_points": default("tile_points"),
        "primitive_tile": default("primitive_tile"),
        "integral_terms": default("integral_terms"),
        "additional_device_bytes": default("max_device_bytes"),
        "additional_host_bytes": default("max_host_bytes"),
        "gate_order": [
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
        ],
    }


def _grid_spec(payload: dict[str, Any]) -> GridSpec:
    values = dict(payload)
    values["element_radii"] = tuple(tuple(item) for item in values["element_radii"])
    return GridSpec(**values)


def _basis_layout_contract(repository: Path) -> dict[str, str]:
    source = (repository / "python/generativeqc_compiler/dft/ao.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NativeAO"
    ]
    if len(classes) != 1:
        raise RuntimeError("NativeAO capacity owner is missing or ambiguous")
    constructors = [
        node
        for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    ]
    if len(constructors) != 1:
        raise RuntimeError("NativeAO capacity constructor is missing or ambiguous")
    constructor_digest = _source_node_sha256(source, constructors[0])
    packed = [
        ast.unparse(node.value)
        for node in ast.walk(constructors[0])
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "packed"
    ]
    numeric = [
        ast.unparse(node.value)
        for node in ast.walk(constructors[0])
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Attribute)
        and ast.unparse(node.targets[0]) == "self.numeric_bytes"
    ]
    if packed != [BASIS_PACKED_CAPACITY_DEFINITION]:
        raise RuntimeError("NativeAO packed capacity definition changed")
    if numeric != [BASIS_NUMERIC_CAPACITY_DEFINITION]:
        raise RuntimeError("NativeAO numeric capacity definition changed")
    if constructor_digest != NATIVE_AO_CONSTRUCTOR_CONTRACT_SHA256:
        raise RuntimeError("NativeAO constructor contract changed")

    def function_digest(
        relative: str,
        function_name: str,
        expected: str,
        *,
        class_name: str | None = None,
    ) -> str:
        owner_source = (repository / relative).read_text(encoding="utf-8")
        owner_tree = ast.parse(owner_source)
        body: list[ast.stmt] = owner_tree.body
        if class_name is not None:
            owners = [
                node
                for node in owner_tree.body
                if isinstance(node, ast.ClassDef) and node.name == class_name
            ]
            if len(owners) != 1:
                raise RuntimeError(f"{class_name} basis owner is missing or ambiguous")
            body = owners[0].body
        functions = [
            node
            for node in body
            if isinstance(node, ast.FunctionDef) and node.name == function_name
        ]
        if len(functions) != 1:
            raise RuntimeError(
                f"{function_name} basis lowering is missing or ambiguous"
            )
        digest = _source_node_sha256(owner_source, functions[0])
        if digest != expected:
            raise RuntimeError(f"{function_name} basis lowering contract changed")
        return digest

    snapshot_digest = function_digest(
        "python/generativeqc/_model_resolution.py",
        "snapshot_basis",
        BASIS_SNAPSHOT_CONTRACT_SHA256,
    )
    expansion_digest = function_digest(
        "python/generativeqc/basis.py",
        "shells_for",
        BASIS_SHELL_EXPANSION_CONTRACT_SHA256,
        class_name="BasisSet",
    )
    shell_forwarding_digest = function_digest(
        "python/generativeqc/calculator.py",
        "_shells_for_atoms",
        CALCULATOR_SHELL_FORWARDING_CONTRACT_SHA256,
        class_name="Calculator",
    )
    native_system_digest = function_digest(
        "python/generativeqc/calculator.py",
        "_create_native_system",
        NATIVE_SYSTEM_BASIS_FORWARDING_CONTRACT_SHA256,
        class_name="Calculator",
    )
    return {
        "packed_capacity_definition": packed[0],
        "numeric_capacity_definition": numeric[0],
        "native_ao_constructor_contract_sha256": constructor_digest,
        "basis_snapshot_contract_sha256": snapshot_digest,
        "basis_shell_expansion_contract_sha256": expansion_digest,
        "calculator_shell_forwarding_contract_sha256": shell_forwarding_digest,
        "native_system_basis_forwarding_contract_sha256": native_system_digest,
        "production_shell_expansion": (
            "snapshot_basis('def2-svp', 'spherical').shells_for(atoms)"
        ),
    }


def _spd_expansion_contract(repository: Path) -> dict[str, Any]:
    """Bind frozen s/p/d term counts to the native public-AO expansion table."""

    source = (repository / "src/molecule/basis.cpp").read_text(encoding="utf-8")
    try:
        begin = source.index("std::vector<CartesianComponent> cartesian_components")
        end = source.index("  if (l == 3)", begin)
    except ValueError as error:
        raise RuntimeError("native s/p/d expansion owner is missing") from error
    digest = _lf_sha256(source[begin:end].encode())
    if digest != SPD_EXPANSION_CONTRACT_SHA256:
        raise RuntimeError("native s/p/d expansion contract changed")

    packer_source = (repository / "src/dft/ao_grid.cpp").read_text(encoding="utf-8")
    try:
        packer_begin = packer_source.index("AoBasis::AoBasis(")
        packer_end = packer_source.index("void AoBasis::evaluate(", packer_begin)
    except ValueError as error:
        raise RuntimeError("native packed-AO owner is missing") from error
    packer_digest = _lf_sha256(packer_source[packer_begin:packer_end].encode())
    if packer_digest != AO_PACKER_CONTRACT_SHA256:
        raise RuntimeError("native packed-AO contract changed")

    bridge_source = (repository / "src/dft/bridge.cpp").read_text(encoding="utf-8")
    try:
        bridge_begin = bridge_source.index(
            "GENERATIVEQC_API int generativeqc_grid_basis_create_v1"
        )
        bridge_end = bridge_source.index(
            "GENERATIVEQC_API int generativeqc_grid_ao_v1", bridge_begin
        )
    except ValueError as error:
        raise RuntimeError("native AO pack bridge is missing") from error
    bridge_digest = _lf_sha256(bridge_source[bridge_begin:bridge_end].encode())
    if bridge_digest != AO_PACK_BRIDGE_CONTRACT_SHA256:
        raise RuntimeError("native packed-AO contract changed")

    ao_count_digest = _source_span_sha256(
        source,
        begin="std::size_t ao_count(",
        end="std::size_t cartesian_ao_count(",
        label="native spherical AO count",
    )
    if ao_count_digest != NATIVE_SPHERICAL_AO_COUNT_CONTRACT_SHA256:
        raise RuntimeError("native spherical AO count contract changed")

    stationary_source = (
        repository / "python/generativeqc/_stationary_cuda.py"
    ).read_text(encoding="utf-8")
    stationary_tree = ast.parse(stationary_source)
    layouts = [
        node
        for node in stationary_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_layout"
    ]
    if len(layouts) != 1:
        raise RuntimeError("stationary layout owner is missing or ambiguous")
    layout_digest = _source_node_sha256(stationary_source, layouts[0])
    if layout_digest != STATIONARY_LAYOUT_CONTRACT_SHA256:
        raise RuntimeError("stationary layout contract changed")

    return {
        "spd_expansion_contract_sha256": digest,
        "ao_packer_contract_sha256": packer_digest,
        "ao_pack_bridge_contract_sha256": bridge_digest,
        "native_spherical_ao_count_contract_sha256": ao_count_digest,
        "stationary_layout_contract_sha256": layout_digest,
        "sparse_spherical_component_terms": {
            "s": SPARSE_SPHERICAL_COMPONENT_TERMS[0],
            "p": SPARSE_SPHERICAL_COMPONENT_TERMS[1],
            "d": SPARSE_SPHERICAL_COMPONENT_TERMS[2],
        },
        "spd_expansion_owner": (
            "basis.cpp::cartesian_components+ao_expansions -> "
            "ao_grid.cpp::AoBasis -> bridge.cpp::generativeqc_grid_basis_pack_v1 -> "
            "_stationary_cuda.py::_layout"
        ),
    }


def _basis_shape(
    atoms: tuple[Atom, ...], charge: int, multiplicity: int
) -> tuple[dict[str, Any], Any]:
    basis = snapshot_basis("def2-svp", "spherical")
    if not isinstance(basis, BasisSet) or basis.representation != "spherical":
        raise RuntimeError(
            "production basis snapshot did not retain spherical def2-SVP"
        )
    shells = basis.shells_for(atoms)
    if any(
        shell.angular_momentum not in SPARSE_SPHERICAL_COMPONENT_TERMS
        for shell in shells
    ):
        raise NotImplementedError(
            "DFT-MP-v1 capacity audit only covers the frozen s/p/d def2-SVP domain"
        )
    ao_count = sum(2 * shell.angular_momentum + 1 for shell in shells)
    primitive_count = sum(len(shell.primitives) for shell in shells)
    ao_primitive_peak = max(len(shell.primitives) for shell in shells)
    component_primitive_sum = sum(
        len(shell.primitives) * SPARSE_SPHERICAL_COMPONENT_TERMS[shell.angular_momentum]
        for shell in shells
    )
    packed = np.empty(3 * len(atoms) + 2 * primitive_count + 16 * ao_count)
    numeric_bytes = (
        2 * packed.nbytes + 32 * len(atoms) + 32 * len(shells) + 16 * primitive_count
    )
    synthetic = SimpleNamespace(
        nao=ao_count,
        natom=len(atoms),
        nprimitive=primitive_count,
        numeric_bytes=numeric_bytes,
        packed=packed,
    )
    record = _named_basis_record("def2-svp", "spherical")
    metadata = resolved_basis_metadata(
        record,
        shells,
        atoms,
        representation="spherical",
        charge=charge,
        multiplicity=multiplicity,
    )
    return (
        {
            "atom_count": len(atoms),
            "ao_count_spherical": ao_count,
            "shell_count": len(shells),
            "basis_primitive_count": primitive_count,
            "ao_primitive_count_peak": ao_primitive_peak,
            "component_primitive_sum": component_primitive_sum,
        },
        (synthetic, metadata),
    )


def _method_resources(
    basis: Any,
    *,
    atom_count: int,
    functional: int,
    spin: str,
    limits: dict[str, Any],
    plan: Any | None = None,
) -> tuple[dict[str, Any], Any]:
    if plan is None:
        plan = _qualified_aot_plan(functional, spin)
    source_names = stationary_runtime_sources(plan)
    tile_points = limits["tile_points"]
    primitive_tile = limits["primitive_tile"]
    integral_terms = limits["integral_terms"]
    grid_plan = plan_tiles(
        basis,
        backend="cuda",
        order=1 if functional == 0 else 2,
        tile_points=tile_points,
        active_ao_capacity=basis.nao,
        # Estimate even a losing case so the report cannot hide it by raising
        # from the live budget gate before recording the required bytes.
        budget_bytes=(1 << 63) - 1,
    )
    shape = {
        "atoms": atom_count,
        "aos": basis.nao,
        "primitives": basis.nprimitive,
        "points": tile_points,
        "tasks": primitive_tile,
        "spins": plan.spin_blocks,
        "sources": len(source_names),
    }
    minimum = stationary_cuda_allocation_bytes(
        **shape, geometry_lanes=min(32, tile_points)
    )
    native_host_reserve = stationary_native_pair_reserve(
        atoms=atom_count, aos=basis.nao, primitives=basis.nprimitive
    )
    native_reserve = min(
        max(0, limits["additional_device_bytes"] - grid_plan.peak_bytes - minimum),
        native_host_reserve,
    )
    resources = plan_stationary_cuda_resources(
        **shape,
        target=cuda_target_info("sm_120"),
        # Preserve a losing minimum-byte requirement instead of hiding it by
        # throwing before the capacity report records the failure.
        budget_bytes=max(
            minimum,
            limits["additional_device_bytes"] - grid_plan.peak_bytes - native_reserve,
        ),
    )
    source_bytes = resources.allocation_bytes
    device_bound = grid_plan.peak_bytes + source_bytes
    host_bound = grid_plan.host_bytes + 8 * (
        34 * primitive_tile
        + 4 * plan.spin_blocks * basis.nao * basis.nao
        + 120 * atom_count
        + 12 * (len(source_names) - len(STATIONARY_RUNTIME_SOURCE_NAMES)) * atom_count
        + 26 * integral_terms
        + len(QUALIFIED_SPD_COMPONENTS) ** 4
        + 3 * len(QUALIFIED_SPD_COMPONENTS) ** 2
        + 3 * tile_points
        + 2 * basis.nprimitive
        + 4 * basis.nao
        + 80
    )
    return (
        {
            "resource_scope": (
                "additional stationary/grid and Direct paired-provider numeric owners; "
                "excludes retained SCF state, snapshot exports, Python/compiler objects and driver/modules"
            ),
            "grid_tile_peak_bytes": grid_plan.peak_bytes,
            "stationary_source_bytes": source_bytes,
            "stationary_geometry_lanes": resources.geometry_lanes,
            "stationary_geometry_scratch_bytes": resources.geometry_scratch_bytes,
            "stationary_center_geometry_bytes": resources.center_geometry_bytes,
            "stationary_grid_device_peak_bound": device_bound,
            # The provider has not executed. Charge its reserved allowance,
            # rather than reporting the stationary/grid owners as the full peak.
            "additional_device_peak_bound": device_bound + native_reserve,
            "native_integral_device_budget": max(
                0, limits["additional_device_bytes"] - device_bound
            ),
            "minimum_additional_device_bytes": grid_plan.peak_bytes + minimum,
            "stationary_native_pair_reserve_bytes": native_reserve,
            "additional_device_budget": limits["additional_device_bytes"],
            "stationary_native_integral_host_reserve_bytes": native_host_reserve,
            "additional_host_numeric_bound": host_bound + native_host_reserve,
            "additional_host_budget": limits["additional_host_bytes"],
        },
        plan,
    )


def _failure(
    gate: str,
    message: str,
    *,
    required: Any,
    cap: Any,
    exceeded: list[str] | None = None,
) -> dict[str, Any]:
    resource_function = (
        "stationary_cuda_requires_native_integrals"
        if gate == "native_owner_capacity"
        else "plan_stationary_cuda_grid_work"
        if gate
        in (
            "grid_work_capacity",
            "grid_point_work_budget",
            "grid_pair_work_budget",
            "pending_grid_pair_budget",
        )
        else None
    )
    result = {
        "gate": gate,
        "owner": STATIONARY_OWNER
        if resource_function is None
        else {
            "file": "python/generativeqc_compiler/method/stationary_resources.py",
            "function": resource_function,
        },
        "message": message,
        "required": required,
        "cap": cap,
    }
    if exceeded is not None:
        result["exceeded"] = exceeded
    return result


def _case_failures(
    shape: dict[str, Any],
    requirements: dict[str, Any],
    memory: dict[str, Any],
    limits: dict[str, Any],
    *,
    work_mode: str = "public",
) -> list[dict[str, Any]]:
    failures = []
    capacity = limits["native_owner_capacity"]
    for name, actual in (
        ("atom_count", shape["atom_count"]),
        ("ao_count", shape["ao_count_spherical"]),
        ("basis_primitive_count", shape["basis_primitive_count"]),
    ):
        if not 1 <= actual <= capacity[name]:
            failures.append(
                _failure(
                    "native_owner_capacity",
                    "stationary CUDA "
                    + {
                        "atom_count": "atoms",
                        "ao_count": "aos",
                        "basis_primitive_count": "primitives",
                    }[name]
                    + " exceeds resource caps",
                    required={name: actual},
                    cap={name: capacity[name]},
                    exceeded=[name],
                )
            )
    for key in ("grid_points", "grid_pair_visits"):
        minimum = 1 if key == "grid_points" else 0
        if not minimum <= requirements[key] <= limits["grid_work_capacity"][key]:
            failures.append(
                _failure(
                    "grid_work_capacity",
                    "stationary grid work exceeds resource/metric range",
                    required={key: requirements[key]},
                    cap={key: limits["grid_work_capacity"][key]},
                )
            )
    for key, gate, message in (
        ("grid_points", "grid_point_work_budget", "grid point work budget exceeded"),
        ("grid_pair_visits", "grid_pair_work_budget", "grid work budget exceeded"),
    ):
        cap = limits[f"{work_mode}_work_limits"][key]
        if cap is not None and requirements[key] > cap:
            failures.append(
                _failure(gate, message, required=requirements[key], cap=cap)
            )
    tile_visits = (
        2
        * min(requirements["grid_points"], limits["tile_points"])
        * shape["atom_count"]
        * (shape["atom_count"] - 1)
        // 2
    )
    if tile_visits > limits["pending_grid_pair_visits"]:
        failures.append(
            _failure(
                "pending_grid_pair_budget",
                "stationary grid tile exceeds pending pair-visit budget",
                required=tile_visits,
                cap=limits["pending_grid_pair_visits"],
            )
        )
    if requirements["primitive_records"] > limits["primitive_logical_metric_limit"]:
        failures.append(
            _failure(
                "primitive_logical_metric_range",
                "primitive work count exceeds uint64 metric range",
                required=requirements["primitive_records"],
                cap=limits["primitive_logical_metric_limit"],
            )
        )
    if (
        memory.get(
            "minimum_additional_device_bytes", memory["additional_device_peak_bound"]
        )
        >= limits["additional_device_bytes"]
        or memory["additional_device_peak_bound"] > limits["additional_device_bytes"]
    ):
        failures.append(
            _failure(
                "additional_device_budget",
                "stationary additional-device budget exceeded",
                required=memory["additional_device_peak_bound"],
                cap=limits["additional_device_bytes"],
            )
        )
    if memory["additional_host_numeric_bound"] > limits["additional_host_bytes"]:
        failures.append(
            _failure(
                "additional_host_budget",
                "stationary additional-host byte budget exceeded",
                required=memory["additional_host_numeric_bound"],
                cap=limits["additional_host_bytes"],
            )
        )
    fallback_capacity = limits["ao_task_fallback_capacity"]
    requires_native = (
        shape["atom_count"] > fallback_capacity["atom_count"]
        or shape["ao_count_spherical"] > fallback_capacity["ao_count"]
        or shape["basis_primitive_count"] > fallback_capacity["basis_primitive_count"]
    )
    if (
        not requires_native
        and requirements["primitive_descriptor_peak_records"]
        > limits["primitive_records"]
    ):
        failures.append(
            _failure(
                "primitive_descriptor_page_budget",
                "stationary CUDA descriptor exceeds primitive page work budget",
                required=requirements["primitive_descriptor_peak_records"],
                cap=limits["primitive_records"],
            )
        )
    return failures


def _admission_record(failures: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "outcome": "blocked" if failures else "passes_static_stationary_caps",
        "first_blocker": failures[0] if failures else None,
        "failures": failures,
        "scope": (
            "static source/resource preflight only; no SCF, CUDA execution, "
            "native derivative-provider availability, AOT binary, numerical, or performance qualification"
        ),
    }


def _source_package_inventory(repository: Path) -> dict[str, Any]:
    cmake = (repository / "cmake/GenerativeQCCuda.cmake").read_text(encoding="utf-8")
    names = (
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
    )
    required = (
        "generativeqc_stationary_spd_primitives",
        'OUTPUT_NAME "generativeqc_stationary_${_generativeqc_stationary_name}_spd"',
        *names,
    )
    missing = [token for token in required if token not in cmake]
    if missing:
        raise RuntimeError(
            "stationary s/p/d package declaration is incomplete: " + ", ".join(missing)
        )
    contract_digest = _source_span_sha256(
        cmake,
        begin=("    # Component-expanded s/p/d derivatives are shared compiler output"),
        end=(
            "  if(GENERATIVEQC_PYTHON_WHEEL)\n    generativeqc_attach_cuda_implib(${target})"
        ),
        label="stationary packaged-AOT CMake",
    )
    if contract_digest != STATIONARY_AOT_CMAKE_CONTRACT_SHA256:
        raise RuntimeError("stationary packaged-AOT CMake contract changed")
    return {
        "cmake_contract_sha256": contract_digest,
        "profiles": list(names),
        "component_domain": "spd",
    }


def _source_public_route(repository: Path) -> dict[str, Any]:
    """Fail closed if the source predicates supporting the reported route move."""

    calculator = (repository / "python/generativeqc/calculator.py").read_text(
        encoding="utf-8"
    )
    batch = (repository / "python/generativeqc/batch.py").read_text(encoding="utf-8")
    calculator_tree = ast.parse(calculator)
    calculator_classes = [
        node
        for node in calculator_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Calculator"
    ]
    if len(calculator_classes) != 1:
        raise RuntimeError("public Calculator owner is missing or ambiguous")
    constructors = [
        node
        for node in calculator_classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    ]
    if len(constructors) != 1:
        raise RuntimeError("public Calculator constructor is missing or ambiguous")
    semilocal_assignments = [
        node
        for node in ast.walk(constructors[0])
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "semilocal_force"
            for target in node.targets
        )
    ]
    hybrid_assignments = [
        node
        for node in ast.walk(constructors[0])
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "cuda_hybrid_force"
            for target in node.targets
        )
    ]
    force_promotions = [
        node
        for node in constructors[0].body
        if isinstance(node, ast.If)
        and any(
            isinstance(item, ast.Name) and item.id == "semilocal_force"
            for item in ast.walk(node.test)
        )
    ]
    if (
        len(semilocal_assignments) != 1
        or len(hybrid_assignments) != 1
        or len(force_promotions) != 1
    ):
        raise RuntimeError("public CUDA force capability owner is ambiguous")
    semilocal_digest = _source_node_sha256(calculator, semilocal_assignments[0])
    hybrid_digest = _source_node_sha256(calculator, hybrid_assignments[0])
    promotion_digest = _source_node_sha256(calculator, force_promotions[0])
    if semilocal_digest != PUBLIC_SEMILOCAL_FORCE_CONTRACT_SHA256:
        raise RuntimeError("public semilocal force predicate changed")
    if hybrid_digest != PUBLIC_CUDA_HYBRID_FORCE_CONTRACT_SHA256:
        raise RuntimeError("public global-hybrid force predicate changed")
    if promotion_digest != PUBLIC_FORCE_PROMOTION_CONTRACT_SHA256:
        raise RuntimeError("public force capability promotion changed")

    batch_tree = ast.parse(batch)
    batch_classes = [
        node
        for node in batch_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PreparedBatch"
    ]
    if len(batch_classes) != 1:
        raise RuntimeError("public PreparedBatch owner is missing or ambiguous")
    force_methods = [
        node
        for node in batch_classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == "_public_dft_cuda_force"
    ]
    if len(force_methods) != 1:
        raise RuntimeError("public CUDA force route is missing or ambiguous")
    kwargs_nodes = [
        node.value
        for node in ast.walk(force_methods[0])
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "kwargs"
            for target in node.targets
        )
        and isinstance(node.value, ast.Dict)
    ]
    if len(kwargs_nodes) != 1:
        raise RuntimeError("public stationary kwargs are missing or ambiguous")
    kwargs = {
        ast.literal_eval(key): value
        for key, value in zip(kwargs_nodes[0].keys, kwargs_nodes[0].values, strict=True)
        if isinstance(key, ast.Constant)
    }
    work_limits = {}
    for key in ("grid_points", "grid_pair_visits"):
        node = kwargs.get(f"max_{key}")
        if not isinstance(node, ast.Constant) or node.value is not None:
            raise RuntimeError("public complete-grid work override changed")
        work_limits[key] = None
    batch_digest = _source_node_sha256(batch, force_methods[0])
    if batch_digest != PUBLIC_CUDA_FORCE_METHOD_CONTRACT_SHA256:
        raise RuntimeError("public CUDA force route changed")
    return {
        "semilocal_force_predicate_sha256": semilocal_digest,
        "global_hybrid_force_predicate_sha256": hybrid_digest,
        "force_capability_promotion_sha256": promotion_digest,
        "cuda_force_method_sha256": batch_digest,
        "whole_grid_work_limits": work_limits,
    }


def _grid_count_contract(repository: Path) -> dict[str, str]:
    """Bind the source-only point count to the native CUDA grid owner."""

    python_grid = (repository / "python/generativeqc_compiler/dft/grid.py").read_text(
        encoding="utf-8"
    )
    python_tree = ast.parse(python_grid)
    grid_classes = [
        node
        for node in python_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MolecularGrid"
    ]
    if len(grid_classes) != 1:
        raise RuntimeError("source-only MolecularGrid owner is missing or ambiguous")
    post_init = [
        node
        for node in grid_classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == "__post_init__"
    ]
    if len(post_init) != 1:
        raise RuntimeError(
            "source-only MolecularGrid constructor is missing or ambiguous"
        )
    python_digest = _source_node_sha256(python_grid, post_init[0])

    public_ks = (repository / "python/generativeqc/ks.py").read_text(encoding="utf-8")
    public_ks_tree = ast.parse(public_ks)
    public_lowerers = [
        node
        for node in public_ks_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "native_ks_options"
    ]
    if len(public_lowerers) != 1:
        raise RuntimeError("public GridSpec ABI lowerer is missing or ambiguous")
    public_abi_digest = _source_node_sha256(public_ks, public_lowerers[0])

    quadrature_source = (
        repository / "python/generativeqc_compiler/xc/quadrature_cuda.py"
    ).read_text(encoding="utf-8")
    quadrature_tree = ast.parse(quadrature_source)
    layouts = [
        node
        for node in quadrature_tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_LAYOUT"
            for target in node.targets
        )
    ]
    if len(layouts) != 1:
        raise RuntimeError("generated quadrature layout owner is missing or ambiguous")
    layout_digest = _source_node_sha256(quadrature_source, layouts[0])

    cuda_source = (repository / "src/dft/cuda_quadrature.cu").read_text(
        encoding="utf-8"
    )
    cuda_digest = _source_span_sha256(
        cuda_source,
        begin="MolecularGrid MolecularGrid::from_cuda(",
        end="}  // namespace generativeqc::dft",
        label="native CUDA grid",
    )
    route_source = (repository / "src/methods/dft_method.cpp").read_text(
        encoding="utf-8"
    )
    route_digest = _source_span_sha256(
        route_source,
        begin="dft::MolecularGrid ks_molecular_grid(",
        end="class KsPreparedCalculation",
        label="native CUDA grid route",
    )
    native_abi_digest = _source_span_sha256(
        route_source,
        begin="dft::GridSpec ks_grid_options(",
        end="Result adapt_result(",
        label="native GridSpec ABI lowerer",
    )
    header_source = (repository / "src/dft/grid.hpp").read_text(encoding="utf-8")
    point_count_digest = _source_span_sha256(
        header_source,
        begin="  std::size_t point_count()",
        end="  const std::vector<double>& points()",
        label="native grid point-count publication",
    )
    contracts = {
        "source_only_molecular_grid_sha256": python_digest,
        "public_grid_abi_sha256": public_abi_digest,
        "native_grid_abi_sha256": native_abi_digest,
        "generated_quadrature_layout_sha256": layout_digest,
        "native_cuda_grid_sha256": cuda_digest,
        "native_cuda_grid_route_sha256": route_digest,
        "native_grid_point_count_sha256": point_count_digest,
    }
    expected = {
        "source_only_molecular_grid_sha256": PYTHON_GRID_CONTRACT_SHA256,
        "public_grid_abi_sha256": PUBLIC_GRID_ABI_CONTRACT_SHA256,
        "native_grid_abi_sha256": NATIVE_GRID_ABI_CONTRACT_SHA256,
        "generated_quadrature_layout_sha256": QUADRATURE_LAYOUT_CONTRACT_SHA256,
        "native_cuda_grid_sha256": NATIVE_CUDA_GRID_CONTRACT_SHA256,
        "native_cuda_grid_route_sha256": NATIVE_GRID_ROUTE_CONTRACT_SHA256,
        "native_grid_point_count_sha256": NATIVE_GRID_POINT_COUNT_CONTRACT_SHA256,
    }
    moved = [name for name, digest in contracts.items() if digest != expected[name]]
    if moved:
        raise RuntimeError("grid point-count contract changed: " + ", ".join(moved))
    return {
        **contracts,
        "point_count_definition": (
            "atom_count * radial_points * angular_polar * angular_azimuth"
        ),
    }


def _prepared_aot_route_contract(repository: Path) -> str:
    """Bind no-runtime-compilation claims to the prepared production owner."""

    source = (repository / "python/generativeqc/_stationary_cuda.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "PreparedStationaryCudaExecution"
    ]
    methods = (
        []
        if len(classes) != 1
        else [
            node
            for node in classes[0].body
            if isinstance(node, ast.FunctionDef) and node.name == "ensure"
        ]
    )
    if len(methods) != 1:
        raise RuntimeError("prepared stationary AOT owner is missing or ambiguous")
    digest = _source_node_sha256(source, methods[0])
    if digest != PREPARED_AOT_SELECTION_CONTRACT_SHA256:
        raise RuntimeError("prepared stationary AOT selection contract changed")
    requests = [
        node
        for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == "_request"
    ]
    if (
        len(requests) != 1
        or _source_node_sha256(source, requests[0])
        != PREPARED_AO_REQUEST_CONTRACT_SHA256
    ):
        raise RuntimeError("prepared stationary AO request contract changed")
    return digest


def _public_selector_contract(
    selector: str,
    *,
    expected_functional: int,
    expected_spin: str,
    stationary_plan: Any,
    grid_spec: GridSpec | None = None,
) -> dict[str, Any]:
    """Prove that a public selector resolves to the audited packaged plan."""

    method_ir, functional = resolve_ks_method(selector)
    expected_coefficients = ks_coefficients(method_ir)
    hybrid = expected_coefficients[2] != 0.0
    registry_selector = native_dft_carrier(selector) if hybrid else selector
    try:
        metadata = generated_methods.METHOD_METADATA[registry_selector]
        expected_abi = int(metadata["abi_id"])
    except KeyError as error:
        raise RuntimeError(f"unrecognized frozen public selector {selector}") from error
    if hybrid:
        if grid_spec is None:
            raise RuntimeError(
                "global-hybrid selector contract requires the frozen grid"
            )
        options = resolve_ks_options(selector, KsOptions(grid=grid_spec))
    else:
        options = resolve_ks_options(selector)
    native_functional = int(native_xc_functional_code(selector))
    public_stationary = StationaryGradientPlan(
        method_ir,
        StationaryMeanField(SCF_POINT_MODEL),
    )
    execution_plan = options.execution_plan
    exchange = tuple(execution_plan.exchange)
    failures = []
    if metadata["family"] != "density_functional" or metadata["provider"] != "dft":
        failures.append("family/provider")
    if not hybrid and SEMILOCAL_ABI_IDS.get(selector) != expected_abi:
        failures.append("native ABI ID")
    if expected_abi not in generated_methods.NATIVE_DFT_METHOD_IDS:
        failures.append("native DFT eligibility")
    if not metadata["supports_batch"] or "energy" not in metadata["properties"]:
        failures.append("batch/energy registry eligibility")
    if method_ir.spin != expected_spin or functional.spin != expected_spin:
        failures.append("spin")
    if native_functional != expected_functional:
        failures.append("native functional-family lowering")
    if options.coefficients != expected_coefficients:
        failures.append(
            "scientific coefficients" if hybrid else "semilocal coefficients"
        )
    if (
        execution_plan.method.identity != method_ir.identity
        or execution_plan.nonlocal_correlation is not None
        or execution_plan.post_scf
    ):
        failures.append("KS execution plan")
    if hybrid:
        if (
            len(exchange) != 1
            or exchange[0].operator != "full-range"
            or float(exchange[0].fock_coefficient) != expected_coefficients[2]
            or exchange[0].omega != 0
        ):
            failures.append("exact-exchange execution plan")
    elif exchange:
        failures.append("KS execution plan")
    if public_stationary.identity != stationary_plan.identity:
        failures.append("stationary plan identity")
    if failures:
        raise RuntimeError(
            f"public selector {selector} disagrees with packaged DFT plan: "
            + ", ".join(failures)
        )
    return {
        "selector": selector,
        "registry_selector": registry_selector,
        "native_abi_id": expected_abi,
        "native_dft_eligible": True,
        "supports_batch": True,
        "native_functional_code": native_functional,
        "spin": expected_spin,
        "coefficients": list(options.coefficients),
        "exchange": [term.semantic_payload() for term in exchange],
        "method_ir_identity": method_ir.identity,
        "ks_execution_plan_identity": execution_plan.identity,
        "stationary_plan_identity": public_stationary.identity,
    }


def _artifact_verification(
    directory: Path | None,
    *,
    functional: int,
    spin: str,
    plan: Any,
) -> dict[str, Any]:
    if directory is None:
        return {"status": "not_checked", "detail": "no AOT directory supplied"}
    try:
        artifact = load_stationary_aot_artifact(
            directory,
            functional=functional,
            spin=spin,
            plan=plan,
            architecture="sm_120",
            component_domain=QUALIFIED_SPD_COMPONENTS,
        )
    except OSError as error:
        return {"status": "missing_or_invalid", "detail": str(error)}
    except (NotImplementedError, TypeError, ValueError) as error:
        return {"status": "missing_or_invalid", "detail": str(error)}
    except KeyError as error:
        field = error.args[0] if error.args else "unknown"
        return {
            "status": "missing_or_invalid",
            "detail": f"missing AOT manifest field: {field}",
        }
    except AttributeError as error:
        return {
            "status": "missing_or_invalid",
            "detail": f"invalid AOT manifest schema: {error}",
        }
    return {
        "status": "verified",
        "detail": "packaged s/p/d AOT contract and binary identity verified",
        "library": str(Path(artifact.library).resolve()),
        "binary_sha256": artifact.metadata["binary_sha256"],
        "artifact_key": artifact.metadata["key"],
        "code_kinds": list(artifact.metadata["identity"]["target"]["code_kinds"]),
        "driver_ptx_jit_required": artifact.metadata["driver_ptx_jit_required"],
    }


def _validate_aot_evidence_directory(
    aot_directory: Path | None,
) -> Path | None:
    if aot_directory is None:
        return None
    aot_path = Path(aot_directory).absolute()

    def linked(candidate: Path) -> bool:
        junction = getattr(candidate, "is_junction", None)
        return candidate.is_symlink() or (junction is not None and junction())

    if any(linked(candidate) for candidate in (aot_path, *aot_path.parents)) or any(
        linked(candidate) for candidate in aot_path.rglob("*")
    ):
        raise ValueError("AOT evidence must not contain symlinks")
    return aot_path


def _build_report(
    repository: Path,
    *,
    source_sha: str,
    aot_directory: Path | None = None,
) -> dict[str, Any]:
    repository = Path(repository).resolve()
    _assert_local_imports()
    if repository != SOURCE_REPOSITORY:
        raise ValueError(
            "capacity report must run against the checkout containing this tool"
        )
    if re.fullmatch(r"[0-9a-f]{40}", source_sha) is None:
        raise ValueError("source_sha must be a full lowercase Git commit SHA")
    _validate_aot_evidence_directory(aot_directory)
    _basis_pack.cache_clear()
    _named_basis_record.cache_clear()
    root = repository / "tools/dft_mp_v1"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    frozen_hash = manifest.get("contract_sha256")
    unhashed = dict(manifest)
    unhashed.pop("contract_sha256", None)
    if frozen_hash != _canonical_sha256(unhashed):
        raise ValueError("DFT-MP-v1 frozen contract hash mismatch")

    basis_pack = repository / "python/generativeqc/data/basis_pack.json"
    basis_pack_sha = _lf_sha256(basis_pack.read_bytes())
    basis_pack_matches = basis_pack_sha == manifest["basis"]["basis_pack_sha256"]
    if not basis_pack_matches:
        raise ValueError(
            "current basis pack differs from the frozen DFT-MP-v1 contract"
        )

    limits = _source_limits(repository)
    basis_layout = _basis_layout_contract(repository)
    spd_expansion = _spd_expansion_contract(repository)
    source_package = _source_package_inventory(repository)
    public_route = _source_public_route(repository)
    limits["public_work_limits"] = dict(public_route["whole_grid_work_limits"])
    public_route["registry_manifest_sha256"] = _lf_sha256(
        (repository / "manifests/public_methods.json").read_bytes()
    )
    public_route["prepared_aot_selection_sha256"] = _prepared_aot_route_contract(
        repository
    )
    grid_contract = _grid_count_contract(repository)
    grid_spec = _grid_spec(manifest["model"]["grid_spec"])

    required_rows = [
        row
        for row in manifest["rows"]
        if row["required"]
        and row["method"] in FP64_FORCE_METHODS
        and row["level"] == "fp64_energy_forces"
    ]
    rows_by_case: dict[str, list[dict[str, Any]]] = {}
    for row in required_rows:
        rows_by_case.setdefault(row["case"], []).append(row)

    cases = []
    case_work: dict[str, dict[str, Any]] = {}
    ao_counts_match = True
    for case_name, frozen in sorted(manifest["cases"].items()):
        input_path = root / frozen["input"]
        raw = input_path.read_bytes()
        if _lf_sha256(raw) != frozen["input_sha256"]:
            raise ValueError(f"frozen input hash mismatch for {case_name}")
        value = json.loads(raw)
        atoms = tuple(Atom.from_value(atom) for atom in value["atoms"])
        changed_path = root / frozen["changed_input"]
        changed_raw = changed_path.read_bytes()
        if _lf_sha256(changed_raw) != frozen["changed_input_sha256"]:
            raise ValueError(f"frozen changed input hash mismatch for {case_name}")
        changed_value = json.loads(changed_raw)
        changed_atoms = tuple(Atom.from_value(atom) for atom in changed_value["atoms"])
        shape, (basis, basis_metadata) = _basis_shape(
            atoms, value["charge"], value["multiplicity"]
        )
        ao_match = shape["ao_count_spherical"] == frozen["ao_count_spherical"]
        ao_counts_match = ao_counts_match and ao_match
        if not ao_match:
            raise ValueError(
                f"actual spherical AO count differs from manifest for {case_name}"
            )
        grid = MolecularGrid(
            atoms,
            grid_spec,
            charge=value["charge"],
            multiplicity=value["multiplicity"],
        )
        if grid.identity != frozen["grid_identity"]:
            raise ValueError(
                f"current grid identity differs from manifest for {case_name}"
            )
        changed_grid = MolecularGrid(
            changed_atoms,
            grid_spec,
            charge=changed_value["charge"],
            multiplicity=changed_value["multiplicity"],
        )
        if changed_grid.identity != frozen["changed_grid_identity"]:
            raise ValueError(
                f"current changed grid identity differs from manifest for {case_name}"
            )
        native_grid_points = (
            len(atoms)
            * grid_spec.radial_points
            * grid_spec.angular_polar
            * grid_spec.angular_azimuth
        )
        changed_native_grid_points = (
            len(changed_atoms)
            * grid_spec.radial_points
            * grid_spec.angular_polar
            * grid_spec.angular_azimuth
        )
        if grid.npoint != native_grid_points or (
            changed_grid.npoint != changed_native_grid_points
        ):
            raise RuntimeError("source-only grid count disagrees with native contract")

        atom_pairs = shape["atom_count"] * (shape["atom_count"] - 1) // 2
        primitive_sum = shape["component_primitive_sum"]
        requirements = {
            "primitive_records": primitive_sum**4
            + (shape["atom_count"] + 2) * primitive_sum**2
            + atom_pairs,
            "primitive_descriptor_peak_records": shape["ao_primitive_count_peak"] ** 4,
            "grid_points": native_grid_points,
            "grid_pair_visits": (1 + 2 * native_grid_points) * atom_pairs,
        }
        grid_work = plan_stationary_cuda_grid_work(
            atoms=shape["atom_count"],
            grid_points=native_grid_points,
            tile_points=limits["tile_points"],
            max_grid_points=limits["public_work_limits"]["grid_points"],
            max_grid_pair_visits=limits["public_work_limits"]["grid_pair_visits"],
            max_pending_tiles=limits["pending_grid_tiles"],
            max_pending_pair_visits=limits["pending_grid_pair_visits"],
        )
        if grid_work.grid_pair_visits != requirements["grid_pair_visits"]:
            raise RuntimeError("compiler grid work disagrees with native census")
        native_integral_admission = {
            "required": stationary_cuda_requires_native_integrals(
                atoms=shape["atom_count"],
                aos=shape["ao_count_spherical"],
                primitives=shape["basis_primitive_count"],
            ),
            "provider_and_budget_qualification": "NOT_RUN",
            "enlarged_domain_failure_behavior": (
                "fail closed; enlarged domains cannot use AO-task fallback"
            ),
            "small_domain_fallback": "bounded AO-task route remains available",
            "stationary_primitive_records_if_native_complete": atom_pairs,
            "ao_task_descriptors_if_native_complete": 0,
        }
        method_memory = {}
        method_plans = {}
        method_requirements = {}
        for row in rows_by_case.get(case_name, ()):
            spin = "unpolarized" if row["spin"] == "rks" else "polarized"
            selector = f"{row['method']}-{row['spin']}"
            key = f"{row['method']}/{row['spin']}"
            if key not in method_memory:
                method_ir, _ = resolve_ks_method(selector)
                plan = StationaryGradientPlan(
                    method_ir,
                    StationaryMeanField(SCF_POINT_MODEL),
                )
                method_requirements[key] = {
                    **requirements,
                    "primitive_records": (
                        (1 + int(bool(method_ir.full_range_exact_exchange)))
                        * primitive_sum**4
                        + (shape["atom_count"] + 2) * primitive_sum**2
                        + atom_pairs
                    ),
                }
                method_memory[key], method_plans[key] = _method_resources(
                    basis,
                    atom_count=shape["atom_count"],
                    functional=int(native_xc_functional_code(selector)),
                    spin=spin,
                    limits=limits,
                    plan=plan,
                )
        admission_by_method_spin = {
            key: _admission_record(
                _case_failures(shape, method_requirements[key], memory, limits)
            )
            for key, memory in method_memory.items()
        }
        maximum_memory = {
            "minimum_additional_device_bytes": max(
                item["minimum_additional_device_bytes"]
                for item in method_memory.values()
            ),
            "additional_device_peak_bound": max(
                item["additional_device_peak_bound"] for item in method_memory.values()
            ),
            "additional_host_numeric_bound": max(
                item["additional_host_numeric_bound"] for item in method_memory.values()
            ),
        }
        maximum_requirements = {
            **requirements,
            "primitive_records": max(
                item["primitive_records"] for item in method_requirements.values()
            ),
        }
        failures = _case_failures(shape, maximum_requirements, maximum_memory, limits)
        record = {
            "id": case_name,
            "classification": frozen["classification"],
            "charge": value["charge"],
            "multiplicity": value["multiplicity"],
            "shape": {**shape, "grid_points": native_grid_points},
            "identities": {
                "input_sha256": frozen["input_sha256"],
                "changed_input_sha256": frozen["changed_input_sha256"],
                "basis": basis_metadata["mathematical_identity"],
                "basis_pack_sha256": basis_pack_sha,
                "ao_representation": "real_spherical/libcint-PySCF-order",
                "grid": grid.identity,
                "changed_grid": changed_grid.identity,
            },
            "requirements": {
                **maximum_requirements,
                "primitive_records_scope": "maximum logical AO-task reference across requested methods; not executed native work",
                "grid_work_plan": asdict(grid_work),
                "native_integral_admission": native_integral_admission,
                "memory_by_method_spin": method_memory,
            },
            "diagnostic_admission": _admission_record(
                _case_failures(
                    shape,
                    maximum_requirements,
                    maximum_memory,
                    limits,
                    work_mode="diagnostic",
                )
            ),
            "requested_rows": sorted(
                row["id"] for row in rows_by_case.get(case_name, ())
            ),
            "admission": _admission_record(failures),
            "admission_by_method_spin": admission_by_method_spin,
        }
        cases.append(record)
        case_work[case_name] = {
            "basis": basis,
            "plans": method_plans,
            "requirements": method_requirements,
            "memory": method_memory,
            "admission_by_method_spin": admission_by_method_spin,
            "record": record,
        }

    artifact_cache: dict[tuple[int, str, str], dict[str, Any]] = {}
    selector_cache: dict[str, dict[str, Any]] = {}
    rows = []
    for frozen_row in sorted(required_rows, key=lambda item: item["id"]):
        method = frozen_row["method"]
        spin = "unpolarized" if frozen_row["spin"] == "rks" else "polarized"
        selector = f"{method}-{'rks' if spin == 'unpolarized' else 'uks'}"
        functional = int(native_xc_functional_code(selector))
        method_key = f"{method}/{frozen_row['spin']}"
        plan = case_work[frozen_row["case"]]["plans"][method_key]
        aot_key = (functional, spin, plan.identity)
        if aot_key not in artifact_cache:
            artifact_cache[aot_key] = _artifact_verification(
                aot_directory,
                functional=functional,
                spin=spin,
                plan=plan,
            )
        if selector not in selector_cache:
            selector_cache[selector] = _public_selector_contract(
                selector,
                expected_functional=functional,
                expected_spin=spin,
                stationary_plan=plan,
                grid_spec=grid_spec,
            )
        native_properties = list(
            generated_methods.METHOD_METADATA[
                selector_cache[selector]["registry_selector"]
            ]["properties"]
        )
        rows.append(
            {
                **frozen_row,
                "stationary_plan": {
                    "identity": plan.identity,
                    "source_names": list(stationary_runtime_sources(plan)),
                    "component_domain": list(QUALIFIED_SPD_COMPONENTS),
                },
                "resource_requirements": {
                    "atom_count": case_work[frozen_row["case"]]["record"]["shape"][
                        "atom_count"
                    ],
                    "ao_count_spherical": case_work[frozen_row["case"]]["record"][
                        "shape"
                    ]["ao_count_spherical"],
                    **{
                        key: case_work[frozen_row["case"]]["requirements"][method_key][
                            key
                        ]
                        for key in (
                            "primitive_records",
                            "primitive_descriptor_peak_records",
                            "grid_points",
                            "grid_pair_visits",
                        )
                    },
                    "primitive_records_scope": "logical AO-task reference; not executed native work",
                    "grid_work_plan": case_work[frozen_row["case"]]["record"][
                        "requirements"
                    ]["grid_work_plan"],
                    "native_integral_admission": case_work[frozen_row["case"]][
                        "record"
                    ]["requirements"]["native_integral_admission"],
                    "primitive_page_work_budget": limits["primitive_records"],
                    **case_work[frozen_row["case"]]["memory"][method_key],
                },
                "public_capability": {
                    "forces": True,
                    "native_registry_properties": native_properties,
                    "promotion": "python direct-CUDA stationary-force predicate",
                    "owner": "python/generativeqc/calculator.py::Calculator.__init__",
                    "source_audited": True,
                    "selector_contract": selector_cache[selector],
                },
                "public_route": {
                    "scientific_runtime_compilation_required": method
                    not in SEMILOCAL_FUNCTIONALS,
                    "selection": (
                        "runtime-compiled stationary CUDA"
                        if method not in SEMILOCAL_FUNCTIONALS
                        else "all-electron packaged stationary CUDA"
                    ),
                    "owner": "python/generativeqc/batch.py::_public_dft_cuda_force",
                    "missing_aot_behavior": (
                        "not selected by this public route"
                        if method not in SEMILOCAL_FUNCTIONALS
                        else "fail closed; no NVCC fallback"
                    ),
                    "source_audited": True,
                },
                "packaged_aot": {
                    "name": _stationary_aot_name(
                        functional,
                        spin,
                        component_domain=QUALIFIED_SPD_COMPONENTS,
                        plan=plan,
                    ),
                    "architecture": "sm_120",
                    "source_package_declared": True,
                    "selected_by_public_route": method in SEMILOCAL_FUNCTIONALS,
                    "scope": "package availability evidence only; not execution qualification",
                    "source_owner": "cmake/GenerativeQCCuda.cmake",
                    "contract_identity": stationary_aot_profile_contract_identity(
                        f"{method}_{'rks' if spin == 'unpolarized' else 'uks'}",
                        component_domain=QUALIFIED_SPD_COMPONENTS,
                    ),
                    "binary_verification": artifact_cache[aot_key],
                },
                "admission": case_work[frozen_row["case"]]["admission_by_method_spin"][
                    method_key
                ],
            }
        )

    owner_files = (
        "python/generativeqc/_stationary_cuda.py",
        "python/generativeqc/_ks_snapshot.py",
        "python/generativeqc/_snapshot_grid_cache.py",
        "python/generativeqc/resources_ks.py",
        "python/generativeqc/calculator.py",
        "python/generativeqc/batch.py",
        "python/generativeqc/ks.py",
        "python/generativeqc_compiler/method/stationary_cuda.py",
        "python/generativeqc_compiler/method/stationary_resources.py",
        "python/generativeqc_compiler/dft/ao.py",
        "python/generativeqc_compiler/dft/grid.py",
        "python/generativeqc_compiler/xc/quadrature_cuda.py",
        "src/molecule/basis.cpp",
        "src/dft/ao_grid.cpp",
        "src/dft/bridge.cpp",
        "src/dft/grid.hpp",
        "src/dft/cuda_quadrature.cu",
        "src/dft/stationary_gradient_cuda.cuh",
        "src/methods/dft_method.cpp",
        "cmake/GenerativeQCCuda.cmake",
    )
    blocked = sum(row["admission"]["outcome"] == "blocked" for row in rows)
    return {
        "schema": SCHEMA,
        "source": {
            "sha": source_sha,
            "qualifier_sha256": _IMPORTED_TOOL_SOURCE_SHA256,
            "imported_module_sha256": {
                name: digest
                for name, (_, digest) in sorted(_IMPORTED_LOCAL_MODULE_SOURCES.items())
            },
            "runtime_owner_sha256": {
                path: _lf_sha256((repository / path).read_bytes())
                for path in owner_files
            },
        },
        "contract": {
            "id": manifest["contract_id"],
            "version": manifest["version"],
            "sha256": frozen_hash,
        },
        "basis": {
            "name": manifest["basis"]["name"],
            "representation": manifest["basis"]["representation"],
            "basis_pack_sha256": basis_pack_sha,
            "basis_pack_sha256_match": basis_pack_matches,
            "manifest_ao_counts_match": ao_counts_match,
            **basis_layout,
            **spd_expansion,
            "component_primitive_sum_derivation": (
                "sum(shell primitive count * sparse public-AO Cartesian term count); "
                "s=1, p=3, spherical d=8 from src/molecule/basis.cpp"
            ),
        },
        "grid": grid_contract,
        "public_route": public_route,
        "stationary_aot_source_package": source_package,
        "admission_limits": limits,
        "aot_binary_directory": None
        if aot_directory is None
        else str(Path(aot_directory).resolve()),
        "cases": cases,
        "rows": rows,
        "summary": {
            "required_fp64_force_rows": len(rows),
            "required_semilocal_fp64_force_rows": sum(
                row["method"] in SEMILOCAL_FUNCTIONALS for row in rows
            ),
            "statically_blocked_rows": blocked,
            "rows_passing_static_stationary_caps": len(rows) - blocked,
            "scientific_qualification": "NOT_RUN",
        },
    }


def _clean_git_sha(
    repository: Path,
    *,
    ignored_path: Path | None = None,
) -> str:
    tagged = subprocess.run(
        ["git", "ls-files", "-v", "--", "."],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    hidden = [
        line[2:] for line in tagged if line and (line[0] == "S" or line[0].islower())
    ]
    if hidden:
        raise RuntimeError(
            "capacity report rejects assume-unchanged/skip-worktree paths: "
            + ", ".join(hidden)
        )
    status_command = [
        "git",
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        ".",
    ]
    if ignored_path is not None:
        resolved_ignored = Path(ignored_path).resolve()
        if resolved_ignored.is_relative_to(repository):
            relative = resolved_ignored.relative_to(repository).as_posix()
            status_command.append(f":(exclude,literal){relative}")
    status = subprocess.run(
        status_command,
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise RuntimeError("capacity report requires a clean Git worktree")
    return _git_head(repository)


def _report_output_exemption(
    repository: Path,
    output_path: Path,
    *,
    aot_directory: Path | None = None,
) -> Path | None:
    """Allow only an untracked JSON report to be ignored inside the checkout."""

    output_path = Path(output_path).resolve()
    if aot_directory is not None:
        aot_path = _validate_aot_evidence_directory(aot_directory)
        assert aot_path is not None
        if output_path.is_relative_to(aot_path.resolve()):
            raise ValueError("capacity report output must not overlap AOT evidence")
    if output_path.exists() and output_path.stat().st_nlink != 1:
        raise ValueError("capacity report output must not be a hard link")
    if not output_path.is_relative_to(repository):
        return None
    if output_path.suffix.lower() != ".json":
        raise ValueError("in-checkout capacity report output must be a JSON file")
    relative = output_path.relative_to(repository).as_posix()
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    if tracked.returncode == 0:
        raise ValueError("capacity report output must not replace a tracked file")
    return output_path


def build_report(
    repository: Path,
    *,
    aot_directory: Path | None = None,
) -> dict[str, Any]:
    """Build a report whose source identity is the clean tool-checkout HEAD."""

    repository = Path(repository).resolve()
    if repository != SOURCE_REPOSITORY:
        raise ValueError(
            "capacity report must run against the checkout containing this tool"
        )
    _assert_local_imports(require_fresh=True)
    return _build_report(
        repository,
        source_sha=_clean_git_sha(repository),
        aot_directory=aot_directory,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        type=Path,
        default=SOURCE_REPOSITORY,
        help="checkout containing this tool; other repositories are rejected",
    )
    parser.add_argument(
        "--aot-directory",
        type=Path,
        help="optionally verify the actual sm_120 packaged s/p/d binaries",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repository = args.repository.resolve()
    output_path = None if args.output is None else args.output.resolve()
    if output_path is None:
        payload = build_report(repository, aot_directory=args.aot_directory)
    else:
        if repository != SOURCE_REPOSITORY:
            raise ValueError(
                "capacity report must run against the checkout containing this tool"
            )
        _assert_local_imports(require_fresh=True)
        exemption = _report_output_exemption(
            repository,
            output_path,
            aot_directory=args.aot_directory,
        )
        payload = _build_report(
            repository,
            source_sha=_clean_git_sha(repository, ignored_path=exemption),
            aot_directory=args.aot_directory,
        )
    text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
