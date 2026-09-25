"""Complete numeric-capacity planning for native semilocal prepared KS owners.

The dry run resolves shell metadata and integer shapes, never quadrature,
integrals, density matrices, eigensolvers or a CUDA execution context. CPU
workspace bounds cover the existing dense reference route. CUDA device state
uses allocator-owned layout queries and the common #202 direct-J boundary.
All estimates compose through the existing #203 ResourcePlan and one budget.
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import typing
from dataclasses import asdict, replace

from vibeqc_compiler.common.resources import (
    ResourceBudget,
    ResourceCandidate,
    ResourceEstimate,
    ResourceIdentity,
    ResourceRequest,
    byte_product,
    checked_bytes,
    plan_resources,
)

from ._cpu_force_resources import CPU_FORCE_HOST_CAP, qualified_basis
from .basis import BasisSet
from .basis_capabilities import require_basis
from .calculator import Atom, _snapshot_basis
from .elements import electron_state
from .ks import resolve_ks_options
from .resources_hf import _basis_record, _cuda_library_identity, _ecp_workspace

_CPU_AO_GRID_CACHE_CAP = 64 << 20

_METHODS = (
    "lda-rks",
    "pbe-rks",
    "lda-uks",
    "pbe-uks",
    "pbe0-rks",
    "pbe0-uks",
    "b3lyp-rks",
    "b3lyp-uks",
)


def _item_host_inventory(
    item: typing.Any,
    *,
    diis_history: typing.Any,
    max_iterations: typing.Any,
    pbe: typing.Any,
    backend: typing.Any,
    model: typing.Any,
) -> typing.Any:
    """Bound numeric and LP64 value metadata by actual execution lifetimes.

    Every plan retains its host grid, basis, warm/source metadata and provider.
    Preparation and CPU SCF are serialized across items; CUDA SCF state and
    histories coexist for all active streams. A geometry rebuild destroys its
    old owner before allocating the new one, retaining only its last-good seed.
    """
    orbital = item["orbital"]
    a, n, c = item["atoms"], orbital["nbf"], orbital["cartesian_nbf"]
    s, p, spins = orbital["shells"], orbital["primitives"], item["spins"]
    n2, c2 = byte_product(n, n), byte_product(c, c)
    points = item["grid_points"]
    packed = 3 * a + 2 * p + 16 * n
    # One owned grid has xyz/weights/owner. Source-system copies, AO records,
    # descriptors and plan objects have a conservative LP64 value allowance.
    # Include fixed element-radius tables copied into method/grid/XC snapshots.
    metadata = 8192 + 512 * (a + s + p + n + c + orbital.get("ecp_terms", 0))
    grid = byte_product(36, points)
    basis = byte_product(8, packed)
    warm_and_matrices = byte_product(8, n2, 4 + 4 * spins) + 8 * 3 * a * 4
    # 128 bytes covers ScfIteration (checked by native inventory v1); the
    # factor two includes CPU vector growth or the CUDA owner plus its exported
    # history. The adapter/ABI move that export through to its cached handle.
    history = byte_product(256, max_iterations)
    provider = byte_product(8, 2 * n2 + (n2 * n2 if backend == "cpu" else 0))
    # The provider retains S/H; do not count those again as a second owner.
    retained = metadata + grid + basis + warm_and_matrices + history + provider
    # Ordinary KS reuses one serialized Xsyevd workspace across spins. Match
    # the shared native admission bound; actual provider queries are checked
    # before either allocation. Small native solves need no provider workspace.
    solver_host = (
        1024 * 1024 + byte_product(16, 8, n2) if backend == "cuda" and n > 16 else 0
    )
    retained += solver_host
    quadrature = 16 * (model.grid.radial_points + model.grid.angular_polar) + 8 * a
    if backend == "cuda":
        # Shared rule/center/radius upload coexists with the small rule vectors.
        quadrature += 8 * (4 * a + 2 * (512 + 256))
    matrix_work = byte_product(8, spins, n2, 128 + 2 * (diis_history + 1))
    matrix_work += byte_product(16, diis_history + 1, diis_history + 1)
    host_unfused = backend == "cuda" and model.xc_schedule == "host_unfused"
    xc_tile = (
        byte_product(8, min(points, model.tile_points), n, 4 if pbe else 1)
        + byte_product(8, spins, n2)
        if backend == "cpu" or host_unfused
        else 0
    )
    # run_rks owns this optional cache only for the current serialized solve.
    # Reserve its full eligible footprint even when an experimental XC route
    # declines it; never charge all batch caches as persistent/coexisting state.
    ao_grid_cache = 0
    if (
        backend == "cpu"
        and pbe
        and spins == 1
        and n > 0
        and points <= _CPU_AO_GRID_CACHE_CAP // 32 // n
    ):
        ao_grid_cache = byte_product(32, points, n)
    # Match CudaKsPlan's retained host staging exactly. Host-unfused owns one
    # density and one Vxc matrix per spin; UKS additionally owns split alpha/
    # beta matrices for the audited CPU integrator.
    xc_schedule_staging = (
        byte_product(8, n2, 2 * spins + (2 if spins == 2 else 0)) if host_unfused else 0
    )
    nonlocal_provider = 0
    nonlocal_work = 0
    if model.has_nonlocal_correlation:
        # Vv10Plan retains omega/kappa/weighted-density and three local
        # derivatives: six FP64 arrays. The AO bridge separately owns rho,
        # grad-rho, vrho/vsigma, one first-derivative AO tile and V_nlc.
        nonlocal_provider = byte_product(8, points, 6)
        if nonlocal_provider > model.nonlocal_memory_budget_bytes:
            raise ValueError(
                "KS nonlocal provider workspace exceeds nonlocal_memory_budget_bytes"
            )
        matrix_factor = 3 if spins == 2 else 1
        nonlocal_work = (
            byte_product(8, points, 6)
            + byte_product(8, min(points, model.tile_points), n, 4)
            + byte_product(8, n2, matrix_factor)
        )
        retained += nonlocal_provider
    if backend == "cpu":
        # Value-only Jet objects retain no derivative arrays. Raw Cartesian
        # Jet integrals coexist with unpacked and spherical transform buffers.
        angular = orbital["maximum_angular"]
        hermite = 6 * (angular + 3) ** 2 * (2 * angular + 4)
        recurrence = 64 * ((4 * angular + 1) ** 4 + hermite + 256 + a)
        integrals = byte_product(64, 2 * c2 + byte_product(c2, c2))
        integrals += max(recurrence, 32 * (c2 * c2 + n2 * n2 + c * n + n2))
        setup = integrals + quadrature + basis
    else:
        pairs = s * (s + 1) // 2
        # Common HostBatch packing retains primitive/shell-pair metadata and
        # may form PSSS task descriptors. Bound vector growth, source copies,
        # AO transforms and Cartesian one-electron downloads without packing.
        setup = 2048 * (1 + a + s + p + n + c + pairs + pairs * pairs)
        setup += 64 * (c2 + n2 + n * c) + quadrature + basis
    setup += _ecp_workspace(item)
    return {
        key: checked_bytes(value, f"KS {key}")
        for key, value in {
            "metadata": metadata,
            "grid": grid,
            "basis": basis,
            "warm_and_matrices": warm_and_matrices,
            "history": history,
            "provider": provider,
            "solver_host": solver_host,
            "xc_schedule_staging": xc_schedule_staging,
            "nonlocal_provider": nonlocal_provider,
            "ao_grid_cache": ao_grid_cache,
            "retained": retained,
            "setup_workspace": setup,
            "scf_workspace": matrix_work + xc_tile + nonlocal_work + ao_grid_cache,
        }.items()
    }


def _cuda_item_inventory(
    library: typing.Any,
    item: typing.Any,
    *,
    diis_history: typing.Any,
    pbe: typing.Any,
    tile: typing.Any,
) -> typing.Any:
    query = getattr(library, "vibeqc_resource_ks_cuda_v1", None)
    if query is None:
        raise NotImplementedError(
            "native library has no KS CUDA allocation inventory v1"
        )
    orbital = item["orbital"]
    args = (
        orbital["nbf"],
        item["atoms"],
        orbital["shells"],
        orbital["primitives"],
        item["grid_points"],
        diis_history,
        item["spins"],
        int(pbe),
        tile,
    )
    if any(value > 2 ** (8 * ctypes.sizeof(ctypes.c_size_t)) - 1 for value in args):
        raise ValueError("KS resource shape exceeds the host size_t ABI")
    query.argtypes = [ctypes.c_size_t] * 9 + [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
    ]
    query.restype = ctypes.c_int
    output = (ctypes.c_uint64 * 3)()
    if query(*args, output, len(output)):
        raise NotImplementedError(
            "KS CUDA inventory requires a CUDA build and valid launch/DIIS dimensions"
        )
    n, a, s, p = (
        orbital["cartesian_nbf"],
        item["atoms"],
        orbital["shells"],
        orbital["primitives"],
    )
    pairs = s * (s + 1) // 2
    # The existing one-electron exporter uploads Cartesian metadata, shell/AO
    # pair indices and S/H/nuclear output, then frees them before preparing J.
    # This bound includes every explicit device buffer of that value-only path.
    setup = byte_product(64, 1 + a + s + p + n + pairs + n * n)
    setup = max(setup, _ecp_workspace(item, cuda=True))
    quadrature_query = getattr(library, "vibeqc_resource_quadrature_cuda_v1", None)
    if quadrature_query is None:
        raise NotImplementedError(
            "native library has no CUDA quadrature allocation inventory"
        )
    quadrature_query.argtypes = [
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    quadrature_query.restype = ctypes.c_int
    quadrature = ctypes.c_uint64()
    if quadrature_query(a, item["grid_points"], ctypes.byref(quadrature)):
        raise NotImplementedError("invalid CUDA quadrature resource shape/build")
    # Quadrature is built after the Coulomb provider, before KS/XC buffers.
    # Count the coexistence explicitly, including changed-geometry rebuilds.
    setup = max(setup, checked_bytes(int(output[2]) + quadrature.value))
    return {
        "state": checked_bytes(int(output[0])),
        "xc": checked_bytes(int(output[1])),
        "coulomb": checked_bytes(int(output[2])),
        "setup": checked_bytes(setup),
        "quadrature_setup": checked_bytes(quadrature.value),
    }


def ks_resource_request(
    systems: typing.Any,
    *,
    method: typing.Any = "pbe-rks",
    basis: typing.Any = "sto-3g",
    backend: typing.Any = "cpu",
    precision: typing.Any = "fp64",
    basis_representation: typing.Any = None,
    charges: typing.Any = None,
    multiplicities: typing.Any = None,
    diis_history: typing.Any = 8,
    max_iterations: typing.Any = 100,
    energy_tolerance: typing.Any = 1e-10,
    density_tolerance: typing.Any = 1e-8,
    screening_tolerance: typing.Any = 1e-12,
    ks_options: typing.Any = None,
    device_id: typing.Any = 0,
    library: typing.Any = None,
    name: typing.Any = "ks",
    first_phase: typing.Any = 0,
    last_phase: typing.Any = 0,
) -> typing.Any:
    """Resolve one complete energy-only KS request for the shared global planner."""
    if method not in _METHODS or backend not in ("cpu", "cuda"):
        raise NotImplementedError(
            "KS planning supports native CPU LDA/PBE/PBE0 and CUDA LDA/PBE RKS/UKS energies"
        )
    precision = str(precision).lower()
    if precision not in ("fp64", "auto"):
        raise ValueError("KS precision must be 'fp64' or 'auto'")
    if precision == "auto" and backend != "cuda":
        raise NotImplementedError("KS automatic precision currently requires CUDA")
    model = resolve_ks_options(method, ks_options)
    if backend == "cuda" and model.has_nonlocal_correlation:
        raise NotImplementedError(
            "self-consistent nonlocal correlation currently requires CPU"
        )
    if backend == "cuda" and model.has_nondefault_composition:
        raise NotImplementedError(
            "CUDA KS planning does not claim scaled/global-hybrid execution"
        )
    systems = tuple(tuple(Atom.from_value(a) for a in atoms) for atoms in systems)
    if not systems or any(not atoms for atoms in systems):
        raise ValueError("KS resource planning requires nonempty systems")
    charges = (0,) * len(systems) if charges is None else tuple(charges)
    multiplicities = (
        (1,) * len(systems) if multiplicities is None else tuple(multiplicities)
    )
    if len(charges) != len(systems) or len(multiplicities) != len(systems):
        raise ValueError("charges and multiplicities must match the KS batch")
    for label, value in (
        ("DIIS history", diis_history),
        ("maximum iterations", max_iterations),
        ("device", device_id),
    ):
        checked_bytes(value, label)
    # Match descriptor defaults, including explicitly supplied zero controls.
    diis_history, max_iterations = diis_history or 8, max_iterations or 100
    for value in (energy_tolerance, density_tolerance, screening_tolerance):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("KS numerical tolerances must be positive finite")
    selected = _snapshot_basis(basis, basis_representation)
    cpu_forces = backend == "cpu" and qualified_basis(selected)
    pbe, unrestricted = bool(model.ao_order), method.endswith("uks")
    items = []
    for atoms, charge, multiplicity in zip(
        systems, charges, multiplicities, strict=True
    ):
        electrons = electron_state(
            atoms,
            charge=charge,
            multiplicity=multiplicity,
            element_metadata=selected.by_element
            if isinstance(selected, BasisSet)
            else {},
        )
        if not unrestricted and (
            multiplicity != 1 or electrons.nalpha != electrons.nbeta
        ):
            raise ValueError("RKS requires a closed-shell singlet")
        orbital = _basis_record(
            atoms,
            selected,
            basis_representation,
            backend,
            charge,
            multiplicity,
            "orbital",
            derivative_order=0,
        )
        if (
            not electrons.nalpha
            or max(electrons.nalpha, electrons.nbeta) > orbital["nbf"]
        ):
            raise ValueError("KS spin populations do not fit the AO space")
        require_basis(
            selected,
            atoms,
            backend=backend,
            operator="ao",
            derivative_order=int(pbe),
            representation=orbital["representation"],
            role="orbital",
        )
        items.append(
            {
                "atoms": len(atoms),
                "orbital": orbital,
                "electrons": asdict(electrons),
                "spins": 2 if unrestricted else 1,
                "grid_points": byte_product(
                    len(atoms),
                    model.grid.radial_points,
                    model.grid.angular_polar,
                    model.grid.angular_azimuth,
                ),
            }
        )
    controls = {
        "diis_history": diis_history,
        "max_iterations": max_iterations,
        "energy_tolerance": energy_tolerance,
        "density_tolerance": density_tolerance,
        "screening_tolerance": screening_tolerance,
        "ks_options": model.to_payload(),
        "outputs": "energy+forces" if backend == "cuda" or cpu_forces else "energy",
        "inventory_version": 1,
        "schedule": "ordinary-stream-round-robin"
        if backend == "cuda"
        else "serialized-native",
    }
    if backend == "cpu":
        controls["ao_grid_cache_cap"] = _CPU_AO_GRID_CACHE_CAP
    if backend == "cuda":
        controls.update(
            device_id=device_id,
            one_electron_mapping=os.environ.get("VIBEQC_ONE_ELECTRON_VALUE_MAPPING"),
        )
    # AUTO runs two separately bounded nonlinear stages. Reserve the native
    # owner and exported history for both stages plus strict closure corrections.
    history_iterations = max_iterations
    if precision == "auto":
        history_iterations = checked_bytes(2 * max_iterations + 4, "KS mixed history")
        controls["history_iterations"] = history_iterations
    identity = ResourceIdentity(
        method,
        "native-ks-direct-v1",
        backend,
        precision,
        json.dumps({"items": items}),
        ("energy", "forces") if backend == "cuda" or cpu_forces else ("energy",),
        json.dumps(controls, sort_keys=True),
    )
    exclusions = (
        "caller-owned inputs and previously retained explicit output/snapshot objects",
        "Python objects, allocator bookkeeping, arenas and page rounding",
    )
    if backend == "cuda":
        exclusions += (
            "CUDA driver/context/modules, compiler-managed stacks and pool/page retention",
        )
    host = [
        _item_host_inventory(
            item,
            diis_history=diis_history,
            max_iterations=history_iterations,
            pbe=pbe,
            backend=backend,
            model=model,
        )
        for item in items
    ]
    estimates = []
    for key in (
        "metadata",
        "grid",
        "basis",
        "warm_and_matrices",
        "history",
        "provider",
        "solver_host",
        "xc_schedule_staging",
        "nonlocal_provider",
    ):
        estimates.append(
            ResourceEstimate(
                f"all KS host {key}",
                checked_bytes(sum(x[key] for x in host)),
                "pageable",
                first_phase,
                last_phase,
                kind="persistent",
            )
        )
    estimates.append(
        ResourceEstimate(
            "serialized KS setup/SCF/XC host workspace",
            max(max(x["setup_workspace"], x["scf_workspace"]) for x in host),
            "pageable",
            first_phase,
            last_phase,
        )
    )
    device = []
    if cpu_forces:
        estimates.append(
            ResourceEstimate(
                "serialized generated KS CPU force host staging cap",
                CPU_FORCE_HOST_CAP,
                "pageable",
                first_phase,
                last_phase,
            )
        )
        exclusions += (
            "force JIT/compiler processes, loaded code, BLAS/runtime internal storage",
        )
    try:
        if backend == "cuda" and library is None:
            from . import _native

            library = _native.load_library(device="cpu")
        if library is not None:
            options_version = library.vibeqc_ks_options_version
            options_version.argtypes, options_version.restype = [], ctypes.c_uint32
            if options_version() != 1:
                raise NotImplementedError(
                    "native library does not support the current semantic KS execution-plan ABI"
                )
            version = getattr(library, "vibeqc_ks_resource_inventory_version_v1", None)
            if version is not None:
                version.argtypes, version.restype = [], ctypes.c_int
            if version is None or version() != 1:
                raise NotImplementedError(
                    "native library does not implement KS allocation inventory v1"
                )
        if backend == "cuda":
            identity = replace(
                identity,
                schedule=json.dumps(
                    {**controls, **_cuda_library_identity(library)}, sort_keys=True
                ),
            )
            device = [
                _cuda_item_inventory(
                    library,
                    item,
                    diis_history=diis_history,
                    pbe=pbe,
                    tile=model.tile_points,
                )
                for item in items
            ]
            if model.xc_schedule == "host_unfused":
                for item in device:
                    item["xc"] = 0
            for key in ("state", "xc", "coulomb"):
                estimates.append(
                    ResourceEstimate(
                        f"all KS device {key}",
                        checked_bytes(sum(x[key] for x in device)),
                        f"device:{device_id}",
                        first_phase,
                        last_phase,
                        kind="persistent",
                    )
                )
            # At most one owner is rebuilt at a time. Its retired allocation
            # need not coexist with its setup temporary. Quadrature setup
            # already includes the new owner's live Coulomb allocation.
            extra = max(
                max(0, x["setup"] - sum(x[k] for k in ("state", "xc", "coulomb")))
                for x in device
            )
            estimates.append(
                ResourceEstimate(
                    "KS setup excess over retired owner",
                    extra,
                    f"device:{device_id}",
                    first_phase,
                    last_phase,
                )
            )
            # C2 executes one generated-force item at a time after the resident
            # SCF owner. The public consumer enforces these same staging caps.
            estimates.append(
                ResourceEstimate(
                    "serialized generated KS force device staging cap",
                    512 << 20,
                    f"device:{device_id}",
                    first_phase,
                    last_phase,
                )
            )
            estimates.append(
                ResourceEstimate(
                    "serialized generated KS force host staging cap",
                    256 << 20,
                    "pageable",
                    first_phase,
                    last_phase,
                )
            )
    except (NotImplementedError, RuntimeError, OSError) as error:
        return ResourceRequest(
            name, identity, (), exclusions, unsupported_reason=str(error)
        )
    candidate = ResourceCandidate(
        f"{backend}-ks-resident",
        "resident",
        tuple(estimates),
        decisions=(
            ("batch_execution", controls["schedule"]),
            ("cpu_worker_limit", "1"),
            ("item_host_inventory", json.dumps(host, sort_keys=True)),
            ("item_device_inventory", json.dumps(device, sort_keys=True)),
            (
                "preparation",
                "included in common observation and persistent device ledger",
            ),
        ),
    )
    return ResourceRequest(name, identity, (candidate,), exclusions)


def estimate_ks_resources(
    systems: typing.Any, *, budget: typing.Any = None, **options: typing.Any
) -> typing.Any:
    """Plan native KS energy capacity without creating scientific arrays."""
    return plan_resources(
        (ks_resource_request(systems, **options),),
        ResourceBudget() if budget is None else budget,
    )
