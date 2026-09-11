"""Compact HF topology resolution and conservative CPU allocation inventory.

Only shell/primitive metadata is built here. In particular, a dry run never
constructs overlap, ERI, DF, density or derivative arrays. CUDA HF needs its
native arena/provider query; absence of that query is reported explicitly.
"""

from __future__ import annotations

import ctypes
import json
import math
import os
from dataclasses import asdict, replace
from pathlib import Path

from .basis import BasisSet
from .basis_capabilities import require_basis, resolved_basis_metadata
from .calculator import Atom, _snapshot_basis
from .elements import electron_state
from .resources import (
    ResourceBudget,
    ResourceCandidate,
    ResourceEstimate,
    ResourceIdentity,
    ResourceRequest,
    byte_product,
    checked_bytes,
    plan_resources,
)

# These runtime switches are read by scf/cuda/rhf_policy.cpp. Preserve even
# switches dormant for a small shape: a prepared resource identity must not
# silently accept a different arithmetic, profiling or scheduling policy.
_CUDA_SCHEDULE_VARIABLES = (
    "VIBEQC_FINAL_FOCK_REBUILD",
    "VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD",
    "VIBEQC_GRAPH_EIGENSOLVER_OVERRIDE",
    "VIBEQC_XSYEV_PROBE_SKIP_DIAGNOSTIC",
    "VIBEQC_BOUNDED_DIRECT_STREAMING",
    "VIBEQC_BOUNDED_DIRECT_COUNT_DIAGNOSTIC",
    "VIBEQC_BOUNDED_DIRECT_AOT_ONLY_DIAGNOSTIC",
    "VIBEQC_BOUNDED_DIRECT_FOCK_ONLY_DIAGNOSTIC",
    "VIBEQC_BOUNDED_DIRECT_FOCK_CLASS_PROFILE",
    "VIBEQC_DIRECT_TILE_VALIDATION",
    "VIBEQC_FORCE_DENSITY_PRODUCT_SCREENING",
    "VIBEQC_PPPS_RESIDENT_BRA",
    "VIBEQC_PPPS_SIGNATURE_BUCKETING",
    "VIBEQC_PSPS_SIGNATURE_BUCKETING",
    "VIBEQC_PPSS_SIGNATURE_BUCKETING",
    "VIBEQC_PPPS_BLOCK_THREADS",
    "VIBEQC_ONE_ELECTRON_FORCE_SCALAR",
    "VIBEQC_ONE_ELECTRON_VALUE_MAPPING",
    "VIBEQC_ONE_ELECTRON_DERIVATIVES",
    "VIBEQC_ONE_ELECTRON_DERIVATIVE_MAPPING",
    "VIBEQC_PSSS_RESIDENT_BRA",
    "VIBEQC_PSSS_WEIGHTED",
    "VIBEQC_DF_VALUE_MAPPING",
    "VIBEQC_DF_DERIVATIVE_MAPPING",
)


def _cuda_library_identity(library):
    """Hash an explicitly loaded artifact once without selecting/probing a GPU."""
    from .profiles import file_hash

    digest = getattr(library, "_vibeqc_resource_binary_sha256", None)
    path = getattr(library, "_name", None)
    if digest is None and path is not None:
        digest = file_hash(Path(path))
        library._vibeqc_resource_binary_sha256 = digest
    profile = getattr(library, "_vibeqc_profile_diagnostics", {})
    return {"native_binary_sha256": digest, "profile_identity": profile.get("identity")}


def _basis_record(atoms, basis, representation, backend, charge, multiplicity, role):
    selected = _snapshot_basis(basis, representation)
    mode = (
        selected.representation
        if isinstance(selected, BasisSet)
        else representation or "cartesian"
    )
    require_basis(
        selected,
        atoms,
        backend=backend,
        operator="eri",
        derivative_order=1,
        representation=mode,
        role=role,
    )
    shells = selected.shells_for(atoms) if isinstance(selected, BasisSet) else selected
    metadata = resolved_basis_metadata(
        selected,
        shells,
        atoms,
        representation=mode,
        charge=charge,
        multiplicity=multiplicity,
        role=role,
    )
    cartesian = sum(
        (s.angular_momentum + 1) * (s.angular_momentum + 2) // 2 for s in shells
    )
    nbf = (
        cartesian
        if mode == "cartesian"
        else sum(2 * s.angular_momentum + 1 for s in shells)
    )
    return {
        "nbf": checked_bytes(nbf, "AO dimension"),
        "cartesian_nbf": checked_bytes(cartesian),
        "shells": len(shells),
        "primitives": sum(len(s.primitives) for s in shells),
        "maximum_angular": max(s.angular_momentum for s in shells),
        "representation": mode,
        "basis_hash": metadata["mathematical_identity"],
    }


def _cpu_item_inventory(item, diis_history):
    """Bound the existing resident CPU route by its preparation/iteration phases.

    The CPU DF implementation currently retains the ordinary four-center
    integral record while preparing RI tensors. That real allocation belongs
    here even though DF contractions do not use its ERI values. Forward-mode
    Jet arrays coexist with unpacked tensors during preparation; spherical
    conversion can retain Cartesian and public tensors simultaneously.

    Bounds include requested numeric storage and conservative LP64 C++ value
    metadata. Malloc bookkeeping, allocator arenas and language/runtime objects
    remain explicit exclusions, covered by user reserve/headroom when needed.
    """
    orbital, auxiliary = item["orbital"], item.get("auxiliary")
    n, c, d = orbital["nbf"], orbital["cartesian_nbf"], 3 * item["atoms"]
    spin = item["spins"]
    n2, c2 = byte_product(n, n), byte_product(c, c)
    conventional = byte_product(8, d + 1, 2 * n2 + byte_product(n2, n2)) + 8 * d
    cartesian = byte_product(8, d + 1, 2 * c2 + byte_product(c2, c2)) + 8 * d
    # Jet: scalar, derivative vector and vector value metadata. The 64-byte
    # per-value allowance exceeds the current LP64 Jet/vector object sizes.
    jet_bytes = 8 * (d + 1) + 64
    jet_arrays = byte_product(jet_bytes, 2 * c2 + byte_product(c2, c2))
    l = orbital["maximum_angular"]
    a_l = auxiliary["maximum_angular"] if auxiliary else 0
    coulomb_order = max(4 * l, 2 * l + a_l, 2 * a_l)
    hermite = 6 * (max(l, a_l) + 3) ** 2 * (2 * max(l, a_l) + 4)
    # fill_coulomb allocates its full four-dimensional recurrence box. Only
    # one primitive quartet executes at a time; contraction length does not
    # multiply this workspace by the primitive-quartet count.
    recurrence = byte_product(jet_bytes, (coulomb_order + 1) ** 4 + hermite + 256 + d)
    preparation = jet_arrays + max(
        recurrence, cartesian + conventional + 8 * (2 * n2 + n2 * n2)
    )
    resident = conventional
    matrix_work = byte_product(8, spin, n2, 64 + 2 * (diis_history + 1))
    matrix_work += byte_product(16, diis_history + 1, diis_history + 1)
    iteration = resident + matrix_work
    finalization = iteration
    if auxiliary:
        a, ac = auxiliary["nbf"], auxiliary["cartesian_nbf"]
        raw_count = byte_product(a, a) + byte_product(n2, a)
        cart_raw_count = byte_product(ac, ac) + byte_product(c2, ac)
        raw = byte_product(8, d + 1, raw_count)
        raw_cart = byte_product(8, d + 1, cart_raw_count)
        df_jets = byte_product(jet_bytes, cart_raw_count)
        transformed = byte_product(8, n2, a)
        preparation = max(
            preparation,
            conventional + df_jets + max(recurrence, raw_cart + raw + 8 * raw_count),
            conventional + raw + transformed + 8 * 16 * a * a,
        )
        resident += raw + transformed
        iteration = resident + matrix_work + 8 * (a + 4 * n2)
        # Metric pseudoinverse differentiation plus two full response tensors
        # coexist with SCF state during analytic DF force assembly.
        finalization = resident + matrix_work + 8 * (32 * a * a + 2 * n2 * a + 8 * d)
    return {
        name: checked_bytes(value, f"CPU {name} capacity")
        for name, value in {
            "preparation": preparation,
            "iteration": iteration,
            "finalization": finalization,
            "integral_state": resident,
        }.items()
    }


def _small_cuda_item_inventory(library, item, diis_history):
    """Query the existing small native route without constructing a CUDA plan.

    Its exact numeric arena needs no external eigensolver workspace. Packed
    host metadata and staging use a conservative LP64 capacity allowance; the
    opaque plan's real C++ object size comes from the same native contract.
    """
    query = getattr(library, "vibeqc_resource_small_hf_cuda_v1", None)
    if query is None:
        raise NotImplementedError(
            "native library has no CUDA HF allocation inventory v1"
        )
    orbital = item["orbital"]
    args = (
        orbital["nbf"],
        orbital["cartesian_nbf"],
        item["atoms"],
        orbital["shells"],
        orbital["primitives"],
        diis_history,
        item["spins"],
    )
    if any(value > 2 ** (8 * ctypes.sizeof(ctypes.c_size_t)) - 1 for value in args):
        raise ValueError("CUDA resource shape exceeds this host's size_t ABI")
    query.argtypes = [ctypes.c_size_t] * 7 + [ctypes.POINTER(ctypes.c_uint64)]
    query.restype = ctypes.c_int
    output = (ctypes.c_uint64 * 2)()
    if query(*args, output):
        raise NotImplementedError(
            "CUDA HF inventory v1 requires a CUDA library, <=16 public AOs and DIIS history <=64"
        )
    pairs = orbital["shells"] * (orbital["shells"] + 1) // 2
    # Four copies allow vector growth, HostBatch staging, the cached topology
    # and the temporary cold retry retained alongside a warm bucket. At most
    # 16 shells reach this route; PSSS metadata is bounded by pair^2 records.
    metadata = 2048 * (
        1
        + item["atoms"]
        + orbital["shells"]
        + orbital["primitives"]
        + orbital["nbf"]
        + orbital["cartesian_nbf"]
        + pairs
        + pairs * pairs
    )
    host = (
        4 * int(output[1])
        + metadata
        + 8 * item["spins"] * orbital["nbf"] ** 2 * 64
        + 8 * 3 * item["atoms"] * 16
    )
    return {"arena": checked_bytes(int(output[0])), "host": checked_bytes(host)}


def hf_resource_request(
    systems,
    *,
    method="rhf",
    basis="sto-3g",
    backend="cpu",
    basis_representation=None,
    charges=None,
    multiplicities=None,
    density_fitting="none",
    auxiliary_basis=None,
    diis_history=8,
    max_iterations=100,
    energy_tolerance=1e-10,
    density_tolerance=1e-8,
    screening_tolerance=1e-12,
    precision="fp64",
    density_fitting_relative_threshold=1e-10,
    density_fitting_memory_budget_bytes=0,
    name="hf",
    first_phase=0,
    last_phase=0,
    device_id=0,
    library=None,
):
    """Resolve real basis/spin inputs into one serialized-fleet provider request.

    Workspace scales with the largest serialized item. Retained item state and
    output storage scale with the entire batch. A geometry-only change reuses
    this topology identity; geometry-derived caches and warm densities still
    obey their normal scientific invalidation rules at execution.
    """
    systems = tuple(tuple(Atom.from_value(a) for a in atoms) for atoms in systems)
    if not systems or any(not atoms for atoms in systems):
        raise ValueError("HF resource planning requires nonempty systems")
    charges = (0,) * len(systems) if charges is None else tuple(charges)
    multiplicities = (
        (1,) * len(systems) if multiplicities is None else tuple(multiplicities)
    )
    if len(charges) != len(systems) or len(multiplicities) != len(systems):
        raise ValueError("charges and multiplicities must match the fleet size")
    if method not in ("rhf", "uhf") or backend not in ("cpu", "cuda"):
        raise NotImplementedError(
            "HF resource planning requires an explicit RHF/UHF CPU/CUDA provider"
        )
    if density_fitting not in ("none", "cpu", "cpu_reference", "cuda", "auto"):
        raise ValueError("unknown density-fitting provider")
    if auxiliary_basis is not None and density_fitting == "none":
        raise ValueError("auxiliary basis requires density fitting")
    checked_bytes(diis_history, "DIIS history")
    checked_bytes(density_fitting_memory_budget_bytes, "DF memory sub-budget")
    checked_bytes(device_id, "device ordinal")
    checked_bytes(max_iterations, "maximum iterations")
    if not max_iterations:
        raise ValueError("maximum iterations must be positive")
    for value in (
        energy_tolerance,
        density_tolerance,
        screening_tolerance,
        density_fitting_relative_threshold,
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("HF numerical controls must be positive finite")
    if density_fitting_relative_threshold >= 1:
        raise ValueError("DF metric threshold must be below one")
    if precision not in ("fp64", "auto"):
        raise ValueError("precision must be 'fp64' or 'auto'")
    fitted = density_fitting != "none"
    executed_backend = "cpu" if density_fitting in ("cpu", "cpu_reference") else backend
    if density_fitting == "cuda" and backend != "cuda":
        raise ValueError("CUDA DF requires a CUDA backend")
    items = []
    for atoms, charge, multiplicity in zip(
        systems, charges, multiplicities, strict=True
    ):
        electrons = electron_state(atoms, charge=charge, multiplicity=multiplicity)
        if method == "rhf" and multiplicity != 1:
            raise ValueError("RHF requires a closed-shell singlet")
        orbital = _basis_record(
            atoms,
            basis,
            basis_representation,
            executed_backend,
            charge,
            multiplicity,
            "orbital",
        )
        if max(electrons.nalpha, electrons.nbeta) > orbital["nbf"]:
            raise ValueError(
                "basis has fewer orbitals than the requested spin population"
            )
        auxiliary = None
        if fitted:
            auxiliary = (
                orbital
                if auxiliary_basis is None
                else _basis_record(
                    atoms,
                    auxiliary_basis,
                    None
                    if isinstance(auxiliary_basis, (BasisSet, os.PathLike))
                    or (
                        isinstance(auxiliary_basis, str)
                        and auxiliary_basis.endswith(".json")
                    )
                    else basis_representation,
                    executed_backend,
                    charge,
                    multiplicity,
                    "auxiliary",
                )
            )
        items.append(
            {
                "atoms": len(atoms),
                "orbital": orbital,
                "auxiliary": auxiliary,
                "electrons": asdict(electrons),
                "spins": 1 if method == "rhf" else 2,
            }
        )
    controls = {
        "diis_history": diis_history,
        "max_iterations": max_iterations,
        "energy_tolerance": energy_tolerance,
        "density_tolerance": density_tolerance,
        "screening_tolerance": screening_tolerance,
        "precision": precision,
        "density_fitting_relative_threshold": density_fitting_relative_threshold,
        "density_fitting": density_fitting,
        "density_fitting_memory_budget_bytes": density_fitting_memory_budget_bytes,
    }
    if executed_backend == "cuda":
        controls.update(
            device_id=device_id,
            runtime_policy={
                key: os.environ.get(key) for key in _CUDA_SCHEDULE_VARIABLES
            },
        )
    identity = ResourceIdentity(
        method,
        "native-hf-df-v1" if fitted else "native-hf-direct-v1",
        executed_backend,
        precision,
        json.dumps({"items": items}),
        ("energy", "forces"),
        json.dumps(controls, sort_keys=True),
    )
    exclusions = (
        "caller-owned inputs and previously retained output objects",
        "Python/runtime objects, allocator bookkeeping, arenas and page rounding",
    )
    if (
        auxiliary_basis is not None
        and len({item["auxiliary"]["basis_hash"] for item in items}) > 1
    ):
        return ResourceRequest(
            name,
            identity,
            (),
            exclusions,
            unsupported_reason="native prepared fleets require one compatible explicit auxiliary template",
        )
    if executed_backend != "cpu":
        cuda_exclusions = exclusions + (
            "CUDA context, modules, graphs, stacks and allocator pool/page retention",
        )
        try:
            if os.environ.get("VIBEQC_BOUNDED_DIRECT_FOCK_CLASS_PROFILE") in (
                "1",
                "profile",
            ):
                raise NotImplementedError(
                    "CUDA resource plans do not include optional class profiling"
                )
            if os.environ.get("VIBEQC_GRAPH_EIGENSOLVER_OVERRIDE") == "graph_native":
                raise NotImplementedError(
                    "CUDA inventory v1 requires the default small native eigensolver"
                )
            if library is None:
                from . import _native

                # CPU loading deliberately avoids select_library's GPU probe
                # and profile selection. The explicit library path can still
                # identify a CUDA-linked library for scalar layout queries.
                library = _native.load_library(device="cpu")
            identity = replace(
                identity,
                schedule=json.dumps(
                    {**controls, **_cuda_library_identity(library)}, sort_keys=True
                ),
            )
            inventories = tuple(
                _small_cuda_item_inventory(library, item, diis_history)
                for item in items
            )
            if fitted:
                from .resources_hf_df import cuda_df_candidates

                candidates = cuda_df_candidates(
                    library,
                    items,
                    diis_history=diis_history,
                    device=device_id,
                    first_phase=first_phase,
                    last_phase=last_phase,
                    requested_budget=density_fitting_memory_budget_bytes,
                )
                return ResourceRequest(
                    name,
                    identity,
                    candidates,
                    cuda_exclusions
                    + (
                        "opaque CUDA library retention above the reported allowance is untracked",
                    ),
                )
        except ValueError as error:
            if "budget" not in str(error):
                raise
            return ResourceRequest(
                name, identity, (), cuda_exclusions, infeasible_reason=str(error)
            )
        except (NotImplementedError, RuntimeError, OSError) as error:
            return ResourceRequest(
                name, identity, (), cuda_exclusions, unsupported_reason=str(error)
            )
        resident = checked_bytes(sum(row["arena"] for row in inventories))
        retry = max(row["arena"] for row in inventories)
        host = checked_bytes(sum(row["host"] for row in inventories))
        candidate = ResourceCandidate(
            "cuda-small-resident",
            "resident",
            (
                ResourceEstimate(
                    "all retained direct-HF bucket arenas",
                    resident,
                    f"device:{device_id}",
                    first_phase,
                    last_phase,
                    kind="persistent",
                ),
                ResourceEstimate(
                    "one cold numerical retry alongside warm bucket",
                    retry,
                    f"device:{device_id}",
                    first_phase,
                    last_phase,
                ),
                ResourceEstimate(
                    "fleet host state, packed topology and staging capacity",
                    host,
                    "pageable",
                    first_phase,
                    last_phase,
                    kind="persistent",
                ),
            ),
            decisions=(
                ("batch_execution", "serialized buckets; every retained arena summed"),
                ("eigensolver", "small native; no external library workspace"),
                ("item_inventory", json.dumps(inventories, sort_keys=True)),
            ),
        )
        return ResourceRequest(name, identity, (candidate,), cuda_exclusions)
    inventories = tuple(_cpu_item_inventory(item, diis_history) for item in items)
    retained = sum(
        8 * (4 * item["spins"] * item["orbital"]["nbf"] ** 2 + 8 * 3 * item["atoms"])
        + 128
        * (item["orbital"]["primitives"] + item["orbital"]["shells"] + item["atoms"])
        + (
            128 * (item["auxiliary"]["primitives"] + item["auxiliary"]["shells"])
            if item["auxiliary"]
            else 0
        )
        for item in items
    )
    peak_work = max(
        max(row["preparation"], row["iteration"], row["finalization"])
        for row in inventories
    )
    candidate = ResourceCandidate(
        "cpu-resident",
        "resident",
        (
            ResourceEstimate(
                "fleet topology, warm state and outputs",
                checked_bytes(retained),
                "pageable",
                first_phase,
                last_phase,
                kind="persistent",
            ),
            ResourceEstimate(
                "largest serialized item workspace",
                peak_work,
                "pageable",
                first_phase,
                last_phase,
            ),
        ),
        decisions=(
            ("batch_execution", "serialized items; retained state summed across fleet"),
            ("cpu_worker_limit", "1"),
            ("item_phase_inventory", json.dumps(inventories, sort_keys=True)),
        ),
    )
    return ResourceRequest(name, identity, (candidate,), exclusions)


def estimate_hf_resources(systems, *, budget=None, **options):
    """Return an HF dry-run plan using compact topology and no native execution."""
    budget = ResourceBudget() if budget is None else budget
    return plan_resources((hf_resource_request(systems, **options),), budget)
