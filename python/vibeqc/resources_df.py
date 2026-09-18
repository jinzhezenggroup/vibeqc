"""Shape-only access to the existing native density-fitting tile planner.

This subplan is not a complete HF memory estimate. Its caller supplies fixed
source/neighbor reservations and accounts host state, provider/runtime overhead
and cross-bucket lifetimes through the common ResourcePlan.
"""

import ctypes
from dataclasses import dataclass

from .resources import checked_bytes


@dataclass(frozen=True)
class DensityFittingResourceTile:
    """The native planner's actual executable dimensions and workspace bound."""

    batch_tile: int
    ao_pair_tile: int
    auxiliary_tile: int
    occupied_tile: int
    peak_workspace_bytes: int
    stores_full_three_center: bool
    budget_bytes: int
    fixed_device_bytes: int
    generated_source: bool = False
    # Dense v1/v2 queries retain their original fields. Explicit packed queries
    # additionally expose the same capacities used by the native allocator.
    pair_storage: str = "dense"
    rank_capacity: int = 0
    stored_factor_bytes: int = 0
    raw_factor_bytes: int = 0
    contraction_scratch_bytes: int = 0
    projection_capacity_elements: int = 0
    panel_capacity_elements: int = 0
    automatic_rhf_rank: int = 0


def density_fitting_source_bytes(
    library, *, batch, atoms, shells, cartesian_aos, primitives, transforms
):
    """Query the native source's upload capacity from compact combined counts.

    Include one dummy shell/AO/primitive per item, and both orbital and
    auxiliary public-to-Cartesian transform matrices. No CUDA runtime call or
    full transform/three-center tensor allocation occurs.
    """
    values = (batch, atoms, shells, cartesian_aos, primitives, transforms)
    for value in values:
        checked_bytes(value, "DF source dimension")
        if value > 2 ** (8 * ctypes.sizeof(ctypes.c_size_t)) - 1:
            raise ValueError("DF source dimension exceeds size_t")
    query = getattr(library, "vibeqc_resource_df_source_bytes_v1", None)
    if query is None:
        raise NotImplementedError("native library has no DF source capacity query")
    query.argtypes = [ctypes.c_size_t] * 6 + [ctypes.POINTER(ctypes.c_uint64)]
    query.restype = ctypes.c_int
    output = ctypes.c_uint64()
    if query(*values, ctypes.byref(output)):
        raise ValueError("DF source metadata capacity overflows")
    return checked_bytes(output.value, "DF source capacity")


def density_fitting_tile_plan(
    library,
    batch,
    nbf,
    naux,
    occupied,
    *,
    budget_bytes,
    fixed_device_bytes,
    generated_source=False,
    pair_storage="dense",
    rhf_occupied=None,
):
    """Compose a fixed reservation with the provider's own tiling decisions.

    A zero native budget means implementation defaults. The global planner
    should supply a positive sub-budget when constraining a calculation. Every
    shape/product is checked by the native implementation before allocation.
    Packed physical sources reserve complete U only up to ``occupied``; zero
    is an explicit bounded-only reservation. Arbitrary tensor queries stay dense.
    ``rhf_occupied`` authorizes automatic SCF storage for a known RHF rank; None
    means unknown reference or UHF. If optional factors would force streaming,
    auto keeps the dense plan instead. Explicit occupied selection is unchanged.
    """
    for name, value in (
        ("batch", batch),
        ("nbf", nbf),
        ("naux", naux),
        ("occupied", occupied),
        ("DF sub-budget", budget_bytes),
        ("fixed device bytes", fixed_device_bytes),
    ):
        checked_bytes(value, name)
        if value > 2 ** (8 * ctypes.sizeof(ctypes.c_size_t)) - 1:
            raise ValueError(f"{name} exceeds this host's size_t ABI")
    if pair_storage not in ("dense", "packed"):
        raise ValueError("DF pair storage must be dense or packed")
    packed = pair_storage == "packed"
    if not all((batch, nbf, naux)) or (not occupied and not packed):
        raise ValueError("DF planner dimensions must be positive")
    if type(generated_source) is not bool:
        raise TypeError("generated_source must be boolean")
    if packed and not generated_source:
        raise ValueError("packed DF storage requires a physical generated source")
    method_aware = rhf_occupied is not None
    if method_aware:
        checked_bytes(rhf_occupied, "RHF occupation")
        if rhf_occupied > 2 ** (8 * ctypes.sizeof(ctypes.c_size_t)) - 1:
            raise ValueError("RHF occupation exceeds size_t")
    # Versioned queries preserve the old shape-only ABI while sharing the
    # method-authorized reservation used by native HF plan construction.
    name = (
        "vibeqc_resource_df_packed_tiles_v2"
        if packed and method_aware
        else "vibeqc_resource_df_packed_tiles_v1"
        if packed
        else "vibeqc_resource_df_tiles_v3"
        if method_aware
        else "vibeqc_resource_df_tiles_v2"
        if generated_source
        else "vibeqc_resource_df_tiles_v1"
    )
    query = getattr(library, name, None)
    if query is None:
        raise NotImplementedError("native library has no matching DF resource query")
    extra_types, extra_values = [], []
    if not packed and (generated_source or method_aware):
        extra_types.append(ctypes.c_uint)
        extra_values.append(int(generated_source))
    if method_aware:
        extra_types.append(ctypes.c_size_t)
        extra_values.append(rhf_occupied)
    query.argtypes = (
        [ctypes.c_size_t] * 6
        + extra_types
        + [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
    )
    query.restype = ctypes.c_int
    values = (ctypes.c_uint64 * ((10 if packed else 6) + int(method_aware)))()
    error = ctypes.create_string_buffer(2048)
    if query(
        batch,
        nbf,
        naux,
        occupied,
        budget_bytes,
        fixed_device_bytes,
        *extra_values,
        values,
        len(values),
        error,
        len(error),
    ):
        raise ValueError(error.value.decode())
    return DensityFittingResourceTile(
        *values[:5],
        bool(values[5]),
        budget_bytes,
        fixed_device_bytes,
        generated_source,
        pair_storage=pair_storage,
        rank_capacity=occupied if packed else 0,
        stored_factor_bytes=values[6] if packed else 0,
        raw_factor_bytes=values[6] if packed else 0,
        contraction_scratch_bytes=values[7] if packed else 0,
        projection_capacity_elements=values[8] if packed else 0,
        panel_capacity_elements=values[9] if packed else 0,
        automatic_rhf_rank=values[-1] if method_aware else 0,
    )


def density_fitting_diis_bytes(library, batch, nbf, history):
    """Query the native RHF/UHF history reservation without a CUDA context."""
    for value in (batch, nbf, history):
        checked_bytes(value, "DF DIIS shape")
    if (
        not batch
        or not nbf
        or max(batch, nbf) > 2 ** (8 * ctypes.sizeof(ctypes.c_size_t)) - 1
    ):
        raise ValueError("DF DIIS dimensions exceed the positive size_t domain")
    if history > 2 ** (8 * ctypes.sizeof(ctypes.c_uint)) - 1:
        raise ValueError("DF DIIS history exceeds the native unsigned domain")
    query = getattr(library, "vibeqc_resource_df_diis_bytes_v1", None)
    if query is None:
        raise NotImplementedError("native library has no DF DIIS capacity query")
    query.argtypes = [
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    query.restype = ctypes.c_int
    output = ctypes.c_uint64()
    if query(batch, nbf, history, ctypes.byref(output)):
        raise ValueError("DF DIIS capacity overflows")
    return checked_bytes(output.value, "DF DIIS capacity")
