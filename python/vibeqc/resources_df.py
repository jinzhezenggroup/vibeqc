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
    library, batch, nbf, naux, occupied, *, budget_bytes, fixed_device_bytes
):
    """Compose a fixed reservation with the provider's own tiling decisions.

    A zero native budget means implementation defaults. The global planner
    should supply a positive sub-budget when constraining a calculation. Every
    shape/product is checked by the native implementation before allocation.
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
    if not all((batch, nbf, naux, occupied)):
        raise ValueError("DF planner dimensions must be positive")
    query = getattr(library, "vibeqc_resource_df_tiles_v1", None)
    if query is None:
        raise NotImplementedError("native library has no shape-only DF resource query")
    query.argtypes = [ctypes.c_size_t] * 6 + [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    query.restype = ctypes.c_int
    values = (ctypes.c_uint64 * 6)()
    error = ctypes.create_string_buffer(2048)
    if query(
        batch,
        nbf,
        naux,
        occupied,
        budget_bytes,
        fixed_device_bytes,
        values,
        len(values),
        error,
        len(error),
    ):
        raise ValueError(error.value.decode())
    return DensityFittingResourceTile(
        *values[:5], bool(values[5]), budget_bytes, fixed_device_bytes
    )
