"""Explicit same-AO orbital transport/extrapolation for traditional baselines."""

import time

import numpy as np

from .proposals import project_occupied
from .state import metric_root, spin_counts, validate_density


def transported_density(previous, model, overlap, *, older=None):
    """Build a new, validated determinant from one/two prior geometry states.

    Compatible basis/charge/spin/Hamiltonian identities are required. Geometry
    is intentionally allowed to differ and every source identity is recorded.
    With two states, transport both occupied projectors, extrapolate 2P1-P0,
    then explicitly project onto the highest occupied metric eigenvectors.
    This projection is a timed repair, not an unreported ensemble assumption.
    """
    started = time.perf_counter()
    target = model.to_dict()
    for key in ("geometry_hash", "identity"):
        target.pop(key)
    sources = (previous,) if older is None else (previous, older)
    counts, weight = spin_counts(model)
    densities = []
    for state in sources:
        origin = state.model.to_dict()
        for key in ("geometry_hash", "identity"):
            origin.pop(key)
        if origin != target or state.overlap.shape != np.shape(overlap):
            raise ValueError(
                "orbital transport requires compatible basis/charge/spin/model"
            )
        root = metric_root(state.overlap)
        x = np.linalg.inv(root)
        blocks = []
        for density, electrons in zip(state.density, counts, strict=True):
            no = int(electrons / weight)
            occupations, natural = np.linalg.eigh(root @ density @ root)
            if 0 < no < len(root) and occupations[-no] - occupations[-no - 1] < 1e-8:
                raise ValueError("ambiguous transported occupied subspace")
            columns = x @ natural[:, -no:] if no else np.zeros((len(root), 0))
            projected, _ = project_occupied(columns, overlap)
            blocks.append(weight * projected @ projected.T)
        densities.append(np.asarray(blocks))
    result = densities[0]
    if older is not None:
        extrapolated = 2 * densities[0] - densities[1]
        root = metric_root(overlap)
        x = np.linalg.inv(root)
        blocks = []
        for density, electrons in zip(extrapolated, counts, strict=True):
            no = int(electrons / weight)
            occupations, natural = np.linalg.eigh(root @ density @ root)
            if 0 < no < len(root) and occupations[-no] - occupations[-no - 1] < 1e-8:
                raise ValueError("ambiguous extrapolated occupied subspace")
            c = x @ natural[:, -no:] if no else np.zeros((len(root), 0))
            blocks.append(weight * c @ c.T)
        result = np.asarray(blocks)
    result = validate_density(result, overlap, model, determinant=True)
    return result, {
        "repair": "same_ao_occupied_lowdin"
        if older is None
        else "projected_density_extrapolation",
        "repair_seconds": time.perf_counter() - started,
        "source_snapshot_ids": [s.identity for s in sources],
        "target_model_id": model.identity,
        "derivative_status": "unsupported",
    }
