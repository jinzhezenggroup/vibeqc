"""Explicit, checked export from VibeQC's own HF density into owned snapshots."""

from __future__ import annotations

import time
import uuid

import numpy as np

from .reference import ReferenceSnapshot


def conventional_fock(source, density, *, axis_tile=2):
    """Contract the physical unscreened F[D] in one streamed AO-tile pass."""
    _, h = source.one_electron()
    f = h.copy()
    for request in source.requests(
        "four_center_eri", axis_tile=axis_tile, budget_bytes=16 * axis_tile**4
    ):
        tile = source.tile(request)
        begin = source.global_offsets(request)
        u, v, w, x = (slice(b, b + n) for b, n in zip(begin, tile.shape))
        f[u, v] += np.einsum("uvwx,wx->uv", tile, density[w, x], optimize=False)
        f[u, w] -= 0.5 * np.einsum("uvwx,vx->uw", tile, density[v, x], optimize=False)
    return f


def export_rhf(
    source,
    *,
    backend="cpu",
    device_id=0,
    metric=None,
    generation_id=None,
    max_iterations=100,
    tolerance=1e-11,
    axis_tile=2,
):
    """Run native RHF and canonicalize its final physical Fock on the host.

    The existing HF implementation owns its original calculation memory. This
    explicit initial export is a small-system bridge (N<=12); bounded production
    providers accept already validated snapshots without repeating HF/export.
    CPU canonicalization and CUDA density transfers are disclosed in records.
    """
    if source.nbf > 12:
        raise ValueError("initial native HF snapshot exporter supports at most 12 AOs")
    start = time.perf_counter()
    density, hf = source.rhf_density(
        backend=backend,
        device_id=device_id,
        max_iterations=max_iterations,
        tolerance=tolerance,
        df=metric is not None,
        metric_threshold=metric.relative_threshold if metric else 1e-10,
    )
    s, h = source.one_electron()
    if metric is None:
        build = lambda d: conventional_fock(source, d, axis_tile=axis_tile)
        hamiltonian = "conventional-unscreened"
    else:
        # Small-only native-HF export oracle. Production DFProvider streams
        # raw A/B tiles and never calls this dense export routine.
        raw = np.empty((source.nbf, source.nbf, source.naux))
        for request in source.requests(
            "three_center_eri", axis_tile=axis_tile, budget_bytes=16 * axis_tile**3
        ):
            values = source.tile(request)
            begin = source.global_offsets(request)
            raw[tuple(slice(b, b + n) for b, n in zip(begin, values.shape))] = values
        b = (raw.reshape(-1, source.naux) @ metric.inverse_square_root).reshape(
            raw.shape
        )

        def build(d):
            j = np.einsum("uvP,wxP,wx->uv", b, b, d, optimize=True)
            k = np.einsum("uwP,vxP,wx->uv", b, b, d, optimize=True)
            return h + j - 0.5 * k

        hamiltonian = metric.hamiltonian_id
    f = build(density)
    eigen, u = np.linalg.eigh(s)
    if eigen[0] <= 1e-10:
        raise ValueError("linearly dependent AO basis cannot be exported")
    x = (u / np.sqrt(eigen)) @ u.T
    eps, v = np.linalg.eigh(x.T @ f @ x)
    c = x @ v
    for column in range(c.shape[1]):
        if c[np.argmax(np.abs(c[:, column])), column] < 0:
            c[:, column] *= -1
    occupation = np.zeros(source.nbf)
    occupation[: source.electron_count // 2] = 2
    canonical_density = (c * occupation) @ c.T
    physical_residual = float(np.max(np.abs(f @ density @ s - s @ density @ f)))
    density_drift = float(np.max(np.abs(canonical_density - density)))
    fock_drift = float(np.max(np.abs(build(canonical_density) - f)))
    snapshot = ReferenceSnapshot(
        s,
        h,
        f,
        c,
        eps,
        occupation,
        source.electron_count,
        hf["energy"],
        max(physical_residual, density_drift, fock_drift),
        source.geometry_hash,
        source.basis_hash,
        generation_id or str(uuid.uuid4()),
        hamiltonian_id=hamiltonian,
        representation=source.representation,
        hf_backend="native-" + backend,
        device_id=device_id if backend == "cuda" else None,
    )
    return snapshot, {
        **hf,
        "snapshot_id": snapshot.identity,
        "physical_residual": physical_residual,
        "canonical_density_drift": density_drift,
        "physical_fock_drift": fock_drift,
        "canonicalization_backend": "cpu-numpy",
        "density_host_staging": backend == "cuda",
        "export_seconds": time.perf_counter() - start,
        "snapshot_numeric_bytes": snapshot.numeric_bytes,
    }
