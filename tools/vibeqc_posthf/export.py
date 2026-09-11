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


def _canonical_orbitals(overlap, fock):
    """Canonicalize one symmetric AO Fock matrix in the shared AO metric."""
    eigenvalues, eigenvectors = np.linalg.eigh(overlap)
    if eigenvalues[0] <= 1e-10:
        raise ValueError("linearly dependent AO basis cannot be exported")
    orthogonalizer = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    energies, rotated = np.linalg.eigh(orthogonalizer.T @ fock @ orthogonalizer)
    coefficients = orthogonalizer @ rotated
    for column in range(coefficients.shape[1]):
        if coefficients[np.argmax(np.abs(coefficients[:, column])), column] < 0:
            coefficients[:, column] *= -1
    return energies, coefficients


def conventional_uhf_fock(source, alpha_density, beta_density, *, axis_tile=2):
    """Contract the direct UHF Focks with total-density J and spin-local K."""
    _, hcore = source.one_electron()
    alpha = hcore.copy()
    beta = hcore.copy()
    total = alpha_density + beta_density
    for request in source.requests(
        "four_center_eri", axis_tile=axis_tile, budget_bytes=16 * axis_tile**4
    ):
        tile = source.tile(request)
        begin = source.global_offsets(request)
        u, v, w, x = (
            slice(offset, offset + size) for offset, size in zip(begin, tile.shape)
        )
        alpha[u, v] += np.einsum("uvwx,wx->uv", tile, total[w, x], optimize=False)
        beta[u, v] += np.einsum("uvwx,wx->uv", tile, total[w, x], optimize=False)
        alpha[u, w] -= np.einsum(
            "uvwx,vx->uw", tile, alpha_density[v, x], optimize=False
        )
        beta[u, w] -= np.einsum("uvwx,vx->uw", tile, beta_density[v, x], optimize=False)
    return alpha, beta


def export_uhf(
    source,
    *,
    generation_id=None,
    max_iterations=100,
    tolerance=1e-11,
    axis_tile=2,
):
    """Export a converged CPU direct-UHF solution as a response snapshot.

    The bridge deliberately restricts this first UHF endpoint to the direct
    CPU Hamiltonian. The existing CUDA DF response plan is RHF-only and must
    not be presented as spin-resolved UHF response evidence.
    """
    from tools.vibeqc_response.uhf import UHFReferenceSnapshot

    if source.nbf > 12:
        raise ValueError("initial native UHF snapshot exporter supports at most 12 AOs")
    started = time.perf_counter()
    densities, hf = source.uhf_density(
        backend="cpu", max_iterations=max_iterations, tolerance=tolerance
    )
    alpha_density, beta_density = densities
    overlap, hcore = source.one_electron()
    alpha_fock, beta_fock = conventional_uhf_fock(
        source, alpha_density, beta_density, axis_tile=axis_tile
    )
    alpha_energies, alpha_coefficients = _canonical_orbitals(overlap, alpha_fock)
    beta_energies, beta_coefficients = _canonical_orbitals(overlap, beta_fock)
    alpha_occupied = (source.electron_count + source.multiplicity - 1) // 2
    beta_occupied = source.electron_count - alpha_occupied
    alpha_occupations = np.zeros(source.nbf)
    beta_occupations = np.zeros(source.nbf)
    alpha_occupations[:alpha_occupied] = 1.0
    beta_occupations[:beta_occupied] = 1.0
    canonical_alpha = (alpha_coefficients * alpha_occupations) @ alpha_coefficients.T
    canonical_beta = (beta_coefficients * beta_occupations) @ beta_coefficients.T
    physical_residual = max(
        float(
            np.max(
                np.abs(
                    alpha_fock @ alpha_density @ overlap
                    - overlap @ alpha_density @ alpha_fock
                )
            )
        ),
        float(
            np.max(
                np.abs(
                    beta_fock @ beta_density @ overlap
                    - overlap @ beta_density @ beta_fock
                )
            )
        ),
    )
    density_drift = max(
        float(np.max(np.abs(canonical_alpha - alpha_density))),
        float(np.max(np.abs(canonical_beta - beta_density))),
    )
    rebuilt_alpha, rebuilt_beta = conventional_uhf_fock(
        source, canonical_alpha, canonical_beta, axis_tile=axis_tile
    )
    fock_drift = max(
        float(np.max(np.abs(rebuilt_alpha - alpha_fock))),
        float(np.max(np.abs(rebuilt_beta - beta_fock))),
    )
    snapshot = UHFReferenceSnapshot(
        overlap=overlap,
        hcore=hcore,
        fock_alpha=alpha_fock,
        fock_beta=beta_fock,
        coefficients_alpha=alpha_coefficients,
        coefficients_beta=beta_coefficients,
        orbital_energies_alpha=alpha_energies,
        orbital_energies_beta=beta_energies,
        occupations_alpha=alpha_occupations,
        occupations_beta=beta_occupations,
        reference_energy=hf["energy"],
        scf_residual=max(physical_residual, density_drift, fock_drift),
        geometry_hash=source.geometry_hash,
        basis_hash=source.basis_hash,
        generation_id=generation_id or str(uuid.uuid4()),
        representation=source.representation,
        hf_backend="native-cpu",
    )
    return snapshot, {
        **hf,
        "snapshot_id": snapshot.identity,
        "physical_residual": physical_residual,
        "canonical_density_drift": density_drift,
        "physical_fock_drift": fock_drift,
        "canonicalization_backend": "cpu-numpy",
        "density_host_staging": False,
        "export_seconds": time.perf_counter() - started,
    }
