"""Matrix-free RHF/CPKS orbital-response operators."""

from __future__ import annotations

import time

import numpy as np
from vibeqc.profiles import canonical_hash

from .problem import (
    ResponseProblem,
    ResponseUnsupported,
    RotationLayout,
)


def rhf_operator_identity(backend):
    """Stable operator key for one RHF J/K backend."""
    return canonical_hash(
        {
            "method": "rhf",
            "backend": backend.identity,
            "exchange_fraction": 0.5,
            "parameterization": "density-response-symmetric-ov",
        }
    )


def cpks_operator_identity(backend, xc_kernel):
    """Stable operator key for one CPKS semilocal kernel."""
    return canonical_hash(
        {
            "method": "cpks",
            "backend": backend.identity,
            "xc_kernel": xc_kernel.identity,
            "exchange_fraction": 0.0,
            "parameterization": "density-response-symmetric-ov",
        }
    )


class _BaseResponseOperator:
    """Shared validation/statistics for density-response operators."""

    def __init__(self, problem: ResponseProblem, backend):
        if not isinstance(problem, ResponseProblem):
            raise TypeError("expected ResponseProblem")
        self.problem = problem
        self.backend = backend
        validate = getattr(backend, "validate_reference", None)
        if validate is not None:
            validate(problem.reference)
        self.nbf = problem.reference.nmo
        self.dimension = problem.dimension
        self.statistics = {
            "actions": 0,
            "transpose_actions": 0,
            "seconds": 0.0,
            "backend_seconds": 0.0,
            "peak_workspace_bytes": 0,
        }

    @property
    def identity(self):
        return self.problem.operator_identity

    def _record(self, started, backend_started):
        self.statistics["actions"] += 1
        self.statistics["seconds"] += time.perf_counter() - started
        self.statistics["backend_seconds"] += time.perf_counter() - backend_started
        workspace = (
            3 * self.nbf * self.nbf * 8
            + 2 * self.dimension * 8
            + getattr(self.backend, "statistics", {}).get("peak_bytes", 0)
        )
        self.statistics["peak_workspace_bytes"] = max(
            self.statistics["peak_workspace_bytes"], workspace
        )

    def _delta_density_ao(self, vector):
        x = self.problem.layout.validate_vector(vector)
        delta_mo = self.problem.layout.density_matrix(x)
        c = self.problem.reference.coefficients
        return x, c @ delta_mo @ c.T

    def _base_action(self, vector, *, transpose=False):
        started = time.perf_counter()
        x, delta_ao = self._delta_density_ao(vector)
        backend_started = time.perf_counter()
        coulomb, exchange = self.backend.coulomb_exchange(delta_ao)
        xc = self._xc_response(delta_ao, transpose=transpose)
        response_ao = coulomb - self.exchange_fraction * exchange + xc
        response_mo = (
            self.problem.reference.coefficients.T
            @ response_ao
            @ self.problem.reference.coefficients
        )
        nocc = self.problem.layout.nocc
        nvirt = self.problem.layout.nvirt
        occupied = self.problem.layout.occupied
        virtual = self.problem.layout.virtual
        x_ia = x.reshape(nocc, nvirt)
        eps = self.problem.reference.orbital_energies
        response = (
            eps[list(virtual)][None, :] - eps[list(occupied)][:, None]
        ) * x_ia + response_mo[np.ix_(virtual, occupied)].T
        self._record(started, backend_started)
        if transpose:
            self.statistics["transpose_actions"] += 1
        return self.problem.layout.validate_vector(response.reshape(-1))

    def _xc_response(self, delta_ao, *, transpose=False):
        """Semilocal XC response; RHF has none."""
        del delta_ao, transpose
        return 0.0

    def apply(self, vector):
        """Apply the Jacobian action to one response vector."""
        return self._base_action(vector)

    def apply_transpose(self, vector):
        """Apply the transpose action under the Euclidean response metric.

        The real closed-shell RHF/CPKS Jacobian is symmetric for the canonical
        nonredundant parameterization.  The explicit entry point keeps the
        contract testable and leaves room for future nonsymmetric backends.
        """
        return self._base_action(vector, transpose=True)

    def apply_many(self, matrix):
        """Apply the operator to every column while sharing operator state."""
        values = self.problem.validate_rhs(matrix)
        return np.column_stack(
            [self.apply(values[:, column]) for column in range(values.shape[1])]
        )

    def dot_identity(self, left, right):
        """Return the JVP/VJP dot-product identity error."""
        left = self.problem.layout.validate_vector(left)
        right = self.problem.layout.validate_vector(right)
        lhs = float(np.dot(left, self.apply(right)))
        rhs = float(np.dot(self.apply_transpose(left), right))
        scale = max(1.0, abs(lhs), abs(rhs))
        return abs(lhs - rhs) / scale

    def to_dense(self):
        """Materialize the operator for tiny explicit-oracle tests only."""
        if self.dimension > 4096:
            raise ValueError("dense response materialization is tiny-system only")
        result = np.empty((self.dimension, self.dimension))
        for column in range(self.dimension):
            basis = np.zeros(self.dimension)
            basis[column] = 1.0
            result[:, column] = self.apply(basis)
        return result


class RHFResponseOperator(_BaseResponseOperator):
    """Matrix-free closed-shell RHF orbital-response Jacobian.

    The action follows the canonical nonredundant occupied-virtual
    parameterization documented in :class:`RotationLayout`.  It is explicitly
    symmetric for real RHF and exposes both JVP and VJP entry points.
    """

    exchange_fraction = 0.5

    def __init__(self, problem, backend):
        super().__init__(problem, backend)
        if problem.method != "rhf":
            raise ResponseUnsupported("RHFResponseOperator requires an RHF problem")
        expected = rhf_operator_identity(backend)
        if problem.operator_identity != expected:
            raise ValueError("problem operator_identity does not match its RHF backend")

    @classmethod
    def build_problem(
        cls,
        reference,
        backend,
        *,
        rhs_layout="ov-response-vector",
        perturbation_labels=(),
    ):
        """Create the exact problem snapshot used by this operator."""
        return ResponseProblem.from_reference(
            reference,
            method="rhf",
            operator_identity=rhf_operator_identity(backend),
            rhs_layout=rhs_layout,
            perturbation_labels=perturbation_labels,
        )


class CPKSResponseOperator(_BaseResponseOperator):
    """Matrix-free closed-shell CPKS action with a semilocal XC kernel.

    The kernel must describe the same basis, grid, functional and reference
    density as the converged KS snapshot.  Hybrid/exact-exchange and
    unsupported meta-GGA modes fail closed rather than silently dropping a
    derivative contribution.
    """

    exchange_fraction = 0.0

    def __init__(self, problem, backend, xc_kernel):
        super().__init__(problem, backend)
        if problem.method != "cpks":
            raise ResponseUnsupported("CPKSResponseOperator requires a CPKS problem")
        if problem.reference.algorithm != "KS":
            raise ResponseUnsupported("CPKS requires a converged KS reference")
        validate = getattr(xc_kernel, "validate_reference", None)
        if validate is not None:
            validate(problem.reference)
        else:
            for name in ("grid_identity", "functional_identity"):
                if getattr(xc_kernel, name, None) != getattr(
                    problem.reference, name, None
                ):
                    raise ValueError(f"XC kernel/reference {name} mismatch")
            if (
                getattr(xc_kernel, "basis_identity", None)
                != problem.reference.basis_hash
            ):
                raise ValueError("XC kernel/reference basis_identity mismatch")
        self.xc_kernel = xc_kernel
        expected = cpks_operator_identity(backend, xc_kernel)
        if problem.operator_identity != expected:
            raise ValueError(
                "problem operator_identity does not match its CPKS backend/kernel"
            )

    @classmethod
    def build_problem(
        cls,
        reference,
        backend,
        xc_kernel,
        *,
        rhs_layout="ov-response-vector",
        perturbation_labels=(),
    ):
        """Create the exact CPKS problem snapshot used by this operator."""
        return ResponseProblem.from_reference(
            reference,
            method="cpks",
            operator_identity=cpks_operator_identity(backend, xc_kernel),
            rhs_layout=rhs_layout,
            perturbation_labels=perturbation_labels,
        )

    def _xc_response(self, delta_ao, *, transpose=False):
        if transpose:
            return self.xc_kernel.apply_transpose(delta_ao)
        return self.xc_kernel.apply(delta_ao)


class DenseMatrixResponseOperator:
    """Explicit matrix action for GPU-transformed control evidence only.

    The production shared path is :class:`RHFResponseOperator`, which never
    materializes this matrix.  This wrapper exists so a CUDA MO-block transform
    can be compared against the same solver/recycling code without pretending
    that a dense action is matrix-free.
    """

    def __init__(self, problem, matrix, *, backend_identity):
        if not isinstance(problem, ResponseProblem):
            raise TypeError("expected ResponseProblem")
        value = np.asarray(matrix, dtype=np.float64)
        if value.shape != (problem.dimension, problem.dimension):
            raise ValueError("dense response matrix has the wrong shape")
        if not np.isfinite(value).all():
            raise ValueError("dense response matrix must be finite")
        self.problem = problem
        self.matrix = value
        self.dimension = problem.dimension
        self.backend_identity = backend_identity
        self.statistics = {
            "actions": 0,
            "transpose_actions": 0,
            "seconds": 0.0,
            "peak_workspace_bytes": value.nbytes,
        }

    @property
    def identity(self):
        return canonical_hash(
            {
                "problem": self.problem.identity,
                "backend": self.backend_identity,
                "matrix": "dense-explicit-control",
            }
        )

    def apply(self, vector):
        started = time.perf_counter()
        value = self.matrix @ np.asarray(vector, dtype=np.float64)
        self.statistics["actions"] += 1
        self.statistics["seconds"] += time.perf_counter() - started
        return value

    def apply_transpose(self, vector):
        started = time.perf_counter()
        value = self.matrix.T @ np.asarray(vector, dtype=np.float64)
        self.statistics["transpose_actions"] += 1
        self.statistics["seconds"] += time.perf_counter() - started
        return value

    def dot_identity(self, left, right):
        lhs = float(np.dot(left, self.apply(right)))
        rhs = float(np.dot(self.apply_transpose(left), right))
        return abs(lhs - rhs) / max(1.0, abs(lhs), abs(rhs))


def validate_rotation_layout(problem):
    """Public checked accessor for callers implementing #153-style RHS code."""
    if not isinstance(problem.layout, RotationLayout):
        raise TypeError("problem layout is not a RotationLayout")
    return problem.layout
