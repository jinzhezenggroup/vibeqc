"""CPU-only adapter contracts; fake plans never count as GPU numerical evidence."""

import typing
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from tools.vibeqc_posthf.fixtures import (
    fixture_snapshot,
    load_fixture,
    source_arguments,
)
from tools.vibeqc_posthf.reference import immutable
from tools.vibeqc_posthf.sources import NativeSource
from tools.vibeqc_response import CudaDirectJKBackend, direct_cuda


@pytest.fixture
def source() -> typing.Any:
    meta, _ = load_fixture("h2")
    args = source_arguments(meta)
    args.pop("auxiliary_basis")
    with NativeSource(**args) as value:
        yield value


@pytest.fixture
def fake_provider(monkeypatch: typing.Any, source: typing.Any) -> typing.Any:
    plans = []

    class AO:
        def __init__(self, atoms: typing.Any, **kwargs: typing.Any) -> None:
            assert atoms == source.atoms
            assert kwargs["basis"] == source.shells
            assert kwargs["representation"] == source.representation
            self.nao = source.nbf

        def __enter__(self) -> typing.Any:
            return self

        def __exit__(self, *_: object) -> None:
            pass

    class Plan:
        identity = "fake-mathematical-plan"
        execution_identity = "fake-device-plan"

        def __init__(
            self, basis: typing.Any, spec: typing.Any, **kwargs: typing.Any
        ) -> None:
            assert kwargs["device"] == "cuda"
            assert kwargs["screening_tolerance"] == 0
            assert spec.derivative_order == 0
            self.diagnostics = {
                "backend": "cuda",
                "source_schedule": "cuda_independent",
                "precision": "float64",
                "resolved": spec.to_dict(),
                "screening_tolerance": 0.0,
                "nbf": basis.nao,
                "auxiliary_rank": 0,
                "device_bytes": 4096,
            }
            self.closed = False
            self.fail = False
            self.override = None
            plans.append(self)

        def evaluate(self, d: typing.Any, *, derivative: typing.Any) -> typing.Any:
            assert not derivative and not self.closed
            assert d.dtype == np.float64
            if self.fail:
                raise RuntimeError("injected device failure")
            result = SimpleNamespace(
                coulomb=immutable(2 * d),
                exchange=immutable(3 * d),
                diagnostics=deepcopy(self.diagnostics),
            )
            if self.override:
                setattr(result, *self.override)
            return result

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(direct_cuda, "NativeAO", AO)
    monkeypatch.setattr(direct_cuda, "FockPlan", Plan)
    return plans


@pytest.mark.parametrize(
    "kwargs",
    [
        {"device_id": -1},
        {"device_id": True},
        {"device_id": 2**31},
        {"device_budget_bytes": 0},
        {"device_budget_bytes": True},
        {"device_budget_bytes": 2**64},
    ],
)
def test_invalid_controls_fail_before_provider_creation(
    source: typing.Any, fake_provider: typing.Any, kwargs: typing.Any
) -> None:
    with pytest.raises(ValueError):
        CudaDirectJKBackend(source, **kwargs)
    assert not fake_provider


def test_raw_signed_density_and_detached_provenance(
    source: typing.Any, fake_provider: typing.Any
) -> None:
    d = np.array([[-2, 0.2], [0.2, 1]], dtype=np.float32)
    with CudaDirectJKBackend(source) as backend:
        j, k = backend.coulomb_exchange(d)
        np.testing.assert_array_equal(j, 2 * d.astype(np.float64))
        np.testing.assert_array_equal(k, 3 * d.astype(np.float64))
        with pytest.raises(ValueError):
            j.setflags(write=True)
        diag = backend.diagnostics
        assert diag["jk_execution"] == "cuda"
        assert not diag["gpu_resident_response"]
        assert not diag["molecular_hessian"]
        diag["provider"]["backend"] = "cpu"
        assert backend.diagnostics["provider"]["backend"] == "cuda"
        assert backend.statistics["actions"] == 1
    assert fake_provider[0].closed
    source._check_open()
    backend.close()
    with pytest.raises(RuntimeError, match="closed"):
        backend.coulomb_exchange(d)
    with pytest.raises(RuntimeError, match="closed"):
        backend.__enter__()


@pytest.mark.parametrize(
    "field,value",
    [
        ("geometry_hash", "other"),
        ("basis_hash", "other"),
        ("representation", "real_spherical"),
        ("nmo", 3),
        ("electron_count", 4),
        ("hamiltonian_id", "density-fitting:other"),
        ("algorithm", "KS"),
    ],
)
def test_reference_binding_does_not_accept_matching_dimensions_only(
    source: typing.Any, fake_provider: typing.Any, field: typing.Any, value: typing.Any
) -> None:
    meta, arrays = load_fixture("h2")
    reference = fixture_snapshot(meta, arrays)
    names = (
        "geometry_hash",
        "basis_hash",
        "representation",
        "nmo",
        "electron_count",
        "hamiltonian_id",
        "algorithm",
    )
    bad = SimpleNamespace(**{name: getattr(reference, name) for name in names})
    setattr(bad, field, value)
    with CudaDirectJKBackend(source) as backend:
        assert backend.validate_reference(reference) is backend
        with pytest.raises(ValueError, match=field + " mismatch"):
            backend.validate_reference(bad)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((3, 3)),
        np.array([[0, 1], [0, 0]]),
        np.full((2, 2), np.nan),
        np.eye(2, dtype=complex),
    ],
)
def test_invalid_density_does_not_count_or_poison_actions(
    source: typing.Any, fake_provider: typing.Any, bad: typing.Any
) -> None:
    with CudaDirectJKBackend(source) as backend:
        with pytest.raises(ValueError):
            backend.coulomb_exchange(bad)
        assert backend.statistics["actions"] == 0
        backend.coulomb_exchange(np.eye(2))
        assert backend.statistics["actions"] == 1


def test_failure_is_not_replaced_with_cpu_and_replay_is_clean(
    source: typing.Any, fake_provider: typing.Any
) -> None:
    with CudaDirectJKBackend(source) as backend:
        plan = fake_provider[0]
        plan.fail = True
        with pytest.raises(RuntimeError, match="injected device failure"):
            backend.coulomb_exchange(np.eye(2))
        assert backend.statistics["actions"] == 0
        plan.fail = False
        np.testing.assert_array_equal(
            backend.coulomb_exchange(np.eye(2))[0], 2 * np.eye(2)
        )


@pytest.mark.parametrize(
    "bad", [None, np.eye(3), np.eye(2, dtype=np.float32), np.full((2, 2), np.nan)]
)
def test_incomplete_results_fail_without_publishing_success(
    source: typing.Any, fake_provider: typing.Any, bad: typing.Any
) -> None:
    with CudaDirectJKBackend(source) as backend:
        fake_provider[0].override = ("exchange", bad)
        with pytest.raises(RuntimeError, match="invalid J/K"):
            backend.coulomb_exchange(np.eye(2))
        assert backend.statistics["actions"] == 0
        fake_provider[0].override = None
        backend.coulomb_exchange(np.eye(2))


def test_source_close_and_mutation_invalidate_prepared_adapter(
    source: typing.Any, fake_provider: typing.Any
) -> None:
    with CudaDirectJKBackend(source) as backend:
        with pytest.raises(AttributeError, match="immutable"):
            source.geometry_hash = "different"
        backend.coulomb_exchange(np.eye(2))
        source.close()
        with pytest.raises(RuntimeError, match="closed"):
            backend.coulomb_exchange(np.eye(2))
        assert backend.statistics["actions"] == 1


def test_cpu_or_altered_provider_is_rejected_and_closed(
    source: typing.Any, fake_provider: typing.Any, monkeypatch: typing.Any
) -> None:
    constructor = direct_cuda.FockPlan

    def changed(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        plan = constructor(*args, **kwargs)
        plan.diagnostics["backend"] = "cpu"
        return plan

    monkeypatch.setattr(direct_cuda, "FockPlan", changed)
    with pytest.raises(RuntimeError, match="semantics mismatch"):
        CudaDirectJKBackend(source)
    assert fake_provider[0].closed


def test_provider_creation_failure_is_not_suppressed(
    source: typing.Any, fake_provider: typing.Any, monkeypatch: typing.Any
) -> None:
    def unavailable(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        raise NotImplementedError("CUDA unavailable sentinel")

    monkeypatch.setattr(direct_cuda, "FockPlan", unavailable)
    with pytest.raises(NotImplementedError, match="CUDA unavailable sentinel"):
        CudaDirectJKBackend(source)
