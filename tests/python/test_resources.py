"""Global lifetime composition, coupled budgets and side-effect-free planning."""

import json
from dataclasses import replace

import pytest
from vibeqc.resources import (
    MAX_BYTES,
    ResourceBudget,
    ResourceCandidate,
    ResourceEstimate,
    ResourceIdentity,
    ResourcePlan,
    ResourceRequest,
    byte_product,
    lower_memory_plans,
    plan_resources,
)


def request(name, candidates):
    identity = ResourceIdentity(
        "hf",
        name,
        "cpu",
        "fp64",
        json.dumps({"nbf": 4, "batch": [1]}),
        ("energy", "forces"),
        "fixed",
    )
    return ResourceRequest(name, identity, tuple(candidates))


def allocation(name, size, begin, end, **kwargs):
    return ResourceEstimate(
        name, size, kwargs.pop("space", "pageable"), begin, end, **kwargs
    )


def test_shared_state_plus_serial_phases_does_not_sum_phase_maxima():
    hf = request(
        "hf",
        [
            ResourceCandidate(
                "default",
                "resident",
                (
                    allocation("density", 50, 0, 4, kind="persistent"),
                    allocation("iteration", 80, 1, 1),
                    allocation("forces", 60, 2, 2),
                ),
            )
        ],
    )
    tensor = request(
        "tensor",
        [
            ResourceCandidate(
                "default",
                "tiled",
                (
                    allocation("panels", 70, 3, 3),
                    allocation("output", 10, 3, 4, kind="output"),
                ),
            )
        ],
    )
    plan = plan_resources([tensor, hf], ResourceBudget(host_bytes=130))
    assert plan.status == "feasible"
    assert plan.peak_bytes["host"] == 130
    assert plan.resident_bytes["host"] == 60
    assert plan.identity == plan_resources([hf, tensor], plan.budget).identity


def test_concurrent_stream_lifetimes_and_host_pinned_total_are_composed():
    a = request(
        "a",
        [
            ResourceCandidate(
                "default", "streamed", (allocation("pin", 40, 1, 3, space="pinned"),)
            )
        ],
    )
    b = request(
        "b", [ResourceCandidate("default", "resident", (allocation("host", 70, 2, 4),))]
    )
    plan = plan_resources([a, b], ResourceBudget(host_bytes=100, pinned_host_bytes=50))
    assert plan.status == "infeasible"
    assert plan.peak_bytes["host"] == 110
    assert "dominant allocations" in plan.diagnostic
    with pytest.raises(MemoryError, match="no supported plan fits"):
        plan.require_feasible()


def test_coupled_host_device_tradeoff_selects_supported_streaming():
    resident = ResourceCandidate(
        "resident",
        "resident",
        (allocation("B", 90, 0, 3, space="device:0", kind="cache", cacheable=True),),
    )
    streamed = ResourceCandidate(
        "stream",
        "streamed",
        (
            allocation("B-host", 60, 0, 3, kind="cache"),
            allocation("B-tile", 20, 1, 3, space="device:0", streamed_bytes=180),
        ),
        relative_cost=3,
    )
    consumer = request("df", [resident, streamed])
    other = request(
        "hf",
        [
            ResourceCandidate(
                "fixed",
                "resident",
                (allocation("density", 20, 0, 3, space="device:0", kind="persistent"),),
            )
        ],
    )
    plan = plan_resources(
        [consumer, other], ResourceBudget(host_bytes=60, device_bytes=80)
    )
    assert plan.selections == (("df", "stream"), ("hf", "fixed"))
    assert plan.peak_bytes["device"] == 40
    assert (
        plan_resources(
            [consumer, other], ResourceBudget(host_bytes=59, device_bytes=80)
        ).status
        == "infeasible"
    )


def test_each_device_and_combined_device_caps_both_apply():
    r = request(
        "tensor",
        [
            ResourceCandidate(
                "fixed",
                "resident",
                (
                    allocation("a", 50, 0, 2, space="device:0"),
                    allocation("b", 60, 0, 2, space="device:1"),
                ),
            )
        ],
    )
    assert plan_resources([r], ResourceBudget(device_bytes=100)).status == "infeasible"
    assert (
        plan_resources(
            [r], ResourceBudget(device_bytes=120, per_device_bytes=((1, 59),))
        ).status
        == "infeasible"
    )
    assert (
        plan_resources(
            [r], ResourceBudget(device_bytes=110, per_device_bytes=((1, 60),))
        ).status
        == "feasible"
    )


def test_recomputation_keeps_outputs_and_fixed_state_reserved():
    r = request(
        "tensor",
        [
            ResourceCandidate("keep", "resident", (allocation("shared", 150, 1, 3),)),
            ResourceCandidate(
                "again",
                "recomputed",
                (
                    allocation("shared-1", 50, 1, 1, recomputable=True),
                    allocation("shared-2", 50, 3, 3, recomputable=True),
                ),
                relative_cost=2,
            ),
        ],
    )
    state = request(
        "state",
        [
            ResourceCandidate(
                "fixed",
                "resident",
                (allocation("amplitudes", 80, 0, 4, kind="persistent"),),
            )
        ],
    )
    p = plan_resources([r, state], ResourceBudget(host_bytes=130))
    assert dict(p.selections)["tensor"] == "again"
    assert p.peak_bytes["host"] == 130
    assert (
        plan_resources([r, state], ResourceBudget(host_bytes=129)).status
        == "infeasible"
    )


def test_headroom_reserve_zero_and_unlimited_are_distinct():
    assert (
        ResourceBudget(
            host_bytes=100, host_reserve_bytes=5, headroom_fraction=0.25
        ).limits()["host"]
        == 70
    )
    r = request(
        "a", [ResourceCandidate("one", "resident", (allocation("x", 1, 0, 0),))]
    )
    assert plan_resources([r], ResourceBudget()).status == "feasible"
    assert plan_resources([r], ResourceBudget(host_bytes=0)).status == "infeasible"
    assert (
        ResourceBudget(host_bytes=MAX_BYTES, headroom_fraction=0.5).limits()["host"]
        == MAX_BYTES // 2
    )


@pytest.mark.parametrize("bad", [-1, True, 1.5, MAX_BYTES + 1])
def test_sizes_and_products_cannot_wrap(bad):
    with pytest.raises(ValueError):
        ResourceBudget(host_bytes=bad)
    with pytest.raises(ValueError):
        byte_product(8, bad)
    with pytest.raises(ValueError):
        byte_product(MAX_BYTES, 2)


def test_unsupported_provider_and_search_limit_are_not_false_oom():
    r = request(
        "a", [ResourceCandidate("one", "resident", (allocation("x", 1, 0, 0),))]
    )
    absent = replace(
        r, candidates=(), unsupported_reason="provider workspace query unavailable"
    )
    assert plan_resources([absent], ResourceBudget()).status == "unsupported"
    assert (
        plan_resources([r], ResourceBudget(), maximum_combinations=0).status
        == "unsupported"
    )


def test_sparse_phase_ids_do_not_allocate_a_dense_timeline():
    r = request(
        "sparse",
        [
            ResourceCandidate(
                "one",
                "resident",
                (
                    allocation("first", 10, 0, 0),
                    allocation("last", 20, MAX_BYTES, MAX_BYTES),
                ),
            )
        ],
    )
    assert plan_resources([r], ResourceBudget(host_bytes=20)).peak_bytes["host"] == 20


def test_scientific_and_schedule_changes_invalidate_plan_identity():
    r = request(
        "a", [ResourceCandidate("one", "resident", (allocation("x", 1, 0, 0),))]
    )
    base = plan_resources([r], ResourceBudget()).identity
    for field, value in (
        ("precision", "fp32"),
        ("schedule", "new"),
        ("topology", json.dumps({"nbf": 5, "basis_hash": "changed"})),
        ("observables", ("energy",)),
    ):
        changed = replace(r, identity=replace(r.identity, **{field: value}))
        assert plan_resources([changed], ResourceBudget()).identity != base


def test_portable_plan_rechecks_derived_bytes_even_after_rehashed_tampering():
    from vibeqc.profiles import canonical_hash

    r = request(
        "a", [ResourceCandidate("one", "resident", (allocation("x", 20, 0, 0),))]
    )
    plan = plan_resources([r], ResourceBudget(host_bytes=20))
    payload = json.loads(json.dumps(plan.to_dict()))
    assert ResourcePlan.from_dict(payload).identity == plan.identity
    payload["peak_bytes"]["host"] = 0
    payload.pop("identity")
    payload["identity"] = canonical_hash(payload)
    with pytest.raises(ValueError, match="derived accounting"):
        ResourcePlan.from_dict(payload)


def test_allocation_retry_only_returns_enumerated_lower_memory_candidates():
    r = request(
        "a",
        [
            ResourceCandidate("fast", "resident", (allocation("resident", 80, 0, 2),)),
            ResourceCandidate(
                "slow",
                "recomputed",
                (allocation("recompute", 40, 0, 2),),
                relative_cost=1,
            ),
        ],
    )
    plan = plan_resources([r], ResourceBudget(host_bytes=100))
    alternatives = lower_memory_plans(plan, "host")
    assert len(alternatives) == 1 and alternatives[0].selections == (("a", "slow"),)
    assert not lower_memory_plans(alternatives[0], "host")


def test_session_releases_failed_group_before_retry_and_enforces_lifetimes():
    from vibeqc.resources import ResourceAllocationError, ResourceSession

    events = []

    class Owner:
        def __init__(self, name):
            self.name = name
            events.append(("allocate", name))

        def close(self):
            events.append(("close", self.name))

    a = request(
        "a", [ResourceCandidate("fixed", "resident", (allocation("a", 10, 0, 0),))]
    )
    b = request(
        "b",
        [
            ResourceCandidate("large", "resident", (allocation("b", 80, 0, 0),)),
            ResourceCandidate(
                "small", "recomputed", (allocation("b", 40, 0, 0),), relative_cost=1
            ),
        ],
    )
    c = request(
        "c", [ResourceCandidate("fixed", "resident", (allocation("c", 90, 1, 1),))]
    )

    def factory_b(plan):
        if dict(plan.selections)["b"] == "large":
            events.append(("failed", "b"))
            raise ResourceAllocationError("host", "allocator rejected b")
        return Owner("b")

    plan = plan_resources([a, b, c], ResourceBudget(host_bytes=90))
    # The future phase maximum is unchanged; preparation still needs the
    # lower-memory active-phase candidate after a real allocator rejection.
    with ResourceSession(
        plan, {"a": lambda p: Owner("a"), "b": factory_b, "c": lambda p: Owner("c")}
    ) as session:
        session.advance(0)
        assert events == [
            ("allocate", "a"),
            ("failed", "b"),
            ("close", "a"),
            ("allocate", "a"),
            ("allocate", "b"),
        ]
        assert len(session.fallbacks) == 1
        assert session.fallbacks[0]["to_plan"] == session.plan.identity
        assert session.provider("b").name == "b"
        session.advance(1)
        assert events[-3:] == [("close", "b"), ("close", "a"), ("allocate", "c")]
        with pytest.raises(RuntimeError, match="not live"):
            session.provider("a")
        with pytest.raises(ValueError, match="backwards"):
            session.advance(0)
    assert events[-1] == ("close", "c")


def test_session_never_retries_numerical_errors_or_changes_live_owners():
    from vibeqc.resources import ResourceAllocationError, ResourceSession

    retained = request(
        "a",
        [
            ResourceCandidate("large", "resident", (allocation("a", 80, 0, 2),)),
            ResourceCandidate(
                "small", "resident", (allocation("a", 40, 0, 2),), relative_cost=1
            ),
        ],
    )
    later = request(
        "b", [ResourceCandidate("fixed", "resident", (allocation("b", 10, 1, 1),))]
    )
    plan = plan_resources([retained, later], ResourceBudget(host_bytes=100))
    closed = []

    class Retained:
        def close(self):
            closed.append("a")

    for failure in (
        ResourceAllocationError("host", "out of memory"),
        RuntimeError("numerical failure"),
    ):
        attempts = []

        def fail(selected, attempts=attempts, failure=failure):
            attempts.append(selected.identity)
            raise failure

        with ResourceSession(plan, {"a": lambda p: Retained(), "b": fail}) as session:
            session.advance(0)
            with pytest.raises(type(failure), match=str(failure)):
                session.advance(1)
            assert attempts == [plan.identity]
            assert session.plan.identity == plan.identity
            assert bool(session.fallbacks) == isinstance(
                failure, ResourceAllocationError
            )
    assert closed == ["a", "a"]


def test_resource_session_rejects_internal_phases_it_cannot_enforce():
    from vibeqc.resources import ResourceSession

    r = request(
        "a",
        [
            ResourceCandidate(
                "fixed",
                "resident",
                (
                    allocation("persistent", 10, 0, 2),
                    allocation("workspace", 20, 1, 1),
                ),
            )
        ],
    )
    with pytest.raises(ValueError, match="uniform provider lifetime"):
        ResourceSession(plan_resources([r], ResourceBudget()), {"a": lambda p: None})


def test_session_exhausts_alternating_host_device_failures_without_cycling():
    from vibeqc.resources import ResourceAllocationError, ResourceSession

    choices = request(
        "a",
        [
            ResourceCandidate(
                "device-heavy",
                "resident",
                (
                    allocation("host", 10, 0, 0),
                    allocation("device", 80, 0, 0, space="device:0"),
                ),
            ),
            ResourceCandidate(
                "host-heavy",
                "streamed",
                (
                    allocation("host", 80, 0, 0),
                    allocation("device", 10, 0, 0, space="device:0"),
                ),
                relative_cost=1,
            ),
        ],
    )
    attempts = []

    def fail(plan):
        choice = dict(plan.selections)["a"]
        attempts.append(choice)
        # Fail fast if the executor regresses into a retry cycle.
        assert len(attempts) <= 2
        space = "device:0" if choice == "device-heavy" else "host"
        raise ResourceAllocationError(space, "allocator rejected selected storage")

    with ResourceSession(
        plan_resources([choices], ResourceBudget(host_bytes=100, device_bytes=100)),
        {"a": fail},
    ) as session:
        with pytest.raises(ResourceAllocationError, match="allocator rejected"):
            session.advance(0)
        assert attempts == ["device-heavy", "host-heavy"]
        assert len(session.fallbacks) == 2
        assert session.fallbacks[-1]["to_plan"] is None


def test_native_ledger_metadata_needs_no_gpu_and_rejects_concurrent_binding():
    from concurrent.futures import ThreadPoolExecutor

    from vibeqc import Calculator
    from vibeqc.resources import CpuResourceObservation
    from vibeqc.resources_native import NativeDeviceLedger

    library = Calculator()._library
    r = request(
        "hf",
        [
            ResourceCandidate(
                "fixed", "resident", (allocation("arena", 64, 0, 0, space="device:0"),)
            )
        ],
    )
    r = replace(r, identity=replace(r.identity, backend="cuda"))
    ledger = NativeDeviceLedger(library, plan_resources([r], ResourceBudget()))

    def observe():
        with CpuResourceObservation(library, ledger=ledger):
            return ledger.to_dict()

    try:
        with ThreadPoolExecutor(max_workers=1) as worker:
            with CpuResourceObservation(library, ledger=ledger):
                with pytest.raises(RuntimeError, match="scope could not be bound"):
                    worker.submit(observe).result()
                assert ledger.to_dict()["live_bytes"] == 0
            assert worker.submit(observe).result()["peak_bytes"] == 0
        with pytest.raises(ValueError, match="CUDA request"):
            NativeDeviceLedger(
                library,
                plan_resources(
                    [replace(r, identity=replace(r.identity, backend="cpu"))],
                    ResourceBudget(),
                ),
            )
    finally:
        ledger.close()
