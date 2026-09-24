"""Execute the actual Z-owner lease region with injected setup failures."""

from __future__ import annotations

import textwrap
import typing
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "mode",
    [
        "success",
        "host",
        "dimension-read",
        "dimension-mismatch",
        "workspace-read",
        "negative-workspace",
        "bad-workspace-type",
        "over-budget",
        "solver",
        "solve",
        "diagnostics",
    ],
)
def test_resident_lease_closes_on_every_post_allocation_exit(mode: str) -> None:
    source = (ROOT / "tools/vibeqc_cc/triples_orbital_response.py").read_text()
    start = source.index("        resident_owner = None\n")
    stop = source.index("        independent_z_residual =", start)
    region = textwrap.dedent(source[start:stop])
    events: list[str] = []

    class Engine:
        @property
        def dimension(self) -> int:
            if mode == "dimension-read":
                raise RuntimeError("injected dimension read")
            return 3 if mode == "dimension-mismatch" else 2

        @property
        def workspace_bytes(self) -> object:
            if mode == "workspace-read":
                raise RuntimeError("injected workspace read")
            return {
                "negative-workspace": -1,
                "bad-workspace-type": True,
                "over-budget": 1001,
            }.get(mode, 128)

        @property
        def diagnostics(self) -> dict[str, int]:
            if mode == "diagnostics":
                raise RuntimeError("injected diagnostics")
            return {"owned_device_bytes": 128}

        def close(self) -> None:
            events.append("close")

    engine = Engine()

    def prepare(*args: object, **kwargs: object) -> Engine:
        del args
        assert kwargs["device_budget_bytes"] == 936
        events.append("prepare")
        return engine

    def solver(*args: object) -> object:
        del args
        events.append("solver")
        if mode == "solver":
            raise MemoryError("injected solver construction")
        return object()

    def solve(transpose: object, rhs: object, **kwargs: object) -> object:
        del rhs, kwargs
        events.append("solve")
        assert (getattr(transpose, "_krylov_engine", None) is engine) == (mode != "host")
        if mode == "solve":
            raise RuntimeError("injected solve")
        return SimpleNamespace(solution=[1.0, 2.0])

    scope = {
        "typing": typing,
        "ExitStack": ExitStack,
        "baseline": SimpleNamespace(
            response_backend=SimpleNamespace(
                resident_response=prepare, device_resident_bytes=64
            )
        ),
        "operator": SimpleNamespace(
            dimension=2, problem=object(), apply_transpose=lambda value: value
        ),
        "owner": SimpleNamespace(_assert_current=lambda: None),
        "options": SimpleNamespace(z_options=object()),
        "response_execution": "host" if mode == "host" else "cuda-resident",
        "response_device_budget_bytes": 1000,
        "resident_vector_slots": lambda *args: 8,
        "ResponseGMRES": solver,
        "checked_transpose_solve": solve,
        "ImplicitSolveError": RuntimeError,
        "ResponseCompatibilityError": ValueError,
        "rhs": [1.0, 2.0],
    }
    if mode in {"success", "host"}:
        exec(compile(region, "resident-z-lease", "exec"), scope)
        assert scope["z"].solution == [1.0, 2.0]
    else:
        expected = (
            MemoryError
            if mode == "solver"
            else ValueError
            if mode == "dimension-mismatch"
            else RuntimeError
        )
        with pytest.raises(expected):
            exec(compile(region, "resident-z-lease", "exec"), scope)
    assert events.count("close") == (0 if mode == "host" else 1)
    if mode != "host":
        assert events[0] == "prepare" and events[-1] == "close"
