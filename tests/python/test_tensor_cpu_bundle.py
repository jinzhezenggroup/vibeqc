"""High-level native CPU TensorIR bundle ownership and execution gates."""

import typing
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    input_tensor,
    multiply,
)
from vibeqc_compiler.tensor.cpu_bundle import NativeTensorProgramBundle


def _tensor(name: str, size: int = 4) -> typing.Any:
    index = Index("i", IndexSpace("bundle", "batch", size))
    return input_tensor(name, TensorSpec((index,), role="input"))


def _programs() -> tuple[Program, Program]:
    value = _tensor("value")
    return (
        Program({"out": add(value, value)}),
        Program({"out": multiply(value, value)}),
    )


def _bundle(
    programs: typing.Sequence[Program], tmp_path: Path
) -> NativeTensorProgramBundle:
    return NativeTensorProgramBundle(
        programs,
        compiler=CppCompilerAdapter(Path("c++")),
        cache=tmp_path / "native-tensor-bundle",
    )


def test_bundle_prewarms_exact_program_set_into_one_artifact(tmp_path: Path) -> None:
    programs = _programs()
    bundle = _bundle(programs, tmp_path)
    values = np.arange(4, dtype=np.float64)

    assert bundle.program_count == 2
    assert bundle.artifact_count == 1
    assert bundle.program_identities == tuple(
        program.logical_hash for program in programs
    )
    assert bundle.artifact.metadata["source_count"] == 2
    np.testing.assert_array_equal(
        bundle.execute(programs[0], {"value": values})["out"], 2 * values
    )
    np.testing.assert_array_equal(
        bundle.execute(programs[1], {"value": values})["out"], values * values
    )

    replay = _bundle(
        tuple(Program.loads(program.dumps()) for program in programs), tmp_path
    )
    assert replay.artifact.library == bundle.artifact.library


def test_bundle_identity_is_ordered_and_exact(tmp_path: Path) -> None:
    programs = _programs()
    forward = _bundle(programs, tmp_path)
    reverse = _bundle(programs[::-1], tmp_path)

    assert forward.program_identities == tuple(
        program.logical_hash for program in programs
    )
    assert reverse.program_identities == tuple(
        program.logical_hash for program in programs[::-1]
    )
    assert forward.artifact.library != reverse.artifact.library

    absent_value = _tensor("absent")
    absent = Program({"out": add(absent_value, absent_value)})
    with pytest.raises(ValueError, match="not present"):
        forward.execute(absent, {"absent": np.arange(4, dtype=np.float64)})


def test_bundle_rejects_duplicate_program_identity_before_compile(
    tmp_path: Path,
) -> None:
    program = _programs()[0]
    with pytest.raises(ValueError, match="unique program identities"):
        _bundle((program, Program.loads(program.dumps())), tmp_path)


def test_bundle_preserves_native_input_admission(tmp_path: Path) -> None:
    program = _programs()[0]
    bundle = _bundle((program,), tmp_path)

    with pytest.raises(ValueError, match="invalid float64 tensor input"):
        bundle.execute(program, {"value": np.arange(4, dtype=np.float32)})
    with pytest.raises(ValueError, match="missing tensor input"):
        bundle.execute(program, {})
