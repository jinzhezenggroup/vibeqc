"""CPU evidence registration and the explicit IntegralIR tile/weight boundary."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from tools.vibeqc_codegen.blocks import (
    BlockRequest,
    RawBlock,
    ShellTile,
    TensorLayout,
    WeightDescriptor,
    WeightedDerivative,
    WeightTile,
    assemble_raw_block,
    contract_weighted_derivative,
)
from tools.vibeqc_codegen.ir import IntegralIR, OperatorSpec, TranslationInvariant
from tools.vibeqc_codegen.shell_signature import (
    BasisShell,
    CenterBinding,
    ShellSignature,
)
from tools.vibeqc_tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    execute,
    input_tensor,
)
from tools.vibeqc_validation.schema import validate_evidence


def test_cli_exports_replayable_examples_and_shared_evidence(tmp_path):
    root = Path(__file__).resolve().parents[2]
    script = root / "tools/tensor_ir_examples.py"
    help_result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--equations-dir" in help_result.stdout
    output, equations = tmp_path / "evidence.json", tmp_path / "equations"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--output",
            str(output),
            "--equations-dir",
            str(equations),
        ],
        check=True,
        cwd=tmp_path,
    )
    records = json.loads(output.read_text())
    assert len(records) == 4
    for record in records:
        validate_evidence(record)
        assert record["tier"] == "cpu"
        assert record["stages"]["numerical"]["status"] == "pass"
        assert record["stages"]["production"]["status"] == "not-run"
        assert record["performance"]["status"] == "not-run"
        assert record["hashes"]["equation"] and record["settings"]["source_files"]
    for path in equations.glob("*.json"):
        payload = json.loads(path.read_text())
        program = Program.from_payload(payload["program"])
        inputs = {
            name: np.array(data["values"], dtype=data["dtype"]).reshape(data["shape"])
            for name, data in payload["inputs"].items()
        }
        np.testing.assert_allclose(
            execute(program, inputs).outputs["value"],
            payload["reference"],
            atol=1e-11,
            rtol=1e-10,
        )


def test_integral_raw_tile_to_tensor_weights_and_back_has_explicit_order_and_sign():
    """Exchange a bounded two-center tile; no shell semantics enter TensorIR."""
    signature = ShellSignature(
        (BasisShell(0, 0, 1), BasisShell(1, 1, 1)),
        (CenterBinding(0, 0), CenterBinding(1, 1)),
    )
    operator = OperatorSpec("overlap", (0, 1), (TranslationInvariant(),))
    layout = TensorLayout(signature.tensor_indices, (2, 2), (3, 1))
    integral = IntegralIR(signature, operator, None, (RawBlock(layout, 4096),))
    tile = ShellTile((1, 0), (2, 2))
    request = BlockRequest("overlap_tile", integral, tile)
    response = assemble_raw_block(request, (1.0, 0.2, -0.3, 0.8))
    # The adapter owns the physical layout and AO offsets. TensorIR sees only
    # a logical array and explicit populations/ranges, never ShellClassSpec.
    space = IndexSpace("ao", "ao", 3)
    spec = TensorSpec((Index("p", space, 1, 3), Index("q", space, 0, 2)), role="input")
    source = input_tensor("raw", spec)
    program = Program({"weights": add(source, coefficients=("1/2",))})
    raw = np.array([response.values[offset] for offset in layout.offsets()]).reshape(
        2, 2
    )
    weights = execute(program, {"raw": raw}).outputs["weights"]
    packed = np.zeros(layout.storage_elements)
    packed[list(layout.offsets())] = weights.ravel()
    consumer = WeightedDerivative(
        WeightDescriptor("tensor_weights", layout),
        TensorLayout(("atom", "xyz"), (2, 3)),
        4096,
        output="atomic_force",
        output_sign=-1,
    )
    derivative_ir = replace(
        integral, derivative=operator.nuclear_derivative(), contractions=(consumer,)
    )
    derivative_request = BlockRequest("weighted_tile", derivative_ir, tile)
    d_a = np.arange(12.0).reshape(3, 2, 2) / 10
    result = contract_weighted_derivative(
        derivative_request,
        {0: d_a.ravel()},
        lambda descriptor, request: WeightTile(layout, packed),
    )
    expected = np.zeros((2, 3))
    for atom, sign in ((0, -1), (1, 1)):
        for xyz in range(3):
            for p in range(2):
                for q in range(2):
                    expected[atom, xyz] += sign * weights[p, q] * d_a[xyz, p, q]
    np.testing.assert_allclose(
        np.array(result.values).reshape(2, 3), expected, atol=1e-11, rtol=1e-10
    )
