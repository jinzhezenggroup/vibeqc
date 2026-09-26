"""Audit retained diagnostic hashes and force contractions without CUDA/PySCF."""

import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np

if not __debug__:
    raise RuntimeError("evidence audit requires assertions; do not use python -O")


def maximum(value):
    """Use the complete force array, including both atoms and all components."""
    return float(np.max(np.abs(value)))


def audit(root):
    """Check preserved failure, final densities, derivatives and response weights."""
    manifest = json.loads((root / "manifest.json").read_text())
    assert len(manifest["records"]) == 39
    payloads = {}
    for record in manifest["records"]:
        relative = Path(record["retained"])
        assert not relative.is_absolute() and ".." not in relative.parts
        assert record["source"] not in payloads
        stored = (root / relative).read_bytes()
        assert len(stored) == record["stored_bytes"]
        assert hashlib.sha256(stored).hexdigest() == record["stored_sha256"]
        assert record["encoding"] in ("identity", "gzip")
        data = gzip.decompress(stored) if record["encoding"] == "gzip" else stored
        assert len(data) == record["original_bytes"]
        assert hashlib.sha256(data).hexdigest() == record["original_sha256"]
        payloads[record["source"]] = data

    def read(name):
        return json.loads(payloads[name])

    def array(name):
        return np.load(io.BytesIO(payloads[name]), allow_pickle=False)

    captured = read("final-density-v1/report.json")
    reconstructed = read("retained-density-reconstruction-v1.json")
    oracle_path = next(k for k in payloads if k.endswith("/cpu-diagnosis-v1.json"))
    oracle = {r["case"]: r for r in read(oracle_path)["rows"]}
    assert captured["status"] == "failed diagnostic"
    assert "positional argument" in captured["exception"]
    assert len(captured["rows"]) == len(reconstructed["rows"]) == 2
    assert {r["case"] for r in captured["rows"]} == set(oracle)
    state_errors = {}
    for row, cpu in zip(captured["rows"], reconstructed["rows"], strict=True):
        name = row["case"]
        assert name == cpu["case"]
        density_bytes = payloads["final-density-v1/" + name + "-density.npy"]
        assert hashlib.sha256(density_bytes).hexdigest() == cpu["density_sha256"]
        density = array("final-density-v1/" + name + "-density.npy")
        seed = array("final-density-v1/" + name + "-seed.npy")
        assert (
            maximum(density - seed) == row["state_checks"]["seed_to_output_density_max"]
        )
        native = np.asarray(row["native_forces"])
        reference = np.asarray(oracle[name]["forces_hartree_per_bohr"])
        assert maximum(native - reference) == row["native_vs_oracle"]
        assert set(cpu["reconstructed"]) == {"canonical", "consistent"}
        for entry in cpu["reconstructed"].values():
            force = np.asarray(entry["forces"])
            assert maximum(force - native) == entry["vs_native"]
            assert maximum(force - reference) == entry["vs_oracle"]
        state_errors[name] = cpu["reconstructed"]["consistent"]["vs_oracle"]

    components = read("oh-force-components-v1/report.json")
    assert components["status"].startswith("completed")
    fixture = array("oh-force-components-v1/fixed-weights-and-reference.npz")
    expected = {
        "one_electron": np.einsum(
            "axkij,kij->ax", fixture["derivatives"], fixture["one_weights"]
        ),
        "three_center": np.einsum("axijp,ijp->ax", fixture["da"], fixture["bar_a"]),
        "metric": np.einsum("axpq,pq->ax", fixture["dm"], fixture["bar_m"]),
        "nuclear": np.asarray(components["cpu_components"]["nuclear"]),
    }
    for name, values in expected.items():
        np.testing.assert_allclose(
            values, components["cpu_components"][name], atol=1e-13, rtol=0
        )
    cpu_force = -sum(expected.values())
    np.testing.assert_allclose(cpu_force, components["cpu_force"], atol=1e-13, rtol=0)
    # NPY retains numeric values, not arbitrary source-array strides. A fresh
    # einsum may reduce in a different order; verify its physical contraction
    # above, then check recorded residuals against the exact retained arrays.
    recorded_cpu_force = np.asarray(components["cpu_force"])
    assert len(components["cuda"]) == 8
    oh = next(r for r in captured["rows"] if r["case"] == components["case"])
    oh_oracle = np.asarray(oracle[components["case"]]["forces_hartree_per_bohr"])
    for schedule in (0, 1):
        rows = {
            r["component"]: r for r in components["cuda"] if r["schedule"] == schedule
        }
        assert set(rows) == {
            "one_electron",
            "three_center",
            "metric",
            "assembled_full_force",
        }
        force = -expected["nuclear"].copy()
        for name in ("one_electron", "three_center", "metric"):
            values = np.asarray(rows[name]["gradient"])
            recorded_reference = np.asarray(components["cpu_components"][name])
            assert (
                maximum(values - recorded_reference)
                == rows[name]["max_error_vs_cpu_fixed_weights"]
            )
            force -= values
        row = rows["assembled_full_force"]
        np.testing.assert_allclose(force, row["force"], atol=4e-15, rtol=0)
        stored = np.asarray(row["force"])
        assert maximum(stored - recorded_cpu_force) == row["vs_cpu_fixed_weights"]
        assert maximum(stored - oh["native_forces"]) == row["vs_native_endpoint"]
        assert maximum(stored - oh_oracle) == row["vs_oracle"]

    response = read("oh-response-weights-v1/report.json")
    assert response["status"].startswith("completed")
    assert (
        np.finfo(np.longdouble).nmant == response["metric"]["longdouble_mantissa"] == 63
    )
    arrays = {
        name.removesuffix(".npy"): array(
            "oh-response-weights-v1/response-arrays.npz#" + name
        )
        for name in manifest["containers"][0]["retained_members"]
    }
    assert len(response["rows"]) == 5
    response_errors = {}
    for row in response["rows"]:
        label = row["label"]
        weight_a, weight_m = arrays[label + "_a"], arrays[label + "_m"]
        assert maximum(weight_a - fixture["bar_a"]) == row["a_weight_error"]
        assert maximum(weight_m - fixture["bar_m"]) == row["m_weight_error"]
        da = np.einsum("axijp,ijp->ax", fixture["da"], weight_a - fixture["bar_a"])
        dm = np.einsum("axpq,pq->ax", fixture["dm"], weight_m - fixture["bar_m"])
        # These contractions involve small weight differences; allow only
        # 1e-18 Eh/Bohr absolute reduction drift, well below every scientific
        # gate and the reported response discrepancy.
        for actual, recorded in (
            (maximum(da), row["three_center_force_error"]),
            (maximum(dm), row["metric_force_error"]),
            (maximum(da + dm), row["combined_force_error"]),
        ):
            np.testing.assert_allclose(actual, recorded, atol=1e-18, rtol=1e-10)
        np.testing.assert_allclose(
            np.asarray(da, dtype=float),
            row["three_center_delta"],
            atol=1e-18,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            np.asarray(dm, dtype=float), row["metric_delta"], atol=1e-18, rtol=1e-10
        )
        response_errors[label] = row["combined_force_error"]
    return {
        "records": len(payloads),
        "cpu_force_at_native_density_errors": state_errors,
        "response_only_force_errors": response_errors,
        "conclusion": "Untimed diagnosis only; original failures unchanged; keeps #206 open.",
    }


if __name__ == "__main__":
    print(json.dumps(audit(Path(__file__).resolve().parent), indent=2))
