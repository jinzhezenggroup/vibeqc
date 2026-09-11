"""Independent radial limits and separately referenced LR/SR primitive ERIs."""

import ctypes
import math
import os
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.integral.capabilities import query_integral_capability
from vibeqc_compiler.integral.ir import four_center_eri_operator
from vibeqc_compiler.integral.ir_serialization import (
    integral_from_payload,
    integral_to_payload,
)
from vibeqc_compiler.integral.range_separation import CoulombKernel, reference_moments
from vibeqc_compiler.integral.weighted_eri import (
    build_weighted_eri_ir,
    build_weighted_eri_kernel,
)
from vibeqc_compiler.integral.weighted_eri_native import (
    emit_weighted_eri_primitive_header,
)

from tools.vibeqc_validation.weighted_eri import primitive_variables

# Reference engines are optional for ordinary package/test installations; CI
# explicitly installs the pinned reference-test extra to exercise these gates.
special = pytest.importorskip("scipy.special")
gammainc, gammaln = special.gammainc, special.gammaln


def native_weighted_evaluator(kernel, folder, backend):
    """Compile the real emitted callable; independent references stay in Python."""
    cuda = backend == "cuda"
    if cuda and os.environ.get("VIBEQC_TEST_RANGE_CUDA") != "1":
        pytest.skip("set VIBEQC_TEST_RANGE_CUDA=1 inside a Slurm GPU job")
    if cuda and not os.environ.get("SLURM_JOB_ID"):
        pytest.fail("native CUDA validation requires a Slurm allocation")
    compiler = shutil.which("nvcc" if cuda else "c++")
    if compiler is None:
        pytest.skip("native compiler unavailable")
    source = emit_weighted_eri_primitive_header(
        ((kernel, "weighted"),), backend=backend
    )
    count = 16 + len(kernel.component_indices)
    if cuda:
        source += r"""
__global__ void probe(const double* inputs, double* output) {
  namespace eri = vibeqc::scf::generated_weighted_eri;
  eri::Gradient result{};
  output[13] = eri::weighted_primitive(inputs, inputs + 4, inputs + 16, result);
  output[0] = result.value;
  for (unsigned c = 0; c < 4; ++c)
    for (unsigned a = 0; a < 3; ++a) output[1 + 3*c + a] = result.center[c][a];
}
extern "C" int evaluate(const double* inputs, double* output) {
  double *in = nullptr, *out = nullptr;
  if (cudaMalloc(&in, INPUT_COUNT * sizeof(double)) != cudaSuccess) return -1;
  if (cudaMalloc(&out, 14 * sizeof(double)) != cudaSuccess) { cudaFree(in); return -1; }
  auto status = cudaMemcpy(in, inputs, INPUT_COUNT * sizeof(double), cudaMemcpyHostToDevice);
  if (status == cudaSuccess) { probe<<<1,1>>>(in, out); status = cudaGetLastError(); }
  if (status == cudaSuccess)
    status = cudaMemcpy(output, out, 14 * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(in); cudaFree(out);
  return status == cudaSuccess ? static_cast<int>(output[13]) : -1;
}
""".replace("INPUT_COUNT", str(count))
    else:
        source += r"""
extern "C" int evaluate(const double* inputs, double* output) {
  namespace eri = vibeqc::scf::generated_weighted_eri;
  eri::Gradient result{};
  if (!eri::weighted_primitive(inputs, inputs + 4, inputs + 16, result)) return 0;
  output[0] = result.value;
  for (unsigned c = 0; c < 4; ++c)
    for (unsigned a = 0; a < 3; ++a) output[1 + 3*c + a] = result.center[c][a];
  return 1;
}
"""
    path = folder / ("weighted.cu" if cuda else "weighted.cpp")
    path.write_text(source)
    library = folder / "weighted.so"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-shared",
            *(
                ["-arch=sm_120", "--fmad=false", "-Xcompiler=-fPIC"]
                if cuda
                else ["-fPIC", "-ffp-contract=off"]
            ),
            "-I" + str(Path(__file__).resolve().parents[2] / "src"),
            str(path),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    owner = ctypes.CDLL(str(library))
    pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    owner.evaluate.argtypes = [pointer, pointer]
    owner.evaluate.restype = ctypes.c_int

    def evaluate(exponents, centers, weights):
        inputs = np.r_[exponents, centers.flat, weights]
        assert inputs.shape == (count,)
        output = np.full(14, 123.0)
        assert owner.evaluate(inputs, output) == 1
        return output[0], output[1:13].reshape(4, 3)

    return evaluate


def full_moments(maximum_order, argument):
    """Independent complete-interval incomplete-gamma reference."""
    return np.array(
        [
            1 / (2 * n + 1)
            if argument == 0
            else math.exp(
                gammaln(n + 0.5) - math.log(2) - (n + 0.5) * math.log(argument)
            )
            * gammainc(n + 0.5, argument)
            for n in range(maximum_order + 1)
        ]
    )


@pytest.mark.parametrize("argument", [0.0, 1e-12, 0.1, 10.0, 100.0, 1e6])
@pytest.mark.parametrize("omega", [0.0, 1e-12, 0.2, 1.0, 1e6, 1e150])
def test_independent_radial_limits_and_sum(argument, omega):
    rho = 0.73
    reference = full_moments(13, argument)
    full = reference_moments(13, argument, rho, CoulombKernel())
    lr = np.array(
        reference_moments(13, argument, rho, CoulombKernel("long_range", omega))
    )
    sr = np.array(
        reference_moments(13, argument, rho, CoulombKernel("short_range", omega))
    )
    np.testing.assert_allclose(full, reference, atol=0, rtol=4e-13)
    np.testing.assert_allclose(lr + sr, reference, atol=0, rtol=4e-13)
    assert np.all(lr >= 0) and np.all(sr >= 0)
    if omega == 0:
        np.testing.assert_array_equal(lr, np.zeros(14))
        np.testing.assert_array_equal(sr, full)
    if omega == 1e150 and argument == 0:
        # The interval width remains representable even though sqrt(theta)
        # rounds to one. Subtraction-based short-range moments would be zero.
        np.testing.assert_allclose(sr, rho / (2 * omega) / omega, atol=0, rtol=4e-15)


def primitive_ssss(exponents, centers, kernel):
    """Normalized primitive Gaussian formula, separate from any native code."""
    a, b, c, d = exponents
    centers = np.asarray(centers, dtype=float)
    p, q = a + b, c + d
    first = (a * centers[0] + b * centers[1]) / p
    second = (c * centers[2] + d * centers[3]) / q
    rho = p * q / (p + q)
    argument = rho * np.sum((first - second) ** 2)
    decay = math.exp(
        -a * b / p * np.sum((centers[0] - centers[1]) ** 2)
        - c * d / q * np.sum((centers[2] - centers[3]) ** 2)
    )
    normalization = math.prod(
        (2 * exponent / math.pi) ** 0.75 for exponent in exponents
    )
    prefactor = 2 * math.pi**2.5 / (p * q * math.sqrt(p + q))
    return (
        normalization
        * decay
        * prefactor
        * reference_moments(0, argument, rho, kernel)[0]
    )


@pytest.mark.parametrize("omega", [0.03, 0.7, 4.0, 1e4])
@pytest.mark.parametrize("family", ["long_range", "short_range"])
def test_primitive_parts_against_independent_libcint(family, omega):
    pyscf = pytest.importorskip("pyscf")
    exponents = [0.13, 0.7, 2.3, 12.0]
    centers = [(0.1, 0.2, -0.1), (0.4, -0.3, 0.2), (-0.2, 0.6, 0.3), (0.3, 0.2, 0.8)]
    labels = [f"H{i}" for i in range(4)]
    mol = pyscf.gto.M(
        atom=list(zip(labels, centers)),
        basis={
            label: [[0, [exponent, 1.0]]] for label, exponent in zip(labels, exponents)
        },
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    signed_omega = omega if family == "long_range" else -omega
    with mol.with_range_coulomb(signed_omega):
        reference = mol.intor_by_shell("int2e", (0, 1, 2, 3)).item()
    actual = primitive_ssss(exponents, centers, CoulombKernel(family, omega))
    assert actual == pytest.approx(reference, abs=1e-15, rel=2e-10)


@pytest.mark.parametrize("family", ["long_range", "short_range"])
def test_nuclear_moment_chain_rule_holds_at_fixed_omega(family):
    kernel = CoulombKernel(family, 0.8)
    argument, rho = 1.3, 0.71
    target = -np.array(reference_moments(5, argument, rho, kernel))[1:]
    errors = []
    for step in (0.01, 0.003, 0.001):
        plus = np.array(reference_moments(4, argument + step, rho, kernel))
        minus = np.array(reference_moments(4, argument - step, rho, kernel))
        errors.append(np.max(np.abs((plus - minus) / (2 * step) - target)))
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 2e-8


@pytest.mark.parametrize("family", ["long_range", "short_range"])
def test_range_ir_identity_and_legacy_executor_rejection(family):
    """New radial semantics cannot alias v1 IR or enter a full-Coulomb executor."""
    operator = four_center_eri_operator(CoulombKernel(family, np.float32(0.7)))
    integral = build_weighted_eri_ir((1, 0, 0, 0), operator=operator)
    payload = integral_to_payload(integral)
    assert payload["schema_version"] == 2
    assert payload["operator"]["omega"] == float(np.float32(0.7))
    assert integral_from_payload(payload) == integral
    different = build_weighted_eri_ir(
        (1, 0, 0, 0), operator=four_center_eri_operator(CoulombKernel(family, 0.8))
    )
    assert canonical_hash(payload) != canonical_hash(integral_to_payload(different))
    assert build_weighted_eri_kernel(integral).integral == integral
    for backend in ("cuda", "cuda_weighted_eri"):
        assert not query_integral_capability(integral, backend=backend).supported
    malformed = deepcopy(payload)
    malformed["schema_version"] = 1
    with pytest.raises(ValueError):
        integral_from_payload(malformed)
    malformed["operator"].pop("omega")
    with pytest.raises(ValueError, match="schema version 2"):
        integral_from_payload(malformed)
    missing = deepcopy(payload)
    missing["operator"].pop("omega")
    with pytest.raises(ValueError, match="missing"):
        integral_from_payload(missing)


def test_ordinary_coulomb_payload_preserves_v1_contract():
    integral = build_weighted_eri_ir((1, 0, 0, 0))
    payload = integral_to_payload(integral)
    assert payload["schema_version"] == 1
    assert "omega" not in payload["operator"]
    assert integral_from_payload(payload) == integral


@pytest.mark.parametrize("family", ["long_range", "short_range"])
@pytest.mark.parametrize("backend", ["graph", "cpu", "cuda"])
@pytest.mark.parametrize(
    "angular", [(1, 0, 0, 0), (2, 1, 0, 1), (3, 0, 0, 0), (3, 3, 3, 3)]
)
def test_generated_range_values_and_all_center_derivatives_against_libcint(
    family, angular, backend, tmp_path
):
    """Compare each radial part independently, with arbitrary signed weights.

    Libcint Cartesian normalization is removed using its overlap diagonal and
    the analytic norm of each unnormalized monomial Gaussian. This avoids
    assuming that d/f Cartesian components have individually unit norms.
    """
    pyscf = pytest.importorskip("pyscf")
    exponents = (0.6, 0.8, 1.1, 0.9)
    centers = np.array(
        [
            (0.13, -0.31, 0.24),
            (-0.43, 0.27, 0.51),
            (0.68, -0.14, -0.22),
            (-0.21, 0.48, -0.63),
        ]
    )
    radial = CoulombKernel(family, 0.63)
    integral = build_weighted_eri_ir(angular, operator=four_center_eri_operator(radial))
    count = math.prod(integral.signature.component_shape)
    subset = (0, 1, 15, 99, count - 1) if count > 64 else None
    kernel = build_weighted_eri_kernel(integral, subset)
    weights = np.zeros(count)
    weights[list(kernel.component_indices)] = np.random.default_rng(166).normal(
        size=len(kernel.component_indices)
    )

    def variables(positions):
        values = primitive_variables(
            exponents, positions, integral.maximum_coulomb_order
        )
        argument = values["rho"] * sum(
            values[f"difference_{axis}"] ** 2 for axis in "xyz"
        )
        values.update(
            {
                f"boys_{n}": value
                for n, value in enumerate(
                    reference_moments(
                        integral.maximum_coulomb_order, argument, values["rho"], radial
                    )
                )
            }
        )
        values.update(
            {f"component_weight_{i}": weights[i] for i in kernel.component_indices}
        )
        return values

    if backend == "graph":

        def evaluate(positions):
            inputs = variables(positions)
            return kernel.graph.evaluate(kernel.value, inputs), np.array(
                [
                    [kernel.graph.evaluate(entry, inputs) for entry in row]
                    for row in kernel.gradients
                ]
            )
    else:
        native = native_weighted_evaluator(kernel, tmp_path, backend)

        def evaluate(positions):
            return native(exponents, positions, weights[list(kernel.component_indices)])

    actual, derivative = evaluate(centers)
    labels = [f"H{i}" for i in range(4)]
    mol = pyscf.gto.M(
        atom=list(zip(labels, centers)),
        basis={
            label: [[l, [exponent, 1.0]]]
            for label, l, exponent in zip(labels, angular, exponents)
        },
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    normalization = []
    from vibeqc_compiler.integral.shell_spec import cartesian_components

    for slot, (l, exponent) in enumerate(zip(angular, exponents)):
        norm = []
        for component in cartesian_components(l):
            powers = tuple(component.count(axis) for axis in "xyz")
            norm.append(
                (math.pi / (2 * exponent)) ** 1.5
                * math.prod(math.prod(range(1, 2 * power, 2)) for power in powers)
                / (4 * exponent) ** l
            )
        normalization.append(
            np.sqrt(np.diag(mol.intor_by_shell("int1e_ovlp", (slot, slot))) / norm)
        )
    factors = np.einsum("i,j,k,l->ijkl", *normalization)
    shaped_weights = weights.reshape(integral.signature.component_shape)
    with mol.with_range_coulomb(
        radial.omega if family == "long_range" else -radial.omega
    ):
        raw_reference = mol.intor_by_shell("int2e", (0, 1, 2, 3)) / factors
        reference = np.sum(shaped_weights * raw_reference)
        gradients = []
        raw_gradients = []
        for permutation in ((0, 1, 2, 3), (1, 0, 2, 3), (2, 3, 0, 1), (3, 2, 0, 1)):
            block = mol.intor_by_shell("int2e_ip1", permutation)
            block = np.transpose(
                block, (0, *(permutation.index(i) + 1 for i in range(4)))
            )
            # int2e_ip1 differentiates the electron coordinate of the first
            # Gaussian; its nuclear-center derivative has the opposite sign.
            raw_gradients.append(-block / factors)
            gradients.append(
                np.sum(raw_gradients[-1] * shaped_weights, axis=(1, 2, 3, 4))
            )
    np.testing.assert_allclose(actual, reference, atol=1e-11, rtol=1e-10)
    np.testing.assert_allclose(derivative, gradients, atol=1e-11, rtol=1e-10)
    np.testing.assert_allclose(derivative.sum(axis=0), 0, atol=1e-12)
    if backend != "graph":
        # A weighted sum alone can hide compensating component errors. Unit
        # cotangents expose raw diagnostics through the exact same callable.
        raw_gradients = np.asarray(raw_gradients).reshape(4, 3, -1)
        for packed, component in enumerate(kernel.component_indices):
            unit = np.zeros(len(kernel.component_indices))
            unit[packed] = 1
            raw, response = native(exponents, centers, unit)
            np.testing.assert_allclose(
                raw, raw_reference.flat[component], atol=1e-11, rtol=1e-10
            )
            np.testing.assert_allclose(
                response, raw_gradients[:, :, component], atol=1e-11, rtol=1e-10
            )
    errors = []
    for step in (0.003, 0.001, 0.0003):
        plus, minus = centers.copy(), centers.copy()
        plus[0, 2] += step
        minus[0, 2] -= step
        finite = (evaluate(plus)[0] - evaluate(minus)[0]) / (2 * step)
        errors.append(abs(finite - derivative[0, 2]))
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 2e-7


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
@pytest.mark.parametrize("family", ["long_range", "short_range"])
@pytest.mark.parametrize("name", ["psss", "dpsp", "fsss"])
def test_contracted_range_stream_and_public_basis_against_libcint(
    backend, family, name, tmp_path
):
    """Use the established primitive normalization and public cotangent pullback."""
    pytest.importorskip("pyscf")
    from vibeqc_compiler.integral.blocks import WeightTile
    from vibeqc_compiler.integral.weighted_eri_inputs import (
        PRIMITIVE_RANGE_RECORD,
        prepare_weighted_eri_stream,
    )

    from tools.validate_weighted_eri import make_fixture

    radial = CoulombKernel(family, 0.63)
    integral = build_weighted_eri_ir(
        tuple("spdf".index(c) for c in name), operator=four_center_eri_operator(radial)
    )
    kernel = build_weighted_eri_kernel(integral)
    native = native_weighted_evaluator(kernel, tmp_path, backend)
    indices = {
        tuple(
            component.count(axis) for component in components for axis in "xyz"
        ): packed
        for packed, components in enumerate(kernel.spec.components)
    }
    for variant in ("cartesian", "spherical", "coincident", "long"):
        fixture = make_fixture(name, variant, coulomb_kernel=radial)
        stream = prepare_weighted_eri_stream(
            fixture["request"],
            fixture["primitives"],
            fixture["centers"],
            lambda descriptor, _, fixture=fixture: WeightTile(
                descriptor.layout, fixture["weights"].ravel()
            ),
            projections=fixture["projections"],
        )
        result = np.zeros(13)
        packed = np.zeros(len(kernel.component_indices))
        previous = None

        def accumulate(key, packed=packed, result=result):
            value, gradient = native(key[:4], np.array(key[4:]).reshape(4, 3), packed)
            result[:] += np.r_[value, gradient.ravel()]

        # The validation driver groups adjacent equal primitive geometries
        # to exercise the generated shared-weight DAG, after the production
        # stream has supplied all normalization and projection coefficients.
        for blob in stream.records():
            fields = PRIMITIVE_RANGE_RECORD.unpack(blob)
            assert fields[0] >> 8 == (1 if family == "long_range" else 2)
            assert fields[-2:] == (radial.omega, 2)
            key = fields[14:30]
            if previous is not None and key != previous:
                accumulate(previous)
                packed[:] = 0
            previous = key
            if fields[0] & 255:
                assert name == "psss"
                packed[:] += fields[30:33]
            else:
                packed[indices[fields[2:14]]] += fields[30]
        if previous is not None:
            accumulate(previous)
        np.testing.assert_allclose(
            result,
            fixture["reference"],
            atol=1e-11,
            rtol=1e-10,
            err_msg=f"{name}/{variant}/{family}/{backend}",
        )
        np.testing.assert_allclose(result[1:].reshape(4, 3).sum(axis=0), 0, atol=1e-12)
