# D3(BJ) migration and qualification boundary

The compiler can represent a geometry-only `DispersionCorrectionPrimitive`
containing an immutable `D3Spec`. The repository also contains an executable,
xTBloom-derived native CPU/CUDA **qualification baseline**. Neither capability
registers a public DFT+D3 calculator or completes issue #492.

## Model semantics

The initial model is nonperiodic, real FP64, two-body D3(BJ), `s9=0`.
`D3Spec` records explicit `s6/s8/a1/a2`, source-data SHA-256 identities,
coordination and pair cutoffs, and the pair-switch width. It rejects nonzero ATM,
zero damping, unsupported versions, invalid coefficients and missing data
identity. CN uses the pinned exponential convention (steepness 16) and Gaussian
reference weights (factor 4). Changing those equations requires a new version.

`None` cutoffs mean no cutoff. The separate `gfn1_compatibility()` helper specifies
the GFN1 damping parameters, 25-bohr CN cutoff, 50-bohr pair cutoff and 0.05-bohr
switch. These are not silently applied to arbitrary DFT parameter sets. Hard
cutoff models are only piecewise differentiable; gradient tests do not claim
smoothness at a discontinuous CN or hard pair cutoff.

Energy is in Hartree; coordinates are in bohr. The baseline returns
**gradient = dE/dR**, including explicit distance and reference-CN interpolation
response. Forces have the opposite sign. The unrelated GFN1 halogen correction,
Hamiltonian, SCC state and D4 terms are not imported.

## Method composition

```python
from dataclasses import replace
from vibeqc_compiler.method import METHOD_CATALOG, resolve_method
from tools.vibeqc_d3.reference import gfn1_compatibility

# An explicit composition example, not a validated PBE parameterization:
spec = replace(METHOD_CATALOG["PBE"], dispersion=gfn1_compatibility())
graph = resolve_method(spec)
assert graph.primitives[-1].kind == "dispersion_correction"
```

The graph keeps table/parameter/cutoff identity separate from descriptive names.
Existing pure LDA/PBE/PBE0 graph payloads are unchanged. The native LDA/PBE KS
boundary still rejects graphs with correction nodes; it cannot silently execute
the semilocal part while omitting D3. Generic named DFT+D3 public manifests and
current-state/provider binding remain separate work.

## Reproduction

No xTBloom or simple-dftd3 runtime dependency is added. From a source checkout:

```sh
PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1   python -m pytest tests/python/test_d3_reference.py   tests/python/test_dft_method_ir.py tests/python/test_compiler_structure.py -q
```

This explicitly compiles a tiny C++ diagnostic library in the test temporary
directory. To exercise CUDA, build the diagnostic library with
`tools.vibeqc_d3.reference.build_reference(path, backend="cuda", nvcc=..., cuda_arch="sm_120")`,
then run the same suite on an allocated GPU with `VIBEQC_D3_LIBRARY` set to its
absolute path and `VIBEQC_D3_EXPECT_BACKEND=cuda`. CUDA compilation is separate
from source import/generation. The test wrapper never falls back to CPU.

`tools/vibeqc_d3/generate_goldens.py` requires **dftd3==1.4.0** only when
regenerating the nine tiny independent fixtures. It explicitly disables ATM,
including for the upstream PBE/PBE0 parameter entries whose default `s9` is one.
The tests also use multi-step coordinate finite differences, mixed-element
permutations, translation/rotation covariance, cutoff switching, parameter
scaling and invalid-input controls.

## Resource and production boundaries

The baseline is synchronous and single-molecule, with a serial GPU worker. It
expands `[npair,49]` reference data, caps fixtures at 512 atoms and preflights a
native logical-allocation budget. That budget excludes Python/JSON host overhead
and is not a measured process/device peak. It is an O(N^2) correctness baseline,
not a performance promotion or the final memory-bounded production design.

Native state/stream ownership, ragged batching, generated production lowering,
pair-parallel scheduling, compact shared tables, ATM, zero damping and public
DFT energy/force registration remain open. CPU and CUDA intentionally share the
migrated arithmetic; agreement between them is not an independent oracle.

See [data provenance](../external/xtbloom-d3/README.md) and the
[architecture decision](../.agents/notes/implemented/architecture/2026-09-19-d3-xtbloom-baseline.md).
