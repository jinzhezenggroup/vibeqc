# CG01 protocol evidence

These records exercise the shared protocol at implementation revision
`54b7a892357ff9a23d94c877e2c6a07094c217e7`. They are numerical/runner controls;
performance and production promotion remain `not-run`. Workspace dirtiness is
retained in metadata, alongside exact scientific-source and native-library
hashes. Both native builds record `VIBEQC_CUDA_FAST_COMPILE=OFF`.

| Artifact | Executed scope | Result |
| --- | --- | --- |
| [CPU HF](hf-cpu.json) | H2, five A/B samples per available workload; three finite-difference steps | Energy error `2.04e-14 Eh`, force max error `5.41e-15 Eh/bohr` |
| [CUDA HF](hf-cuda-sm120.json) | Same H2 control on the Slurm-assigned RTX 5090, actual backend `cuda` | Energy error `1.95e-14 Eh`, force max error `5.31e-15 Eh/bohr` |
| [CUDA compilation](compile-sm120.json) | Deterministic generated SSSS source, NVCC 12.9, `-O3 -arch=sm_120`; four PTXAS resource rows | Compiled successfully; no GPU numerical or production claim |
| [Class numerical tier](gpu-not-run.json) | Full #135 class-isolated f-shell matrix | Explicitly not scheduled by this small protocol control |
| [Capability table](capabilities.json) | Projection of the existing 55-class catalog, including all 34 f classes | Six distinct stages; FPPS selection explicitly provisional |

Each HF artifact contains 40 raw synchronized samples across cold native plan
construction, unchanged-geometry replay, changed geometry and energy-plus-force
execution. The compared implementations are identical native HF controls.
Energy-only execution, allocator peaks, compilation cost and complete solver
iteration histories are explicitly unavailable in this API. Changed geometry
is additionally checked against a native one-shot call; that is a replay
consistency check rather than an independent displaced libcint reference.

The CPU finite-difference errors at 0.01, 0.003 and 0.001 Bohr are respectively
`1.66e-5`, `1.49e-6` and `1.66e-7 Eh/bohr`. The CUDA curve agrees to roundoff.
All steps are reported, including the larger-step errors above the analytic
gradient gate. No single favorable finite-difference step establishes promotion.

The [reference stability artifact](../../../tests/reference_data/validation/stability.json)
records two traceable generations of all 12 references with identical data
hashes. The independent NH3 `(T)` correction is `-1.122922812723688e-4 Eh`.

Reproduce the HF records with the commands in the
[protocol documentation](../../../docs/validation.md). To reproduce the small
compile smoke without introducing another CUDA compiler driver, emit source
through the existing generator, then use the shared command wrapper:

```bash
python - <<'PY'
from pathlib import Path
from tools.vibeqc_codegen.benchmark import emit_shell_class_resource_cuda
from tools.vibeqc_codegen.fused_schedule import build_fused_shell_plan
from tools.vibeqc_codegen.shell_spec import FUSED_SHELL_SPEC_BY_NAME
spec = FUSED_SHELL_SPEC_BY_NAME['ssss']
plan = build_fused_shell_plan(spec)
source = emit_shell_class_resource_cuda(spec, plan)
assert source == emit_shell_class_resource_cuda(spec, plan)
Path('/tmp/vibeqc-138-ssss.cu').write_text(source)
PY
CUDA_PATH=/group/software/cuda-12.9.1 python benchmarks/validation_gate.py run \
  --tier cuda-compile --subject ssss-sm120-resource-smoke --output /tmp/compile.json -- \
  /group/software/cuda-12.9.1/bin/nvcc -std=c++17 -arch=sm_120 -O3 -Xptxas=-v \
  -c /tmp/vibeqc-138-ssss.cu -o /tmp/vibeqc-138-ssss.o
```

The committed compilation record additionally registers the emitted-source,
equation, serialized IR, schedule and object hashes, and resources parsed with
the existing `parse_ptxas_resources` helper. Its complete IR and schedule are
saved under `settings`; the compiler command and raw PTXAS diagnostics are
retained. These measurements do not alter the historical capability catalog.
