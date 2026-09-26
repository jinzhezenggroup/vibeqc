# CG08 CPU tensor equation evidence

[`tensor-ir-cpu.json`](tensor-ir-cpu.json) contains four `vibeqc.validation`
records from clean source commit `b983e482f995140423eb44359d919e4994bf622f`.
The NumPy CPU interpreter was checked against separately written coordinate
loops for matrix multiplication, an MP2-like energy fragment, a CC-like virtual
residual term, and a restricted spatial pair update.

Each record checks the original equation, JSON replay, every conservative
rewrite, and optimized replay at `atol=1e-11, rtol=1e-10`. The largest absolute
error was `3.469446951953614e-18`. Two independent runner invocations produced
identical records, including equation, input, IR, and source hashes.

```bash
python tools/tensor_ir_examples.py --output /tmp/tensor-ir-cpu.json \
  --equations-dir /tmp/tensor-ir-equations
```

The artifact records seed 145, Python/NumPy versions, the exact source-file
manifest, and numerical errors. The optional equation exports contain input
arrays, loop references, replayable programs, and signed packing maps.

These are CPU tensor-fragment checks. Compilation, GPU, molecular endpoints,
and production selection are explicitly `not-run`. Logical retained bytes are
an interpreter accounting estimate; allocated/peak memory and performance are
unmeasured. No complete MP2/CCSD implementation or speedup is claimed.
