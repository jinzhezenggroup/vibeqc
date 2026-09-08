# TensorIR: typed equations and a CPU reference interpreter

`tools/vibeqc_tensor/` implements the CG08 foundation in issue #145. It describes
real tensor equations, validates them before execution, and replays them with
NumPy on CPU. It includes independent loop references for a matrix product, a
spin-orbital MP2-like energy fragment, one virtual Fock contribution to a CC-like
residual, and a restricted spatial pair-symmetrized update. These fragments do
not implement a complete MP2 or CCSD method.

`IntegralIR` owns integral operators, shell/center identities, nuclear
derivatives, and bounded integral providers. `TensorIR` owns tensor index
populations and algebra over supplied arrays. Neither inherits from the other;
TensorIR does not import `ShellClassSpec` or require CUDA. Physical strides,
device placement and contraction planning live in the separate
[prepared FP64 CUDA executor](tensor_cuda.md) (#146). Derivative rules remain
subsequent work (#151). The table below describes the original CG08 boundary;
the CUDA document records the subsequent execution capabilities.

| Stage | CG08 capability |
| --- | --- |
| Mathematical representation and legality | Implemented for the primitives below |
| CPU interpretation and debug intermediates | Implemented and checked against explicit loops |
| JSON replay and logical equation hashing | Implemented, versioned, fail closed |
| Generated source | Stable node-name/type/primitive contract only; no source emitter |
| CUDA compilation and GPU numerical validation | Not implemented by this issue |
| Complete molecular method/endpoint/production selection | Not implemented by these fragments |

## Types and index conventions

`IndexSpace(name, kind, size, spin=None)` names an explicitly sized population.
Kinds are `occupied`, `virtual`, `ao`, `auxiliary`, `batch`, and `spin`;
optional spin labels are `alpha` and `beta`. Distinct populations remain distinct
even when their extents match. Space names must have one definition throughout
a program. Nuclear centers and atom identities remain on the integral side.

`Index(name, space, start=0, stop=None, selection=None)` selects a half-open
range. Slice coordinates are local to that range. A gather retains its ordered
global coordinates, including repeats. Equating differently ordered gathers or
different blocks merely because their shapes agree is illegal. Einstein labels
are local to each contraction; reusing one label for different populations,
spins, or selected ranges is an error. Tensor axis names are notation and do
not alter a logical hash.

`TensorSpec` records ordered indices, real `float64` or `float32` dtype, declared
symmetries, orbital representation, parameter role, and differentiability.
Roles are `input`, `parameter`, `constant`, and `intermediate`. Constants cannot
be differentiable. Result differentiability propagates from operands; this
marks future AD inputs without claiming implemented derivatives.

Representations are `general`, `restricted_spatial`, and `spin_orbital`.
Operands of an operation must agree on representation and dtype. In particular,
restricted spatial amplitudes obey the simultaneous exchange

```text
t[i,j,a,b] = t[j,i,b,a]             Symmetry((1,0,3,2), +1)
```

This does not assert separate occupied or virtual antisymmetry. Spin-orbital
antisymmetries require explicit declarations:

```text
t[i,j,a,b] = -t[j,i,a,b]           Symmetry((1,0,2,3), -1)
t[i,j,a,b] = -t[i,j,b,a]           Symmetry((0,1,3,2), -1)
```

The interpreter checks declared input symmetries. Transpose transports them;
addition retains declarations shared by all operands. Other operations drop
unproved declarations. Equal shapes cannot justify extra symmetry.

## Construct and replay an equation

Run this from the repository root:

```python
import numpy as np
from tools.vibeqc_tensor import (
    Index, IndexSpace, Program, TensorSpec, einsum, execute, input_tensor,
)

ao = IndexSpace("ao", "ao", 2)
aux = IndexSpace("aux", "auxiliary", 3)
a = input_tensor("a", TensorSpec(
    (Index("p", ao), Index("P", aux)), role="input"))
b = input_tensor("b", TensorSpec(
    (Index("P", aux), Index("q", ao)), role="parameter", differentiable=True))
program = Program({"c": einsum("pP,Pq->pq", a, b)},
                  provenance={"equation_version": 1})
replayed = Program.loads(program.dumps())
result = execute(replayed, {"a": np.ones((2, 3)), "b": np.ones((3, 2))}, debug=True)
assert np.array_equal(result.outputs["c"], np.full((2, 2), 3.0))
assert replayed.logical_hash == program.logical_hash
```

`Execution.intermediates` maps stable, source-safe `Program.debug_names` to
snapshots of every executed node. `Program.nodes` retains explicit definitions;
`live_nodes` contains output-reachable definitions. Definitions may be retained
for inspecting the original mathematical DAG even after producing an optimized
program.

All values follow immutable SSA semantics. There are no in-place destinations,
so a transpose/slice/broadcast view cannot be overwritten by another node.
Inputs may be noncontiguous, negative-stride, read-only, or share memory with
other inputs. Each returned output/debug array is detached from all inputs and
other returned arrays. Extra unused feeds are allowed when comparing original
and optimized programs. Missing feeds, wrong dtypes/shapes, nonfinite values,
division by zero, and arithmetic overflow fail explicitly.

## Primitive contracts

| Factory | Semantics and constraints |
| --- | --- |
| `constant(values, spec=None)` | Exact scalar or flattened C-order literals; default is an FP64 scalar |
| `add(*values, coefficients=...)` | Ordered rational-scaled sum; equal domains, no implicit broadcast |
| `multiply(a,b)`, `divide(a,b)` | Elementwise operations on equal domains |
| `einsum("...->...", *values, coefficient=...)` | Explicit-output alphabetic labels; traces/repeated input labels and scalar terms supported; literal ellipses unsupported |
| `transpose(value, axes)` | Full permutation of logical axes |
| `reshape(value, indices)` | Explicit C-order logical reshape with equal element count; may require a physical copy; does not transform an orbital basis |
| `slice_tensor(value, ranges)` | One nonnegative unit-step half-open local range per axis |
| `gather(value, axis, positions)` | Static validated positions, including repeated and reordered positions |
| `reduce_sum(value, axes)` | Sum specified axes; full reduction yields a rank-zero scalar |
| `broadcast(value, indices, axes)` | Insert new axes using an explicit input-to-output map; existing axis domains remain unchanged |

An existing singleton cannot silently become a different population. Reduce
away that axis before explicitly broadcasting it. Empty tensors and zero-length
contractions are valid; empty sums are zero. Dimension/byte products use a
checked signed-64-bit contract before allocation. `execute(max_bytes=...)`
defaults to 256 MiB and bounds logical retained arrays plus returned snapshots.
It does not claim a bound on NumPy internal scratch, Python objects, or process
RSS. CUDA allocation planning has its own explicit [budget scope](tensor_cuda.md).

Compile-time coefficients accept integers, `Fraction`, or exact rational
strings such as `"1/4"`. Float coefficients are rejected. Serialization stores
reduced numerator/positive-denominator pairs; conversion happens in the chosen
real dtype during interpretation. No complex values, conjugation, arbitrary
expression evaluation, SCF/CC loops, or mutable scatter operations are supported.

`PRIMITIVES` declares future differentiable-operand and accumulation contracts.
Shared-operand contributions accumulate, and repeated gathers will require
scatter-add in a VJP. Shapes live in each typed node; no AD rules or derivative
execution are introduced here.

## Independent/packed amplitudes

`PackedLayout.from_spec(spec, max_elements=...)` enumerates signed orbits of the
declared symmetry generators. The bounded CPU reference defaults to one million
dense elements. `pack` checks symmetry rather than projecting an arbitrary
tensor; `unpack` expands independent coordinates. Antisymmetric fixed points
are structural zeros. Singleton orbits, empty tensors, and all-zero orbits are
supported.

`dense_to_packed`, `signs`, and `representatives` specify the mapping.
`weights` holds orbit multiplicities:

```text
sum_dense unpack(x) * unpack(y) = sum_packed weights * x * y
```

`inner_product(x,y)` implements this metric. For two occupied and three virtual
orbitals, the spatial pair layout has 21 independent coordinates with weights
1 or 2, while separate spin-orbital antisymmetries have 3 coordinates with weight
4. Later VJPs must use these weights, not the unweighted packed dot product.
`to_payload`/`from_payload` preserve and verify the complete map and metric.

## Rewrites and serialization

`rewrite(program, pass_name)` exposes `dead_nodes`, `identity_transposes`,
`exact_cse`, and `scalar_constants` separately. `optimize` applies those passes
and a final duplicate/dead cleanup, returning a new program with source-hash
provenance. It never overwrites the original DAG. CSE includes spaces, ranges,
spins, dtype, symmetry, representation, input identity/role, and differentiability.
No floating-point reassociation, contraction-tree search, or implied symmetry
rewrite is performed.

Constant folding is deliberately limited to scalar rational add/multiply/divide
subgraphs. The replacement must reproduce the original dtype result bit for
bit; cancellation, overflow, signed-zero changes, or invalid division cannot
be hidden by rational simplification. Array constant folding is deferred.

The `vibeqc.tensor` schema and primitive definitions are versioned independently.
Replay validates every node, shape, attribute, dependency, convention, logical
hash, and stable debug name. It rejects unknown versions/primitives, forward
references, duplicate JSON keys, and incompatible declarations. Serialization
contains data only and never uses `eval` or pickle.

Logical identity includes exact factors, conventions, versions, ordered
operations, and output names. Dummy Einstein labels are normalized by first
occurrence; tensor axis spelling, dictionary insertion order, construction
order of independent nodes, dead definitions, and provenance do not alter that
identity. This is structural equation identity, not a proof that differently
associated algebra has the same floating-point value.

## Integral block and external-weight exchange

The adapter consumes `BlockResponse.layout` element strides and logical offsets
to form an array with a `TensorSpec`. Each tensor index names the appropriate AO
or auxiliary population and explicitly selects the block's range. It must also
honor the response status, AO order/normalization, tile identity, and center maps.
The test `test_integral_raw_tile_to_tensor_weights_and_back_has_explicit_order_and_sign`
shows a complete bounded exchange, including padded integral storage:

1. Assemble an overlap raw block for a partial AO tile.
2. Gather its logical elements from the declared physical layout.
3. Compute external weights as a TensorIR equation.
4. Pack the result into a `WeightTile` matching the integral `WeightDescriptor`.
5. Contract first derivatives with `WeightedDerivative`, applying its explicit
   force sign and translation/atom mappings on the integral side.

For example, after validating the descriptor and response, the array boundary is:

```python
raw = np.array([response.values[k] for k in layout.offsets()]).reshape(spec.shape)
weights = execute(weight_program, {"raw": raw}).outputs["weights"]
buffer = np.zeros(layout.storage_elements)
buffer[list(layout.offsets())] = weights.ravel()
tile = WeightTile(layout, buffer)
```

A scalar prefactor already included in TensorIR must not be applied twice in
the weight descriptor. Arbitrary weights need not factor into HF density
products. This boundary allocates only requested tiles and does not construct
a molecular N⁴ integral or derivative tensor.

## Validation and evidence

```bash
python -m pytest tests/python/test_tensor_ir.py tests/python/test_tensor_execution.py \
  tests/python/test_tensor_examples.py -q
python tools/tensor_ir_examples.py --output /tmp/tensor-evidence.json \
  --equations-dir /tmp/tensor-equations
```

The runner uses the existing `vibeqc.validation` schema and controlled FP64
`atol=1e-11, rtol=1e-10` gates from #138. It checks original execution, JSON
replay, each rewrite, and optimized replay against independent loops. Exported
examples include actual inputs, reference values, the program, packing map,
versions, and seed. Evidence records include equation/IR/source hashes,
revision/dirty state, NumPy/Python versions, per-block errors, and explicit
not-run statuses for later stages. The source-file manifest identifies dirty
source bytes as well as committed code. The runner does not issue a performance
pass or present logical byte accounting as measured allocation/peak memory.

The [archived CG08 CPU records](../benchmarks/results/cg08/README.md) report a
maximum absolute loop-reference error of `3.469446951953614e-18` and identical
equation/input hashes across two generations from the clean implementation
commit.
