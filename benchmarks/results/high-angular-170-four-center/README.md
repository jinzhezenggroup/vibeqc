# BASIS02 generic g four-center qualification

This snapshot qualifies the generic full-range four-center g-shell bounded
component path added for #170. It is intentionally separate from the historical
initial g-shell snapshot in `../high-angular-170/`.

Source base: `9d6d44236e4a09205afbdd1cbcc6f43b8014084d`
Measured implementation commit: `1c551ddb46bdf377fcae573eb5e9ca746af0d081`
Final synchronized PR candidate: `af5313627c178ab4a1cd791093dff8dc6e5f6c92`.

The final candidate was not re-executed on a GPU. A restart of the dedicated
CUDA-12.4 4090 notebook and fresh CUDA-13.2 4090/H200 notebook requests were all
left unschedulable by the platform under the available node-memory/priority
constraints, and all pending requests were stopped. The existing RTX4090
measurements below remain directly applicable to the scientific change because
the three GPU-relevant source/test files are byte-identical between measured
commit `1c551ddb` and final candidate `af531362`:

- `bounded_component.py`: `9aa34bc4689a0391376c49e5f45102c12fb0d495a9e6fadd6301d57fee14cb48`
- `shell_class.py`: `2d32caa12b457af45c44a45c9eb1fa85ad640a3529ff6cb5398a6794af3e4239`
- `test_high_angular.py`: `63a949350c23b481f7de05dbfb63cb897ad66b13a2a7dcc5f9070967ea28248a`

`git diff --exit-code 1c551ddb..af531362 -- <those three paths>` passed. This
source-equivalence statement is intentionally not labeled as a final-head GPU
rerun.

## CPU/reference gates

On the qz CPU notebook, Python 3.11.16 and PySCF/libcint 2.14.0 were used with
one BLAS/OpenMP thread. A fresh Release CPU `libvibeqc` was built from the same
source.

- focused generic g-s-s-s capability + independent libcint gate: 2 passed;
- complete `tests/python/test_high_angular.py` with molecular gates enabled:
  38 passed, 6 CUDA-only skipped;
- loaded orbital-g HeH+ RHF maximum force errors: 6.1446e-11 Hartree/Bohr
  (Cartesian) and 7.3039e-12 Hartree/Bohr (real spherical);
- g-auxiliary DF-RHF maximum force errors: 1.9163e-13 Hartree/Bohr
  (Cartesian) and 1.3667e-13 Hartree/Bohr (real spherical);
- CPU g-s-s-s `xxxx` scalar compilation: 0.1998 s, 26,987 B source,
  19,824 B shared object.

The g-s-s-s oracle compares the raw primitive value and all 12 nuclear-center
first derivatives at two geometries to independent libcint blocks. It also checks
translation invariance and three arbitrary signed derivative weights.

## Allocated CUDA gate

The existing Inspire notebook `issue-0170-higher-l-4090` on node
`qb-prod-4090-gpu117` supplied one NVIDIA GeForce RTX 4090 (48 GB), UUID
`GPU-21fdb213-4048-e5f6-3bb2-ea28f46230c5`, compute capability 8.9 and driver
550.163.01. The compiler was CUDA 12.9.86 targeting `sm_89`; Python was 3.11.16
and PySCF was 2.14.0. The notebook was stopped after qualification.

Six selected CUDA tests passed: existing g overlap, kinetic, nuclear-attraction,
DF metric and three-center arithmetic gates plus the new full-range g-s-s-s
four-center value/first-derivative gate. For the new component:

- compile: 2.0742 s;
- generated source: 50,160 B;
- shared object: 1,210,768 B;
- PTXAS: 128 registers, 48 B stack frame, zero spill stores/loads;
- `cuobjdump --dump-resource-usage`: 0 B shared memory and 0 B local memory.

This is real-device numerical/resource qualification, not compile-only evidence.
The test process executed the generated CUDA kernel and compared the value and all
12 center derivatives against independent libcint at changed geometries.
It is not a complete CUDA molecular endpoint and therefore does not promote g
into native CUDA HF/DF, the 55-class catalog, production manifests, or default
binaries. The complete declared higher-l molecular method endpoint remains the
independently qualified CPU reference RHF/DF-RHF path.

See `results.json` for the machine-readable snapshot and
`../../../docs/high_angular_momentum.md` for the current capability matrix.
