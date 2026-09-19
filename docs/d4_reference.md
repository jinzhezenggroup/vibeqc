# D4 numerical qualification and EEQ integration

`src/dft/dispersion/d4_reference.hpp` retains the bounded CPU/device fixed-charge
D4 baseline migrated from xTBloom. `src/dft/dispersion/d4_eeq.hpp` adds the
generic molecular EEQ2019 charge provider, its analytic coordinate response,
and a complete CPU/CUDA D4 gradient qualification endpoint. Neither file is a
public Calculator registration or a performance-promoted production scheduler.

## Scientific contract

The fixed-charge evaluator accepts explicit D4 reference tables, damping/zeta
parameters, atomic positions in bohr and independent partial charges. It returns
two-body and zero-charge-reference ATM energies, `(partial E/partial R)_q`, and
`partial E/partial q`. GFN2 compatibility remains explicit and separate.

The EEQ provider reproduces the pinned multicharge EEQ2019 equations: the 25-bohr
error-function coordination number, CN saturation at 8, element-specific
`chi/eta/kcnchi/radius`, the constrained Coulomb matrix, and the complete
`dq/dR` response from the differentiated linear system. The complete endpoint
then evaluates

```text
dE/dR = (partial E/partial R)_q + (dq/dR)^T (partial E/partial q)
force = -dE/dR
```

and publishes outputs only after the charge and response solves succeed.

## Reference models and r2SCAN-3c

`d4_eeq_data.hpp` contains independently generated EEQ reference charges and
polarizability/C6 tables. Standard D4 uses the pinned `ga=3, gc=2` profile.
Canonical r2SCAN-3c uses a separate table generated with `ga=2, gc=1`; mixing
a profile and the wrong zeta parameters fails explicitly.

`r2scan3c_d4_parameters()` and the Python `r2scan3c_d4_eeq()` MethodIR spec pin
`s6=1, s8=0, s9=2, a1=0.42, a2=5.65`, the DFT-D4 4.2.0 molecular default
cutoffs `CN=30`, `two-body=60`, `ATM=40` bohr, and the EEQ charge-CN cutoff
`25` bohr. This is the D4 component needed by #172; it does not by itself add
the electronic r2SCAN functional, the r2SCAN-3c basis, or gCP.

## Validation

Independent fixtures are generated with the DFT-D4 4.2.0 executable, not with
VibeQC mathematics. They cover r2SCAN-3c water, an asymmetric PBE molecule and
a charged Zn-ammonia PBE case. Tests compare EEQ charges, complete energies and
Cartesian gradients. Separate multi-step finite differences validate `dq/dR`
and the complete chain-rule gradient; charge and response conservation are also
checked.

The existing GFN2 fixed-charge CPU/CUDA oracle suite remains unchanged and
continues to qualify the shared pair/ATM mathematics and actual device path.

## Memory and execution boundary

All molecular qualification paths are bounded to 256 atoms. The fixed-charge
D4 baseline uses `27*N` doubles and materializes no pair/triple tensors. The
EEQ provider uses a dense `(N+1)^2` constrained matrix and a bounded
`3*N*(N+1)` response block; it is a correctness implementation, not a promoted
large-system schedule. Its asymptotic linear solves are cubic.

The same bounded EEQ charge/response mathematics is qualified on CPU and CUDA.
The CUDA test uploads immutable EEQ/D4 tables, evaluates both standard and
r2SCAN-3c profiles on a real device, and covers ragged members plus peer-local
failure. The one-worker-per-molecule route is a correctness baseline; a
production GPU scheduler, generated derivative lowering and performance
qualification remain open under #493. There is no PBC, Hessian, public SCF
integration, or complete r2SCAN-3c method registration in this slice.

## Reproduction

Regenerate the retained GFN2 compatibility table:

```sh
python tools/parameters/generate_d4.py \
  --source-git-dir /path/to/dftd4/.git \
  --revision 6e1f59c3f39d919a2dbef0601d2576727c8b30e8 \
  --output-dir src/dft/dispersion
```

Regenerate the EEQ tables from pinned dftd4, multicharge and mctc-lib sources:

```sh
python tools/parameters/generate_d4_eeq.py \
  --dftd4-git-dir /path/to/dftd4/.git \
  --multicharge-git-dir /path/to/multicharge/.git \
  --mctc-git-dir /path/to/mctc-lib/.git \
  --output-dir src/dft/dispersion
```

CTest targets are `vibeqc_d4_reference_tests`, `vibeqc_d4_eeq_tests`, and on
CUDA builds `vibeqc_d4_reference_cuda_tests` plus `vibeqc_d4_eeq_cuda_tests`.
Regenerate the independent EEQ
fixtures with:

```sh
python tools/oracle/generate_d4_eeq_reference.py --prefix /path/to/dftd4-4.2.0-prefix
```

Source revisions, blob hashes, licenses, oracle hashes and generated-table
digests are retained in the D4 manifests.
