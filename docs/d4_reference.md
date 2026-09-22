# D4 numerical qualification and EEQ integration

`src/dft/dispersion/d4_reference.hpp` retains the bounded CPU/device fixed-charge
D4 baseline migrated from xTBloom. `src/dft/dispersion/d4_eeq.hpp` adds the
generic molecular EEQ2019 charge provider, its analytic coordinate response,
and a complete CPU/CUDA D4 gradient qualification endpoint. The handwritten
complete composition in that header is qualification-only. Production is owned
by `D4Plan`, the compiler-generated derivative lowering and the bounded runtime.

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

## Production ownership

Production never calls `evaluate_complete_d4_eeq*`. The compiler module
`vibeqc_compiler.method.d4_derivative` generates the only production
composition of the EEQ response with fixed-charge D4 partial derivatives.
EEQ2019 and fixed-charge D4 remain separately qualified custom scientific
primitives; the generated VJP computes
`(partial E/partial R)_q + (dq/dR)^T (partial E/partial q)` and publishes only
after all stages succeed.

`D4Plan` owns immutable model/profile identity, atomic numbers and total
charges, bounded host/device resources, prepared coordinates, replay state and
peer-local failure publication. CUDA production uses at most 32 ragged EEQ
workers. Each worker reuses one worst-system EEQ workspace plus one `dq/dR`
scratch. Fixed-charge pair/ATM work is delegated to the block-cooperative
scheduler in `d4_cuda.cu`, whose workspace is linear in total atoms. Unchanged
replays reuse resident coordinates; changed geometry explicitly uploads the
packed coordinate buffer. Diagnostics expose exact workspace slots, H2D bytes,
kernel launches and host/device capacity bounds.

The public named endpoint `pbe-d4-rks` executes native PBE RKS plus
D4(BJ-EEQ-ATM). Native energy remains exact-once owned by the prepared KS/D4
composition. In the bounded qualified force domain, the stationary derivative
projects that composite state onto its electronic PBE MethodIR, evaluates the
existing PBE stationary force, and combines it with the existing D4 analytic
gradient as `F_total = F_PBE - dE_D4/dR`. The auxiliary D4 force owner publishes
the correction component/gradient but never adds its energy a second time.
Unsupported PBE-D4/ECP force combinations remain fail-closed.
`D4CorrectionBatch` also continues to expose standalone D4 energy, analytic
gradient and EEQ charges.

There is no runtime dependency on xTBloom or an external dftd4 executable.
Standard EEQ uses `ga/gc=3/2`; r2SCAN-3c uses the separate `2/1` profile. GFN2
remains qualification compatibility data, not a generic DFT-D4 default.

## Memory and execution boundary

All molecular qualification paths are bounded to 256 atoms. The fixed-charge
D4 baseline uses `27*N` doubles and materializes no pair/triple tensors. The
EEQ provider uses a dense `(N+1)^2` constrained matrix and a bounded
`3*N*(N+1)` response block; it is a correctness implementation, not a promoted
large-system schedule. Its asymptotic linear solves are cubic.

The same bounded EEQ charge/response mathematics is qualified on CPU and CUDA.
Production CUDA runs multiple ragged EEQ systems concurrently through bounded
workers, delegates fixed-charge work to the 256-lane block-cooperative
scheduler, then applies the generated response VJP. Failed and inactive
members publish zero outputs without poisoning successful peers. Production
tests cover independent fixtures, multi-step finite differences, changed and
unchanged replay, a 4100-member ragged fleet and Compute Sanitizer. PBC and
Hessian execution remain unsupported.

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

Qualification CTest targets are `vibeqc_d4_reference_tests`,
`vibeqc_d4_eeq_tests` and their CUDA variants, plus
`vibeqc_d4_schedule_cuda_tests`. Production additionally runs
`vibeqc_d4_production_tests`, `vibeqc_d4_ragged_tests` and their CUDA
invocations. `benchmarks/d4_production_gate.py` records cold, warm, changed
geometry and ragged endpoint timings together with resource diagnostics.
Regenerate the independent EEQ
fixtures with:

```sh
python tools/oracle/generate_d4_eeq_reference.py --prefix /path/to/dftd4-4.2.0-prefix
```

Source revisions, blob hashes, licenses, oracle hashes and generated-table
digests are retained in the D4 manifests.
