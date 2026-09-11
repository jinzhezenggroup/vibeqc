# Local basis data and execution contracts

`Calculator` accepts immutable `BasisSet` records, canonical JSON files (`Path`
or a string ending in `.json`), bundled names, and explicit `Shell` sequences.
Loading is offline. Element symbols H–Og identify nuclei independently of the
available basis data; the unchanged bundled STO-3G/def2-SVP/def2-TZVP tables
cover H–Ar. Existing bundled defaults remain Cartesian.

## Import and use

Convert a locally supplied **MolSSI Basis Set Exchange complete JSON 0.1** file:

```bash
python tools/import_basis.py input.bse.json local-basis.json \
  --source 'local BSE export' --source-version '<exact release or git revision>' \
  --license BSD-3-Clause --representation spherical
```

The source, version and redistribution license are explicit declarations. The
importer hashes the source bytes and retains the basis-data version as well.
It never queries BSE or installs an external quantum-chemistry backend.

```python
from pathlib import Path
from vibeqc import Calculator, load_basis, basis_capability

basis = load_basis(Path("local-basis.json"))
atoms = [("O", (0, 0, 0)), ("H", (0, -1.43, 1.1)), ("H", (0, 1.43, 1.1))]
report = basis_capability(basis, atoms, backend="cuda",
                          operator="eri", derivative_order=1)
if report["eligible"]:
    result = Calculator(basis=basis, device="cuda").singlepoint(atoms)
    print(result.energy, result.basis_metadata)
```

A record carries its AO representation. An explicit conflicting
`basis_representation` raises an error. To deliberately reinterpret all shells,
use `dataclasses.replace(basis, representation="cartesian")`; this creates a
new identity. A local auxiliary record retains its own representation in the
HF native system descriptor. The experimental post-HF source ABI currently
requires the same representation for both spaces and rejects a conflict.

## Canonical schema version 1

`BasisSet.to_payload()` serializes the following fields; `BasisSet.write(path)`
adds a SHA-256 checksum over canonical sorted JSON excluding `checksum`.
`load_basis` verifies it before constructing immutable owned records.

| Field | Contract |
| --- | --- |
| `schema`, `schema_version` | `vibeqc.basis`, integer `1` |
| `name`, `elements` | Nonempty name; unique element records sorted by Z |
| `representation` | `cartesian` or `spherical` (real AOs) |
| `exponent_units` | `bohr^-2`; coordinates supplied to calculations are Bohr |
| `normalization` | `normalized-primitives; native-normalized-contractions` |
| `ordering` | `CCA-cartesian; libcint-real-spherical` |
| `provenance` | `source`, `version`, `license`, source SHA-256 `checksum` |
| element fields | `atomic_number`, `nuclear_charge`, `ecp_core_electrons`, `ecp_data`, `shells` |
| shell fields | `angular_momentum`, `exponents`, `coefficients`, `source_group` |

Each coefficient **row is one general contraction**, each column corresponds to
one primitive exponent. Decimal strings are retained without context-dependent
rounding; conversion to FP64 occurs during native expansion. Exact zeros and
small representable coefficients survive. Nonzero values that underflow to
FP64 zero, nonfinite numbers, nonpositive exponents, dimension mismatches and
identically zero contractions are rejected. A contraction that cannot be
normalized at native FP64 precision fails explicitly during native validation.

BSE combined SP records split into distinct s and p records sharing
`source_group`. Single-l general contractions remain matrices. A combined
record must have exactly one row per distinct angular label; ambiguous combined
general layouts are rejected. A mixed Cartesian/spherical source requires an
explicit representation choice. Unknown function types, malformed arrays,
duplicate JSON keys/elements and files larger than 64 MiB are rejected.

Coefficients multiply normalized primitives. The importer never applies radial
normalization. The native constructor applies the existing contraction and
radial primitive normalization once to its owned copy; independently normalized
Cartesian components and libcint real-spherical ordering stay unchanged.
Already radially normalized coefficients need an explicit conversion before
import; relabeling their normalization would change the mathematical basis.

## Nuclei, ECP cores and occupations

`atomic_number` identifies the element; `nuclear_charge` describes its nucleus.
An ECP additionally removes `ecp_core_electrons` from the active population.
`electron_state(atoms, charge=..., multiplicity=..., element_metadata=basis.by_element)`
computes

```text
N = sum(nuclear_charge) - sum(ecp_core_electrons) - ionic_charge
nalpha = (N + multiplicity - 1) / 2
nbeta  = (N - multiplicity + 1) / 2
```

An optional `electron_count` must agree with this population. Exact integer
fields, native integer bounds and nonnegative integral spin occupations are
checked. This inspection does not execute an ECP Hamiltonian. For example, the
Au def2-TZVP fixture has Z=79 and 60 ECP core electrons, hence 19 active electrons
at zero ionic charge. It is loadable but every calculation rejects its ECP.
Changing the ionic charge to imitate an ECP is not a supported conversion.

## Loadable data versus executable operations

Data records retain l=0…32, including g/h shells. Capability preflight inspects
every expanded contraction on every actual atom, separately for orbital and
auxiliary roles. Diagnostics name atom, Z, shell, contraction, l, backend,
operator and derivative order where applicable.

| Requested operator | Derivative meaning/order | Generic CPU/CUDA shell boundary |
| --- | --- | --- |
| `overlap`, `kinetic`, `nuclear_attraction`, `eri` | Nuclear 0 or 1 | s through f |
| `df_metric`, `df_three_center` | Nuclear 0 or 1 | s through f in both spaces |
| `ao` | Spatial 0 through 3 | s through f |
| ECP, modified nuclear charge, other derivatives/operators/backends | — | Rejected |

Public HF energy-plus-force endpoints preflight both values and first nuclear
derivatives. `eligible` states that a generic mathematical route exists. Native
shape/resource constraints, device availability, SCF convergence and the
[bounded f-shell batch limitations](f_shell_validation.md) still apply. The
report explicitly separates this from element-model validation and AOT
promotion. It does not advertise DFT, relativity or broad transition-metal
production support.

The realistic Fe cc-pVTZ fixture contains 20 expanded shells including g and
fails intact; no orbital or auxiliary shell is cut to f. Supported Fe tests use
an original small diagnostic basis for Fe24+ and FeH25+, each with two active
all-electron electrons. These are implementation/convention checks, not an
assessment of neutral Fe chemistry or basis completeness.

## Identities and prepared state

Results, prepared batches and benchmark convergence records expose resolved
orbital/auxiliary provenance. `basis_identity` covers the entire canonical
record, including coefficients, zeros, representation, nuclei/ECP metadata and
provenance. `mathematical_identity` covers the expanded FP64 native input, so
an imported bundled equivalent has identical mathematics despite distinct
source provenance. The model identity also includes charge, multiplicity,
method, screening and DF choices. Geometry remains a separate prepared input.

Calculators load files once and own explicit primitive storage. Editing the
source file or the caller's lists does not mutate a prepared model. Load the
new file into a new calculator and prepare a new batch to change the basis.
Replacing calculator model state is detected before prepared execution and
cannot reuse incompatible densities/Fock/DIIS state. Coordinate updates remain
supported. Result metadata is detached from the prepared snapshot.

NumPy integral charge/spin scalars remain JSON-safe, including invalid-occupation
diagnostics. Native method preparation rejects invalid RHF occupations before
execution. Coordinate-update and convergence failures remain isolated per item
on both backends. Metadata preserves those existing behaviors.

## Validation and follow-ups

`tests/data/external_basis/manifest.json` pins BSE git revision
`4adaf1372c7101620ca1a9f3130be9ae97fb8f30`, its BSD-3-Clause license bytes,
PySCF 2.14.0, NumPy, fixture bytes and the independent generator. Regenerate with
`tools/generate_external_basis_references.py --bse-source <pinned-checkout>`.
The generator does not import a VibeQC parser or evaluator. It supplies overlap,
kinetic, hcore, RHF energies and forces at original and changed geometries in
both representations. CPU AO quadrature checks S/T with separate radial/angular
refinement; the HF gates are 1e-8 Eh and 1e-7 Eh/Bohr.

`tools/validate_external_basis.py` records public CPU/CUDA cold, prepared, warm,
changed-geometry and ragged batch evidence, with complete basis identities and
actual backend labels. CUDA runs require a finite Slurm GPU allocation. See
[archived evidence](../benchmarks/results/external-basis-169/README.md).

Higher-l work (#170) must add each needed value/derivative/backend route and
independent conventions tests before extending the corresponding capability
entry. ECP work (#171) needs an explicit potential-bearing native Hamiltonian,
consistent active-electron bookkeeping and full gradients. Neither can be
enabled by editing a global maximum or by discarding unsupported metadata.
