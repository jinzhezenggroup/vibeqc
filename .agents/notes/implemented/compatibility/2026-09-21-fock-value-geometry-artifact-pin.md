# Decision: advance the value-only Fock generated-source pin

Status: implemented
Date: 2026-09-21

## Contract

The #806 value-only FP64 geometry helper deliberately changes generated source.
Regenerate the four sm_120 shard hashes while retaining the exact byte-identity
regression, unchanged manifest/catalog and the six unchanged registry/stub files.
This does not select a new production kernel or change force/mixed equations.

## Independent review evidence

Separate checkouts regenerated all ten artifacts. The base exactly reproduces
the old pins. Removing only the new value-only helper definitions and restoring
their call names makes every new artifact byte-identical to the base. Thus the
pin update cannot hide unrelated generated changes.

The old strict snapshot test fails on the new helper; after the audited update,
59 host contract/codegen tests pass and 3 opt-in CUDA compile tests skip. A
separate finite Slurm allocation on RTX 5090/CUDA 12.9 executes the generated
swapped-DPDS Fock benchmark and its independent oracle successfully; both it
and the value-geometry gate pass. This is numerical/compiler qualification,
not a new whole-molecule performance claim or default selector promotion.

Agent: ChatGPT
Model: GPT-6 Astra Pro

## Additional independent geometry verification

A second review preserves the already-published artifact pin correction rather
than replacing it. Its new test compiles both emitted geometry producers and
the actual retained Boys table implementation on the host. Across ssss, psss,
dppp and dddd, 640 signed-weight, pair-orientation and geometric cases preserve
all finite live state and leave NaN canaries in omitted force state untouched.
Independent 256-point Gauss-Legendre integration verifies the live Boys values.

The 55 IR/artifact/geometry tests pass; a separate focused Fock/psss selection
passes 48 tests with six explicit CUDA opt-in skips. These additional results
are host/compiler validation, not another GPU or molecular performance run.

Agent: ChatGPT
Model: GPT-6 Astra Pro
