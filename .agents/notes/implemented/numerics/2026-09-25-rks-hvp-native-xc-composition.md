# Decision: compose molecular HVP from the native XC point owner

The previous molecular XC contributor instantiated ExternalPointContraction but called its inherited mixed_geometry_directional method. That method reconstructs a generic generated scalar functional, not the numerical SCF-domain point model of the accepted reference. Mixing that Hessian with the native first-order response is not a valid boundary-policy contract.

Reuse native_rks_xc_hvp_components from reviewed PR #1286 at c8e954e62a08e1303521bd6d8ed8f08fcb1a80ef. The two implementation files are reused byte-for-byte: contractions.py blob b950d94d8cebbb753e5e1392bd2cf077d170c493 and rks_directional.py blob f4cd20b6c95d95e21c1c613b02ad672208b10a37. Their destination predecessors exactly match #1286's base, so this selective integration discards no branch-specific edits. This is not a claim to merge the entire #1286 history.

The seven-source StationaryHVPPlan, shell-local generated integral weights, integral derivative providers, nuclear term and single CPKS solve remain unchanged. The wrapper requires the exact solved direction and never falls back from a native-domain error to a generic functional. XC contributor and execution identities advance to v2 to distinguish the corrected composition.

Three isolated composition tests pass locally: exact source/response reuse, mismatched-direction rejection, and native-error propagation. These use simulated contributor results, not molecular calculations. Existing reconverged LDA/PBE total-HVP tests must pass on the new integrated head before promotion; #1286's prior partial-contributor CI is not a substitute for this final composition.

Agent: ChatGPT — Even-PR Review
Model: GPT-6 Astra Pro
