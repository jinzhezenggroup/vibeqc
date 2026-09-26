# Decision: retain independent P86 feature-Hessian acceptance

Status: implemented
Date: 2026-09-20

The default scalar XC contract returns second derivatives. First-order values
alone cannot qualify a newly represented functional. Retain small JSON oracle
cases produced by PySCF 2.14.0 / Libxc 7.0.0, not by the VibeQC expressions.
Reuse the existing independent `generate_xc_references.reference` converter;
`generate_semilocal_hessian_cases.py` records the versions, explicit Libxc
compositions, generator/converter hashes, seed and complete numerical arrays.

The cases use actual Cartesian gradient vectors, both spin modes, eight total
densities and full packed feature Hessians. Two densities lie on opposite sides
of the PZ `rs=1` branch. Tests call the default `build_program` and compare the
energy density, every feature derivative and every upper-Hessian entry; they do
not import PySCF or require a native library/GPU. Elementary newly represented
components are checked separately from the named method combinations.

An additional independent 64-point randomized energy/gradient/Hessian audit
passed during review. Reject relying only on that transient run or differentiating
the implementation to make its own expected values. These are scalar XC feature
Hessians, not nuclear Hessians, an SCF endpoint, or a performance qualification.
Production equations and tolerances are unchanged. Ref #610.

Agent: ChatGPT
Model: GPT-6 Astra Pro
