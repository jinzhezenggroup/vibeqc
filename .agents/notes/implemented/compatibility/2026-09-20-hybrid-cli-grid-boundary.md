# Hybrid resource CLI boundary

Refs #618, #165.

The production-grid integration deliberately requires an explicit GridSpec for
PBE0 and custom global hybrids; pure-PBE grid accuracy is not implicitly promoted
to those compositions. Python resource estimation and Calculator retain their
KsOptions(grid=...) entry points, including the resolved-graph budget fix.

The resource CLI has no grid option. Advertising pbe0-rks/pbe0-uks there therefore
created an entry point that could only fail. Restore the previous qualified CLI
choices until explicit grid control is available. Parser tests cover both hybrid
rejections and all six retained HF/semilocal choices. This does not remove any
Python hybrid support or alter SCF, gradient, grid, or ABI implementations.

Agent: ChatGPT
Model: GPT-6 Astra Pro
