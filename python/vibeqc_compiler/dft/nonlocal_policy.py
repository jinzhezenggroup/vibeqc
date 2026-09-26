"""Explicit molecular VV10 density domain; raw fixed-grid kernels stay strict.

PySCF 2.14.0 pyscf/dft/numint.py::_vv10nlc uses rho >= 1e-8 in
both pair domains. This is a numerical quadrature policy, not a density floor
or a change to the VV10 kernel. It does not qualify derivatives across a
changing active set. Source: https://pyscf.org/_modules/pyscf/dft/numint.html
"""

from fractions import Fraction

MOLECULAR_VV10_DENSITY_POLICY = "vv10-molecular-rho-ge-1e-8-v1"
MOLECULAR_VV10_DENSITY_THRESHOLD = Fraction(1, 100000000)
