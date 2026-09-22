"""Generated XC method aliases from pinned PySCF/Libxc metadata.

Do not edit by hand. Regenerate with tools/generate_xc_aliases.py.
"""

from types import MappingProxyType

UPSTREAM_XC_ALIAS_PROVENANCE = MappingProxyType(
    {
        "pyscf_version": "2.14.0",
        "libxc_version": "7.0.0",
        "module": "pyscf.dft.libxc",
        "module_sha256": "0d3cc988782a90775c525eb348aea63f047433191ce01482c641bbfb5e80f2c7",
    }
)

METHOD_ALIASES = MappingProxyType(
    {
        "B3LYPG": "B3LYP",
        "B3P86G": "B3P86",
        "CAMB3LYP": "CAM-B3LYP",
        "CAM_B3LYP": "CAM-B3LYP",
        "PBE1PBE": "PBE0",
        "PBEH": "PBE0",
        "X3LYPG": "X3LYP",
    }
)
