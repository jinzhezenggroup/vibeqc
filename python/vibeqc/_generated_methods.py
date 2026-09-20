"""Generated public method identity metadata; do not edit by hand."""

# fmt: off
from types import MappingProxyType

METHOD_RHF = 1
METHOD_UHF = 2
METHOD_WB97M_V = 3
METHOD_RCCSD_T = 4
METHOD_MP2 = 5
METHOD_LDA_RKS = 6
METHOD_PBE_RKS = 7
METHOD_LDA_UKS = 8
METHOD_PBE_UKS = 9
METHOD_R2SCAN_RKS = 10
METHOD_R2SCAN_UKS = 11
METHOD_RCCSD = 12
METHOD_PBE0_RKS = 13
METHOD_PBE0_UKS = 14
METHOD_GFN2_XTB = 15

METHOD_CONSTANTS = MappingProxyType({
    "METHOD_RHF": METHOD_RHF,
    "METHOD_UHF": METHOD_UHF,
    "METHOD_WB97M_V": METHOD_WB97M_V,
    "METHOD_RCCSD_T": METHOD_RCCSD_T,
    "METHOD_MP2": METHOD_MP2,
    "METHOD_LDA_RKS": METHOD_LDA_RKS,
    "METHOD_PBE_RKS": METHOD_PBE_RKS,
    "METHOD_LDA_UKS": METHOD_LDA_UKS,
    "METHOD_PBE_UKS": METHOD_PBE_UKS,
    "METHOD_R2SCAN_RKS": METHOD_R2SCAN_RKS,
    "METHOD_R2SCAN_UKS": METHOD_R2SCAN_UKS,
    "METHOD_RCCSD": METHOD_RCCSD,
    "METHOD_PBE0_RKS": METHOD_PBE0_RKS,
    "METHOD_PBE0_UKS": METHOD_PBE0_UKS,
    "METHOD_GFN2_XTB": METHOD_GFN2_XTB,
})

METHOD_METADATA = MappingProxyType({
    'rhf': MappingProxyType({"abi_id": 1, "family": 'hartree_fock', "provider": 'hf', "properties": ('energy', 'forces'), "supports_batch": True, "aliases": ()}),
    'uhf': MappingProxyType({"abi_id": 2, "family": 'hartree_fock', "provider": 'hf', "properties": ('energy', 'forces'), "supports_batch": True, "aliases": ()}),
    'wb97m-v': MappingProxyType({"abi_id": 3, "family": 'density_functional', "provider": 'reserved', "properties": (), "supports_batch": False, "aliases": ()}),
    'rccsd(t)': MappingProxyType({"abi_id": 4, "family": 'coupled_cluster', "provider": 'reserved', "properties": (), "supports_batch": False, "aliases": ('ccsd(t)',)}),
    'mp2': MappingProxyType({"abi_id": 5, "family": 'perturbation', "provider": 'mp2', "properties": ('energy', 'forces'), "supports_batch": True, "aliases": ()}),
    'lda-rks': MappingProxyType({"abi_id": 6, "family": 'density_functional', "provider": 'dft', "properties": ('energy',), "supports_batch": True, "aliases": ()}),
    'pbe-rks': MappingProxyType({"abi_id": 7, "family": 'density_functional', "provider": 'dft', "properties": ('energy',), "supports_batch": True, "aliases": ()}),
    'lda-uks': MappingProxyType({"abi_id": 8, "family": 'density_functional', "provider": 'dft', "properties": ('energy',), "supports_batch": True, "aliases": ()}),
    'pbe-uks': MappingProxyType({"abi_id": 9, "family": 'density_functional', "provider": 'dft', "properties": ('energy',), "supports_batch": True, "aliases": ()}),
    'r2scan-rks': MappingProxyType({"abi_id": 10, "family": 'density_functional', "provider": 'dft', "properties": ('energy',), "supports_batch": True, "aliases": ()}),
    'r2scan-uks': MappingProxyType({"abi_id": 11, "family": 'density_functional', "provider": 'dft', "properties": ('energy',), "supports_batch": True, "aliases": ()}),
    'rccsd': MappingProxyType({"abi_id": 12, "family": 'coupled_cluster', "provider": 'rccsd', "properties": ('energy',), "supports_batch": True, "aliases": ()}),
    'pbe0-rks': MappingProxyType({"abi_id": 13, "family": 'density_functional', "provider": 'dft', "properties": ('energy',), "supports_batch": True, "aliases": ()}),
    'pbe0-uks': MappingProxyType({"abi_id": 14, "family": 'density_functional', "provider": 'dft', "properties": ('energy',), "supports_batch": True, "aliases": ()}),
    'gfn2-xtb': MappingProxyType({"abi_id": 15, "family": 'semiempirical', "provider": 'xtb', "properties": ('energy', 'forces'), "supports_batch": False, "aliases": ('gfn2',)}),
})

METHOD_NAME_TO_ID = MappingProxyType({
    'rhf': METHOD_RHF,
    'uhf': METHOD_UHF,
    'wb97m-v': METHOD_WB97M_V,
    'rccsd(t)': METHOD_RCCSD_T,
    'ccsd(t)': METHOD_RCCSD_T,
    'mp2': METHOD_MP2,
    'lda-rks': METHOD_LDA_RKS,
    'pbe-rks': METHOD_PBE_RKS,
    'lda-uks': METHOD_LDA_UKS,
    'pbe-uks': METHOD_PBE_UKS,
    'r2scan-rks': METHOD_R2SCAN_RKS,
    'r2scan-uks': METHOD_R2SCAN_UKS,
    'rccsd': METHOD_RCCSD,
    'pbe0-rks': METHOD_PBE0_RKS,
    'pbe0-uks': METHOD_PBE0_UKS,
    'gfn2-xtb': METHOD_GFN2_XTB,
    'gfn2': METHOD_GFN2_XTB,
})
METHOD_ID_TO_NAME = MappingProxyType({
    METHOD_RHF: 'rhf',
    METHOD_UHF: 'uhf',
    METHOD_WB97M_V: 'wb97m-v',
    METHOD_RCCSD_T: 'rccsd(t)',
    METHOD_MP2: 'mp2',
    METHOD_LDA_RKS: 'lda-rks',
    METHOD_PBE_RKS: 'pbe-rks',
    METHOD_LDA_UKS: 'lda-uks',
    METHOD_PBE_UKS: 'pbe-uks',
    METHOD_R2SCAN_RKS: 'r2scan-rks',
    METHOD_R2SCAN_UKS: 'r2scan-uks',
    METHOD_RCCSD: 'rccsd',
    METHOD_PBE0_RKS: 'pbe0-rks',
    METHOD_PBE0_UKS: 'pbe0-uks',
    METHOD_GFN2_XTB: 'gfn2-xtb',
})

HF_METHOD_IDS = frozenset((METHOD_RHF, METHOD_UHF,))
NATIVE_DFT_METHOD_IDS = frozenset((METHOD_LDA_RKS, METHOD_PBE_RKS, METHOD_LDA_UKS, METHOD_PBE_UKS, METHOD_R2SCAN_RKS, METHOD_R2SCAN_UKS, METHOD_PBE0_RKS, METHOD_PBE0_UKS,))
# fmt: on
