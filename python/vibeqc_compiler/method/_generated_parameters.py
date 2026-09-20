"""Generated from method_parameters.json; do not edit by hand."""

import math
from types import MappingProxyType
from typing import TypedDict

PARAMETER_SOURCE_SHA256 = '27e63f7c814a8d7f5af7c3eeadea7ef38256967c15b3e6cdcf2086e5e97e2881'
D3_TABLE_SHA256 = '9ff932ea598f690c1fb599a67762060ba1907102d5ec132164f2a7e8886cd22e'
D3_RADII_SHA256 = '92b32fada844a337204b84f2d961473bad5737240765eb8d0727a62827de5111'

D3_BJ_PARAMETER_SETS = MappingProxyType({
    'PBE-D3(BJ)': MappingProxyType({
        's6': 1.0,
        's8': 0.7875,
        'a1': 0.4289,
        'a2': 4.4407,
        's9': 0.0,
        'table_sha256': '9ff932ea598f690c1fb599a67762060ba1907102d5ec132164f2a7e8886cd22e',
        'radii_sha256': '92b32fada844a337204b84f2d961473bad5737240765eb8d0727a62827de5111',
    }),
    'PBE0-D3(BJ)': MappingProxyType({
        's6': 1.0,
        's8': 1.2177,
        'a1': 0.4145,
        'a2': 4.8593,
        's9': 0.0,
        'table_sha256': '9ff932ea598f690c1fb599a67762060ba1907102d5ec132164f2a7e8886cd22e',
        'radii_sha256': '92b32fada844a337204b84f2d961473bad5737240765eb8d0727a62827de5111',
    }),
})

D4_PARAMETER_SETS = MappingProxyType({
    'r2SCAN-3c': MappingProxyType({
        's6': 1.0,
        's8': 0.0,
        's9': 2.0,
        'a1': 0.42,
        'a2': 5.65,
        'ga': 2.0,
        'gc': 1.0,
        'profile': 'r2scan3c',
        'reference_model': 'eeq',
        'charge_model': 'eeq2019',
        'cn_cutoff': 30.0,
        'pair_cutoff': 60.0,
        'atm_cutoff': 40.0,
        'charge_cn_cutoff': 25.0,
        'table_sha256': 'd1691a6cf08748e7c35a340f78824a4a8da1813b0ca32d1074c346bfb3874871',
        'charge_parameter_sha256': '02b8bee49c10b4c31914caf149d9f58164e58d6dc7ae2ab21e9d12f5bc22797a',
    }),
})

GCP_PARAMETER_SETS = MappingProxyType({
    'r2SCAN-3c': MappingProxyType({
        'basis': 'def2-mTZVPP',
        'sigma': 1.0,
        'eta': 1.315,
        'eta_spec': 1.15,
        'alpha': 0.941,
        'beta': 1.4636,
        'damping_scale': 4.0,
        'damping_exponent': 6.0,
        'damping': True,
        'parameter_sha256': '3edc7b569cff3cf47dffd06395de02b3d45aabb4d4be368debf87fbafd56fa4e',
        'implementation_sha256': 'c1d69f640c9a7498618998a97b40a1629554ede31001ceab3060dc2f84c8be1e',
        'vdw_radii_sha256': '3f08b5755bfd643d6dbb56fd544c117145473a4b27138978a25d0475af985575',
        'data_sha256': 'c1cede24b2527a2b688981b651d91da7206d8da1a223c3f217a86800c47eded2',
        'source': 'dftd3/simple-dftd3',
        'source_revision': '41d5a07b98ce15e97bec7a1815869725f6c7b0c2',
        'license': 'LGPL-3.0-or-later',
        'profile': 'r2scan3c',
        'supported_atomic_numbers': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18),
    }),
})

PARAMETER_PROVENANCE = MappingProxyType({
    'd3_bj:PBE-D3(BJ)': MappingProxyType({
        'source': 'simple-dftd3',
        'version': '1.4.0',
    }),
    'd3_bj:PBE0-D3(BJ)': MappingProxyType({
        'source': 'simple-dftd3',
        'version': '1.4.0',
    }),
    'd4:GFN2-xTB': MappingProxyType({
        'source': 'xTBloom',
        'revision': '2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3',
    }),
    'd4:r2SCAN-3c': MappingProxyType({
        'source': 'DFT-D4',
        'version': '4.2.0',
    }),
    'gcp:r2SCAN-3c': MappingProxyType({
        'source': 'dftd3/simple-dftd3',
        'revision': '41d5a07b98ce15e97bec7a1815869725f6c7b0c2',
    }),
})

def _parameter_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise TypeError('parameter must be a finite real scalar')
    return float(value)

def _parameter_str(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError('parameter must be a nonempty string')
    return value

def _parameter_bool(value: object) -> bool:
    if type(value) is not bool:
        raise TypeError('parameter must be a bool')
    return value

def _parameter_ints(value: object) -> tuple[int, ...]:
    if not isinstance(value, tuple) or not value or any(type(x) is not int for x in value):
        raise TypeError('parameter must be a nonempty integer tuple')
    return tuple(int(x) for x in value)

class D3Parameters(TypedDict):
    s6: float
    s8: float
    a1: float
    a2: float
    s9: float
    table_sha256: str
    radii_sha256: str

def d3_parameters(name: str) -> D3Parameters:
    value = D3_BJ_PARAMETER_SETS[name]
    return {
        's6': _parameter_float(value['s6']),
        's8': _parameter_float(value['s8']),
        'a1': _parameter_float(value['a1']),
        'a2': _parameter_float(value['a2']),
        's9': _parameter_float(value['s9']),
        'table_sha256': _parameter_str(value['table_sha256']),
        'radii_sha256': _parameter_str(value['radii_sha256']),
    }

class D4Parameters(TypedDict):
    s6: float
    s8: float
    s9: float
    a1: float
    a2: float
    ga: float
    gc: float
    profile: str
    reference_model: str
    charge_model: str
    cn_cutoff: float
    pair_cutoff: float
    atm_cutoff: float
    charge_cn_cutoff: float
    table_sha256: str
    charge_parameter_sha256: str

def d4_parameters(name: str) -> D4Parameters:
    value = D4_PARAMETER_SETS[name]
    return {
        's6': _parameter_float(value['s6']),
        's8': _parameter_float(value['s8']),
        's9': _parameter_float(value['s9']),
        'a1': _parameter_float(value['a1']),
        'a2': _parameter_float(value['a2']),
        'ga': _parameter_float(value['ga']),
        'gc': _parameter_float(value['gc']),
        'profile': _parameter_str(value['profile']),
        'reference_model': _parameter_str(value['reference_model']),
        'charge_model': _parameter_str(value['charge_model']),
        'cn_cutoff': _parameter_float(value['cn_cutoff']),
        'pair_cutoff': _parameter_float(value['pair_cutoff']),
        'atm_cutoff': _parameter_float(value['atm_cutoff']),
        'charge_cn_cutoff': _parameter_float(value['charge_cn_cutoff']),
        'table_sha256': _parameter_str(value['table_sha256']),
        'charge_parameter_sha256': _parameter_str(value['charge_parameter_sha256']),
    }

class GCPParameters(TypedDict):
    basis: str
    sigma: float
    eta: float
    eta_spec: float
    alpha: float
    beta: float
    damping_scale: float
    damping_exponent: float
    damping: bool
    parameter_sha256: str
    implementation_sha256: str
    vdw_radii_sha256: str
    data_sha256: str
    source: str
    source_revision: str
    license: str
    profile: str
    supported_atomic_numbers: tuple[int, ...]

def gcp_parameters(name: str) -> GCPParameters:
    value = GCP_PARAMETER_SETS[name]
    return {
        'basis': _parameter_str(value['basis']),
        'sigma': _parameter_float(value['sigma']),
        'eta': _parameter_float(value['eta']),
        'eta_spec': _parameter_float(value['eta_spec']),
        'alpha': _parameter_float(value['alpha']),
        'beta': _parameter_float(value['beta']),
        'damping_scale': _parameter_float(value['damping_scale']),
        'damping_exponent': _parameter_float(value['damping_exponent']),
        'damping': _parameter_bool(value['damping']),
        'parameter_sha256': _parameter_str(value['parameter_sha256']),
        'implementation_sha256': _parameter_str(value['implementation_sha256']),
        'vdw_radii_sha256': _parameter_str(value['vdw_radii_sha256']),
        'data_sha256': _parameter_str(value['data_sha256']),
        'source': _parameter_str(value['source']),
        'source_revision': _parameter_str(value['source_revision']),
        'license': _parameter_str(value['license']),
        'profile': _parameter_str(value['profile']),
        'supported_atomic_numbers': _parameter_ints(value['supported_atomic_numbers']),
    }
