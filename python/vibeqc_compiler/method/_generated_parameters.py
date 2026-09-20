"""Generated from method_parameters.json; do not edit by hand."""

from types import MappingProxyType

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
