"""Build-time-safe identity and coefficients for the RHF DF response slice."""

import hashlib
import json
from fractions import Fraction

RESPONSE_SLICE = "df-stationary-source-weights-v1"
RHF_COULOMB_COEFFICIENT = Fraction(1, 1)
RHF_EXCHANGE_COEFFICIENT = Fraction(1, 4)
SOURCE_WEIGHT_NAMES = ("metric", "one_electron", "overlap", "three_center")

_CONTRACT = {
    "schema": "vibeqc.df_hf_stationary_response",
    "version": 1,
    "response_slice": RESPONSE_SLICE,
    "coulomb_coefficient": [
        RHF_COULOMB_COEFFICIENT.numerator,
        RHF_COULOMB_COEFFICIENT.denominator,
    ],
    "exchange_coefficient": [
        RHF_EXCHANGE_COEFFICIENT.numerator,
        RHF_EXCHANGE_COEFFICIENT.denominator,
    ],
    "source_weights": list(SOURCE_WEIGHT_NAMES),
}
CONTRACT_IDENTITY = hashlib.sha256(
    json.dumps(_CONTRACT, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
