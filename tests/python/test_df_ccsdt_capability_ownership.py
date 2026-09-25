"""Energy-only capability records must own immutable property sets."""

from dataclasses import replace
from typing import cast

import pytest

from tools.vibeqc_cc.df_api import DFRCCSDTCapabilities


@pytest.mark.parametrize("operation", ["clear", "force"])
def test_capability_detaches_a_mutable_property_set(operation: str) -> None:
    properties = {"energy"}
    capability = DFRCCSDTCapabilities(
        supported_properties=cast("frozenset[str]", properties)
    )
    if operation == "clear":
        properties.clear()
    else:
        properties.add("forces")
    assert capability.supported_properties == frozenset({"energy"})
    assert isinstance(capability.supported_properties, frozenset)
    assert hash(capability) == hash(DFRCCSDTCapabilities())


def test_replaced_capability_remains_owned_and_hashable() -> None:
    properties = {"energy"}
    capability = replace(
        DFRCCSDTCapabilities(),
        supported_properties=cast("frozenset[str]", properties),
    )
    before = hash(capability)
    properties.add("forces")
    assert hash(capability) == before
    assert capability == DFRCCSDTCapabilities()


@pytest.mark.parametrize("properties", [set(), {"forces"}, {"energy", "forces"}])
def test_invalid_property_sets_remain_rejected(properties: set[str]) -> None:
    with pytest.raises(ValueError, match="energy-only"):
        DFRCCSDTCapabilities(supported_properties=cast("frozenset[str]", properties))


def test_default_capability_is_still_energy_only() -> None:
    capability = DFRCCSDTCapabilities()
    assert capability.available and not capability.supports_batch
    assert not capability.native_public
    assert capability.supported_properties == frozenset({"energy"})
