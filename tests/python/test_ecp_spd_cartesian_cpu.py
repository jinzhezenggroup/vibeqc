"""Complete cartesian s/p/d CPU ECP force qualification worker."""

import typing

import pytest
import test_ecp_public_cpu as gates

retain_endpoint_evidence = gates.retain_endpoint_evidence


@pytest.mark.parametrize("method", ["lda-rks", "pbe-rks", "lda-uks", "pbe-uks"])
def test_spd_force_analytic_and_reconverged_fd(
    method: str,
    record_property: typing.Any,
    tmp_path: typing.Any,
) -> None:
    gates.test_public_ecp_force_analytic_and_reconverged_fd(
        method, "cartesian", record_property, tmp_path, d_shell=True
    )


@pytest.mark.parametrize("method", ["pbe-rks", "pbe-uks"])
def test_spd_budgeted_ragged_replay_and_failure_recovery(
    method: str,
    record_property: typing.Any,
) -> None:
    gates.test_public_ecp_budgeted_ragged_replay_and_failure_recovery(
        method, "cartesian", record_property, d_shell=True
    )


def test_spd_arbitrary_ordered_weights_against_libcint_energy_differences() -> None:
    gates.check_spd_arbitrary_ordered_weights_against_libcint_energy_differences(
        "cartesian"
    )
