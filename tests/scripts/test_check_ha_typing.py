"""Ensure the HA type gate fails when the checker stops rejecting bad code."""

import subprocess

import pytest

from scripts.check_ha_typing import valid_control_results


@pytest.mark.parametrize(
    ("positive_code", "negative_code", "errors", "expected"),
    [
        (0, 1, "[index]\n[arg-type]\n[arg-type]", True),
        (0, 0, "", False),
        (1, 1, "[index]\n[arg-type]\n[arg-type]", False),
        (0, 1, "[import-not-found]", False),
        (0, 1, "[index]\n[arg-type]", False),
    ],
)
def test_contract_gate_requires_valid_code_and_all_negative_controls(
    positive_code, negative_code, errors, expected
):
    positive = subprocess.CompletedProcess([], positive_code, "", "")
    negative = subprocess.CompletedProcess([], negative_code, errors, "")
    assert valid_control_results(positive, negative) is expected
