"""Guard against unit doubles accidentally breaking repair reporting."""

import logging

import pytest


@pytest.fixture
def expected_repair_errors():
    """Allow only exact exceptions deliberately injected by a failure-path test."""
    return []


@pytest.fixture(autouse=True)
def verify_repair_registry_errors(caplog, expected_repair_errors):
    """Caught repair errors must not silently make a passing test incomplete."""
    yield
    records = [
        record
        for phase in ("setup", "call", "teardown")
        for record in caplog.get_records(phase)
        if record.name == "custom_components.groq.repairs"
        and record.levelno >= logging.ERROR
    ]
    assert len(records) == len(expected_repair_errors), [
        (record.getMessage(), record.exc_info) for record in records
    ]
    for record, expected in zip(records, expected_repair_errors, strict=True):
        assert record.getMessage() == "Could not update the Groq repair registry"
        assert record.exc_info is not None
        assert record.exc_info[1] is expected
