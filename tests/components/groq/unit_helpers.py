"""Explicit boundary doubles for tests that do not exercise HA registries."""

from unittest.mock import Mock

from homeassistant.helpers import issue_registry as ir


def unit_hass_data():
    """Keep isolated API/audio tests out of HA's persistent registry storage.

    The fake belongs to this lightweight hass instance. Tests using the real hass
    fixture continue to exercise HA's actual issue registry and persistence APIs.
    """
    return {ir.DATA_REGISTRY: Mock(spec=ir.IssueRegistry, issues={})}
