"""Estimate JSON encoding work before choosing an execution context."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

JSON_EXECUTOR_THRESHOLD = 64 * 1024


def estimated_json_size_exceeds(
    payload: dict[str, Any], limit: int, *, use_encoded_upper_bound: bool = False
) -> bool:
    """Check the size threshold without encoding strings or copying containers.

    The conservative estimate accounts for escaping, numeric rendering and
    string conversion of values that JSON cannot encode directly.
    """
    remaining = limit
    pending: list[Iterator[Any]] = [iter((payload,))]
    while pending:
        for value in pending[-1]:
            if isinstance(value, str):
                escapes = (
                    (6 if value.isascii() else 12) if use_encoded_upper_bound else 1
                )
                remaining -= len(value) * escapes + 2
            else:
                remaining -= 8
                if use_encoded_upper_bound:
                    if isinstance(value, int):
                        remaining -= (value.bit_length() + 2) // 3
                    elif isinstance(value, float):
                        remaining -= 24
                    elif value is not None and not isinstance(
                        value, (dict, list, tuple)
                    ):
                        return True
            if remaining <= 0:
                return True
            if isinstance(value, (dict, list, tuple)):
                pending.append(
                    iter(value.items() if isinstance(value, dict) else value)
                )
                break
        else:
            pending.pop()
    return False
