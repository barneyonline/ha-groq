"""Local JSON Schema validation shared by generation entrypoints."""

from __future__ import annotations

from typing import Any

import jsonschema
from referencing import Registry
from referencing.exceptions import Unresolvable

from .errors import translated_error


def validate_json_schema_data(
    data: Any,
    schema: dict[str, Any],
) -> Any:
    """Validate parsed AI task data against a service-level JSON Schema."""
    try:
        jsonschema.validate(data, schema, registry=Registry())
    except (jsonschema.SchemaError, jsonschema.ValidationError, Unresolvable) as err:
        raise translated_error(
            "Groq returned data that did not match the requested structure",
            "structured_response_invalid",
        ) from err
    return data
