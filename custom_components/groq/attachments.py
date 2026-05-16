"""Helpers for Groq image attachments."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError

MAX_IMAGE_ATTACHMENT_BYTES = 10 * 1024 * 1024


def attachment_mime_type(attachment: Any) -> str | None:
    """Return a resolved Home Assistant attachment MIME type."""
    if isinstance(attachment, dict):
        value = attachment.get("mime_type") or attachment.get("media_content_type")
    else:
        value = getattr(attachment, "mime_type", None) or getattr(
            attachment, "media_content_type", None
        )
    return value if isinstance(value, str) and value else None


def attachment_path(attachment: Any) -> Path | None:
    """Return a resolved local attachment path."""
    if isinstance(attachment, dict):
        value = attachment.get("path")
    else:
        value = getattr(attachment, "path", None)
    if value is None:
        return None
    return Path(value)


def _read_attachment_data_url(path: Path, mime_type: str) -> str:
    """Read an attachment as a data URL for OpenAI-compatible vision input."""
    if not path.exists():
        raise HomeAssistantError("Groq image attachment file does not exist")
    if not path.is_file():
        raise HomeAssistantError("Groq image attachment must be a file")
    size = path.stat().st_size
    if size > MAX_IMAGE_ATTACHMENT_BYTES:
        raise HomeAssistantError(
            "Groq image attachment exceeds the 10 MB integration limit"
        )
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{data}"


async def async_attachment_content_parts(
    hass: HomeAssistant,
    attachments: Any,
    *,
    text: str,
) -> list[dict[str, Any]] | None:
    """Return OpenAI-compatible multimodal content parts for image attachments."""
    if not attachments:
        return None

    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for attachment in attachments:
        mime_type = attachment_mime_type(attachment)
        path = attachment_path(attachment)
        if mime_type is None or not mime_type.startswith("image/"):
            raise HomeAssistantError("Groq attachments must be image files")
        if path is None:
            raise HomeAssistantError("Groq image attachments must resolve to files")
        data_url = await hass.async_add_executor_job(
            _read_attachment_data_url,
            path,
            mime_type,
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": data_url},
            }
        )
    return content
