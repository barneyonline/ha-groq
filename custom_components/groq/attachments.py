"""Helpers for Groq image attachments."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from homeassistant.core import HomeAssistant

from .errors import translated_error

MAX_IMAGE_ATTACHMENT_BYTES = 10 * 1024 * 1024
MAX_IMAGE_ATTACHMENTS = 4
MAX_IMAGE_ATTACHMENT_TOTAL_BYTES = 20 * 1024 * 1024


def read_bounded_file(path: Path, limit: int) -> bytes:
    """Read at most limit+1 bytes; callers translate I/O and size errors."""
    with path.open("rb") as source:
        content = source.read(limit + 1)
    if len(content) > limit:
        raise ValueError("File exceeds byte limit")
    return content


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


def _read_attachment_data_url(path: Path, mime_type: str) -> tuple[str, int]:
    """Read an attachment as a data URL for OpenAI-compatible vision input."""
    if not path.exists():
        raise translated_error(
            "Groq image attachment file does not exist", "attachment_file_missing"
        )
    if not path.is_file():
        raise translated_error(
            "Groq image attachment must be a file", "attachment_not_file"
        )
    size = path.stat().st_size
    if size > MAX_IMAGE_ATTACHMENT_BYTES:
        raise translated_error(
            "Groq image attachment exceeds the 10 MB integration limit",
            "attachment_too_large",
            limit_mb=MAX_IMAGE_ATTACHMENT_BYTES // 1024 // 1024,
        )
    try:
        content = read_bounded_file(path, MAX_IMAGE_ATTACHMENT_BYTES)
    except ValueError as err:
        raise translated_error(
            "Groq image attachment exceeds the 10 MB integration limit",
            "attachment_too_large",
            limit_mb=MAX_IMAGE_ATTACHMENT_BYTES // 1024 // 1024,
        ) from err
    except OSError as err:
        raise translated_error(
            "Groq image attachment could not be read", "attachment_file_missing"
        ) from err
    data = base64.b64encode(content).decode("ascii")
    return f"data:{mime_type};base64,{data}", len(content)


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
    total_bytes = 0
    for attachment in attachments:
        if len(content) > MAX_IMAGE_ATTACHMENTS:
            raise translated_error(
                f"Groq accepts at most {MAX_IMAGE_ATTACHMENTS} image attachments",
                "too_many_attachments",
                limit=MAX_IMAGE_ATTACHMENTS,
            )
        mime_type = attachment_mime_type(attachment)
        path = attachment_path(attachment)
        if mime_type is None or not mime_type.startswith("image/"):
            raise translated_error(
                "Groq attachments must be image files", "attachment_not_image"
            )
        if path is None:
            raise translated_error(
                "Groq image attachments must resolve to files",
                "attachment_file_required",
            )
        data_url, attachment_bytes = await hass.async_add_executor_job(
            _read_attachment_data_url,
            path,
            mime_type,
        )
        total_bytes += attachment_bytes
        if total_bytes > MAX_IMAGE_ATTACHMENT_TOTAL_BYTES:
            raise translated_error(
                "Groq image attachments exceed the combined attachment size limit",
                "attachments_too_large",
                limit_mb=MAX_IMAGE_ATTACHMENT_TOTAL_BYTES // 1024 // 1024,
            )
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": data_url},
            }
        )
    return content
