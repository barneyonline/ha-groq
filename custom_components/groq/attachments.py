"""Helpers for Groq image attachments."""

from __future__ import annotations

import base64
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from stat import S_ISREG
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError

from .errors import GroqError, translated_error

MAX_IMAGE_ATTACHMENT_BYTES = 10 * 1024 * 1024
MAX_IMAGE_ATTACHMENTS = 4
MAX_IMAGE_ATTACHMENT_TOTAL_BYTES = 20 * 1024 * 1024

type ImageAttachmentSource = tuple[str | None, Path | None]
type ImageAttachmentSources = tuple[ImageAttachmentSource, ...]


class ImageAttachmentBudgetExceeded(GroqError):
    """An image group cannot fit within the request-wide budget."""


@dataclass(frozen=True, slots=True)
class ImageAttachmentGroup:
    """Keep converted images together with their measured source size."""

    images: list[dict[str, Any]]
    source_bytes: int

    def with_text(self, text: str) -> list[dict[str, Any]]:
        """Build multimodal message content."""
        return [{"type": "text", "text": text}, *self.images]


@dataclass(slots=True)
class ImageAttachmentBudget:
    """Track the count and source bytes available in one request."""

    remaining_count: int = field(default_factory=lambda: MAX_IMAGE_ATTACHMENTS)
    remaining_bytes: int = field(
        default_factory=lambda: MAX_IMAGE_ATTACHMENT_TOTAL_BYTES
    )

    def admit(self, attachments: Iterable[Any]) -> ImageAttachmentSources:
        """Check the count before snapshotting sources for caching and file reads."""
        selected = list(islice(attachments, self.remaining_count + 1))
        self.check_count(len(selected))
        return tuple(
            (attachment_mime_type(attachment), attachment_path(attachment))
            for attachment in selected
        )

    def check_count(self, count: int) -> None:
        """Reject an attachment group that exceeds the remaining image count."""
        if count > self.remaining_count:
            raise ImageAttachmentBudgetExceeded(
                f"Groq accepts at most {MAX_IMAGE_ATTACHMENTS} image attachments",
                translation_key="too_many_attachments",
                translation_placeholders={"limit": str(MAX_IMAGE_ATTACHMENTS)},
            )

    def charge(self, group: ImageAttachmentGroup) -> None:
        """Count newly read and cached images against the same request budget."""
        count = len(group.images)
        self.check_count(count)
        if group.source_bytes > self.remaining_bytes:
            raise _attachments_too_large()
        self.remaining_count -= count
        self.remaining_bytes -= group.source_bytes


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


def _attachments_too_large() -> ImageAttachmentBudgetExceeded:
    return ImageAttachmentBudgetExceeded(
        "Groq image attachments exceed the combined attachment size limit",
        translation_key="attachments_too_large",
        translation_placeholders={
            "limit_mb": str(MAX_IMAGE_ATTACHMENT_TOTAL_BYTES // 1024 // 1024)
        },
    )


def _attachment_too_large(*, also_exceeds_budget: bool) -> GroqError:
    error_type = ImageAttachmentBudgetExceeded if also_exceeds_budget else GroqError
    return error_type(
        "Groq image attachment exceeds the 10 MB integration limit",
        translation_key="attachment_too_large",
        translation_placeholders={
            "limit_mb": str(MAX_IMAGE_ATTACHMENT_BYTES // 1024 // 1024)
        },
    )


def _attachment_access_error(error: OSError) -> HomeAssistantError:
    """Distinguish a missing image from one that cannot be accessed."""
    if isinstance(error, (FileNotFoundError, NotADirectoryError)):
        return translated_error(
            "Groq image attachment file does not exist", "attachment_file_missing"
        )
    return translated_error(
        "Groq image attachment could not be read", "attachment_file_unreadable"
    )


def _attachment_file_size(path: Path) -> int:
    """Return the size of a regular image file before opening it."""
    try:
        info = path.stat()
    except OSError as err:
        raise _attachment_access_error(err) from err
    if not S_ISREG(info.st_mode):
        raise translated_error(
            "Groq image attachment must be a file", "attachment_not_file"
        )
    return info.st_size


def _image_attachment_details(source: ImageAttachmentSource) -> tuple[Path, str, int]:
    """Validate one attachment and collect its on-disk size."""
    mime_type, path = source
    if mime_type is None or not mime_type.startswith("image/"):
        raise translated_error(
            "Groq attachments must be image files", "attachment_not_image"
        )
    if path is None:
        raise translated_error(
            "Groq image attachments must resolve to files", "attachment_file_required"
        )
    return path, mime_type, _attachment_file_size(path)


def _read_attachment_bytes(
    path: Path, checked_size: int, remaining_bytes: int
) -> bytes:
    """Read within the remaining limits even if an attachment changed size."""
    if checked_size > remaining_bytes:
        raise _attachments_too_large()
    limit = min(MAX_IMAGE_ATTACHMENT_BYTES, remaining_bytes)
    try:
        return read_bounded_file(path, limit)
    except ValueError as err:
        if limit < MAX_IMAGE_ATTACHMENT_BYTES:
            raise _attachments_too_large() from err
        raise _attachment_too_large(
            also_exceeds_budget=remaining_bytes == limit
        ) from err
    except OSError as err:
        raise _attachment_access_error(err) from err


def _read_attachment_group(
    sources: ImageAttachmentSources, remaining_bytes: int, historical: bool = False
) -> ImageAttachmentGroup:
    """Require the whole group to fit before encoding any of its images."""
    projected_bytes = 0
    images: list[tuple[Path, str, int]] = []
    first_error: HomeAssistantError | None = None
    for source in sources:
        try:
            path, mime_type, size = _image_attachment_details(source)
            projected_bytes += size
            if size > MAX_IMAGE_ATTACHMENT_BYTES:
                raise _attachment_too_large(
                    also_exceeds_budget=projected_bytes > remaining_bytes
                )
        except ImageAttachmentBudgetExceeded:
            raise
        except HomeAssistantError as err:
            if not historical:
                raise
            # A later readable file can establish that this whole group will
            # be omitted. Keep the first error if overflow remains unproven.
            first_error = first_error or err
            continue
        if projected_bytes > remaining_bytes:
            raise _attachments_too_large()
        images.append((path, mime_type, size))
    if first_error is not None:
        raise first_error

    raw_images: deque[tuple[str, bytes]] = deque()
    total_bytes = 0
    for path, mime_type, size in images:
        content = _read_attachment_bytes(path, size, remaining_bytes - total_bytes)
        total_bytes += len(content)
        raw_images.append((mime_type, content))

    parts = []
    while raw_images:
        mime_type, content = raw_images.popleft()
        encoded = base64.b64encode(content).decode("ascii")
        data_url = f"data:{mime_type};base64,{encoded}"
        parts.append({"type": "image_url", "image_url": {"url": data_url}})
        del content, encoded
    return ImageAttachmentGroup(parts, total_bytes)


async def async_load_image_attachments(
    hass: HomeAssistant,
    sources: ImageAttachmentSources,
    *,
    budget: ImageAttachmentBudget,
    historical: bool = False,
) -> ImageAttachmentGroup | None:
    """Load admitted images within the request's remaining byte budget."""
    if not sources:
        return None

    return await hass.async_add_executor_job(
        _read_attachment_group, sources, budget.remaining_bytes, historical
    )
