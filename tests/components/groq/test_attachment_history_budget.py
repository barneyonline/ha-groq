"""Offline regressions for the request-wide image history budget."""

import base64
from pathlib import Path
from unittest.mock import patch

import pytest
from homeassistant.components import conversation
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError

from custom_components.groq import attachments
from custom_components.groq.chat import ChatHistoryCache, _async_chat_log_messages
from custom_components.groq.model_registry import GroqModelRegistry

from .test_foundation import DummyHass

VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


class OverCountAttachments(list[conversation.Attachment]):
    """Detect metadata inspection before the count guard."""

    def __repr__(self):
        raise AssertionError("attachments were rendered before count admission")

    def __eq__(self, _other):
        raise AssertionError("attachments were compared before count admission")


def _attachment(path: Path, mime_type: str = "image/png") -> conversation.Attachment:
    return conversation.Attachment(
        media_content_id=f"media-source://media_source/local/{path.name}",
        mime_type=mime_type,
        path=path,
    )


def _image(path: Path, size: int) -> conversation.Attachment:
    path.write_bytes(b"x" * size)
    return _attachment(path)


def _user(text: str, *images: conversation.Attachment) -> conversation.UserContent:
    return conversation.UserContent(content=text, attachments=list(images) or None)


def _chat_log(
    hass: HomeAssistant, *users: conversation.UserContent
) -> conversation.ChatLog:
    log = conversation.ChatLog(hass, "fixture")
    for user in users:
        log.async_add_user_content(user)
    return log


@pytest.mark.asyncio
async def test_over_count_history_skips_cache_key_construction(
    hass, tmp_path, monkeypatch
):
    monkeypatch.setattr(attachments, "MAX_IMAGE_ATTACHMENTS", 2)
    old = _image(tmp_path / "old.png", 1)
    current = _image(tmp_path / "current.png", 1)
    log = conversation.ChatLog(
        hass,
        "fixture",
        content=[
            conversation.SystemContent(content=""),
            conversation.UserContent(
                content="old", attachments=OverCountAttachments([old, old])
            ),
            _user("current", current),
        ],
    )

    messages = await _async_chat_log_messages(
        hass, GroqModelRegistry(), VISION_MODEL, log, "current"
    )

    assert messages[0] == {"role": "user", "content": "old"}
    assert messages[1]["content"][1]["type"] == "image_url"


@pytest.mark.asyncio
async def test_over_count_current_input_skips_attachment_comparison(
    hass, tmp_path, monkeypatch
):
    monkeypatch.setattr(attachments, "MAX_IMAGE_ATTACHMENTS", 1)
    image = _image(tmp_path / "current.png", 1)
    log = conversation.ChatLog(
        hass,
        "fixture",
        content=[
            conversation.SystemContent(content=""),
            conversation.UserContent(
                content="current", attachments=OverCountAttachments([image, image])
            ),
        ],
    )

    with pytest.raises(HomeAssistantError) as error:
        await _async_chat_log_messages(
            hass,
            GroqModelRegistry(),
            VISION_MODEL,
            log,
            "current",
            OverCountAttachments([image, image]),
        )

    assert error.value.translation_key == "too_many_attachments"


@pytest.mark.asyncio
async def test_attachment_admission_snapshots_the_group_before_executor(
    hass, tmp_path, monkeypatch
):
    monkeypatch.setattr(attachments, "MAX_IMAGE_ATTACHMENTS", 1)
    original = _image(tmp_path / "original.png", 3)
    added = _image(tmp_path / "added.png", 4)
    supplied = [original]
    executor = hass.async_add_executor_job

    def submit(job, *args):
        supplied.append(added)
        return executor(job, *args)

    monkeypatch.setattr(hass, "async_add_executor_job", submit)
    budget = attachments.ImageAttachmentBudget()
    group = await attachments.async_load_image_attachments(
        hass, budget.admit(supplied), budget=budget
    )

    assert group is not None
    assert len(group.images) == 1
    assert group.source_bytes == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("constraint", ["count", "bytes"])
async def test_history_attachment_budget_prefers_latest_and_skips_old_reads(
    hass, tmp_path, monkeypatch, constraint
):
    """Charge images against one budget, including cached images on repeat turns."""
    if constraint == "count":
        monkeypatch.setattr(attachments, "MAX_IMAGE_ATTACHMENTS", 2)
    else:
        monkeypatch.setattr(attachments, "MAX_IMAGE_ATTACHMENT_TOTAL_BYTES", 7)
    old = _image(tmp_path / "old.png", 2)
    recent = _image(tmp_path / "recent.png", 4)
    current = _image(tmp_path / "current.png", 3)
    log = _chat_log(
        hass, _user("old", old), _user("recent", recent), _user("current", current)
    )
    cache = ChatHistoryCache()
    with patch.object(
        attachments, "read_bounded_file", wraps=attachments.read_bounded_file
    ) as read:
        for _ in range(2):
            messages = await _async_chat_log_messages(
                hass,
                GroqModelRegistry(),
                VISION_MODEL,
                log,
                "current",
                [current],
                history_cache=cache,
            )
            assert messages[0] == {"role": "user", "content": "old"}
            assert [message["content"][0]["text"] for message in messages[1:]] == [
                "recent",
                "current",
            ]
            encoded_images = [
                part["image_url"]["url"].partition(",")[2]
                for message in messages[1:]
                for part in message["content"]
                if part["type"] == "image_url"
            ]
            assert [len(image) for image in encoded_images] == [8, 4]
            decoded_sizes = [
                len(base64.b64decode(image, validate=True)) for image in encoded_images
            ]
            assert sum(decoded_sizes) == 7
        assert [call.args[0] for call in read.call_args_list] == [
            current.path,
            recent.path,
        ]


@pytest.mark.asyncio
async def test_history_attachment_budget_checks_only_retained_groups(
    hass, tmp_path, monkeypatch
):
    """Keep validation errors for retained images and omit groups over budget."""
    missing = _attachment(tmp_path / "missing.png")
    current = _image(tmp_path / "current.png", 3)
    log = _chat_log(hass, _user("old", missing), _user("new", current))
    with pytest.raises(HomeAssistantError, match="attachment file does not exist"):
        await _async_chat_log_messages(
            hass, GroqModelRegistry(), VISION_MODEL, log, "new"
        )

    monkeypatch.setattr(attachments, "MAX_IMAGE_ATTACHMENTS", 1)
    messages = await _async_chat_log_messages(
        hass, GroqModelRegistry(), VISION_MODEL, log, "new"
    )
    assert messages[0] == {"role": "user", "content": "old"}
    assert messages[1]["content"][1]["type"] == "image_url"

    with pytest.raises(HomeAssistantError, match="attachment file does not exist"):
        await _async_chat_log_messages(
            hass, GroqModelRegistry(), VISION_MODEL, log, "new", [missing]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("limit_name", "limit", "translation_key"),
    [
        ("MAX_IMAGE_ATTACHMENTS", 1, "too_many_attachments"),
        ("MAX_IMAGE_ATTACHMENT_TOTAL_BYTES", 7, "attachments_too_large"),
    ],
)
async def test_current_image_group_over_budget_is_rejected_before_opening(
    hass, tmp_path, monkeypatch, limit_name, limit, translation_key
):
    monkeypatch.setattr(attachments, limit_name, limit)
    first = (
        _attachment(tmp_path / "missing.png")
        if limit_name == "MAX_IMAGE_ATTACHMENTS"
        else _image(tmp_path / "first.png", 4)
    )
    second = _image(tmp_path / "second.png", 4)
    log = _chat_log(hass, _user("current", first, second))

    with (
        patch.object(attachments, "read_bounded_file") as read,
        pytest.raises(attachments.ImageAttachmentBudgetExceeded) as error,
    ):
        await _async_chat_log_messages(
            hass, GroqModelRegistry(), VISION_MODEL, log, "current", [first, second]
        )

    assert error.value.translation_key == translation_key
    read.assert_not_called()


@pytest.mark.asyncio
async def test_attachment_source_is_extracted_once_on_cache_miss(hass, tmp_path):
    path_reads = []

    class ObservedAttachment(conversation.Attachment):
        def __getattribute__(self, name):
            if name == "path":
                path_reads.append(name)
            return super().__getattribute__(name)

    path = tmp_path / "camera.png"
    path.write_bytes(b"x")
    image = ObservedAttachment("media-source://camera/snapshot", "image/png", path)
    log = conversation.ChatLog(
        hass,
        "fixture",
        content=[conversation.UserContent(content="current", attachments=[image])],
    )

    messages = await _async_chat_log_messages(
        hass, GroqModelRegistry(), VISION_MODEL, log, "current"
    )

    assert messages[0]["content"][1]["image_url"]["url"] == "data:image/png;base64,eA=="
    assert path_reads == ["path"]


@pytest.mark.asyncio
async def test_attachment_cache_reuses_sources_across_different_prompts(hass, tmp_path):
    image = _image(tmp_path / "camera.png", 3)
    same_source = conversation.Attachment(
        media_content_id="media-source://media_source/local/alias.png",
        mime_type=image.mime_type,
        path=image.path,
    )
    log = _chat_log(hass, _user("first", image), _user("second", same_source))
    with patch.object(
        attachments, "read_bounded_file", wraps=attachments.read_bounded_file
    ) as read:
        messages = await _async_chat_log_messages(
            hass, GroqModelRegistry(), VISION_MODEL, log, "second"
        )

    assert [message["content"][0]["text"] for message in messages] == [
        "first",
        "second",
    ]
    assert messages[0]["content"][1] == messages[1]["content"][1]
    read.assert_called_once()


@pytest.mark.asyncio
async def test_empty_attachment_iterable_stays_plain_text(hass):
    log = _chat_log(
        hass, conversation.UserContent(content="current", attachments=iter(()))
    )

    messages = await _async_chat_log_messages(
        hass, GroqModelRegistry(), VISION_MODEL, log, "current"
    )

    assert messages == [{"role": "user", "content": "current"}]


@pytest.mark.asyncio
async def test_duplicate_cached_image_still_costs_request_bytes(
    hass, tmp_path, monkeypatch
):
    """Reusing an encoded image in history still adds its bytes to the payload."""
    monkeypatch.setattr(attachments, "MAX_IMAGE_ATTACHMENT_TOTAL_BYTES", 7)
    repeated = _image(tmp_path / "repeated.png", 3)
    current = _image(tmp_path / "current.png", 3)
    log = _chat_log(
        hass, _user("old", repeated), _user("recent", repeated), _user("new", current)
    )
    with patch.object(
        attachments, "read_bounded_file", wraps=attachments.read_bounded_file
    ) as read:
        messages = await _async_chat_log_messages(
            hass, GroqModelRegistry(), VISION_MODEL, log, "new"
        )
    assert messages[0] == {"role": "user", "content": "old"}
    assert messages[1]["content"][1]["type"] == "image_url"
    assert [call.args[0] for call in read.call_args_list] == [
        current.path,
        repeated.path,
    ]


@pytest.mark.asyncio
async def test_over_budget_history_group_is_rejected_before_opening(
    hass, tmp_path, monkeypatch
):
    """A larger group cannot consume the budget ahead of another eligible turn."""
    monkeypatch.setattr(attachments, "MAX_IMAGE_ATTACHMENT_TOTAL_BYTES", 7)
    small = _image(tmp_path / "small.png", 2)
    first = _image(tmp_path / "first.png", 4)
    second = _image(tmp_path / "second.png", 4)
    log = _chat_log(
        hass, _user("small", small), _user("large", first, second), _user("new")
    )
    with patch.object(
        attachments, "read_bounded_file", wraps=attachments.read_bounded_file
    ) as read:
        messages = await _async_chat_log_messages(
            hass, GroqModelRegistry(), VISION_MODEL, log, "new"
        )
    read.assert_called_once_with(small.path, 7)
    assert messages[0]["content"][1]["type"] == "image_url"
    assert messages[1:] == [
        {"role": "user", "content": "large"},
        {"role": "user", "content": "new"},
    ]

    with pytest.raises(HomeAssistantError, match="vision-capable model"):
        await _async_chat_log_messages(
            hass, GroqModelRegistry(), "llama-3.1-8b-instant", log, "new"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("mime_type", ["image/png", "audio/wav"])
@pytest.mark.parametrize("unusable_first", [False, True])
async def test_proven_history_overflow_omits_group_with_unusable_attachment(
    hass, tmp_path, monkeypatch, mime_type, unusable_first
):
    """An omitted history group must not fail the current request."""
    monkeypatch.setattr(attachments, "MAX_IMAGE_ATTACHMENT_TOTAL_BYTES", 7)
    old = _image(tmp_path / "old.png", 2)
    missing = _attachment(tmp_path / "missing", mime_type)
    current = _image(tmp_path / "current.png", 6)
    historical = (missing, old) if unusable_first else (old, missing)
    log = _chat_log(hass, _user("old", *historical), _user("new", current))

    with patch.object(
        attachments, "read_bounded_file", wraps=attachments.read_bounded_file
    ) as read:
        messages = await _async_chat_log_messages(
            hass, GroqModelRegistry(), VISION_MODEL, log, "new"
        )

    read.assert_called_once_with(current.path, 7)
    assert messages[0] == {"role": "user", "content": "old"}
    assert messages[1]["content"][1]["type"] == "image_url"


@pytest.mark.asyncio
@pytest.mark.parametrize("oversized_first", [False, True])
async def test_historical_group_overflow_is_independent_of_image_size_order(
    hass, tmp_path, monkeypatch, oversized_first
):
    monkeypatch.setattr(attachments, "MAX_IMAGE_ATTACHMENT_BYTES", 10)
    monkeypatch.setattr(attachments, "MAX_IMAGE_ATTACHMENT_TOTAL_BYTES", 20)
    oversized = _image(tmp_path / "oversized.png", 11)
    smaller = _image(tmp_path / "smaller.png", 8)
    current = _image(tmp_path / "current.png", 2)
    historical = (oversized, smaller) if oversized_first else (smaller, oversized)
    log = _chat_log(hass, _user("old", *historical), _user("new", current))
    registry = GroqModelRegistry()

    messages = await _async_chat_log_messages(hass, registry, VISION_MODEL, log, "new")
    assert messages[0] == {"role": "user", "content": "old"}
    assert messages[1]["content"][1]["type"] == "image_url"

    monkeypatch.setattr(attachments, "MAX_IMAGE_ATTACHMENT_TOTAL_BYTES", 21)
    with pytest.raises(HomeAssistantError) as error:
        await _async_chat_log_messages(hass, registry, VISION_MODEL, log, "new")
    assert error.value.translation_key == "attachment_too_large"


@pytest.mark.asyncio
async def test_current_group_preserves_first_validation_error(
    hass, tmp_path, monkeypatch
):
    monkeypatch.setattr(attachments, "MAX_IMAGE_ATTACHMENT_TOTAL_BYTES", 7)
    missing = _attachment(tmp_path / "missing.png")
    later = _image(tmp_path / "over-budget.png", 8)
    log = _chat_log(hass, _user("current", missing, later))

    with pytest.raises(HomeAssistantError) as error:
        await _async_chat_log_messages(
            hass, GroqModelRegistry(), VISION_MODEL, log, "current"
        )

    assert error.value.translation_key == "attachment_file_missing"


@pytest.mark.asyncio
@pytest.mark.parametrize("per_image_limit", [2, 3])
async def test_attachment_growth_during_read_cannot_exceed_request_budget(
    hass, tmp_path, monkeypatch, per_image_limit
):
    """Keep actual reads bounded even when a file grows after group preflight."""
    monkeypatch.setattr(attachments, "MAX_IMAGE_ATTACHMENT_TOTAL_BYTES", 3)
    monkeypatch.setattr(attachments, "MAX_IMAGE_ATTACHMENT_BYTES", per_image_limit)
    old = _image(tmp_path / "old.png", 2)
    current = _image(tmp_path / "current.png", 1)
    log = _chat_log(hass, _user("old", old), _user("new", current))
    read_bounded_file = attachments.read_bounded_file

    def grow_on_read(path, limit):
        if path == old.path:
            path.write_bytes(b"xxx")
            assert limit == 2
        return read_bounded_file(path, limit)

    monkeypatch.setattr(attachments, "read_bounded_file", grow_on_read)
    messages = await _async_chat_log_messages(
        hass, GroqModelRegistry(), VISION_MODEL, log, "new"
    )
    assert messages[0] == {"role": "user", "content": "old"}
    assert messages[1]["content"][1]["type"] == "image_url"


@pytest.mark.asyncio
async def test_historical_image_over_both_limits_preserves_current_request(
    hass, tmp_path, monkeypatch
):
    monkeypatch.setattr(attachments, "MAX_IMAGE_ATTACHMENT_BYTES", 3)
    monkeypatch.setattr(attachments, "MAX_IMAGE_ATTACHMENT_TOTAL_BYTES", 6)
    old = _image(tmp_path / "old.png", 4)
    current = _image(tmp_path / "current.png", 3)
    log = _chat_log(hass, _user("old", old), _user("new", current))
    registry = GroqModelRegistry()

    messages = await _async_chat_log_messages(hass, registry, VISION_MODEL, log, "new")
    assert messages[0] == {"role": "user", "content": "old"}
    assert messages[1]["content"][1]["type"] == "image_url"

    monkeypatch.setattr(attachments, "MAX_IMAGE_ATTACHMENT_TOTAL_BYTES", 7)
    with pytest.raises(HomeAssistantError) as historical_error:
        await _async_chat_log_messages(hass, registry, VISION_MODEL, log, "new")
    assert historical_error.value.translation_key == "attachment_too_large"

    monkeypatch.setattr(attachments, "MAX_IMAGE_ATTACHMENT_TOTAL_BYTES", 3)
    current_log = _chat_log(hass, _user("current", old))
    with pytest.raises(HomeAssistantError) as current_error:
        await _async_chat_log_messages(
            hass, registry, VISION_MODEL, current_log, "current"
        )
    assert current_error.value.translation_key == "attachment_too_large"


@pytest.mark.asyncio
async def test_attachment_group_growth_skips_later_reads_that_no_longer_fit(
    hass, tmp_path, monkeypatch
):
    """A growth within the first file's read bound can exhaust the group budget."""
    monkeypatch.setattr(attachments, "MAX_IMAGE_ATTACHMENT_TOTAL_BYTES", 6)
    first = _image(tmp_path / "first.png", 3)
    second = _image(tmp_path / "second.png", 3)
    log = _chat_log(hass, _user("old", first, second), _user("new"))
    read_bounded_file = attachments.read_bounded_file
    opened = []

    def grow_on_first_read(path, limit):
        opened.append(path)
        if path == first.path:
            path.write_bytes(b"xxxx")
        return read_bounded_file(path, limit)

    monkeypatch.setattr(attachments, "read_bounded_file", grow_on_first_read)
    with patch.object(
        attachments.base64, "b64encode", wraps=attachments.base64.b64encode
    ) as encode:
        messages = await _async_chat_log_messages(
            hass, GroqModelRegistry(), VISION_MODEL, log, "new"
        )
    encode.assert_not_called()
    assert opened == [first.path]
    assert messages[0] == {"role": "user", "content": "old"}


@pytest.mark.asyncio
async def test_attachment_preflight_translates_os_failure_and_accepts_empty_iterable(
    tmp_path,
):
    """An inaccessible attachment fails as unreadable, while no images need no I/O."""
    current = _image(tmp_path / "current.png", 3)
    budget = attachments.ImageAttachmentBudget()
    with (
        patch.object(type(current.path), "stat", side_effect=PermissionError),
        pytest.raises(
            HomeAssistantError, match="attachment could not be read"
        ) as error,
    ):
        await attachments.async_load_image_attachments(
            DummyHass(), budget.admit((current,)), budget=budget
        )
    assert error.value.translation_key == "attachment_file_unreadable"
    assert (
        await attachments.async_load_image_attachments(
            DummyHass(), budget.admit(iter(())), budget=budget
        )
        is None
    )
