"""Regression tests for security boundaries in the Groq integration."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from homeassistant.exceptions import (
    ConfigEntryAuthFailed,
    HomeAssistantError,
    Unauthorized,
)

from custom_components.groq.api import GroqApiClient, GroqResponseError
from custom_components.groq.attachments import (
    _read_attachment_data_url,
    async_attachment_content_parts,
)
from custom_components.groq.services import (
    ATTR_AUDIO_PATH,
    ATTR_CAMERA_ENTITY_ID,
    ATTR_IMAGE_PATH,
    _audio_from_call,
    _image_from_camera_target,
    _image_url_from_call,
)


class _Auth:
    def __init__(self, user) -> None:
        self._user = user

    async def async_get_user(self, _user_id):
        return self._user


class _Permissions:
    def __init__(self, allowed: bool) -> None:
        self.allowed = allowed
        self.checked: list[tuple[str, str]] = []

    def check_entity(self, entity_id: str, policy: str) -> bool:
        self.checked.append((entity_id, policy))
        return self.allowed


class _Hass:
    def __init__(self, user=None) -> None:
        self.auth = _Auth(user)
        self.config = SimpleNamespace(is_allowed_path=lambda _path: True)

    async def async_add_executor_job(self, func, *args):
        return func(*args)


def _service_call(data, *, user_id: str | None = "user-id"):
    return SimpleNamespace(
        data=data,
        context=SimpleNamespace(user_id=user_id),
    )


@pytest.mark.asyncio
async def test_camera_media_requires_entity_read_permission(monkeypatch) -> None:
    """A caller must be able to read the selected camera."""
    permissions = _Permissions(False)
    user = SimpleNamespace(is_admin=False, permissions=permissions)

    async def unexpected_camera_read(*_args):
        raise AssertionError("camera must not be read before authorization")

    monkeypatch.setattr(
        "custom_components.groq.services.camera.async_get_image",
        unexpected_camera_read,
    )

    with pytest.raises(Unauthorized):
        await _image_from_camera_target(
            _Hass(user),
            _service_call({ATTR_CAMERA_ENTITY_ID: "camera.private"}),
        )

    assert permissions.checked == [("camera.private", "read")]


@pytest.mark.asyncio
async def test_authorized_camera_media_remains_available(monkeypatch) -> None:
    """A caller with entity read permission can still use a camera."""
    permissions = _Permissions(True)
    user = SimpleNamespace(is_admin=False, permissions=permissions)

    async def camera_read(*_args):
        return SimpleNamespace(content=b"image", content_type="image/png")

    monkeypatch.setattr(
        "custom_components.groq.services.camera.async_get_image",
        camera_read,
    )

    result = await _image_from_camera_target(
        _Hass(user),
        _service_call({ATTR_CAMERA_ENTITY_ID: "camera.allowed"}),
    )

    assert result == "data:image/png;base64,aW1hZ2U="
    assert permissions.checked == [("camera.allowed", "read")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("resolver", "field"),
    [
        (_image_url_from_call, ATTR_IMAGE_PATH),
        (_audio_from_call, ATTR_AUDIO_PATH),
    ],
)
async def test_raw_media_paths_require_admin(tmp_path, resolver, field) -> None:
    """Non-admin callers cannot make Home Assistant read local paths."""
    media_path = tmp_path / ("image.png" if field == ATTR_IMAGE_PATH else "audio.wav")
    media_path.write_bytes(b"private")
    user = SimpleNamespace(is_admin=False, permissions=_Permissions(True))

    with pytest.raises(Unauthorized):
        await resolver(_Hass(user), _service_call({field: str(media_path)}))


@pytest.mark.asyncio
async def test_admin_raw_media_path_remains_available(tmp_path) -> None:
    """Administrators can still use explicitly allowlisted local media."""
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"image")
    user = SimpleNamespace(is_admin=True, permissions=_Permissions(True))

    result = await _image_url_from_call(
        _Hass(user),
        _service_call({ATTR_IMAGE_PATH: str(image_path)}),
    )

    assert result == "data:image/png;base64,aW1hZ2U="


@pytest.mark.asyncio
async def test_attachment_count_and_total_size_are_bounded(tmp_path) -> None:
    """Attachment collections are bounded before building the request payload."""
    paths = []
    for index in range(5):
        path = tmp_path / f"{index}.png"
        path.write_bytes(b"abc")
        paths.append(SimpleNamespace(mime_type="image/png", path=path))

    with pytest.raises(HomeAssistantError, match="at most 4 image attachments"):
        await async_attachment_content_parts(_Hass(), paths, text="Describe")

    with (
        patch("custom_components.groq.attachments.MAX_IMAGE_ATTACHMENT_TOTAL_BYTES", 5),
        pytest.raises(HomeAssistantError, match="combined attachment size"),
    ):
        await async_attachment_content_parts(_Hass(), paths[:2], text="Describe")


@pytest.mark.asyncio
async def test_attachment_collection_within_limits_remains_available(tmp_path) -> None:
    """A small legitimate attachment collection still produces content parts."""
    path = tmp_path / "image.png"
    path.write_bytes(b"abc")

    result = await async_attachment_content_parts(
        _Hass(),
        [SimpleNamespace(mime_type="image/png", path=path)],
        text="Describe",
    )

    assert result == [
        {"type": "text", "text": "Describe"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,YWJj"},
        },
    ]


def test_attachment_read_rechecks_size_after_stat(tmp_path) -> None:
    """A file that grows after its stat check cannot bypass the per-file limit."""
    path = tmp_path / "growing.png"
    path.write_bytes(b"123456")
    stat_result = path.stat()

    with (
        patch.object(
            type(path),
            "stat",
            return_value=SimpleNamespace(st_size=3, st_mode=stat_result.st_mode),
        ),
        patch("custom_components.groq.attachments.MAX_IMAGE_ATTACHMENT_BYTES", 5),
        pytest.raises(HomeAssistantError, match="exceeds the 10 MB"),
    ):
        _read_attachment_data_url(path, "image/png")


class _ChunkedContent:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def iter_chunked(self, _size):
        for chunk in self._chunks:
            yield chunk

    def __aiter__(self):
        self._iterator = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._iterator)
        except StopIteration:
            raise StopAsyncIteration


class _Response:
    def __init__(
        self,
        chunks: list[bytes],
        *,
        status: int = 200,
        content_type: str = "application/json",
        content_length: int | None = None,
    ) -> None:
        self.status = status
        self.headers = {"content-type": content_type}
        if content_length is not None:
            self.headers["content-length"] = str(content_length)
        self.content = _ChunkedContent(chunks)

    async def read(self):
        return b"".join(self.content._chunks)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False


class _UnreadableContent:
    async def iter_chunked(self, _size):
        raise AssertionError("authentication responses must not be read")
        yield b""  # pragma: no cover - make this an async generator


class _Session:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls = []

    def request(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_provider_response_bodies_and_streams_are_bounded() -> None:
    """Chunked provider responses cannot exceed configured byte ceilings."""
    session = _Session(
        [
            _Response([b"123", b"45"]),
            _Response([b"audio"]),
            _Response([b'data: {"value":"oversized"}\n']),
        ]
    )
    client = GroqApiClient(_Hass(), api_key="key", session=session)

    with (
        patch("custom_components.groq.api.MAX_JSON_RESPONSE_BYTES", 4),
        pytest.raises(GroqResponseError, match="response exceeded"),
    ):
        await client._request_json("GET", "/models")

    with (
        patch("custom_components.groq.api.MAX_AUDIO_RESPONSE_BYTES", 4),
        pytest.raises(GroqResponseError, match="response exceeded"),
    ):
        await client._request_audio("POST", "/audio/speech", json_payload={})

    with (
        patch("custom_components.groq.api.MAX_STREAM_RESPONSE_BYTES", 8),
        pytest.raises(GroqResponseError, match="stream exceeded"),
    ):
        async for _ in client._request_stream(
            "POST", "/chat/completions", json_payload={}
        ):
            pass


@pytest.mark.asyncio
async def test_provider_content_length_is_rejected_before_reading() -> None:
    """An oversized declared response is rejected without buffering its body."""
    response = _Response([b"{}"], content_length=5)
    session = _Session([response])
    client = GroqApiClient(_Hass(), api_key="key", session=session)

    with (
        patch("custom_components.groq.api.MAX_JSON_RESPONSE_BYTES", 4),
        patch.object(response, "read", side_effect=AssertionError("must not read")),
        pytest.raises(GroqResponseError, match="response exceeded"),
    ):
        await client._request_json("GET", "/models")


@pytest.mark.asyncio
async def test_provider_auth_status_is_handled_before_body_limits() -> None:
    """Authentication failures take precedence over response body limits."""
    json_response = _Response([b"oversized"], status=401, content_length=5)
    audio_response = _Response([b"oversized"], status=401, content_length=5)
    json_response.content = _UnreadableContent()
    audio_response.content = _UnreadableContent()
    session = _Session([json_response, audio_response])
    client = GroqApiClient(_Hass(), api_key="key", session=session)

    with (
        patch("custom_components.groq.api.MAX_JSON_RESPONSE_BYTES", 4),
        pytest.raises(ConfigEntryAuthFailed),
    ):
        await client._request_json("GET", "/models")

    with (
        patch("custom_components.groq.api.MAX_AUDIO_RESPONSE_BYTES", 4),
        pytest.raises(ConfigEntryAuthFailed),
    ):
        await client._request_audio("POST", "/audio/speech", json_payload={})


@pytest.mark.asyncio
async def test_provider_limit_ignores_invalid_content_length() -> None:
    """A malformed length header cannot disable the bounded chunked read."""
    invalid_length = _Response([b"{}"])
    invalid_length.headers["content-length"] = "not-a-number"
    session = _Session([invalid_length])
    client = GroqApiClient(_Hass(), api_key="key", session=session)

    assert await client._request_json("GET", "/models") == {}


@pytest.mark.asyncio
async def test_provider_requests_never_follow_redirects() -> None:
    """Authorization headers cannot be forwarded through provider redirects."""
    session = _Session(
        [
            _Response([b"{}"]),
            _Response([b'data: {"ok": true}\n', b"data: [DONE]\n"]),
            _Response([b"data: [DONE]"]),
            _Response([b"audio"], content_type="audio/wav"),
        ]
    )
    client = GroqApiClient(_Hass(), api_key="key", session=session)

    assert await client._request_json("GET", "/models") == {}
    assert [
        event
        async for event in client._request_stream(
            "POST", "/chat/completions", json_payload={}
        )
    ] == [{"ok": True}]
    assert [
        event
        async for event in client._request_stream(
            "POST", "/chat/completions", json_payload={}
        )
    ] == []
    assert (
        await client._request_audio("POST", "/audio/speech", json_payload={})
        == b"audio"
    )

    assert all(kwargs["allow_redirects"] is False for _, kwargs in session.calls)
