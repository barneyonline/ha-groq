# Groq API Spec

Last reviewed: 2026-05-09

This document captures the Groq API surface used or likely to be used by this
Home Assistant integration. It is intentionally focused on Groq speech/TTS,
model discovery, rate limits, and error handling rather than the full Groq
platform.

Primary sources:

- [Groq Text to Speech](https://console.groq.com/docs/text-to-speech)
- [Groq Orpheus Text to Speech](https://console.groq.com/docs/text-to-speech/orpheus)
- [Groq API Reference](https://console.groq.com/docs/api-reference)
- [Groq Supported Models](https://console.groq.com/docs/models)
- [Groq Rate Limits](https://console.groq.com/docs/rate-limits)
- [Groq Error Codes](https://console.groq.com/docs/errors)
- [Groq OpenAI Compatibility](https://console.groq.com/docs/openai)

## Authentication

Groq uses bearer-token authentication.

```http
Authorization: Bearer ${GROQ_API_KEY}
```

The API key can be created in Groq Console. Groq recommends keeping the key in
the `GROQ_API_KEY` environment variable for SDK and local development usage.

The OpenAI-compatible base URL is:

```text
https://api.groq.com/openai/v1
```

The integration currently stores the API key in the Home Assistant config entry
and sends it as a bearer token on outbound Groq API calls.

## Text To Speech Endpoint

```http
POST https://api.groq.com/openai/v1/audio/speech
Content-Type: application/json
Authorization: Bearer ${GROQ_API_KEY}
```

Purpose: convert text input into generated audio.

### Request Body

| Field | Type | Required | Groq docs behavior | Integration behavior |
| --- | --- | --- | --- | --- |
| `model` | string | Yes | TTS model ID. For Orpheus, use `canopylabs/orpheus-v1-english` or `canopylabs/orpheus-arabic-saudi`. | Configurable at setup/options and overridable per `tts.speak` call. |
| `input` | string | Yes | Text to convert to speech. Orpheus docs limit input to 200 characters. Bracketed directions can be embedded for English Orpheus. | Defaults to Home Assistant `message`; overridable via `options.input`. Vocal directions are prepended before the length check. |
| `voice` | string | Yes | Voice persona ID for the selected model. | Configurable at setup/options and overridable per call. |
| `response_format` | string | Optional | Generic API reference lists `flac`, `mp3`, `mulaw`, `ogg`, and `wav`; Orpheus docs state only `wav` is supported and defaulted for Orpheus. | Currently restricted to `wav` because the integration targets Orpheus. |
| `sample_rate` | integer | Optional | Generic API reference lists allowed values `8000`, `16000`, `22050`, `24000`, `32000`, `44100`, and `48000`, defaulting to `48000`. | Not currently exposed. Do not enable for Orpheus unless verified against Orpheus docs. |
| `speed` | number | Optional | Generic API reference lists range `0.5` to `5`, defaulting to `1`. | Not currently exposed. Do not enable for Orpheus unless verified against Orpheus docs. |

### Response

Successful speech generation returns audio bytes. For Orpheus, Groq documents
`wav` output. The generic API reference also says the endpoint returns an audio
file in `wav` format.

The integration should treat successful audio responses as binary data, not JSON.
If Groq returns JSON from the speech endpoint, treat it as an error payload unless
future docs define a JSON success response for a supported mode.

### Example Request

```bash
curl https://api.groq.com/openai/v1/audio/speech \
  -X POST \
  -H "Authorization: Bearer ${GROQ_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "canopylabs/orpheus-v1-english",
    "input": "[cheerful] Welcome home.",
    "voice": "troy",
    "response_format": "wav"
  }' \
  --output speech.wav
```

## Orpheus Models

Groq currently documents two Orpheus TTS models hosted on GroqCloud.

| Model ID | Language | Vocal directions | Pricing | Production status |
| --- | --- | --- | --- | --- |
| `canopylabs/orpheus-v1-english` | English | Supported | $22.00 per 1M characters | Preview model |
| `canopylabs/orpheus-arabic-saudi` | Arabic, Saudi dialect | Not supported | $40.00 per 1M characters | Preview model |

The Groq models page lists both Orpheus models as preview models. Preview models
are for evaluation and can be changed or removed with less notice than production
models. The integration should avoid assuming these IDs, pricing, or limits are
permanent.

Both model pages describe the model capability as text-to-speech with text input
and audio output.

### Orpheus Constraints

- Input text is limited to 200 characters per request.
- Orpheus `response_format` is documented as `wav` only.
- English Orpheus supports bracketed vocal directions.
- Arabic Saudi Orpheus does not support vocal directions.
- Model access can depend on the Groq organization/project configuration.

## Voices

### English Orpheus Voices

Use these with `canopylabs/orpheus-v1-english`:

| Voice name | Voice ID | Gender |
| --- | --- | --- |
| Autumn | `autumn` | Female |
| Diana | `diana` | Female |
| Hannah | `hannah` | Female |
| Austin | `austin` | Male |
| Daniel | `daniel` | Male |
| Troy | `troy` | Male |

### Arabic Saudi Orpheus Voices

Use these with `canopylabs/orpheus-arabic-saudi`:

| Voice name | Voice ID | Gender |
| --- | --- | --- |
| Abdullah | `abdullah` | Male |
| Fahad | `fahad` | Male |
| Sultan | `sultan` | Male |
| Lulwa | `lulwa` | Female |
| Noura | `noura` | Female |
| Aisha | `aisha` | Female |

Note: the Arabic model page text says the model offers four distinct voices, but
the Orpheus guide lists six Arabic Saudi voices. The integration should follow
the explicit voice table unless Groq updates the model page or API behavior.

## Vocal Directions

Vocal directions are supported by `canopylabs/orpheus-v1-english` only.

Groq documents directions as bracketed descriptors embedded in the input text,
for example:

```text
[cheerful] Welcome home.
```

Implementation rules for this integration:

- Accept user input with or without brackets.
- Normalize `cheerful` to `[cheerful]`.
- Prepend the normalized direction to the final Groq `input`.
- Count the final composed input toward the 200-character Orpheus limit.
- Do not advertise vocal directions as supported for Arabic Saudi Orpheus.

Groq guidance:

- Fewer or no directions produce a more natural conversational cadence.
- More directions produce a more acted or expressive performance.
- One- or two-word directions work best.
- There is no exhaustive official direction list; vague or unfamiliar directions
  may be ignored by the model.

Useful examples to expose in docs or UI hints:

- `cheerful`
- `friendly`
- `casual`
- `warm`
- `professionally`
- `authoritatively`
- `whisper`
- `excited`
- `dramatic`
- `deadpan`
- `sarcastic`
- `gravelly whisper`
- `rapid babbling`
- `singsong`
- `breathy`

## Model Discovery API

### List Models

```http
GET https://api.groq.com/openai/v1/models
Authorization: Bearer ${GROQ_API_KEY}
```

Purpose: return a JSON list of active models available through the GroqCloud
Models API.

Response shape:

```json
{
  "object": "list",
  "data": [
    {
      "id": "model-id",
      "object": "model",
      "created": 1693721698,
      "owned_by": "Provider",
      "active": true,
      "context_window": 8192,
      "public_apps": null
    }
  ]
}
```

Integration behavior:

- Use this endpoint only for discovery.
- Filter discovered models to known TTS-capable model IDs or IDs matching the
  documented Orpheus naming pattern.
- Keep built-in Orpheus IDs available even if discovery fails.
- Discovery failure should not block setup when built-in model IDs are usable.

### Retrieve Model

```http
GET https://api.groq.com/openai/v1/models/{model}
Authorization: Bearer ${GROQ_API_KEY}
```

Purpose: return details for a model ID.

Documented response fields include:

- `id`
- `object`
- `created`
- `owned_by`
- `active`
- `context_window`
- `public_apps`
- `max_completion_tokens`

This integration does not currently call the retrieve endpoint, but it is useful
for future validation or diagnostics.

## Rate Limits

Groq rate limits apply at the organization level. A request can be rejected when
any one applicable limit is exceeded.

Groq documents these rate-limit dimensions:

| Abbreviation | Meaning |
| --- | --- |
| RPM | Requests per minute |
| RPD | Requests per day |
| TPM | Tokens per minute |
| TPD | Tokens per day |
| ASH | Audio seconds per hour |
| ASD | Audio seconds per day |

Groq also states that cached tokens do not count toward rate limits.

### Free Plan Limits For Orpheus

| Model ID | RPM | RPD | TPM | TPD | ASH | ASD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `canopylabs/orpheus-v1-english` | 10 | 100 | 1.2K | 3.6K | - | - |
| `canopylabs/orpheus-arabic-saudi` | 10 | 100 | 1.2K | 3.6K | - | - |

### Developer Plan Limits For Orpheus

The supported models page lists these Developer plan limits for preview Orpheus
models:

| Model ID | TPM | RPM | Context window | Max completion tokens |
| --- | ---: | ---: | ---: | ---: |
| `canopylabs/orpheus-v1-english` | 50K | 250 | 4,000 | 50,000 |
| `canopylabs/orpheus-arabic-saudi` | 50K | 250 | 4,000 | 50,000 |

### Rate Limit Headers

Groq documents these response headers:

| Header | Meaning |
| --- | --- |
| `retry-after` | Seconds to wait before retrying after a 429. Only present when a 429 limit is hit. |
| `x-ratelimit-limit-requests` | Request limit, documented as Requests Per Day. |
| `x-ratelimit-limit-tokens` | Token limit, documented as Tokens Per Minute. |
| `x-ratelimit-remaining-requests` | Remaining requests, documented as Requests Per Day. |
| `x-ratelimit-remaining-tokens` | Remaining tokens, documented as Tokens Per Minute. |
| `x-ratelimit-reset-requests` | Time until request limit reset. |
| `x-ratelimit-reset-tokens` | Time until token limit reset. |

### Required Integration Behavior

- Keep the local free-tier guard conservative for built-in Orpheus models.
- Count uncached requests only; local cache hits should not consume local guard
  counters because they do not call Groq.
- Estimate text token usage conservatively from final `input` length.
- Surface 429 responses as rate-limit errors, including `retry-after` and reset
  headers when Groq provides them.
- Do not auto-retry 429s in Home Assistant service calls unless future UX design
  explicitly adds bounded retry behavior.

## Error Handling

Groq uses standard HTTP status codes and returns JSON error bodies for failures.

| Status | Meaning | Integration handling |
| --- | --- | --- |
| 400 | Bad request, invalid syntax or unsupported fields. | Show/log validation detail when available. |
| 401 | Unauthorized, missing or invalid API key. | Trigger Home Assistant reauthentication. |
| 403 | Forbidden, permission or model access restriction. | Trigger reauthentication or model-access guidance depending on body. |
| 404 | Resource not found. | Treat as configuration/model/endpoint error. |
| 413 | Request body too large. | Treat as user/config input error. |
| 422 | Request understood but semantically invalid. | Surface model/input validation detail. |
| 424 | Failed dependency. | Surface as upstream dependency failure. |
| 429 | Too many requests. | Raise rate-limit error and include retry/reset details. |
| 498 | Flex tier capacity exceeded. | Treat as transient capacity error. |
| 499 | Request cancelled by caller. | Treat as cancellation. |
| 500 | Internal server error. | Transient server error. |
| 502 | Bad gateway. | Transient server/upstream error. |
| 503 | Service unavailable. | Transient server overload or maintenance. |

Groq states server error responses are not charged. The integration should still
avoid tight retry loops inside Home Assistant.

## OpenAI Compatibility Notes

Groq is mostly compatible with OpenAI client libraries when using:

```text
base_url = https://api.groq.com/openai/v1
```

For this integration, the compatibility point that matters most is endpoint
shape:

- Speech endpoint path: `/audio/speech`
- Models endpoint path: `/models`
- Bearer-token authentication
- JSON request body for speech
- Binary audio response for speech

Groq's OpenAI compatibility page lists unsupported OpenAI features for other API
areas. Those unsupported fields are not part of this integration's TTS request
payload and should not be sent.

## Implementation Checklist

- Keep `model`, `input`, `voice`, and `response_format` configurable.
- Keep `response_format` restricted to `wav` while the integration is Orpheus-only.
- Consider adding `sample_rate` and `speed` only if Groq confirms Orpheus support.
- Keep vocal directions English-model-only in docs and UI help text.
- Use `GET /models` for discovery but preserve built-in Orpheus fallback values.
- Preserve current local free-tier protection behavior for Orpheus free-plan
  limits.
- Parse and surface Groq error JSON without logging API keys.
- Treat non-audio successful responses from `/audio/speech` as invalid for the
  current integration.
