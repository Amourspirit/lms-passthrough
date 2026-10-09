# Providers

## Built-in providers

| Route name | Kind | Streaming | Upstream protocol |
| --- | --- | --- | --- |
| `lm-studio` | `lm_studio` | Native | Transparent passthrough of LM Studio's REST API |
| `android` | `openai_compatible` | Synthesized | OpenAI-compatible (`/v1/chat/completions`) |
| `1min-chat` | `one_min_chat` | Native SSE | 1min.ai chat API |
| `1min-code-generator` | `one_min_code_generator` | Synthesized | 1min.ai code generator API |

## Compatibility matrix

Each cell shows whether the endpoint works with the provider, and any
caveats. Features unsupported by a provider are rejected at the capability
validation layer with a `400` error — never silently dropped.

| Endpoint | `lm-studio` | `android` | `1min-chat` | `1min-code-generator` |
| --- | --- | --- | --- | --- |
| `POST /api/v1/chat` | Native passthrough | Translated | Translated | Translated |
| `POST /v1/chat/completions` | Passthrough (when provider is `lm-studio`) | OpenAI-compatible | OpenAI-compatible | OpenAI-compatible |
| `POST /v1/responses` | Passthrough (when provider is `lm-studio`) | OpenAI-compatible | OpenAI-compatible | OpenAI-compatible |
| `POST /v1/embeddings` | Passthrough (when provider is `lm-studio`) | OpenAI-compatible | Rejected (`400`) | Rejected (`400`) |
| `POST /v1/audio/transcriptions` | Rejected (`400`) | OpenAI-compatible | Two-hop Asset API + `SPEECH_TO_TEXT` | Rejected (`400`) |
| `GET /api/v1/models` | Passthrough (when provider is `lm-studio`) | Translated | Translated | Translated |
| `GET /v1/models` | Passthrough (when provider is `lm-studio`) | Translated | Translated | Translated |

### Streaming

| Provider | `stream: true` behavior |
| --- | --- |
| `lm-studio` | Native SSE lifecycle forwarded byte-for-byte (`chat.start` / `message.delta` / `chat.end`) |
| `android` (`openai_compatible`) | Synthesized from a completed non-streaming response (canonical lifecycle → `chat.completion.chunk` frames) |
| `1min-chat` | Real SSE streaming from the upstream API |
| `1min-code-generator` | Synthesized from a completed response (no native streaming endpoint) |

### Feature support per provider

| Feature | `lm-studio` | `android` | `1min-chat` | `1min-code-generator` |
| --- | --- | --- | --- | --- |
| Tools (function calling) | Yes | Yes | No | No |
| Image inputs | Yes | Yes | No | No |
| Structured output (response_format) | Yes | Yes | No | No |
| Embeddings | Yes | Yes | No | No |
| Transcriptions | No | Yes | Yes | No |
| Conversation state (previous_response_id) | Yes | Yes | Yes | Yes |
| Model listing | Yes | Yes | Yes | Yes |

### Speech to text

`POST /v1/audio/transcriptions` serves `response_format=json` and
`response_format=text`. `srt`, `vtt`, `verbose_json`, and `diarized_json` are
rejected with `400`, as is `stream=true` — see
[ADR-0003](adr/0003-transcription-response-formats.md). Model IDs are forwarded
untranslated, so a client has to name a model the selected provider serves.

The two provider paths differ because the upstreams differ:

- `openai_compatible` forwards the multipart body as-is, so every field the
  upstream understands passes through, including ones the proxy has no notion
  of (`word_timestamps`, `max_tokens`).
- `one_min_chat` is not OpenAI-compatible. The proxy uploads the audio to the
  Asset API, then invokes a `SPEECH_TO_TEXT` feature referencing the returned
  asset key, and reads the transcript from
  `aiRecord.aiRecordDetail.resultObject[0]`. 1min exposes no `response_format`
  knob, so `language` and `prompt` are carried in the feature payload only.
  Its `list_models` also queries `?feature=SPEECH_TO_TEXT` and merges the result
  into the chat catalog; a failure there is swallowed so a broken STT listing
  cannot mark the provider Down for chat traffic.

## Capabilities and rejection

Features unsupported by a provider are rejected at the capability validation
layer with a `400` error containing a descriptive message. The validation
runs before any upstream call, so unsupported requests never reach the
provider.

The set of features checked:

- `stream: true` when the provider does not declare `STREAMING`.
- `tools` present when the provider does not declare `TOOLS`.
- `response_format` present when the provider does not declare
  `STRUCTURED_OUTPUT`.
- Image content in messages when the provider does not declare `IMAGES`.

## Conversation state

Conversation state is managed by the proxy, not delegated to provider-native
conversation IDs. When a request includes `previous_response_id`, the proxy
looks up the stored response chain in SQLite and prepends the assistant's
previous outputs to the request's message list. The state store is purged
periodically based on `persistence.retention_days` (see [Configuration](configuration.md)).

## Writing a provider

Third-party providers implement the canonical `provider_api` protocol and run
the reusable contract suite against their implementation. See
[Provider contract kit](provider-contract.md) for the full authoring guide.
