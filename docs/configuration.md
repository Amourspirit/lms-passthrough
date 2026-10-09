# Configuration

The service loads a single YAML configuration file and resolves `${ENV_VAR}`
references into process environment variables at startup. Secrets (API keys,
auth tokens) must live in environment variables or a `.env` file in the
working directory; the file is never committed.

## Quick start

```yaml
host: 0.0.0.0
port: 1234
default_provider: lm-studio
security:
  enabled: false
persistence:
  path: ./storage/state/lms-passthrough.sqlite3
  retention_days: 30
logging:
  level: INFO
  format: json
providers:
  - name: lm-studio
    kind: lm_studio
    base_url: http://127.0.0.1:8999
```

## Top-level keys

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `host` | `str` | `"127.0.0.1"` | Bind address for the HTTP server. |
| `port` | `int` | `8000` | Port for the HTTP server. |
| `default_provider` | `str` | `"lm-studio"` | Name of a configured provider to use when `X-LMS-Provider` is absent. |
| `failover` | `bool` | `false` | Serve a request from a healthy provider when the first-selected one is unreachable. See [Failover](routing.md#failover). |
| `failover_cooldown_seconds` | `float` | `60.0` | How long a provider stays skipped after a transient failure. |
| `readiness_timeout_seconds` | `float` | `2.0` | Budget for a single `/health/ready` probe. See [Readiness](#readiness). |

`default_provider` **must** name a configured provider; validation fails at
startup otherwise. Provider names must be unique.

`failover` is off by default. When off, a provider that cannot be reached is
still selected if it is the best match for the model, and its failure is returned
to the caller — no fallback between providers. The background and the reasoning
behind each limit are in [ADR-0002](adr/0002-failure-driven-failover.md).

## Readiness

`/health/ready` probes **every** provider concurrently — `GET {base_url}{health_path}`,
with that provider's `extra_headers` — and each probe gets
`readiness_timeout_seconds` of its own, not the provider's much longer
`timeout.total_seconds`. The endpoint answers `200` with `status: "ok"` when at
least one provider answered *and* every provider marked `required: true` answered;
otherwise it answers `503` with `status: "unavailable"`. Both bodies carry a
`providers` map of per-provider booleans:

```json
{"status": "ok", "providers": {"lm-studio": true, "android": false}}
```

A provider counts as answered on any status below 500: this is a reachability
signal, so an upstream that answers `404` at its probe path is up. A probe
failure of any kind — unreachable, timed out, malformed URL — counts as not
answered rather than failing the endpoint.

The api key is not sent: the header name differs per kind (`Authorization: Bearer`
for `openai_compatible`, `API-KEY` for `one_min_*`). If the probe path needs
credentials, put the header in the provider's `extra_headers`.

Readiness reflects the probe, not the failover down-set. A provider is only Down
after a *client* request fails transiently, and that state expires on its own, so
folding it into readiness would make the endpoint flap with each cooldown window
rather than report what is reachable now. Probing has no effect on routing.


## `security`

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | `bool` | `false` | Enable Bearer-token authentication on every endpoint except `/health/live` and `/health/ready`. |
| `token_env` | `str` \| `null` | `null` | Name of the environment variable holding the Bearer token (e.g. `LMS_PASSTHROUGH_TOKEN`). Ignored when `enabled: false`. |
| `docs_enabled` | `bool` | `true` | Expose the FastAPI `/docs` and `/openapi.json` endpoints. |
| `cors_origins` | `list[str]` | `[]` | Allowed CORS origins. Use `["*"]` for all origins. Cannot be combined with `cors_allow_credentials: true`. |
| `cors_allow_credentials` | `bool` | `false` | Allow credentials (cookies, auth headers) in CORS requests. Mutually exclusive with a wildcard origin list. |

Example:

```yaml
security:
  enabled: true
  token_env: LMS_PASSTHROUGH_TOKEN
  cors_origins:
    - https://myapp.example.com
  cors_allow_credentials: true
```

### CORS policy

The CORS middleware is only added when `cors_origins` is non-empty. Allowed
request methods are `GET` and `POST`; allowed request headers are
`Authorization`, `Content-Type`, and `X-LMS-Provider`. Any configuration
combining `cors_origins: ["*"]` with `cors_allow_credentials: true` is
rejected at startup with a `ValueError`.

## `persistence`

Response-chain storage for conversation state (the
`previous_response_id` continuation pattern). Backed by SQLite.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `path` | `str` | `"./state/lms-passthrough.sqlite3"` | Path to the SQLite database file. Created on first write if missing. |
| `retention_days` | `int` | `30` | Age after which stored response chains are purged. A periodic cleanup task (every hour) handles this. |

Example:

```yaml
persistence:
  path: ./storage/state/lms-passthrough.sqlite3
  retention_days: 90
```

## `timeout`

Upstream HTTP timeouts applied to every provider's `httpx.AsyncClient`.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `connect_seconds` | `float` | `5.0` | Timeout for TCP connection to upstream. |
| `response_seconds` | `float` | `60.0` | Timeout for the first byte of an upstream response. |
| `stream_idle_seconds` | `float` | `120.0` | Max time between SSE chunks before an active stream is aborted. |
| `total_seconds` | `float` | `600.0` | Overall timeout for any single upstream request (used as the `httpx` client timeout). |
| `catalog_ttl_seconds` | `float` | `300.0` | Time-to-live for the per-provider model-catalog cache. |
| `shutdown_grace_seconds` | `float` | `30.0` | Maximum time to wait for running tasks during shutdown. |

Example:

```yaml
timeout:
  connect_seconds: 10.0
  response_seconds: 120.0
  total_seconds: 900.0
  catalog_ttl_seconds: 120.0
```

## `logging`

Structured log emission per request. Every API request produces one structured
log line containing method, path, provider, status, latency, and request ID
(`X-Request-ID`, propagated from the caller or generated). `Authorization`
headers and provider API keys are redacted; prompt content and generated text
are never logged by default.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `level` | `str` | `"INFO"` | Minimum Python log level. One of `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. Values are uppercased before validation. |
| `format` | `str` | `"json"` | Log output format. `"json"` emits one JSON object per line; `"text"` emits human-readable lines. |

Example:

```yaml
logging:
  level: DEBUG
  format: text
```

## `providers`

A list of provider configurations. At least one entry is required. Each entry
declares how the proxy connects to an upstream inference service.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | `str` | _(required)_ | Unique identifier for the provider. Used in `X-LMS-Provider` header routing and as `default_provider` value. |
| `kind` | `str` | _(required)_ | Built-in provider implementation. One of: `lm_studio`, `openai_compatible`, `one_min_chat`, `one_min_code_generator`. |
| `base_url` | `str` | _(required)_ | Base URL of the upstream service. Must start with `http://` or `https://`. Trailing slashes are stripped. |
| `api_key_env` | `str` \| `null` | `null` | Name of the environment variable holding the provider's API key. Required for `one_min_*` kinds; optional for `openai_compatible`; ignored for `lm_studio`. Missing at startup when required raises `ValueError`. |
| `default_model` | `str` \| `null` | `null` | Default upstream model ID when no `models:` mappings are configured and no model is sent by the client. |
| `models` | `list[object]` | `[]` | Public-to-upstream model mappings (see [Model mappings](#model-mappings)). |
| `required` | `bool` | `false` | When `true`, a failed readiness probe of this provider makes the whole proxy unready. See [Readiness](#readiness). |
| `health_path` | `str` | `"/"` | Path appended to `base_url` for the readiness probe. Must start with `/`. |
| `verify_tls` | `bool` | `true` | Verify upstream TLS certificates. Set `false` for self-signed certs (not recommended for production). |
| `ca_bundle` | `str` \| `null` | `null` | Path to a custom CA bundle file for TLS verification. Overrides `verify_tls` when set. |
| `timeout` | `object` | _(see [Timeouts](#timeout))_ | Per-provider timeout overrides. |
| `max_concurrent` | `int` \| `null` | `null` | Maximum concurrent inference calls. `null` (or non-positive) means unlimited. |
| `queue_timeout_seconds` | `float` | `5.0` | Time to wait in the queue when at capacity before failing with `503`. Must be non-negative. |
| `extra_headers` | `dict[str, str]` \| `null` | `null` | Additional headers forwarded with every upstream request (e.g. Cloudflare Access tokens), including the readiness probe. Honored on inference requests only by `openai_compatible`; other kinds ignore it there. |

### Kind values

| Kind | Description |
| --- | --- |
| `lm_studio` | Transparent passthrough: forwards `/api/v1/*` and `/v1/*` byte-for-byte. No translation layer. |
| `openai_compatible` | OpenAI-compatible endpoint (e.g. Android phone acting as an OpenAI server). Streaming synthesized from completed responses when native SSE is unavailable. |
| `one_min_chat` | 1min.ai chat model. Real SSE streaming. Supports transcriptions. No tools, images, structured output, or embeddings. |
| `one_min_code_generator` | 1min.ai code generator. Streaming synthesized from a completed response. No tools, images, structured output, embeddings, or transcriptions. |

### Model mappings

Each entry in `models:` maps a public-facing model ID to an upstream model ID.
Model discovery merges upstream catalog entries with synthetic entries built
from these mappings.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `public` | `str` | _(required)_ | The model ID exposed to clients (appears in `/v1/models`, used in requests). |
| `upstream` | `str` | _(required)_ | The model ID sent to the upstream provider. |
| `type` | `str` | `"llm"` | Model type: `"llm"` or `"embedding"`. |
| `display_name` | `str` \| `null` | _(generated)_ | Human-readable name for the model. Defaults to `public` if omitted. |
| `capabilities` | `list[str]` | `[]` | Optional capability list to override the kind's default for this model. |

When a provider has `models:` configured, requests for public model IDs not
present in the list are rejected with `400`. The `lm_studio` provider ignores
mappings and passes model IDs through unchanged.

`/v1/audio/transcriptions` is the exception: model IDs reach the provider
untranslated, because a client of the speech-to-text route has to name a model
the selected provider actually serves. Map the model yourself, or use an
`openai_compatible` provider whose model IDs are already OpenAI-shaped.

Example:

```yaml
providers:
  - name: android
    kind: openai_compatible
    base_url: http://192.168.1.50:8080
    models:
      - public: android-chat
        upstream: local-model
        display_name: "Local model on phone"
      - public: android-embed
        upstream: embedding-model
        type: embedding
```

## Provider example

A complete four-provider configuration using environment-variable interpolation
for secrets:

```yaml
host: 0.0.0.0
port: 1234
default_provider: lm-studio
security:
  enabled: true
  token_env: LMS_PASSTHROUGH_TOKEN
  cors_origins:
    - https://myapp.example.com
persistence:
  path: ./storage/state/lms-passthrough.sqlite3
  retention_days: 30
timeout:
  response_seconds: 120.0
  catalog_ttl_seconds: 120.0
logging:
  level: INFO
  format: json
providers:
  - name: lm-studio
    kind: lm_studio
    base_url: http://127.0.0.1:8999
    required: true
  - name: android
    kind: openai_compatible
    base_url: http://192.168.1.50:8080
    api_key_env: ANDROID_API_KEY
    max_concurrent: 4
    queue_timeout_seconds: 2.0
  - name: 1min-chat
    kind: one_min_chat
    base_url: https://api.1min.ai
    api_key_env: ONEMIN_API_KEY
    models:
      - public: 1min-chat
        upstream: ${ONEMIN_CHAT_MODEL}
  - name: 1min-code-generator
    kind: one_min_code_generator
    base_url: https://api.1min.ai
    api_key_env: ONEMIN_API_KEY
    extra_headers:
      X-App-Version: "2.1"
```

## Environment variables

- `LMS_PASSTHROUGH_CONFIG` — override the config file path (not used by the
  YAML loader directly, but available via the CLI `--config` flag).
- Provider API keys come from whatever `api_key_env` names. Missing values
  fail startup.
- `${ENV_VAR}` interpolation in any YAML scalar substitutes the matching
  environment variable at load time. Missing variables raise `ValueError`.

## Validation errors at startup

The following conditions fail startup with a `ValueError`:

- `default_provider` does not name a configured provider.
- Duplicate provider names.
- `base_url` missing `http://` or `https://` prefix.
- `max_concurrent` is zero or negative.
- `cors_origins: ["*"]` combined with `cors_allow_credentials: true`.
- An `api_key_env` provider has no matching environment variable set.
