# lms-passthrough

A FastAPI proxy that presents the LM Studio and OpenAI HTTP surfaces to clients and routes each request to a configured backend. One inbound request shape, several possible backends, header-selected.

## Language

### Boundaries and layers

**HTTP boundary**:
The FastAPI request/response layer where inbound JSON is parsed into typed Pydantic models and outbound JSON is shaped to match the LM Studio or OpenAI spec. Lives in `src/lms_passthrough/api/`.
_Avoid_: request layer, API layer

**Canonical type**:
A backend-agnostic dataclass in `provider_api/base.py` (`ChatRequest`, `ChatResult`, `CanonicalOutput`, `CanonicalUsage`, `StreamEvent`) that every provider sees. The HTTP boundary translates typed HTTP models into canonical types; providers never see raw HTTP payloads.
_Avoid_: internal type, DTO

**Provider protocol**:
The `Provider` `typing.Protocol` in `provider_api/base.py`. Anything that speaks canonical types on both sides and declares a `ProviderInfo` with capabilities. The three concrete providers today are lm-studio, openai_compatible (Android), and one-min (chat + code generator).
_Avoid_: backend interface, adapter

### Routing and translation

**Transparent passthrough**:
The path used when `X-LMS-Provider` resolves to `lm-studio`. The proxy forwards method, body, and filtered headers to the LM Studio server unchanged and streams the response back. No canonical translation. Applied per-route in `app.py`.
_Avoid_: proxy mode, forward mode

**Canonical translation**:
The path used for every non-lm-studio provider. Inbound payload is parsed at the HTTP boundary, converted to `ChatRequest`, dispatched to `provider.chat` / `.stream_chat` / `.responses`, and the result is shaped back into whichever HTTP spec the inbound route promises (LM Studio native or OpenAI).
_Avoid_: adapter mode, translated mode

**Canonical stream**:
The ordered sequence of `StreamEvent`s a provider's `stream_chat` yields: `chat.start`, `message.start`, one or more `message.delta`, `message.end`, `chat.end`. An `error` event is terminal — nothing follows it. Every provider emits this same grammar whether the upstream streams natively or the provider synthesizes it from a buffered result.
_Avoid_: SSE events, event stream, chunk stream

**Advertised contract**:
The shapes the proxy promises on its inbound routes, independent of which provider serves the request or whether the lm-studio bytes pass through unchanged. What the proxy's own OpenAPI document describes — the proxy's contract, never any upstream provider's spec.
_Avoid_: docs surface, self-documentation, spec mirror

**Provider selection**:
Choosing which provider serves a request. An `X-LMS-Provider` header is absolute — the named provider is the only candidate, so a bad model is a 400 and an unavailable one is a 503, never a reroute. Without a header, config order decides and the first provider that *offers* the requested model serves the request; with Failover enabled, the first that offers it *and* is available. Speech-to-text is the one route where *offers* is not a model match: `/v1/audio/transcriptions` reaches whichever provider is first in config order, passing the model id through untranslated, so the id is the only lever a client has. The lm-studio transparent path never applies here — lm_studio does not declare `TRANSCRIPTIONS`. Decisions (`/api/v1/decisions`, `/v1/decisions`) ignore *offers* entirely, because `model` is optional there: they go to the first provider in config order that declares `DECISIONS` and is available, never to `default_provider`. With Failover enabled, every such provider is a candidate.
_Avoid_: routing, dispatching

**Failover**:
Serving a request from a different provider than the one first selected, after a transient failure. Governed by the `failover` config flag. Only `ProviderUnavailableError`, `ProviderOverloadedError`, and 5xx `ProviderHTTPError` count as transient — a 4xx is the request's fault, not the provider's, and never triggers it. See `docs/adr/0002-failure-driven-failover.md`.
_Avoid_: retry, fallback chain, load balancing

### Availability

**Offers**:
A provider's model catalog — its live upstream listing or its configured `models:` mappings — contains a model id. Purely declarative; says nothing about whether the provider can be reached.
_Avoid_: supports, has the model

**Available**:
A provider that is not inside a cooldown window. The two predicates are independent: a provider can *offer* a model it cannot currently *serve*.
_Avoid_: healthy, up, online

**Down**:
The state a provider is in between a transient failure and the expiry of its `failover_cooldown_seconds` window. A Down provider is skipped during provider selection and is not advertised in the model catalog. A failed model probe also marks a provider Down, which is what stops the proxy re-probing a dead host on every request.
_Avoid_: unhealthy, offline, failed

**Ready**:
What `/health/ready` reports: at least one provider answered its readiness probe, and every provider marked `required: true` did. Probes run concurrently against each provider's `health_path` on a short `readiness_timeout_seconds` of their own, so the answer is what is reachable *now* — deliberately not the Down set, which only client traffic opens and which expires on its own. A probe is a reachability signal: any status below 500 counts as answered. Unready answers `503` with `status: "unavailable"`; both bodies carry a per-provider `providers` map.
_Avoid_: healthy, up, online, available

### Capabilities

**Capability**:
A `StrEnum` value in `provider_api/base.Capability` (`CHAT`, `RESPONSES`, `EMBEDDINGS`, `TRANSCRIPTIONS`, `DECISIONS`, `MODELS`, `TOOLS`, `IMAGES`, `STRUCTURED_OUTPUT`, `STREAMING`, `STATE`) that a provider declares in its `ProviderInfo`. Requests asking for a feature the selected provider has not declared are rejected at the HTTP boundary with 400.
_Avoid_: feature flag, support flag

**Capability validation**:
The check performed in `api/capabilities.validate_request_features` before dispatch. Rejects unsupported features rather than silently degrading them.
_Avoid_: feature check

### State

**Response chain**:
A stored sequence of assistant outputs, keyed by `response_id`, that supports `previous_response_id` continuation on the next request. Persisted in SQLite via `storage/state/store.py`. Single-worker deployment is assumed.
_Avoid_: conversation, session, history
