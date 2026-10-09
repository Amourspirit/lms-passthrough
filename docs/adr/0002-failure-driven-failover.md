---
status: accepted
---

# Failure-driven provider failover

`failover: true` lets the proxy serve a request from a provider other than the one
first selected, when the first one fails transiently. This reverses the earlier
rule that the proxy never failed over between providers.

Health is **failure-driven**, not probed. A provider becomes Down when a call to
it raises `ProviderUnavailableError`, `ProviderOverloadedError`, or a 5xx
`ProviderHTTPError`, and stays Down for `failover_cooldown_seconds`. We chose
this over a background poller because a poller needs a `health()` member on the
`Provider` protocol, all four implementations, and a probe timeout — and the
symptom it fixes is already paid once by the first request. The win that
mattered more was negative-caching a *failed model probe*: without it a dead
provider is re-probed on every request at up to `timeout.total_seconds` (600s by
default) per call.

`X-LMS-Provider` is absolute and is never failed over from. A client that names a
provider gets that provider, or a 400/503 — silently serving a different backend
than the one asked for is close to undiagnosable in production. Only header-less
requests are candidates for failover.

Only transient errors fail over. A 4xx is the request's fault, and retrying it
elsewhere just produces a confusing 400 from the wrong provider — this includes
429 and 401/403, which is why a misconfigured `api_key_env` still surfaces
instead of being papered over by silent rerouting.

Two consequences worth stating because they look like bugs:

- **Streaming does not fail over.** The SSE preamble commits the response id
  before the provider emits anything, and a terminal `error` event is the
  documented way a stream ends badly.
- **The transparent lm-studio path does not fail over.** Failing out of it would
  mean translating a raw LM Studio-native body to another backend, which is
  exactly the work passthrough exists to avoid.

When every candidate fails, the *first* error is re-raised rather than replaced,
so the existing `overloaded` / `service_unavailable` / `upstream_error` envelopes
still describe what happened. Every candidate is Down by then, so the next
request gets the routing-time 503 naming them.
