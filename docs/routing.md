# Routing

Requests select a provider through the `X-LMS-Provider` header or via automatic model-based discovery.

| Situation | Behavior |
| --- | --- |
| Header absent, model matched | Uses the first provider in `config.yaml` that offers the model |
| Header absent, no model match | Uses `default_provider` from configuration |
| Header names a configured provider | Uses that provider (validates model support) |
| Header names an unknown provider | `400` error |
| Selected provider is unavailable | `503` error — see [Failover](#failover) |

## Automatic Provider Switching

When the `X-LMS-Provider` header is absent, the proxy attempts to route the request to the best compatible provider based on the requested `model` ID.

### Routing Logic (Priority Order)
1. **Explicit Header**: If `X-LMS-Provider` is present, that provider is used. If the provider does not offer the requested model, a `400` error is returned.
2. **Model Discovery**: The proxy iterates through providers in the order they are defined in `config.yaml`. The first provider whose catalog contains the model ID is selected.
3. **Default Fallback**: If no matching provider is found, the proxy falls back to the `default_provider` specified in the configuration.

### Verifying Model Catalogs

You can inspect the models available to the router using the `/v1/models` endpoint.

**List all available models (across all providers):**
```sh
curl localhost:1234/v1/models
```

**List models for a specific provider:**
```sh
curl -H "X-LMS-Provider: omlx" localhost:1234/v1/models
```

**Test automatic switching:**
Send a request without the provider header to see which one is selected. Every
response carries an `X-LMS-Provider` header naming the provider that actually
served it, whether or not you sent one:
```sh
curl -v -X POST localhost:1234/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "gemma-4-12B-it-oQ8e", "messages": [{"role": "user", "content": "hi"}]}'
```

## Failover

Set `failover: true` to serve a request from a healthy provider when the one that
was first selected is unreachable:

```yaml
failover: true
failover_cooldown_seconds: 60
```

A provider becomes **down** for `failover_cooldown_seconds` after a transient
failure — an unreachable host, a saturated queue, or a `5xx` from upstream. While
down it is skipped during provider selection, is not advertised by `/v1/models`,
and is never re-probed. A `4xx` never counts as transient: a bad request stays a
bad request, so a misconfigured API key surfaces instead of being silently
rerouted somewhere else.

Failover is deliberately limited. The `X-LMS-Provider` header is absolute and is
never failed over from; streaming requests do not fail over (a terminal `error`
event ends the stream); and the transparent `lm-studio` path does not fail over
(it forwards raw bodies, so there is nothing to translate to a different
backend).

When a request is served by a provider other than the first one selected, the
request log carries a `failover` field naming both:

```json
{"path": "/v1/chat/completions", "provider": "1min-chat", "status": 200,
 "failover": "omlx->1min-chat"}
```

If no available provider offers the requested model, the proxy returns `503` and
names the providers that were down.

### Gotchas

- **Case Sensitivity**: Model ID matching is case-sensitive. If a provider reports `Gemma-4` but you request `gemma-4`, the automatic switch will fail and fall back to the default provider.
- **Catalog Caching**: Model lists are cached for `catalog_ttl_seconds` (default 300s). If you add a model upstream, it may not be discoverable for up to 5 minutes unless the proxy is restarted.
- **Config Order**: Since the first match wins, the order of providers in `config.yaml` defines their priority.
- **Failover is off by default**: `failover: false` is the default, and with it a provider that cannot be reached is still selected if it is the best match for the model.

## Model mapping

Each provider may define `models:` entries mapping a public model ID to the upstream model ID:

```yaml
providers:
  - name: android
    kind: openai_compatible
    base_url: http://192.168.1.50:8080
    models:
      - public: android-chat
        upstream: local-model
```

When a provider has `models:` configured, requests for unknown public model IDs are rejected with `400`. The `lm-studio` provider always passes model IDs through unchanged.

