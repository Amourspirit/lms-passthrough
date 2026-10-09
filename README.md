# LMS Passthrough

An unofficial, unaffiliated FastAPI service that presents an LM Studio- and
OpenAI-compatible HTTP surface while routing each request to a configurable
provider (a local LM Studio server, an OpenAI-compatible endpoint, 1min.ai, or a
System 1 decision model). One inbound API, several possible backends, selected
per request.

> Not affiliated with or endorsed by LM Studio, OpenAI, or 1min.ai.

## What it does

- Speaks the **LM Studio native** API (`/api/v1/*`) and the **OpenAI** API
  (`/v1/*`), including chat, responses, embeddings, speech-to-text, model
  listing, and the JEV `/decisions` route.
- Routes by `X-LMS-Provider` header or by model discovery across providers.
- Proxies the LM Studio paths **transparently** when the selected provider is
  `lm-studio` — bodies, headers, statuses, and streams pass through unchanged.
- Translates requests to **canonical types** for every other provider, rejecting
  unsupported features (streaming, tools, images, structured output) with a
  `400` rather than silently dropping them.
- Persists response chains in SQLite to support `previous_response_id`
  continuation, with periodic retention-based cleanup.

## Endpoints

| Method | Path | Notes |
| --- | --- | --- |
| `GET` | `/health/live` | Liveness. |
| `GET` | `/health/ready` | Readiness; probes every provider concurrently. `503` when none answer or a `required` provider is down. |
| `GET` | `/api/v1/models` | LM Studio native model catalog. |
| `GET` | `/v1/models` | OpenAI model list. |
| `POST` | `/api/v1/chat` | LM Studio native chat (streaming supported). |
| `POST` | `/v1/chat/completions` | OpenAI chat completions (streaming supported). |
| `POST` | `/v1/responses` | OpenAI Responses API. |
| `POST` | `/v1/embeddings` | OpenAI embeddings. |
| `POST` | `/v1/audio/transcriptions` | Speech-to-text (`json` and `text` formats only). |
| `POST` | `/api/v1/decisions`, `/v1/decisions` | System 1 `jev_decide` decisions. |

## Providers

| `kind` | Behavior |
| --- | --- |
| `lm_studio` | Transparent passthrough of the LM Studio REST API. |
| `openai_compatible` | OpenAI-compatible upstream (e.g. an Android phone). Streaming synthesized when the upstream has no native SSE. |
| `one_min_chat` | 1min.ai chat, native SSE, supports transcriptions. |
| `one_min_code_generator` | 1min.ai code generator; synthesized streaming. |
| `system1` | JEV `/decisions` only. |

## Quick start

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras
cp .env.example .env          # fill in the keys your config references
uv run lms-passthrough check-config --config config.yaml
uv run lms-passthrough serve --config config.yaml
```

The service listens on `0.0.0.0:1234` by default. The sample `config.yaml`
references `${CF_ACCESS_CLIENT_ID}` / `${CF_OMLX_TOKEN}` and needs
`ONEMIN_API_KEY` set for the 1min.ai providers.

Point any LM Studio- or OpenAI-compatible client at the proxy; use
`X-LMS-Provider: <name>` to pin a backend. Every response is stamped with the
`X-LMS-Provider` that served it and an `X-Request-ID`.

## Routing and failover

- An `X-LMS-Provider` header is **absolute**: an unknown provider is `400`, an
  unavailable one is `503` — never a reroute.
- Without the header, the first provider in config order that *offers* the
  requested model serves the request; otherwise `default_provider` is used.
- With `failover: true`, a transient failure (unreachable, queue-saturated, or
  upstream `5xx`) marks the provider down for `failover_cooldown_seconds` and the
  request is retried on the next candidate. `4xx` never fails over, and neither
  streaming nor the transparent `lm-studio` path does.

See [docs/routing.md](docs/routing.md) and
[docs/adr/0002-failure-driven-failover.md](docs/adr/0002-failure-driven-failover.md).

## Configuration

All configuration lives in a single YAML file with `${ENV_VAR}` interpolation
for secrets. Full reference: [docs/configuration.md](docs/configuration.md).

## Documentation

- [Configuration](docs/configuration.md)
- [Routing](docs/routing.md)
- [Providers](docs/providers.md)
- [Provider contract kit](docs/provider-contract.md)
- [LM Studio conformance](docs/conformance.md)
- [ADRs](docs/adr/)

Build the site locally with `uv run mkdocs build --strict`.

## Development

```bash
uv run ruff check .           # lint
uv run mypy                   # strict typecheck
uv run pytest                 # test suite
uv run pytest --live-lms http://127.0.0.1:8999   # opt-in live LM Studio conformance
```

CI runs `ruff check .` → `mypy` → `pytest` → `mkdocs build --strict`.

## Docker

```bash
docker compose build          # image: lms-passthrough:local
docker compose up
```

The container mounts `config.yaml` read-only and persists state under
`./storage/state`.
