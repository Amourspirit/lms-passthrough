# AGENTS.md — lms-passthrough

Unofficial FastAPI proxy that mimics the LM Studio REST API and routes requests to pluggable providers (LM Studio, Android OpenAI-compatible, 1min.AI chat/code-generator). Early stage; the README and docs/ are scaffolds only — trust the code.

## Commands (all run from repo root, all via `uv`)

```bash
uv sync --all-extras          # one-time / after dependency changes
uv run ruff check .           # lint
uv run mypy                   # strict typecheck (mypy strict = true)
uv run pytest                 # test suite (pytest-asyncio, asyncio_mode=auto)
uv run pytest tests/test_app.py::test_health_ready_shape   # single test
uv run mkdocs build --strict  # docs
```

CI runs exactly: `ruff check .` → `mypy` → `pytest` → `mkdocs build --strict`. Mirror that order before considering work done.

Python 3.12 is pinned by `.python-version`; `uv sync` manages the venv. Do not run `python` directly — the system has no `python` on PATH; use `uv run ...`.

Run the app locally:

```bash
uv run lms-passthrough serve --config config.yaml
uv run lms-passthrough check-config --config config.yaml
```

## Architecture — non-obvious wiring

- `src/lms_passthrough/app.py` — everything is wired in `create_app()` / `AppState`: provider construction from config, auth, routing helpers, endpoints, cleanup task. It is intentionally monolithic right now (~440 lines); split it only with a deliberate refactor.
- `src/lms_passthrough/provider_api/base.py` — the canonical internal contract (`ChatRequest`, `ChatResult`, `Capability`, `StreamEvent`, `Provider` protocol). External protocol parsing happens at the HTTP boundary in `app.py` + `api/schemas/`; providers only see canonical types.
- `src/lms_passthrough/providers/` — `lm_studio.py` (transparent passthrough), `openai_compatible.py` (Android), `one_min.py` (1min chat + code generator). Provider HTTP calls must raise `ProviderHTTPError` / `ProviderUnavailableError`, not raw httpx errors.
- `src/lms_passthrough/api/` — `capabilities.py` (feature rejection), `compat.py` (SSE/response shaping), `passthrough.py` (header filtering), `schemas/` (payload → canonical via `schemas/translate.py`).
- `src/lms_passthrough/state/store.py` — SQLite response-chain persistence with manual versioned migrations (no Alembic).

## Repo-specific behavior an agent must not break

- Routing: `X-LMS-Provider` is absolute — it names the only candidate, so unknown value → 400 and unavailable → 503, never a reroute. Absent header → the first provider in config order that *offers* the requested model, else `default_provider` (config, initially `lm-studio`).
- `failover: false` by default. When true, a transient provider failure (unreachable, queue-saturated, upstream 5xx — never a 4xx) marks the provider down for `failover_cooldown_seconds`; down providers are skipped in routing, hidden from the model catalog, and not re-probed. Streaming and the transparent lm-studio path never fail over. See `docs/adr/0002-failure-driven-failover.md`.
- LM Studio routes (`/api/v1/*`, `/v1/*` when provider is `lm-studio`) are transparently proxied — preserve bodies/headers/status/streams; strip `Authorization`, `X-LMS-Provider`, and hop-by-hop headers via `api/passthrough.filter_headers`.
- Non-LM-Studio providers go through canonical translation; request features the provider lacks (streaming, tools, images, structured output) must be rejected via capability validation, never silently dropped.
- Config (`config.yaml`) uses `${ENV_VAR}` interpolation for secrets; `default_provider` must name a configured provider or validation fails. API keys come from env vars named in `api_key_env` — missing key = startup `ValueError` (the sample config requires `ONEMIN_API_KEY`; tests that don't need 1min use a trimmed config).
- Model discovery is merged across providers with a per-provider TTL cache (`catalog_ttl_seconds`, default 300s) and synthetic entries built from config `models:` mappings.
- State: SQLite stores response chains only (`previous_response_id` continuation); periodic cleanup deletes chains older than `retention_days`. Single-worker deployment is assumed — do not add multi-worker assumptions.

## Gotchas

- Ruff line-length is 100 and strict rule sets are on (A, B, C4, E, F, I, PL, SIM, UP, W); `PLR0913`/`PLR0915` are globally ignored. Tests ignore `PLR2004`.
- Mypy is fully strict; use `Any` deliberately at HTTP boundaries rather than weakening config.
- The `Dockerfile` installs with `uv pip install --system .` (uv binary copied in from the official image) — keep it in sync with `pyproject.toml` when changing the build.
- No git remote, no PyPI/registry destinations configured; image tag is `lms-passthrough:local` only.
- Live LM Studio conformance testing against a real 0.4.23 server is planned but not yet wired; current tests are all offline fakes.

## Agent skills

### Issue tracker

Issues live as local markdown files under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-role vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout: `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
