"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, TypeVar

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from lms_passthrough.api.capabilities import validate_request_features
from lms_passthrough.api.compat import (
    native_sse_from_canonical,
    openai_chat_sse_from_canonical,
    responses_sse_from_canonical,
    result_to_chat_completion,
    result_to_native,
    result_to_responses,
)
from lms_passthrough.api.passthrough import filter_headers
from lms_passthrough.api.schemas.decisions import DecisionsRequest, DecisionsResponse
from lms_passthrough.api.schemas.embeddings import EmbeddingsRequest, EmbeddingsResponse
from lms_passthrough.api.schemas.lm_studio import LMSChatRequest, LMSChatResponse
from lms_passthrough.api.schemas.openai_chat import (
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
)
from lms_passthrough.api.schemas.openai_responses import (
    OpenAIResponsesRequest,
    OpenAIResponsesResponse,
)
from lms_passthrough.api.schemas.transcription import (
    SUPPORTED_RESPONSE_FORMATS,
    TranscriptionResponse,
)
from lms_passthrough.api.schemas.translate import (
    lms_chat_request_to_canonical,
    openai_chat_request_to_canonical,
    openai_responses_request_to_canonical,
)
from lms_passthrough.concurrency import ProviderLimiter
from lms_passthrough.config import AppConfig, load_config_or_default
from lms_passthrough.logging import get_logger, setup_logging
from lms_passthrough.provider_api import (
    CanonicalOutput,
    Capability,
    ChatResult,
    ProviderHTTPError,
    ProviderOverloadedError,
    ProviderUnavailableError,
    ProviderUnsupportedError,
    StreamEvent,
    TranscriptionRequest,
)
from lms_passthrough.providers.lm_studio import LMStudioProvider
from lms_passthrough.providers.one_min import OneMinChatProvider, OneMinCodeGeneratorProvider
from lms_passthrough.providers.openai_compatible import OpenAICompatibleProvider
from lms_passthrough.providers.system1 import System1Provider
from lms_passthrough.state.store import StateStore, StoredResponse

READY_OK_MAX = 500
ERROR_THRESHOLD = 500

#: Upstream statuses that mark a provider as *down* rather than as a bad request.
TRANSIENT_STATUSES = frozenset({500, 502, 503, 504})

T = TypeVar("T")


def _native_error(message: str, error_type: str = "invalid_request") -> dict[str, dict[str, str]]:
    """The LM Studio native error envelope shape."""
    return {"error": {"message": message, "type": error_type}}


def is_transient(exc: Exception) -> bool:
    """Whether `exc` means 'this provider is down, try another' rather than 'bad request'."""
    if isinstance(exc, (ProviderUnavailableError, ProviderOverloadedError)):
        return True
    return isinstance(exc, ProviderHTTPError) and exc.status_code in TRANSIENT_STATUSES


class AppState:
    """Runtime state."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.state_store = StateStore(config.persistence.path)
        self.providers: dict[str, Any] = {}
        self.http_clients: dict[str, httpx.AsyncClient] = {}
        self.limiters: dict[str, ProviderLimiter] = {}
        for provider in config.providers:
            # Concurrency limits apply to translated canonical inference calls
            # only; the transparent LM Studio passthrough path and `/v1/models`
            # discovery / health endpoints bypass the limiter.
            self.limiters[provider.name] = ProviderLimiter(
                provider.max_concurrent,
                provider.queue_timeout_seconds,
            )
            verify: Any = provider.verify_tls
            if provider.ca_bundle:
                verify = provider.ca_bundle
            client = httpx.AsyncClient(timeout=provider.timeout.total_seconds, verify=verify)
            self.http_clients[provider.name] = client
            api_key = os.environ.get(provider.api_key_env) if provider.api_key_env else None
            if provider.kind == "lm_studio":
                self.providers[provider.name] = LMStudioProvider(
                    provider.name,
                    provider.base_url,
                    client,
                )
            elif provider.kind == "openai_compatible":
                self.providers[provider.name] = OpenAICompatibleProvider(
                    provider.name,
                    provider.base_url,
                    client,
                    api_key,
                    extra_headers=provider.extra_headers,
                )
            elif provider.kind == "one_min_chat":
                if not api_key:
                    raise ValueError(f"Missing API key for provider {provider.name}")
                self.providers[provider.name] = OneMinChatProvider(
                    provider.name,
                    provider.base_url,
                    api_key,
                    client,
                )
            elif provider.kind == "one_min_code_generator":
                if not api_key:
                    raise ValueError(f"Missing API key for provider {provider.name}")
                self.providers[provider.name] = OneMinCodeGeneratorProvider(
                    provider.name,
                    provider.base_url,
                    api_key,
                    client,
                )
            elif provider.kind == "system1":
                self.providers[provider.name] = System1Provider(
                    provider.name,
                    provider.base_url,
                    client,
                    api_key,
                    extra_headers=provider.extra_headers,
                )
        self._auth_token = (
            os.environ.get(config.security.token_env) if config.security.token_env else None
        )
        self.model_cache: dict[str, tuple[datetime, list[dict[str, Any]]]] = {}
        self.down_until: dict[str, datetime] = {}
        self._cleanup_task: asyncio.Task[None] | None = None

    def is_down(self, name: str) -> bool:
        """Whether `name` is inside the cooldown window opened by a transient failure."""
        until = self.down_until.get(name)
        return until is not None and datetime.now(tz=UTC) < until

    def mark_down(self, name: str, cooldown_seconds: float) -> None:
        """Open a cooldown window for `name`, during which routing skips it."""
        self.down_until[name] = datetime.now(tz=UTC) + timedelta(seconds=cooldown_seconds)

    async def aclose(self) -> None:
        """Close resources."""
        for client in self.http_clients.values():
            await client.aclose()


def create_app(config_path: Path | None = None) -> FastAPI:
    """Create the FastAPI application."""
    config = load_config_or_default(config_path)
    state = AppState(config)
    setup_logging(config.logging)
    request_logger = get_logger("request")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async def cleanup_loop() -> None:
            while True:
                state.state_store.delete_expired(config.persistence.retention_days)
                await asyncio.sleep(3600)

        state._cleanup_task = asyncio.create_task(cleanup_loop())
        try:
            yield
        finally:
            if state._cleanup_task:
                state._cleanup_task.cancel()
                with suppress(asyncio.CancelledError):
                    await state._cleanup_task
            await state.aclose()

    app = FastAPI(title="lms-passthrough", lifespan=lifespan)
    app.state.state = state

    if config.security.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.security.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type", "X-LMS-Provider"],
            allow_credentials=config.security.cors_allow_credentials,
        )

    async def authenticate(request: Request) -> None:
        if not config.security.enabled:
            return
        if request.url.path in {"/health/live", "/health/ready"}:
            return
        auth = request.headers.get("authorization", "")
        expected = f"Bearer {state._auth_token}" if state._auth_token else ""
        if not state._auth_token or auth != expected:
            raise HTTPException(status_code=401, detail="Unauthorized")

    @app.middleware("http")
    async def add_request_id(request: Request, call_next: Any) -> Any:
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        started = datetime.now(tz=UTC)
        response = await call_next(request)
        latency_ms = (datetime.now(tz=UTC) - started).total_seconds() * 1000

        # Route handlers record the provider that actually served the request;
        # without failover on, that is the header or `default_provider`.
        served = getattr(request.state, "served_provider", None) or provider_from_request(request)
        failover_from = getattr(request.state, "failover_from", None)

        span = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "provider": served,
            "status": response.status_code,
            "latency_ms": round(latency_ms, 3),
        }
        if failover_from:
            span["failover"] = f"{failover_from}->{served}"
        if response.status_code >= ERROR_THRESHOLD:
            request_logger.error("request", extra=span)
        else:
            request_logger.info("request", extra=span)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-LMS-Provider"] = served
        return response


    def provider_from_request(request: Request) -> str:
        return request.headers.get("x-lms-provider") or config.default_provider

    def default_candidate() -> list[str]:
        if config.default_provider not in state.providers:
            raise HTTPException(
                status_code=500,
                detail=f"Default provider {config.default_provider} not found",
            )
        return [config.default_provider]

    async def candidate_order(request: Request, model: str) -> list[str]:
        """Provider names to try for this request, most preferred first.

        An `X-LMS-Provider` header is absolute: it is never failed over from, so
        an unknown or model-less provider is a 400 and a down one is a 503.
        Without a header, config order decides, and `failover` widens the result
        from the single best match to every provider that offers the model.
        """
        header_provider = request.headers.get("x-lms-provider")
        if header_provider:
            if header_provider not in state.providers:
                raise HTTPException(
                    status_code=400, detail=f"Unknown provider: {header_provider}"
                )
            if state.is_down(header_provider):
                raise HTTPException(
                    status_code=503, detail=f"Provider {header_provider} is unavailable"
                )
            if not await is_model_supported(header_provider, model):
                raise HTTPException(
                    status_code=400,
                    detail=f"Provider {header_provider} does not support model: {model}",
                )
            return [header_provider]

        matches = [
            provider_cfg.name
            for provider_cfg in config.providers
            if await is_model_supported(provider_cfg.name, model)
        ]
        if matches:
            return matches if config.failover else matches[:1]

        if config.failover:
            down = [p.name for p in config.providers if state.is_down(p.name)]
            if down:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"No available provider offers model {model}; "
                        f"unavailable: {', '.join(down)}"
                    ),
                )
        return default_candidate()

    async def embedding_candidates(request: Request, model: str) -> list[str]:
        """Candidate providers for `/v1/embeddings`.

        A header still pins. Otherwise `default_provider` stays first (embeddings
        has never routed by model) and `failover` appends the other providers
        that offer the model, minus any that cannot embed.
        """
        header_provider = request.headers.get("x-lms-provider")
        if header_provider:
            if header_provider not in state.providers:
                raise HTTPException(
                    status_code=400, detail=f"Unknown provider: {header_provider}"
                )
            if state.is_down(header_provider):
                raise HTTPException(
                    status_code=503, detail=f"Provider {header_provider} is unavailable"
                )
            return [header_provider]
        order = default_candidate()
        if config.failover:
            for provider_cfg in config.providers:
                name = provider_cfg.name
                if name in order or state.is_down(name):
                    continue
                if Capability.EMBEDDINGS not in state.providers[name].info.capabilities:
                    continue
                if await is_model_supported(name, model):
                    order.append(name)
        return order

    def decision_candidates(request: Request) -> list[str]:
        """Candidate providers for `/decisions`.

        Routed by capability, not model (`model` is optional in `jev_decide`):
        a header still pins; otherwise config order picks the first provider
        declaring `DECISIONS`, and `failover` widens to all available ones.
        """
        header_provider = request.headers.get("x-lms-provider")
        if header_provider:
            if header_provider not in state.providers:
                raise HTTPException(
                    status_code=400, detail=f"Unknown provider: {header_provider}"
                )
            if Capability.DECISIONS not in state.providers[header_provider].info.capabilities:
                raise HTTPException(
                    status_code=400,
                    detail=f"Provider {header_provider} does not support decisions",
                )
            if state.is_down(header_provider):
                raise HTTPException(
                    status_code=503, detail=f"Provider {header_provider} is unavailable"
                )
            return [header_provider]
        capable = [
            p.name
            for p in config.providers
            if Capability.DECISIONS in state.providers[p.name].info.capabilities
        ]
        if not capable:
            raise HTTPException(
                status_code=400, detail="No configured provider supports decisions"
            )
        available = [name for name in capable if not state.is_down(name)]
        if not available:
            raise HTTPException(
                status_code=503,
                detail=f"No available decisions provider; unavailable: {', '.join(capable)}",
            )
        return available if config.failover else available[:1]

    async def dispatch(
        request: Request,
        names: list[str],
        call: Callable[[str, Any], Awaitable[T]],
    ) -> tuple[str, Any, T]:
        """Run `call` against each candidate in turn, failing over on transient errors.

        Each provider is attempted at most once per request and a transient
        failure marks it down for the cooldown window, so a client cannot make
        the proxy retry-loop. Non-transient errors (4xx, capability rejection)
        propagate untouched: those are the request's fault, not the provider's.

        When every candidate has failed the *first* error is re-raised rather
        than replaced, so the existing handlers still produce the real status
        and envelope (`overloaded`, `service_unavailable`, `upstream_error`).
        Every candidate is now marked down, so the next request gets the
        routing-time 503 from `candidate_order` naming them.
        """
        first_error: Exception | None = None
        failed: list[str] = []
        for name in names:
            provider = state.providers[name]
            # Set per attempt, not after the loop: a provider that raises a
            # non-transient error never returns, and the response must still
            # name the provider that actually handled it.
            request.state.served_provider = name
            try:
                return name, provider, await call(name, provider)
            except (ProviderUnavailableError, ProviderOverloadedError, ProviderHTTPError) as exc:
                if not is_transient(exc):
                    raise
                state.mark_down(name, config.failover_cooldown_seconds)
                failed.append(name)
                if first_error is None:
                    first_error = exc
                    request.state.failover_from = name
        assert first_error is not None, "candidates is never empty"
        raise first_error

    async def is_model_supported(provider_name: str, model: str) -> bool:
        # A provider inside its cooldown window offers nothing: it cannot be
        # probed, so do not pay the probe to find that out.
        if state.is_down(provider_name):
            return False

        # Check synthetic models first (config.yaml mapping)
        cfg = provider_config(provider_name)
        if cfg and cfg.map_model(model):
            return True

        # Check cached upstream models; refresh on miss or stale TTL
        cached = state.model_cache.get(provider_name)
        now = datetime.now(tz=UTC)
        ttl = cfg.timeout.catalog_ttl_seconds if cfg else 300.0
        fresh = cached and (now - cached[0]).total_seconds() < ttl

        if not fresh:
            provider = state.providers.get(provider_name)
            if provider is not None:
                try:
                    upstream_models = await provider.list_models()
                except Exception:
                    # Negative cache. Without this a down provider is re-probed on
                    # every single request, paying up to `timeout.total_seconds`
                    # per call against a dead host.
                    if config.failover:
                        state.mark_down(provider_name, config.failover_cooldown_seconds)
                else:
                    state.model_cache[provider_name] = (now, upstream_models)
                    cached = (now, upstream_models)

        if cached:
            models = cached[1]
            return any(
                m.get("key") == model or m.get("modelId") == model or m.get("id") == model
                for m in models
            )

        return False


    def provider_config(name: str) -> Any:

        return next((provider for provider in config.providers if provider.name == name), None)

    def map_model(provider_name: str, model: str) -> str:
        cfg = provider_config(provider_name)
        if cfg is None:
            raise HTTPException(status_code=500, detail="Provider configuration is missing")
        mapped = cfg.map_model(model)
        if mapped is None and cfg.models:
            raise HTTPException(
                status_code=400,
                detail=f"Provider {provider_name} does not support model: {model}",
            )
        return mapped or model

    def apply_state(request: Any) -> None:
        if not request.previous_response_id:
            return
        prior = state.state_store.get_response(request.previous_response_id)
        if prior is None:
            raise HTTPException(status_code=404, detail="Unknown previous_response_id")
        prior_outputs = prior.payload.get("outputs", [])
        history: list[dict[str, Any]] = []
        for output in prior_outputs:
            history.append({"role": "assistant", "content": output.get("content", "")})
        request.messages = [*history, *request.messages]

    async def merged_models(selected_provider_name: str | None) -> list[dict[str, Any]]:
        providers = config.providers
        if selected_provider_name:
            providers = [p for p in providers if p.name == selected_provider_name]
        else:
            # Do not advertise models the proxy knows it cannot currently serve.
            providers = [p for p in providers if not state.is_down(p.name)]
        merged: list[dict[str, Any]] = []
        now = datetime.now(tz=UTC)
        for provider_cfg in providers:
            provider = state.providers[provider_cfg.name]
            cached = state.model_cache.get(provider_cfg.name)
            cache_fresh = (
                cached
                and (now - cached[0]).total_seconds() < provider_cfg.timeout.catalog_ttl_seconds
            )
            if cache_fresh and cached is not None:
                upstream_models = cached[1]
            else:
                try:
                    upstream_models = await provider.list_models()
                except Exception:
                    upstream_models = cached[1] if cached else []
                else:
                    state.model_cache[provider_cfg.name] = (now, upstream_models)
            merged.extend(upstream_models)
            merged.extend(provider_cfg.synthetic_models())
        return merged

    async def transparent_request(request: Request, path: str) -> Any:
        provider_name = provider_from_request(request)
        if provider_name != "lm-studio":
            return None
        provider_cfg = next((p for p in config.providers if p.name == provider_name), None)
        if provider_cfg is None:
            raise HTTPException(status_code=500, detail="LM Studio provider is not configured")
        client = state.http_clients[provider_name]
        method = request.method
        url = f"{provider_cfg.base_url}{path}"
        headers = filter_headers(request.headers)
        # Let httpx recompute these from the actual outgoing URL/body rather than
        # forwarding the client's values, which may not match after re-serialization.
        headers.pop("content-length", None)
        headers.pop("host", None)
        body = None
        if method in {"POST", "PUT", "PATCH"}:
            raw_body = await request.body()
            # Validate the JSON locally (matching the pre-existing 400 contract
            # of the native endpoints, e.g. "Invalid JSON in request body") but
            # forward the original bytes so an arbitrary valid body reaches LM
            # Studio byte-for-byte. This preserves fields we do not (and cannot)
            # model as Pydantic types. (Typed routes reject malformed bodies
            # with a 422 before this helper runs.)
            try:
                json.loads(raw_body)
            except (json.JSONDecodeError, UnicodeDecodeError):
                return JSONResponse(
                    status_code=400,
                    content=_native_error("Invalid JSON in request body"),
                )
            body = raw_body
        if request.headers.get("x-lms-provider"):
            headers.pop("x-lms-provider", None)
        try:
            if body is None:
                upstream = await client.request(method, url, headers=headers)
            else:
                upstream = await client.request(method, url, headers=headers, content=body)
        except httpx.HTTPError:
            return JSONResponse(
                status_code=503,
                content=_native_error("LM Studio unavailable", "service_unavailable"),
            )
        if upstream.headers.get("content-type", "").startswith("text/event-stream"):
            async def stream() -> AsyncIterator[bytes]:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
            passthrough_headers = {
                key: value
                for key, value in upstream.headers.items()
                if key.lower() not in {"content-length", "transfer-encoding", "content-encoding"}
            }
            return StreamingResponse(
                stream(),
                status_code=upstream.status_code,
                headers=passthrough_headers,
                media_type="text/event-stream",
            )
        content_type = upstream.headers.get("content-type", "")
        passthrough_headers = dict(filter_headers(httpx.Headers(upstream.headers)))
        passthrough_headers.pop("content-length", None)
        return Response(
            status_code=upstream.status_code,
            content=upstream.content,
            headers=passthrough_headers,
            media_type=content_type or None,
        )

    @app.middleware("http")
    async def transparent_native_chat(request: Request, call_next: Any) -> Any:
        # The typed `/api/v1/chat` route below parses the body via Pydantic
        # before the handler runs (FastAPI reads the body during routing), so
        # the transparent lm-studio path must be intercepted here: the raw body
        # is forwarded byte-for-byte and malformed JSON yields a 400 native
        # envelope instead of a Pydantic 422, never reaching the upstream.
        # `transparent_request` always returns a response for lm-studio, so
        # nothing falls through to the route here.
        if request.url.path != "/api/v1/chat" or request.method != "POST":
            return await call_next(request)
        if provider_from_request(request) != "lm-studio":
            return await call_next(request)
        try:
            # Middleware runs before Starlette's ExceptionMiddleware, so an
            # HTTPException raised here would otherwise surface as a 500.
            await authenticate(request)
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content=_native_error(exc.detail))
        return await transparent_request(request, "/api/v1/chat")

    def store_result(
        provider_name: str,
        result: ChatResult,
        request_payload: dict[str, Any],
    ) -> str:
        response_id = result.response_id or f"resp_{uuid.uuid4().hex}"
        state.state_store.save_response(
            StoredResponse(
                response_id=response_id,
                provider_name=provider_name,
                model=result.model,
                created_at=datetime.now(tz=UTC),
                payload={
                    "result": result.raw,
                    "outputs": [asdict(output) for output in result.outputs],
                },
                previous_response_id=request_payload.get("previous_response_id"),
            )
        )
        return response_id

    @app.get("/health/live")
    async def health_live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    async def health_ready() -> dict[str, Any]:
        """Whether the proxy can serve traffic right now.

        Every provider is probed concurrently with `GET {base_url}{health_path}`,
        each on its own short `readiness_timeout_seconds` budget, so one hung
        upstream cannot stall the endpoint for the provider's much longer
        request timeout. Answers `200` with `status: "ok"` when at least one
        provider answered and every provider marked `required: true` answered;
        otherwise `503` with `status: "unavailable"`. Both bodies carry a
        `providers` map of per-provider booleans. A provider counts as answered
        on any status below `READY_OK_MAX`: this is a reachability signal, and
        an upstream that answers `404` at its probe path is up.
        """

        # The probe reflects reachability now, deliberately not `down_until`:
        # that is a TTL'd routing hint opened by a *client* request (ADR-0002),
        # so folding it in would make readiness flap with each cooldown window.
        # Probing has no effect on routing. Only `extra_headers` are sent — the
        # api key header differs per kind (`Bearer` vs `API-KEY`) and belongs to
        # the provider classes, so an operator who needs auth at `health_path`
        # configures it there.
        async def probe(provider: Any) -> bool:
            try:
                response = await asyncio.wait_for(
                    state.http_clients[provider.name].get(
                        f"{provider.base_url}{provider.health_path}",
                        headers=dict(provider.extra_headers or {}),
                    ),
                    timeout=config.readiness_timeout_seconds,
                )
            except Exception:
                # Any failure means "not ready" — a readiness endpoint that 500s
                # on a malformed URL or a surprise client error is worse than
                # useless to whatever is deciding whether to send it traffic.
                return False
            return response.status_code < READY_OK_MAX

        probed = await asyncio.gather(*(probe(provider) for provider in config.providers))
        providers = {
            provider.name: is_ready
            for provider, is_ready in zip(config.providers, probed, strict=True)
        }
        required_ok = all(providers[p.name] for p in config.providers if p.required)
        if required_ok and any(providers.values()):
            return {"status": "ok", "providers": providers}
        return JSONResponse(  # type: ignore[return-value]
            status_code=503,
            content={"status": "unavailable", "providers": providers},
        )

    @app.get("/api/v1/models")
    async def native_models(
        request: Request,
    ) -> Any:
        await authenticate(request)
        transparent = await transparent_request(request, "/api/v1/models")
        if transparent is not None:
            return transparent
        selected = request.headers.get("x-lms-provider")
        if selected and selected not in state.providers:
            raise HTTPException(status_code=400, detail=f"Unknown provider: {selected}")
        models = await merged_models(selected)
        return {"models": models}

    @app.get("/v1/models")
    async def openai_models(
        request: Request,
    ) -> Any:
        await authenticate(request)
        transparent = await transparent_request(request, "/v1/models")
        if transparent is not None:
            return transparent
        selected = request.headers.get("x-lms-provider")
        if selected and selected not in state.providers:
            raise HTTPException(status_code=400, detail=f"Unknown provider: {selected}")
        models = await merged_models(selected)
        data = [
            {
                "id": model.get("key") or model.get("modelId") or model.get("id"),
                "object": "model",
            }
            for model in models
        ]
        return {"object": "list", "data": data}

    @app.post(
        "/api/v1/chat",
        response_model=LMSChatResponse,
        responses={
            200: {
                "description": "LM Studio native chat result",
                "model": LMSChatResponse,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/LMSChatResponse"},
                    },
                    "text/event-stream": {
                        "schema": {"type": "string", "format": "binary"},
                        "description": "Server-sent events when stream=true: native LM "
                        "Studio SSE frames (chat.start/message.delta/chat.end).",
                    },
                },
            },
        },
    )
    async def native_chat(
        request: Request,
        request_model: LMSChatRequest,
    ) -> Any:
        """LM Studio native chat route.

        Non-streaming returns a typed ``LMSChatResponse``. FastAPI parses and
        validates the body into ``request_model`` (malformed bodies get a 422).
        The transparent lm-studio path never reaches this handler: the
        ``transparent_native_chat`` middleware runs before routing, so this
        handler only serves translated providers. When ``request_model.stream``
        is set the response is a ``text/event-stream`` of LM Studio native SSE
        frames.
        """
        await authenticate(request)

        candidates = await candidate_order(request, request_model.model)
        payload = request_model.model_dump(mode="json", exclude_none=True)
        canonical = lms_chat_request_to_canonical(request_model)
        apply_state(canonical)

        if request_model.stream:
            # Streaming has no failover: the SSE preamble commits the response
            # id before the provider emits anything, and a terminal `error`
            # event is the documented way a stream ends badly.
            provider_name = candidates[0]
            provider = state.providers[provider_name]
            request.state.served_provider = provider_name
            validate_request_features(provider, payload)
            if provider_name != "lm-studio":
                canonical.model = map_model(provider_name, canonical.model)
            limiter = state.limiters[provider_name]
            await limiter.acquire()
            response_id = f"resp_{uuid.uuid4().hex}" if request_model.store else None

            async def events() -> AsyncIterator[StreamEvent]:
                try:
                    chunks: list[str] = []
                    async for event in provider.stream_chat(canonical):
                        if event.event == "message.delta":
                            chunks.append(str(event.data.get("content", "")))
                        yield event
                    if response_id is not None:
                        stored = ChatResult(
                            model=canonical.model,
                            outputs=[CanonicalOutput(content="".join(chunks))],
                            response_id=response_id,
                        )
                        store_result(provider_name, stored, payload)
                finally:
                    limiter.release()

            return StreamingResponse(
                native_sse_from_canonical(events(), response_id),
                media_type="text/event-stream",
            )

        async def call(name: str, provider: Any) -> Any:
            validate_request_features(provider, payload)
            upstream = canonical
            if name != "lm-studio":
                upstream = replace(canonical, model=map_model(name, canonical.model))
            async with state.limiters[name].slot():
                return await provider.chat(upstream)

        provider_name, _provider, result = await dispatch(request, candidates, call)
        response_id = (
            store_result(provider_name, result, payload)
            if request_model.store
            else None
        )
        return result_to_native(result, response_id)

    @app.post(
        "/v1/chat/completions",
        response_model=OpenAIChatCompletionResponse,
        responses={
            200: {
                "description": "Chat completion result",
                "model": OpenAIChatCompletionResponse,
                "content": {
                    "application/json": {
                        "schema": {
                            "$ref": "#/components/schemas/OpenAIChatCompletionResponse"
                        },
                    },
                    "text/event-stream": {
                        "schema": {"type": "string", "format": "binary"},
                        "description": "Server-sent events when stream=true: each data "
                        "frame is a chat.completion.chunk object, terminated by [DONE].",
                    },
                },
            },
        },
    )
    async def openai_chat(
        request: Request,
        request_model: OpenAIChatCompletionRequest,
    ) -> Any:
        """OpenAI chat completions route.

        Non-streaming returns a typed ``OpenAIChatCompletionResponse``. FastAPI
        parses and validates the body into ``request_model`` first (malformed
        bodies get a 422 regardless of provider); Starlette caches the raw body,
        so for the transparent lm-studio path the downstream transparent helper
        re-reads it (``await request.body()``) and forwards it byte-for-byte
        rather than re-serializing the parsed model. When ``request_model.stream``
        is set the response is a ``text/event-stream`` of ``chat.completion.chunk``
        frames terminated by ``[DONE]``.
        """
        await authenticate(request)
        transparent = await transparent_request(request, "/v1/chat/completions")
        if transparent is not None:
            return transparent

        candidates = await candidate_order(request, request_model.model)
        payload = request_model.model_dump(mode="json", exclude_none=True)
        canonical = openai_chat_request_to_canonical(request_model)
        apply_state(canonical)

        if request_model.stream:
            provider_name = candidates[0]
            provider = state.providers[provider_name]
            request.state.served_provider = provider_name
            validate_request_features(provider, payload)
            if provider_name != "lm-studio":
                canonical.model = map_model(provider_name, canonical.model)
            limiter = state.limiters[provider_name]
            completion_id = f"chatcmpl-{uuid.uuid4().hex}"
            await limiter.acquire()

            async def events() -> AsyncIterator[StreamEvent]:
                try:
                    async for event in provider.stream_chat(canonical):
                        yield event
                finally:
                    limiter.release()

            return StreamingResponse(
                openai_chat_sse_from_canonical(
                    events(),
                    completion_id,
                    canonical.model,
                ),
                media_type="text/event-stream",
            )

        async def call(name: str, provider: Any) -> Any:
            validate_request_features(provider, payload)
            upstream = canonical
            if name != "lm-studio":
                upstream = replace(canonical, model=map_model(name, canonical.model))
            async with state.limiters[name].slot():
                return await provider.chat(upstream)

        provider_name, _provider, result = await dispatch(request, candidates, call)
        result.response_id = result.response_id or f"chatcmpl-{uuid.uuid4().hex}"
        result.raw.setdefault("created", int(datetime.now(tz=UTC).timestamp()))
        return result_to_chat_completion(result)

    @app.post(
        "/v1/responses",
        response_model=OpenAIResponsesResponse,
        responses={
            200: {
                "description": "OpenAI Responses-compatible response",
                "model": OpenAIResponsesResponse,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/OpenAIResponsesResponse"},
                    },
                    "text/event-stream": {
                        "schema": {
                            "type": "string",
                            "format": "binary",
                        },
                        "description": "Server-sent events for streaming responses",
                    },
                },
            },
        },
    )
    async def openai_responses(
        request: Request,
        request_model: OpenAIResponsesRequest,
    ) -> Any:
        """OpenAI Responses API alias.

        Non-streaming returns a typed ``OpenAIResponsesResponse``. When
        ``stream=true`` the response is a ``text/event-stream`` of OpenAI
        Response-style SSE frames.
        """
        await authenticate(request)
        transparent = await transparent_request(request, "/v1/responses")
        if transparent is not None:
            return transparent

        candidates = await candidate_order(request, request_model.model)
        payload = dict(request_model.model_dump(by_alias=True, exclude_unset=False))
        canonical = openai_responses_request_to_canonical(request_model)
        apply_state(canonical)
        response_id = f"resp_{uuid.uuid4().hex}"

        if payload.get("stream"):
            provider_name = candidates[0]
            provider = state.providers[provider_name]
            request.state.served_provider = provider_name
            validate_request_features(provider, payload)
            if provider_name != "lm-studio":
                canonical.model = map_model(provider_name, canonical.model)
            limiter = state.limiters[provider_name]
            await limiter.acquire()

            async def events_with_store() -> AsyncIterator[StreamEvent]:
                try:
                    chunks: list[str] = []
                    async for event in provider.stream_chat(canonical):
                        if event.event == "message.delta":
                            chunks.append(str(event.data.get("content", "")))
                        yield event
                    if payload.get("store", True):
                        stored = ChatResult(
                            model=canonical.model,
                            outputs=[CanonicalOutput(content="".join(chunks))],
                            response_id=response_id,
                        )
                        store_result(provider_name, stored, payload)
                finally:
                    limiter.release()

            return StreamingResponse(
                responses_sse_from_canonical(
                    events_with_store(),
                    response_id,
                    canonical.model,
                ),
                media_type="text/event-stream",
            )

        async def call(name: str, provider: Any) -> Any:
            validate_request_features(provider, payload)
            upstream = canonical
            if name != "lm-studio":
                upstream = replace(canonical, model=map_model(name, canonical.model))
            async with state.limiters[name].slot():
                return await provider.responses(upstream)

        provider_name, _provider, result = await dispatch(request, candidates, call)
        if payload.get("store", True):
            result.response_id = response_id
            store_result(provider_name, result, payload)
        return result_to_responses(result, response_id)

    @app.post(
        "/v1/embeddings",
        response_model=EmbeddingsResponse,
        responses={
            200: {
                "description": "Embeddings result",
                "model": EmbeddingsResponse,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/EmbeddingsResponse"},
                    },
                },
            },
        },
    )
    async def embeddings(
        request: Request,
        request_model: EmbeddingsRequest,
    ) -> Any:
        """OpenAI embeddings route.

        FastAPI parses and validates the body into ``request_model`` first
        (malformed bodies get a 422 regardless of provider); Starlette caches
        the raw body, so for the transparent lm-studio path the downstream
        transparent helper re-reads it (``await request.body()``) and forwards
        it byte-for-byte rather than re-serializing the parsed model. The
        translated path forwards ``request_model.model_dump(exclude_unset=True)``
        so unset optional fields stay absent; ``Provider.embeddings`` keeps its
        ``dict[str, Any]`` signature (ADR-0001).
        """
        await authenticate(request)
        transparent = await transparent_request(request, "/v1/embeddings")
        if transparent is not None:
            return transparent

        candidates = await embedding_candidates(request, request_model.model)
        capable = [
            name
            for name in candidates
            if Capability.EMBEDDINGS in state.providers[name].info.capabilities
        ]
        if not capable:
            raise HTTPException(
                status_code=400,
                detail="Selected provider does not support embeddings",
            )
        payload = request_model.model_dump(exclude_unset=True)

        async def call(name: str, provider: Any) -> Any:
            body = payload
            if "model" in body:
                body = {**body, "model": map_model(name, str(body["model"]))}
            async with state.limiters[name].slot():
                return await provider.embeddings(body)

        provider_name, _provider, result = await dispatch(request, capable, call)
        return result

    @app.post(
        "/v1/audio/transcriptions",
        response_model=TranscriptionResponse,
        response_model_exclude_none=True,
        responses={
            200: {
                "description": "Transcription result",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/TranscriptionResponse"}
                    },
                    "text/plain": {"schema": {"type": "string"}},
                },
            },
        },
    )
    async def audio_transcriptions(
        request: Request,
        file: Annotated[UploadFile, File(description="The audio file to transcribe.")],
        model: Annotated[str, Form(description="Model id, passed to the provider untranslated.")],
        *,
        response_format: Annotated[str, Form()] = "json",
        language: Annotated[str | None, Form()] = None,
        prompt: Annotated[str | None, Form()] = None,
        stream: Annotated[str | None, Form()] = None,
    ) -> Any:
        """OpenAI-compatible speech-to-text route.

        Model ids reach the provider untranslated — there is no rename layer, so
        a client must already speak the selected provider's model vocabulary.
        The `lm-studio` transparent path is not reachable here: lm_studio does
        not declare `TRANSCRIPTIONS`, so the capability filter below empties the
        candidate list and the multipart body never meets `transparent_request`.

        ponytail: no proxy-side upload cap — the upstream's limit is the
        ceiling, and the file is buffered in memory before forwarding. Add a
        streaming-to-tmpfile path here if a client needs audio larger than the
        largest upstream will take.
        """
        await authenticate(request)

        if response_format not in SUPPORTED_RESPONSE_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported response_format: {response_format}. "
                    f"Supported: {', '.join(sorted(SUPPORTED_RESPONSE_FORMATS))}"
                ),
            )
        if stream is not None and stream.lower() in {"1", "true", "yes"}:
            raise HTTPException(
                status_code=400, detail="Streaming transcriptions are not supported"
            )

        candidates = await candidate_order(request, model)
        capable = [
            name
            for name in candidates
            if Capability.TRANSCRIPTIONS in state.providers[name].info.capabilities
        ]
        if not capable:
            raise HTTPException(
                status_code=400,
                detail="Selected provider does not support transcriptions",
            )

        canonical = TranscriptionRequest(
            model=model,
            audio=await file.read(),
            filename=file.filename or "audio.wav",
            content_type=file.content_type,
            language=language,
            prompt=prompt,
            response_format=response_format,
        )

        async def call(name: str, provider: Any) -> Any:
            async with state.limiters[name].slot():
                return await provider.transcriptions(canonical)

        _provider_name, _provider, result = await dispatch(request, capable, call)
        if response_format == "text":
            return PlainTextResponse(result.text)
        return TranscriptionResponse(
            text=result.text,
            language=result.language,
            duration=result.duration,
        )

    @app.post("/api/v1/decisions", response_model=DecisionsResponse)
    @app.post("/v1/decisions", response_model=DecisionsResponse)
    async def decisions(request: Request, request_model: DecisionsRequest) -> Any:
        """System 1 (JEV `jev_decide`) decision route.

        The request is validated against the JEV schema; the upstream
        `{code, message, data}` envelope is returned unchanged, so a non-zero
        `code` reaches the client as-is.
        """
        await authenticate(request)
        candidates = decision_candidates(request)
        # exclude_unset, not exclude_none: `{"billing": null}` criteria must survive.
        payload = request_model.model_dump(mode="json", exclude_unset=True)

        async def call(name: str, provider: Any) -> Any:
            body = payload
            if body.get("model"):
                body = {**body, "model": map_model(name, str(body["model"]))}
            async with state.limiters[name].slot():
                return await provider.decisions(body)

        _provider_name, _provider, result = await dispatch(request, candidates, call)
        return JSONResponse(result)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"message": exc.detail, "type": "invalid_request"}},
        )

    @app.exception_handler(ProviderUnsupportedError)
    async def unsupported_handler(request: Request, exc: ProviderUnsupportedError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": str(exc), "type": "invalid_request"}},
        )

    @app.exception_handler(ProviderHTTPError)
    async def provider_http_handler(request: Request, exc: ProviderHTTPError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "message": "Upstream provider returned an error",
                    "type": "upstream_error",
                    "provider_error": exc.body,
                }
            },
        )

    @app.exception_handler(ProviderOverloadedError)
    async def provider_overloaded_handler(
        request: Request, exc: ProviderOverloadedError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={
                "error": {"message": "Provider overloaded", "type": "overloaded"}
            },
        )

    @app.exception_handler(ProviderUnavailableError)
    async def provider_unavailable_handler(
        request: Request, exc: ProviderUnavailableError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={
                "error": {"message": "Provider unavailable", "type": "service_unavailable"}
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": "Internal server error", "type": "internal_error"}},
        )

    return app
