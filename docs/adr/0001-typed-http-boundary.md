---
status: proposed
---

# Typed HTTP boundary with per-endpoint Pydantic models

## Context

The POST endpoints (`/api/v1/chat`, `/v1/chat/completions`, `/v1/responses`, `/v1/embeddings`) are declared as `async def handler(request: Request)` and read the body via `await request.json()`. FastAPI has no visibility into the request shape, so `openapi.json` reports no body schema and `/docs` renders empty parameter lists. The canonical `ChatRequest` in `provider_api/base.py` is a plain `@dataclass`, not a Pydantic model, and describes the internal contract — not the HTTP contract the client actually sends.

## Decision

Introduce a typed HTTP boundary: one Pydantic model per HTTP spec per endpoint, living under `src/lms_passthrough/api/schemas/` (split by spec: `lm_studio.py`, `openai_chat.py`, `openai_responses.py`, `embeddings.py`, plus `translate.py` for HTTP-model → canonical-type conversion). Handlers take the typed model as a parameter; FastAPI generates a full OpenAPI schema and `/docs` becomes an accurate, live spec.

Non-obvious choices this pins down:

- **Per-spec models, not shared.** `LMSChatRequest` and `OpenAIChatCompletionRequest` are separate types even though their fields overlap ~80%. Sharing via inheritance produces ugly `allOf` schemas in `/docs` and re-introduces the "one schema, many endpoints" confusion this refactor is meant to fix. The duplication cost is small; `translate.py` handles the convergence into canonical `ChatRequest`.
- **`extra="allow"`, not `forbid`.** Unknown fields flow through into `ChatRequest.raw` and thus into the transparent LM Studio passthrough. Forbidding unknown fields would break every future LM Studio feature until we explicitly model it. Revisit once live LM Studio conformance testing lands.
- **Non-streaming responses typed; SSE responses documented in prose.** Swagger UI cannot render event streams meaningfully. Streaming responses declare `text/event-stream` in the `responses={...}` decorator and describe the event schema in the route docstring.
- **Embeddings typed at the HTTP boundary only.** `EmbeddingsRequest` is a Pydantic model, but `Provider.embeddings` keeps its `dict[str, Any]` signature. Canonicalising embeddings is a separate refactor that would touch the provider protocol and is out of scope for this pass.
- **`/v1/responses` request model reflects what we actually translate, not the full OpenAI Responses spec.** Per ADR-followup, `/v1/responses` for non-lm-studio providers is currently an alias for chat; the typed request advertises only the fields we handle (`input`, `instructions`, `previous_response_id`, `store`, `stream`, `tools`, `response_format`). Clients using unmodelled Responses fields (`input_items`, `include`, `truncation`, real tool-call chain replay) are silently forwarded via `extra="allow"` but not advertised in `/docs`.
- **Field commitments.** For OpenAI chat completions: `messages` with a discriminated `TextPart` / `ImageUrlPart` content union; `tools`, `tool_choice`, `parallel_tool_calls`, `response_format` typed as first-class; common-but-low-value fields (`user`, `logit_bias`, `n`, `logprobs`, `top_logprobs`, `service_tier`, `metadata`, `store`) declared with loose types and a `description="Passed through; not validated."` on the Pydantic `Field`.

## Rejected alternatives

- **Reuse the canonical `ChatRequest` dataclass as the HTTP model.** Simplest, but `/docs` would show canonical fields rather than what LM Studio / OpenAI clients actually send. The whole payoff of a typed HTTP boundary is spec fidelity, which this option gives up.
- **One shared `ChatLikeRequest` union.** Matches the current `to_chat_request` translator's shape, but produces a `oneOf` in `/docs` that documents each endpoint as accepting all three shapes when in fact each endpoint has a canonical shape.
- **Schema for documentation only, keep `await request.json()` in handlers.** Doubles the surface area (models drift from handler behaviour) for no runtime benefit; Pydantic validation is free once the models exist.

## Consequences

- Bad inputs get rejected at the HTTP boundary with 422 and a clear field path instead of downstream 500s. This is a client-visible behaviour change.
- The `synthesize_stream` shortcut in `providers/openai_compatible.py` — which declares `Capability.STREAMING` while actually chunking a blocking response — becomes visibly at odds with the typed contract. Removing it (real upstream SSE or drop the capability) is a followup.
- `api/schemas.py` splits into a `schemas/` package. The existing `to_chat_request` function moves to `schemas/translate.py`.
- `required_capability_for_endpoint` (currently dead code in `capabilities.py`) can be wired into each typed handler as a decorator or dependency; not required by this ADR but unblocked by it.
