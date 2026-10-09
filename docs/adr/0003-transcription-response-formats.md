# ADR-0003: Transcription response formats are narrowed to `json` and `text`

- **Status**: Accepted
- **Date**: 2026-09-29
- **Supersedes**: none
- **Related**: [ADR-0001](0001-typed-http-boundary.md), [ADR-0002](0002-failure-driven-failover.md)

## Context

`POST /v1/audio/transcriptions` follows the OpenAI speech-to-text shape, whose
`response_format` parameter accepts six values: `json`, `text`, `srt`, `vtt`,
`verbose_json`, and `diarized_json`. The proxy has to answer a request that
names any of the six.

Two providers sit behind this route, and they disagree about what is cheap:

- The OpenAI-compatible upstream (omlx and friends) forwards `response_format`
  verbatim and returns whatever the chosen model produces. Emitting `srt` or
  `vtt` is a formatting problem for that upstream, not for us.
- The 1min API is not OpenAI-compatible. Transcribing is a two-hop call — upload
  to the Asset API, then invoke a `SPEECH_TO_TEXT` feature whose
  `resultObject[0]` is the transcript. It exposes no `response_format` knob at
  all, so every format other than plain text would have to be synthesised here
  from a single string: 1min gives us no segment boundaries, no speaker labels,
  and no word timings. An `srt` built by slicing one string into arbitrary
  chunks is not the caller's audio, it is a fabrication wearing a timestamp.

Meanwhile the driving client — voxtype's remote mode — asks for
`response_format=json` and reads only the `text` field. Nothing in the request
path needs the other four.

## Decision

Serve exactly two formats, `json` and `text`. Reject the other four with 400.

- `json` → `{"text": ...}`, plus `language` and `duration` when the upstream
  supplies them.
- `text` → the bare transcript as `text/plain`.
- `srt`, `vtt`, `verbose_json`, `diarized_json` → 400.
- `stream=true` → 400. No provider here streams.

Rejecting is the whole point. ADR-0001 established that a feature a provider
cannot serve is rejected at the boundary rather than silently dropped, and this
is that rule applied to a response format instead of a request feature. A 400
tells the caller its request is unsupported; a synthesised `srt` tells it the
proxy succeeded and lies about the timing.

## Consequences

- Clients asking for `srt` or `vtt` against the proxy get a clear 400 they can
  act on, rather than output that looks usable and is not.
- The narrow surface is cheap to keep honest: the proxy never has to invent
  segment or speaker data it does not have.
- Adding a format later means either teaching the 1min provider to produce
  structured output, or adding a real segmentation step. Both are real work
  with a real design, not a formatting flag.
- omlx can already serve `srt`/`vtt` well. Routing those requests around the
  1min provider is possible in future, but only with a way to say so at
  selection time that does not exist today.
