# LM Studio Conformance

The proxy's `lm-studio` route is a transparent passthrough: it forwards the
native LM Studio REST API and preserves bodies, headers, statuses, and streams.
A conformance suite verifies this wiring against a real LM Studio server and,
offline, against a mocked upstream and captured fixtures.

## Reference version

The suite is validated against **LM Studio 0.4.23** (see
`docs/routing.md` / `AGENTS.md`). Protocol assertions are written for shape and
presence — never exact generated wording — so they tolerate patch-level
server changes.

## Running the suite

The live suite is **opt-in**: a default `pytest` run skips it entirely. Enable
it by pointing the `--live-lms` flag at a running LM Studio server:

```bash
uv run pytest --live-lms http://127.0.0.1:8999 tests/live/test_live_lms_conformance.py
```

The `--live-lms` option is registered globally and the `live` marker skips the
suite when the flag is absent:

```bash
uv run pytest          # live tests skipped
uv run pytest --live-lms http://127.0.0.1:8999   # live tests run
```

## What it asserts

| Endpoint / case | Assertion |
| --- | --- |
| `POST /api/v1/chat` (unknown model) | Native error envelope `{"error": {"message", "type"}}` and a `4xx` status |
| `POST /api/v1/chat` (invalid JSON) | Same native error envelope; rejected locally before forwarding |
| `POST /api/v1/chat` (`stream: true`) | Native SSE lifecycle ordering: `chat.start` → `message.delta*` → `chat.end` |
| `GET /api/v1/models` | Byte-equivalence through the proxy vs direct upstream call |

## Fixtures and offline coverage

When the live suite runs it writes sanitized fixtures to
`tests/fixtures/live_lms/`:

- `models.json` — the model catalog in canonical JSON form.

`tests/live/test_live_lms_conformance_offline.py` runs on every `pytest`
invocation with no server: it replays the committed/fixture bodies through the
proxy and asserts the same byte-equivalence and protocol shapes, in addition to
mocked-upstream coverage of all three protocol seams. If the fixture directory
is empty, the fixture-replay test skips.

The committed `models.json` is a reference sample; **regenerate it** against a
real server whenever you bump the pinned reference version:

```bash
uv run pytest --live-lms http://127.0.0.1:8999 \
  tests/live/test_live_lms_conformance.py::test_live_models_byte_equivalent
```

## Shared test helper

`tests/live/_mock.py` centralizes the mocked upstream (routed via `respx`), the
native SSE/error-envelope builders, and the SSE parser shared by the live and
offline suites. Keep protocol shape definitions here so both suites cannot
drift.
