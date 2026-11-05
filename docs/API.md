# ZeroLM HTTP API

`zerolm serve` lets other programs talk to ZeroLM over HTTP. It has two faces:

- an **OpenAI-compatible** API (`/v1/chat/completions`, `/v1/models`), so existing OpenAI SDKs and
  tools work by changing the base URL;
- a **native** API (`/v1/ask`, `/v1/sessions/{tag}/...`) for sessions with memory.

The contract is [`rust/openapi.yaml`](../rust/openapi.yaml) (OpenAPI 3.0, validated in CI). A
running server also serves it at `GET /v1/openapi.yaml`.

> ZeroLM is a small hand-programmed phrase matcher with canned answers, not a general language
> model. The API makes it easy to call; it does not make it smarter. Unrecognised input gets the
> `unknown` answer.

## Start it

```bash
cd rust && cargo build --release && cd ..
rust/target/release/zerolm serve                       # http://127.0.0.1:8080, no token needed on loopback
ZEROLM_TOKEN=a-long-secret-token rust/target/release/zerolm serve --addr 0.0.0.0:8080
```

Or with Docker (loopback port 8080 on the host, token required):

```bash
ZEROLM_TOKEN=a-long-secret-token docker compose up -d zerolm-api
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--addr HOST:PORT` | `127.0.0.1:8080` | Listen address. Port `0` picks a free port; the chosen one is printed as `listening on http://...`. |
| `--token T` / `$ZEROLM_TOKEN` | none | Bearer token. Prefer the environment variable (arguments are visible in `ps`). |
| `--dir DIR` | `.` | Where `life.<tag>.zlm` session files live. |
| `--model FILE` / `$ZEROLM_MODEL` | `weights.zlm` | Model file. |
| `--capacity N`, `--context BYTES` | 64, 16384 | Per-session fact capacity and transcript budget. |
| `--request-timeout-ms N` | 10000 | Total time a client has to send a complete request. |
| `--max-connections N`, `--max-sessions N` | 64, 256 | Concurrent connections and open sessions. |

**Safe by default.** The server refuses to start on a non-loopback address unless it has a token of
at least 16 characters. There is no TLS: to expose it beyond one machine, put a reverse proxy with
TLS in front. There are no CORS headers, so browser pages on other origins cannot call it directly.

## Authentication

When a token is set, send `Authorization: Bearer <token>` on every request except
`GET /v1/health` (left open for probes and load balancers). OpenAI SDKs send their `api_key` this way,
so use the token as the key. Without a token (loopback only) any key, or none, works.

## OpenAI-compatible API

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="your-token")

r = client.chat.completions.create(model="zerolm", messages=[
    {"role": "user", "content": "does it use training"},
    {"role": "user", "content": "why?"},          # follow-up: the messages are the context
])
print(r.choices[0].message.content)
```

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Authorization: Bearer your-token" \
  -d '{"model":"zerolm","messages":[{"role":"user","content":"what license does it use"}]}'
```

### Models

| `model` | Behaviour |
| --- | --- |
| `zerolm` | **Stateless.** The `messages` array is the whole context. Earlier `user` messages are replayed through a throwaway session, so follow-ups (`why?`) and `remember X is Y` followed by `what is X` work within one request. Nothing is stored. |
| `zerolm-live` | **Persistent.** Only the last user message is run, in the ZeroLM Live session named by the request's `user` field (default `default`; 1-32 letters, digits, `-`, `_`). Memory, taught phrases and the transcript persist, exactly as in the CLI. |

### What works and what does not

| OpenAI feature | Support |
| --- | --- |
| `messages` with `user` / `assistant` / `system` / `developer` | Yes, but only `user` messages affect the answer. ZeroLM cannot follow instructions, so system prompts and assistant messages are ignored. The last message must be a non-empty `user` message (at most 4096 bytes, at most 100 messages). |
| `content` as a string or `[{"type":"text","text":...}]` parts | Yes. Other part types (images, audio) get 400. |
| `stream: true` | Yes: server-sent events with one word per chunk, ending `data: [DONE]`. The body is sent in one write, not paced. |
| `max_tokens`, `max_completion_tokens` | Yes, counted in **words**; the reply is cut and `finish_reason` is `length`. |
| `usage` | Counts of whitespace-separated **words**, not model tokens. |
| `temperature`, `top_p`, `seed`, `stop`, penalties, `logit_bias`, ... | Accepted and ignored: answers are deterministic. |
| `tools`, `functions`, `function_call`, `tool_calls`, `tool_choice` other than `none`, `audio` | **Rejected with 400.** Ignoring them silently would mislead agent frameworks. |
| `n` other than 1, `response_format` other than `text` | Rejected with 400. |
| `GET /v1/models`, `GET /v1/models/{id}` | Yes: `zerolm`, `zerolm-live`. |
| Embeddings, images, audio, files, assistants, legacy `/v1/completions` | Not implemented (404). |

Errors on these routes use OpenAI's envelope, `{"error": {"message", "type", "param", "code"}}`, so
SDKs raise their usual typed exceptions (`AuthenticationError`, `NotFoundError`, `BadRequestError`).

One extension: the completion has `"zerolm": {"intent": "license"}` (the classified intent, `null` for
commands and memory lookups, plus `"session"` for `zerolm-live`). Use it to treat ZeroLM as an
intent router.

## Native API

| Method and path | Purpose |
| --- | --- |
| `GET /v1/health` | `{"status":"ok","version":...}`. Open. |
| `GET /v1/model` | Intent names, feature/state/vocabulary counts. |
| `GET /v1/openapi.yaml` | This server's OpenAPI document. |
| `POST /v1/ask` `{"text"}` | One stateless answer: `{"reply","intent","score"}`. |
| `POST /v1/classify` `{"text"}` or `{"texts":[...]}` | Intent routing with ranked candidates, no reply text; optional `top` (1-10) and `session` (apply that session's learned reflexes). See [ROUTER.md](ROUTER.md). |
| `POST /v1/sessions/{tag}/turns` `{"text"}` | One message in a ZeroLM Live session (created on first use): `{"reply","intent","age","memories"}`. Handles `remember`, `forget`, `teach`, `good`/`bad`, `again`, follow-ups and recall. |
| `GET /v1/sessions/{tag}` | `{"tag","age","memories","learned","capacity"}`; 404 if the session never existed. |
| `GET /v1/sessions/{tag}/memories` | Stored facts, strongest first. |
| `DELETE /v1/sessions/{tag}/memories/{key}` | Forget one fact (key percent-encoded); 204 or 404. |
| `POST /v1/sessions/{tag}/teach` `{"phrase","intent"}` | Add a learned phrase for an intent from `GET /v1/model`. |
| `DELETE /v1/sessions/{tag}` | Delete the session and its life file; 204. |

```bash
curl -H "Authorization: Bearer $T" -d '{"text":"remember my name is Vahid"}' localhost:8080/v1/sessions/me/turns
curl -H "Authorization: Bearer $T" -d '{"text":"what is my name"}'           localhost:8080/v1/sessions/me/turns
```

Native errors are `{"error": "<message>"}`. Status codes: 400 bad request or tag, 401 bad token,
404 unknown endpoint/session/memory, 405 wrong method (with `Allow`), 408 request too slow, 409 the
session is in use by another ZeroLM process, 413 body over 64 KiB, 431 headers too large, 501
chunked bodies, 503 too many connections or sessions (with `Retry-After` for connections).

## Sessions, tags and locking

Each tag is an independent ZeroLM Live session stored in `life.<tag>.zlm`. The server holds the same
kernel file lock as the CLI, so **a tag can be used by only one process at a time**: while the server
has a session open, `zerolm live --tag <tag>` (or another container on the same volume) is refused,
and while the CLI holds a tag the API answers 409 for it. The lock is released when the server stops
or is killed. The base model file is never written. See [LIVE.md](LIVE.md).

## Limits and hardening

One request per connection (`Connection: close`); bodies must use `Content-Length`. Headers up to 8 KiB
and 64 fields; bodies up to 64 KiB; JSON nesting up to 8; text up to 4096 bytes. A client that is slow to
send a complete request is cut off after the total request timeout, however slowly it trickles bytes.
Tags cannot contain path characters. Error messages do not include server paths. Bearer tokens are compared
in constant time. Request bodies are never logged.

This is a small standard-library server meant for local use, tools and containers behind a proxy. It
is not a hardened internet-facing server: terminate TLS and apply rate limits in front of it.

## Checking compatibility

- `python3 scripts/check_openai_sdk.py HOST:PORT [KEY]` runs 15 checks with the official OpenAI Python
  SDK (`pip install openai`).
- `openapi-spec-validator rust/openapi.yaml` validates the spec (`pip install openapi-spec-validator`).
- `cargo test --test api` runs the 20 end-to-end API tests.

All three run in CI.
