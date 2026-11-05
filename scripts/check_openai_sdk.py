#!/usr/bin/env python3
"""Check that the official OpenAI Python SDK works against `zerolm serve`.

Usage:  python3 scripts/check_openai_sdk.py HOST:PORT [API_KEY]

Needs `pip install openai`. Exits non-zero on the first failed expectation. This is the
real-world compatibility test: it exercises the SDK's own request building and response
parsing, which hand-written HTTP tests cannot.
"""
from __future__ import annotations

import sys

from openai import (AuthenticationError, BadRequestError, NotFoundError, OpenAI)


def check(cond: bool, what: str) -> None:
    if not cond:
        raise SystemExit(f"FAIL: {what}")
    print(f"ok   {what}")


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    addr = sys.argv[1]
    key = sys.argv[2] if len(sys.argv) > 2 else "not-needed"
    base = f"http://{addr}/v1"
    client = OpenAI(base_url=base, api_key=key, max_retries=0, timeout=10)

    ids = [m.id for m in client.models.list().data]
    check(ids == ["zerolm", "zerolm-live"], f"models.list -> {ids}")
    check(client.models.retrieve("zerolm-live").id == "zerolm-live", "models.retrieve")

    r = client.chat.completions.create(model="zerolm", messages=[{"role": "user", "content": "What license does it use?"}])
    check("MIT" in r.choices[0].message.content, "chat completion answers from the model")
    check(r.choices[0].finish_reason == "stop" and r.object == "chat.completion", "completion shape")
    check(r.usage.total_tokens == r.usage.prompt_tokens + r.usage.completion_tokens > 0, "usage adds up")
    check((r.model_extra or {}).get("zerolm", {}).get("intent") == "license", "zerolm.intent extension")

    first = client.chat.completions.create(model="zerolm", messages=[{"role": "user", "content": "does it use training"}])
    follow = client.chat.completions.create(model="zerolm", messages=[
        {"role": "system", "content": "You are a pirate."},
        {"role": "user", "content": "does it use training"},
        {"role": "assistant", "content": "arr"},
        {"role": "user", "content": "why?"}])
    check(follow.choices[0].message.content == first.choices[0].message.content, "messages array is the context")
    mem = client.chat.completions.create(model="zerolm", messages=[
        {"role": "user", "content": "remember my name is vaheeD"},
        {"role": "user", "content": "what is my name"}])
    check(mem.choices[0].message.content == "my name is vaheeD.", "in-request memory")

    plain = client.chat.completions.create(model="zerolm", messages=[{"role": "user", "content": "who are you"}])
    chunks = list(client.chat.completions.create(model="zerolm", stream=True, messages=[{"role": "user", "content": "who are you"}]))
    text = "".join(c.choices[0].delta.content or "" for c in chunks)
    check(text == plain.choices[0].message.content, "streamed chunks reassemble into the reply")
    check(chunks[-1].choices[0].finish_reason == "stop", "stream ends with finish_reason stop")
    cut = client.chat.completions.create(model="zerolm", max_tokens=3, messages=[{"role": "user", "content": "who are you"}])
    check(len(cut.choices[0].message.content.split()) == 3 and cut.choices[0].finish_reason == "length", "max_tokens truncates")

    tag = f"sdkcheck{abs(hash(addr)) % 100000}"
    client.chat.completions.create(model="zerolm-live", user=tag, messages=[{"role": "user", "content": "remember my cat is Tom"}])
    live = client.chat.completions.create(model="zerolm-live", user=tag, messages=[{"role": "user", "content": "what is my cat"}])
    check(live.choices[0].message.content == "my cat is Tom.", "zerolm-live keeps a session per user")

    def raises(exc, fn, what):
        try:
            fn()
        except exc:
            print(f"ok   {what}")
            return
        raise SystemExit(f"FAIL: {what} did not raise {exc.__name__}")

    raises(NotFoundError, lambda: client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": "hi"}]), "unknown model -> NotFoundError")
    raises(BadRequestError, lambda: client.chat.completions.create(
        model="zerolm", messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}]), "tools -> BadRequestError")
    if len(sys.argv) > 2:
        raises(AuthenticationError, lambda: OpenAI(base_url=base, api_key="wrong", max_retries=0).models.list(), "wrong key -> AuthenticationError")
    print("openai sdk compatibility check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
