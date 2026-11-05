#!/usr/bin/env python3
"""Compile a hand-authored model specification into the ZeroLM weight file.

This is a serializer, not a training program. It performs no fitting,
optimization, gradient calculation, or parameter search.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPEC_PATH = ROOT / "model" / "spec.json"
WEIGHTS_PATH = ROOT / "weights.zlm"


def load_spec(path: Path = SPEC_PATH) -> dict:
    spec = json.loads(path.read_text(encoding="utf-8"))
    intents = spec.get("intents")
    if not isinstance(intents, list) or not intents:
        raise SystemExit("spec must contain a non-empty intents list")
    if len(intents) > 32:
        raise SystemExit("the C runtime supports at most 32 intents")
    names = [item.get("name") for item in intents]
    if len(set(names)) != len(names) or any(not isinstance(x, str) for x in names):
        raise SystemExit("intent names must be unique strings")
    if names[-1] != "unknown":
        raise SystemExit("the final intent must be named 'unknown'")
    feature_count = 0
    for item in intents:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", item["name"]):
            raise SystemExit(f"invalid intent name: {item['name']}")
        response = item.get("response", "")
        if not isinstance(response, str) or not response.strip():
            raise SystemExit(f"intent {item['name']} has an empty response")
        if "\n" in response or "\r" in response:
            raise SystemExit(f"intent {item['name']} response must be a single line")
        for feature in item.get("features", []):
            if not (isinstance(feature, list) and len(feature) == 2):
                raise SystemExit(f"invalid feature in intent {item['name']}")
            phrase, weight = feature
            if (not isinstance(phrase, str) or not phrase.strip() or len(phrase) > 180
                    or any(c in phrase for c in "\r\n")):
                raise SystemExit(f"invalid feature phrase in intent {item['name']}")
            if not isinstance(weight, (int, float)) or not (0.1 <= float(weight) <= 20.0):
                raise SystemExit(f"feature weight out of range for phrase: {phrase}")
            feature_count += 1
    if feature_count > 512:
        raise SystemExit("the C runtime supports at most 512 features")
    return spec


def compile_spec(spec: dict) -> str:
    intents = spec["intents"]
    intent_ids = {item["name"]: idx for idx, item in enumerate(intents)}
    node_rows: list[tuple[int, int, str]] = []
    starts: list[tuple[int, int]] = []
    next_id = 0

    for intent_id, item in enumerate(intents):
        starts.append((intent_id, next_id))
        tokens = item["response"].split()
        for token in tokens:
            if token == "<eos>" or any(c.isspace() for c in token):
                raise SystemExit("response contains a reserved or invalid token")
            node_rows.append((next_id, next_id + 1, token))
            next_id += 1
        node_rows.append((next_id, -1, "<eos>"))
        next_id += 1

    if next_id > 2048:
        raise SystemExit("the C runtime supports at most 2048 recurrent states")

    lines = [
        "# ZeroLM v0.2.0 - hand-authored weights; no training was performed.",
        "# FEATURE: phrase-to-intent input projection (phrase matching is a fixed feature extractor).",
        "# START: intent-to-hidden initialization. NODE: recurrent transition and output token mapping.",
        "# Implicit inactive matrix values are -8.0; declared active connections are +8.0.",
        "ZEROLM 1",
        f"META {len(intents)} {next_id}",
    ]
    for intent_id, item in enumerate(intents):
        lines.append(f"INTENT {intent_id} {item['name']}")
    for item in intents:
        intent_id = intent_ids[item["name"]]
        for phrase, weight in item.get("features", []):
            lines.append(f"FEATURE {intent_id} {float(weight):.4f} {phrase}")
    for intent_id, state_id in starts:
        lines.append(f"START {intent_id} {state_id}")
    for state_id, next_state, token in node_rows:
        lines.append(f"NODE {state_id} {next_state} {token}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Serialize the manually authored ZeroLM model.")
    parser.add_argument("--check", action="store_true", help="fail if weights.zlm is not up to date")
    parser.add_argument("--spec", type=Path, default=SPEC_PATH, help="spec to compile (default model/spec.json)")
    parser.add_argument("--out", type=Path, default=WEIGHTS_PATH, help="weight file to write or check (default weights.zlm)")
    args = parser.parse_args()
    weights_path = args.out
    spec = load_spec(args.spec)
    content = compile_spec(spec)
    states = sum(len(item["response"].split()) + 1 for item in spec["intents"])
    features = sum(len(item.get("features", [])) for item in spec["intents"])
    if args.check:
        if not weights_path.exists() or weights_path.read_text(encoding="utf-8") != content:
            print(f"{weights_path} is stale. Rebuild it from the spec.")
            return 1
        print(f"Generated weights are up to date: {len(spec['intents'])} intents, {features} features, {states} states.")
    else:
        weights_path.write_text(content, encoding="utf-8")
        print(f"Wrote {weights_path}: {len(spec['intents'])} intents, {features} features, {states} recurrent states.")
    print("Mode: manual serialization only; no training or optimization.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
