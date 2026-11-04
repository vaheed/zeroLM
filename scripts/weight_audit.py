#!/usr/bin/env python3
"""Audit the generated ZeroLM weight graph without executing training."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def audit(path: Path):
    header = False
    intents = {}
    features = []
    starts = {}
    nodes = {}
    meta = None
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not header:
            if line != "ZEROLM 1":
                raise ValueError(f"line {line_no}: invalid model header")
            header = True
        elif line.startswith("META "):
            _, a, b = line.split()
            meta = (int(a), int(b))
        elif line.startswith("INTENT "):
            _, idx, name = line.split(maxsplit=2)
            intents[int(idx)] = name
        elif line.startswith("FEATURE "):
            match = re.match(r"FEATURE (\d+) ([0-9.]+) (.+)$", line)
            if not match:
                raise ValueError(f"line {line_no}: invalid feature")
            features.append((int(match.group(1)), float(match.group(2)), match.group(3)))
        elif line.startswith("START "):
            _, intent, state = line.split()
            starts[int(intent)] = int(state)
        elif line.startswith("NODE "):
            match = re.match(r"NODE (\d+) (-?\d+) (\S+)$", line)
            if not match:
                raise ValueError(f"line {line_no}: invalid node")
            nodes[int(match.group(1))] = (int(match.group(2)), match.group(3))
        else:
            raise ValueError(f"line {line_no}: unknown record")

    if not header or meta is None:
        raise ValueError("missing header or META")
    intent_count, state_count = meta
    if len(intents) != intent_count or len(starts) != intent_count:
        raise ValueError("intent/start count does not match META")
    if len(nodes) != state_count or set(nodes) != set(range(state_count)):
        raise ValueError("node IDs must cover the complete state range")
    if any(i not in intents for i, _, _ in features):
        raise ValueError("feature points to an unknown intent")
    if any(state < 0 or state >= state_count for state in starts.values()):
        raise ValueError("start points outside the state range")
    for state, (next_state, token) in nodes.items():
        if next_state != -1 and not (0 <= next_state < state_count):
            raise ValueError(f"node {state} points outside the state range")
        if next_state != -1 and next_state <= state:
            raise ValueError(f"node {state} is not a forward-only transition")
        if not token:
            raise ValueError(f"node {state} has an empty token")
    for intent_id, start in starts.items():
        state = start
        seen = set()
        while state != -1:
            if state in seen:
                raise ValueError(f"cycle found in intent {intent_id}")
            seen.add(state)
            state = nodes[state][0]
        if not seen:
            raise ValueError(f"intent {intent_id} has an empty path")
    vocab = len({token for _, token in nodes.values()})
    dense_slots = intent_count * len(features) + state_count * intent_count + state_count**2 + state_count * vocab
    return {
        "intents": intent_count,
        "features": len(features),
        "states": state_count,
        "vocabulary": vocab,
        "dense_equivalent_slots": dense_slots,
        "weight_file_bytes": path.stat().st_size,
        "unique_intent_names": sorted(intents.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "weights.zlm")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = audit(args.model)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("ZeroLM weight audit: PASS")
        for key in ("intents", "features", "states", "vocabulary", "dense_equivalent_slots", "weight_file_bytes"):
            print(f"{key}: {result[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
