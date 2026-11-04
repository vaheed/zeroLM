#!/usr/bin/env python3
"""Print a compact, human-readable summary of ZeroLM's hand-written weights."""
from __future__ import annotations
import argparse
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "weights.zlm")
    args = parser.parse_args()
    intents = {}
    features = defaultdict(list)
    starts = {}
    nodes = {}
    for raw in args.model.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("INTENT "):
            _, idx, name = line.split(maxsplit=2)
            intents[int(idx)] = name
        elif line.startswith("FEATURE "):
            m = re.match(r"FEATURE (\d+) ([0-9.]+) (.+)$", line)
            if m:
                features[int(m.group(1))].append((m.group(3), float(m.group(2))))
        elif line.startswith("START "):
            _, idx, state = line.split()
            starts[int(idx)] = int(state)
        elif line.startswith("NODE "):
            m = re.match(r"NODE (\d+) (-?\d+) (\S+)$", line)
            if m:
                nodes[int(m.group(1))] = (int(m.group(2)), m.group(3))
    print(f"Model: {args.model}")
    print(f"Intents: {len(intents)} | features: {sum(map(len, features.values()))} | states: {len(nodes)}")
    print("\nIntent feature map:")
    for idx, name in sorted(intents.items()):
        terms = ", ".join(f"{phrase!r} ({weight:g})" for phrase, weight in features[idx])
        print(f"- {name}: {terms or '[fallback only]'}")
    print("\nDecoder paths:")
    for idx, name in sorted(intents.items()):
        state = starts[idx]
        tokens = []
        while state != -1:
            nxt, token = nodes[state]
            if token != "<eos>":
                tokens.append(token)
            state = nxt
        print(f"- {name}: {' '.join(tokens)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
