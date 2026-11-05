#!/usr/bin/env python3
"""Typo robustness: corrupt one word per prompt and see how often the answer survives.

Takes the prompts a binary already routes correctly, damages the longest word of each (a swap of two
adjacent letters, a dropped letter, a doubled letter, or a substituted letter, seeded and deterministic),
and reports how many still route to the same intent. Run it with two binaries to compare, e.g. a build
without spell-snapping against the current one. Measurement only; it never changes weights.
"""
from __future__ import annotations
import argparse
import json
import random
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(path: Path):
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#") and "\t" in line:
            a, b = line.split("\t", 1)
            out.append((a.strip(), b.strip()))
    return out


def classify(binary: str, model: str, texts):
    p = subprocess.run([binary, "classify", "--model", model], input="\n".join(texts) + "\n",
                       capture_output=True, text=True, check=True, cwd=ROOT)
    return [json.loads(l)["intent"] for l in p.stdout.splitlines()]


def damage(text: str, rng: random.Random, kinds=("swap", "drop", "double", "sub")):
    words = text.split()
    cands = [i for i, w in enumerate(words) if sum(c.isalpha() for c in w) >= 6 and w.isalpha()]
    if not cands:
        return None
    i = max(cands, key=lambda k: (len(words[k]), -k))
    w = words[i]
    kind = rng.choice(kinds)
    pos = rng.randrange(1, len(w) - 1)  # keep the first letter: people rarely mistype it
    if kind == "swap":
        w = w[:pos] + w[pos + 1] + w[pos] + w[pos + 2:]
    elif kind == "drop":
        w = w[:pos] + w[pos + 1:]
    elif kind == "double":
        w = w[:pos] + w[pos] + w[pos:]
    else:
        w = w[:pos] + rng.choice([c for c in "abcdefghijklmnopqrstuvwxyz" if c != w[pos]]) + w[pos + 1:]
    words[i] = w
    return " ".join(words), kind


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(ROOT / "weights.zlm"))
    ap.add_argument("--data", type=Path, nargs="+", default=[ROOT / "tests/regression.tsv", ROOT / "tests/heldout_test.tsv"])
    ap.add_argument("--binary", nargs="+", required=True, help="one or more binaries to compare")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min-survival", type=float, help="fail if the LAST binary keeps less than this fraction")
    args = ap.parse_args()

    labelled = [r for d in args.data for r in rows(d) if r[0] != "unknown"]
    results = {}
    for b in args.binary:
        base = classify(b, args.model, [t for _, t in labelled])
        keep = [(l, t) for (l, t), got in zip(labelled, base) if got == l]
        rng = random.Random(args.seed)
        bad = [(l, t, damage(t, rng)) for l, t in keep]
        bad = [(l, t, d) for l, t, d in bad if d]
        got = classify(b, args.model, [d[0] for _, _, d in bad])
        ok = sum(g == l for g, (l, _, _) in zip(got, bad))
        results[b] = (ok, len(bad))
        by = {}
        for g, (l, _, d) in zip(got, bad):
            k = by.setdefault(d[1], [0, 0]); k[1] += 1; k[0] += g == l
        print(f"{b}\n  correct prompts damaged: {len(bad)}   still correct: {ok} = {ok / len(bad):.1%}")
        print("  by damage: " + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(by.items())))
    ok, n = results[args.binary[-1]]
    if args.min_survival is not None and ok / n < args.min_survival:
        print(f"FAIL: survival {ok / n:.1%} < {args.min_survival:.1%}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
