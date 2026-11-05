#!/usr/bin/env python3
"""Targeted defect suites found by the hand-written adversarial cases (all expect rejection).

  defects-003: 'that' after a connector ('A, after that B'): not in the grammar; was silently mis-ordered.
  defects-004: a connector with no clause after it ('A. after', 'A; then').
Seeded; regenerate with:  python3 research/benchmark/defects.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import Profile, build, clause_text  # noqa: E402
from scenarios import Rng  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "tests" / "next"


def main():
    rng, prof = Rng(777), Profile("final")
    d3, d4 = [], []
    for i in range(150):
        b = build(rng, prof, 2, structure=[("seq", [0]), ("seq", [1])])
        ops = [dict(o, negated=False) for o in b["ops"]]
        for o in ops:
            o.pop("condition", None)
        a, c = (clause_text(prof, ops[0], b["styles"][0]), clause_text(prof, ops[1], b["styles"][1]))
        form = rng.below(4)
        text = [f"{a}, after that {c}", f"{a}, before that {c}", f"{a}. then, after that, {c}", f"{a} and after that {c}"][form]
        d3.append({"id": f"d003-{i:04d}", "category": "defect-003", "text": text, "expect": "reject", "gold_ops": [], "world": None})
        conn = ["after", "before", "then", "and", "next"][rng.below(5)]
        sep = [". ", "; ", ". ", "; ; "][rng.below(4)]
        text4 = f"{a}{sep}{conn}" + ("." if rng.chance(0.3) else "")
        d4.append({"id": f"d004-{i:04d}", "category": "defect-004", "text": text4, "expect": "reject", "gold_ops": [], "world": None})
    for name, cases in (("defects-003.jsonl", d3), ("defects-004.jsonl", d4)):
        (OUT / name).write_text("\n".join(json.dumps(c, sort_keys=True) for c in cases) + "\n")
        print(name, len(cases))


main()
